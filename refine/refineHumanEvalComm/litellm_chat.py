"""
Synchronous LiteLLM runner: a chat-completions fallback for when the batch API is unavailable.

The rest of the pipeline is unchanged. This module runs the same request files build_batch /
refine_descriptions produce ({custom_id, body} per line), sends one chat-completions call per
request to the LiteLLM proxy, and writes a result file in the same shape the batch post-processors
read: {custom_id, response: {status_code, body}} per line. Because a request file
"{stem}_request.jsonl" maps to "{stem}_result.jsonl", results land exactly where postprocess_baseline
/ postprocess_questions / postprocess_descriptions / postprocess_oracle / postprocess_refined_candidates
already look, so scoring and aggregation need no changes. Only the submission step differs.

chat_completion retries transient failures (timeouts, connection errors, 429, 5xx) with
exponential back-off, and run_request_file never crashes on one bad request: a request that
still fails after retries is left unwritten and the run continues, then a later pass (or a
re-run, since it is resumable) retries it. Concurrency is a later step.

Requires LITELLM_API_KEY. The proxy base URL is common.API_BASE_LL.
"""
import json
import os
import sys
import random
import time
import threading

import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import API_BASE_LL

CHAT_URL = API_BASE_LL.rstrip("/") + "/v1/chat/completions"
TIMEOUT         = 120    # read timeout per request (s); long enough for a real generation to finish
CONNECT_TIMEOUT = 10     # give up quickly if the connection won't even open

# Roster slug -> the proxy's public model name. The CISPA proxy routes through OpenRouter, so the
# default drops a ":batch" suffix and prefixes "openrouter/" (e.g. the working curl used
# "openrouter/qwen/qwen3.8-27b"). Add an entry for any model whose proxy name differs.
LITELLM_MODEL_OVERRIDES = {}


def litellm_model(slug: str):
    """Maps a roster slug (e.g. "openai/gpt-5.6-luna:batch") to the proxy model name."""
    return slug[:-len(":batch")] if slug.endswith(":batch") else slug
    return LITELLM_MODEL_OVERRIDES.get(slug,
           LITELLM_MODEL_OVERRIDES.get(plain, "openrouter/" + plain))


def _api_key():
    key = os.environ.get("LITELLM_API_KEY")
    if not key:
        raise RuntimeError("Set LITELLM_API_KEY before running.")
    return key


_local = threading.local()


def _session():
    """One requests.Session per worker thread, so connections are kept alive and reused (no fresh TLS
    handshake per request). Thread-local avoids any cross-thread sharing concern."""
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        _local.session = s
    return s


TRANSIENT_STATUS = {408, 409, 425, 429}   # plus any 5xx, plus 0 (no HTTP response: timeout/connection)


def _is_transient(status: int):
    """A status worth retrying: no response (0), a few 4xx, or any server 5xx."""
    return status == 0 or status in TRANSIENT_STATUS or status >= 500


class _RateLimiter:
    """Paces request *starts* to at most `rpm` per minute across all worker threads, so a fast model
    can't exceed the proxy's requests-per-minute limit no matter how quickly responses come back.
    Spaces consecutive starts by 60/rpm seconds (a smooth leaky bucket). rpm None/0 disables it."""
    def __init__(self, rpm):
        self.interval = 60.0 / rpm if rpm else 0.0
        self.lock = threading.Lock()
        self.next_time = 0.0

    def acquire(self):
        if self.interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_time)
            self.next_time = start + self.interval
            wait = start - now
        if wait > 0:
            time.sleep(wait)


def chat_completion(body: dict, timeout: int = TIMEOUT, max_retries: int = 4, backoff: float = 2.0,
                    max_tokens: int = None, connect_timeout: int = CONNECT_TIMEOUT,
                    keep_alive: bool = True, rate_limiter=None):
    """
    Sends one chat-completions request over a kept-alive connection, retrying transient failures
    (timeouts, connection errors, 429, 5xx) up to max_retries with exponential back-off. The model
    slug is mapped to the proxy name; max_tokens, if given, caps the completion length (shorter
    generations, fewer read timeouts). timeout is the read timeout; connect_timeout bounds the
    connection setup so a connection that will not open is abandoned fast. keep_alive=False opens a
    fresh connection per request (Connection: close) instead of reusing a pooled one, to rule out
    stale-socket read timeouts on a flaky proxy.

    Returns (status_code, json). 200 is success; a real 4xx is a permanent error returned at once;
    status_code 0 means no response (timeout/connection) after all retries.
    """
    payload = dict(body)
    payload["model"] = litellm_model(payload["model"])
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    if not keep_alive:
        headers["Connection"] = "close"          # fresh connection each request (avoid pooled/stale sockets)
    client = _session() if keep_alive else requests
    last = (0, {"error": {"message": "no response"}})
    for attempt in range(max_retries + 1):
        retry_after = None
        if rate_limiter is not None:
            rate_limiter.acquire()                       # stay under the proxy's requests-per-minute limit
        try:
            resp = client.post(CHAT_URL, json=payload, headers=headers,
                               timeout=(connect_timeout, timeout))
            if resp.status_code == 200:
                return 200, resp.json()
            try:
                detail = resp.json()
            except Exception:
                detail = {"error": {"message": resp.text[:500]}}
            if not _is_transient(resp.status_code):
                return resp.status_code, detail          # permanent (e.g. 400/401/404): do not retry
            retry_after = resp.headers.get("Retry-After")  # server-requested wait (429/503)
            last = (resp.status_code, detail)
        except requests.exceptions.RequestException as e:
            last = (0, {"error": {"message": f"{type(e).__name__}: {e}"}})
        if attempt < max_retries:
            try:
                ra = float(retry_after) if retry_after is not None else None
            except (TypeError, ValueError):
                ra = None
            wait = (ra if ra is not None else backoff * (2 ** attempt)) + random.uniform(0, 1)
            err = last[1].get("error", {})
            reason = err.get("message", "") if isinstance(err, dict) else str(err)
            print(f"  transient (status {last[0]}: {reason[:70]}), retry {attempt + 1}/{max_retries} in {wait:.0f}s")
            time.sleep(wait)
    return last

def _result_path_for(request_path: Path):
    """The result file the post-processors read (…_request.jsonl -> …_result.jsonl)."""
    name = request_path.name.replace("_request.jsonl", "_result.jsonl")
    return request_path.parent / (name if name != request_path.name else request_path.stem + "_result.jsonl")


def _finish_reason(body):
    """The completion's finish_reason, or None. 'length' means the output was cut off by max_tokens."""
    try:
        return body["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        return None


def run_request_file(request_path, result_path=None, timeout: int = TIMEOUT,
                     max_retries: int = 4, backoff: float = 2.0, max_passes: int = 3,
                     max_workers: int = 8, max_tokens: int = None, keep_alive: bool = True,
                     drop_truncated: bool = True, max_requests_per_min: int = None):
    """
    Runs every request in request_path against the LiteLLM chat endpoint and writes a batch-shaped
    result file. Requests within a pass run concurrently in a thread pool of max_workers (the calls
    are network-bound, so threads parallelise them); results are written in the main thread as each
    completes, so the file has a single writer and stays safe and resumable.

    It does not crash on a bad request: transient failures are retried inside chat_completion, and any
    request still failing is left unwritten so the rest keep going; up to max_passes sweeps then retry
    whatever is still unfinished. Resumable: custom_ids already in the result file are skipped, so
    re-running (or restarting after a cancel) continues where it left off. Permanent errors (a real
    4xx) are written as failed lines; genuine transient stragglers are left for a later pass/run. With
    drop_truncated (default True), a 200 response cut off by length (finish_reason='length') is not
    written and not retried this run, so truncated candidates never reach the result file; re-run with
    a higher or unset max_tokens to regenerate them.
    Returns the result path.
    """
    request_path = Path(request_path)
    result_path  = Path(result_path) if result_path else _result_path_for(request_path)
    requests = [json.loads(l) for l in open(request_path, encoding="utf-8") if l.strip()]
    limiter  = _RateLimiter(max_requests_per_min)   # shared across workers; caps requests/min

    def done_ids():
        if not result_path.exists():
            return set()
        with open(result_path, encoding="utf-8") as f:
            return {json.loads(l)["custom_id"] for l in f if l.strip()}

    truncated = set()          # 200 responses cut off by length: gated out, not retried this run
    for sweep in range(1, max_passes + 1):
        done = done_ids()
        todo = [r for r in requests if r["custom_id"] not in done and r["custom_id"] not in truncated]
        if not todo:
            break
        print(f"{request_path.name}: pass {sweep}/{max_passes}, {len(todo)} to run "
              f"({len(done)} done, {max_workers} workers) -> {result_path}")
        ok = failed = left = trunc = seen = 0
        with open(result_path, "a", encoding="utf-8") as out:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(chat_completion, r["body"], timeout, max_retries, backoff,
                                       max_tokens, keep_alive=keep_alive, rate_limiter=limiter): r
                           for r in todo}
                for fut in as_completed(futures):
                    r = futures[fut]
                    try:
                        status, body = fut.result()
                    except Exception as e:                     # never let one request kill the pass
                        status, body = 0, {"error": {"message": f"{type(e).__name__}: {e}"}}
                    if status == 200 and drop_truncated and _finish_reason(body) == "length":
                        truncated.add(r["custom_id"]); trunc += 1   # cut off by max_tokens: drop, don't retry
                    elif status == 200:
                        out.write(json.dumps({"custom_id": r["custom_id"],
                                              "response": {"status_code": status, "body": body}}) + "\n")
                        out.flush(); ok += 1
                    elif _is_transient(status):
                        left += 1                              # leave unwritten -> retried next pass/run
                    else:
                        out.write(json.dumps({"custom_id": r["custom_id"],
                                              "response": {"status_code": status, "body": body},
                                              "error": body}) + "\n")
                        out.flush(); failed += 1
                    seen += 1
                    if seen % 25 == 0:
                        print(f"  {seen}/{len(todo)} done ({ok} ok, {failed} failed, {trunc} truncated, {left} left)")
        print(f"{request_path.name}: pass {sweep} done ({ok} ok, {failed} permanent-failed, "
              f"{trunc} truncated (raise max_tokens to regenerate), {left} left for retry)")
        if left == 0:
            break
        if ok == 0 and sweep < max_passes:                     # a whole pass made no headway: wait longer
            wait = backoff * (2 ** sweep)
            print(f"  no progress this pass; waiting {wait:.0f}s before pass {sweep + 1}")
            time.sleep(wait)

    remaining = len([r for r in requests if r["custom_id"] not in done_ids()])
    note = ""
    if remaining:
        note = " (re-run to retry them"
        note += f"; {len(truncated)} were cut off by length, raise/unset max_tokens)" if truncated else ")"
    print(f"{request_path.name}: finished, {remaining} still unfinished" + note)
    return result_path

if __name__ == "__main__":
    import common

    # --- One request file -> its result file (then post-process as usual) ---
    # req = common.EXPERIMENT / "LLM1" / "1c" / "baseline_batch_request.jsonl"
    # run_request_file(req)                         # writes baseline_batch_result.jsonl next to it
    # then in batch_processing.py: postprocess_baseline("LLM1", ["1c"])

    pass
