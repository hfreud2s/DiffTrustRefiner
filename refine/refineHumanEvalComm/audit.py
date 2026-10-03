"""
LLM auditor for the HumanEvalComm refinement pipeline.

A single fixed auditor LLM, kept separate from the coder LLMs and from the oracle (to avoid
self-evaluation bias): 
    flag oracle answers that leak information beyond what answers the
    question and suggest a tightened rewrite for a human to review.

Auditing runs through the OpenRouter Batch API like every other step: build a request file, submit
it with batch_processing.create_batch, then post-process the results back into
questions_and_descriptions.json.
"""
import json
import re
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import EXPERIMENT, CATEGORIES, DATASET_NAME, AUDITOR_MODEL, load_instances, spec_for, provider_obj



def generate_audit_oracle_prompt(manipulated: str, question: str, oracle_description: str):
    """
    Prompt asking the auditor whether the oracle's short clarification (its answer to ONE question)
    leaks information beyond that answer. The auditor sees ONLY the ambiguous task and the question,
    never the ground truth, so it judges leakage, not correctness. Downstream the clarification is
    appended verbatim to the original task, so it should add ONLY this question's answer. When it
    leaks, the auditor supplies a tightened clarification (still a short neutral answer) for a human
    to review; it never edits anything itself.
    """
    return (
        "You are auditing a short clarification that a reference ('oracle') wrote to answer one "
        "yes/no question about an under-specified Python task. Downstream this clarification is "
        "appended verbatim to the original task, so it must add only the answer to this question.\n\n"
        f"The task (as the coder saw it, including any examples it contains):\n{manipulated}\n\n"
        f"The one question this clarification answers:\n{question}\n\n"
        f"The oracle's clarification:\n{oracle_description}\n\n"
        "The clarification may contain ONLY the answer to THIS question, stated as neutral behavior. "
        "Anything else is leakage:\n"
        "- resolving a different open point of the task, or adding a constraint the task did not "
        "state (input lengths, non-emptiness, types, ordering, edge cases)\n"
        "- revealing or restating the reference solution, or quoting or introducing any held-out "
        "test case or specific input/output values.\n"
        "NOT leakage: stating the behavior this question resolves, even when that behavior matches "
        "what the task's own examples already show. Those examples are part of the task the coder "
        "already sees, so relying on them is fine. Judge by meaning.\n\n"
        "If it leaks, name the specific leaked content and give a tightened clarification: a short "
        "neutral answer to THIS question only, one or two sentences, with everything else removed. "
        "Do NOT turn it into a full task description.\n\n"
        "Output exactly three lines, nothing else:\n"
        "leak: YES or NO\n"
        "leaked: the specific leaked information, or NONE\n"
        "rewrite: the tightened clarification, or NONE\n"
    )


def build_audit_oracle_batch(llm_dir: str, category: str, model: str = AUDITOR_MODEL,
                             dataset_name: str = DATASET_NAME, provider=None,
                             temperature: float = 0.0):
    """
    Writes the batch that asks the auditor to leak-check every filled oracle_description in
    {category}/refined/questions_and_descriptions.json (one request per (task, question) that has an
    oracle_description, including q_auditor). custom_id
    "audit_oracle__humanevalcomm_{task_id}__{qkey}"; output
    {category}/refined/audit_oracle_batch_request.jsonl. Submit with batch_processing.create_batch;
    post-process with postprocess_audit_oracle.
    """
    variant     = CATEGORIES[category]
    refined_dir = EXPERIMENT / llm_dir / category / "refined"
    with open(refined_dir / "questions_and_descriptions.json", "r", encoding="utf-8") as f:
        entries = json.load(f)
    instances = load_instances(dataset_name, {e["task_id"] for e in entries})

    out_path = refined_dir / "audit_oracle_batch_request.jsonl"
    written = skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            inst = instances.get(e["task_id"])
            spec = spec_for(inst, variant) if inst is not None else None
            if spec is None:
                skipped += 1
                continue
            for qkey, q in e["questions"].items():
                oracle = q.get("oracle_description")
                if not oracle:
                    continue
                content = generate_audit_oracle_prompt(spec.description, q["question"], oracle)
                body = {"model": model, "messages": [{"role": "user", "content": content}]}
                if temperature is not None:
                    body["temperature"] = temperature
                if provider is not None:
                    body["provider"] = provider_obj(provider)
                f.write(json.dumps({
                    "custom_id": f"audit_oracle__humanevalcomm_{e['task_id']}__{qkey}",
                    "body":      body,
                }) + "\n")
                written += 1
    print(f"{llm_dir}/{category}: {written} oracle-audit request(s) -> {out_path} ({skipped} task(s) skipped)")
    return out_path


_LEAK_RE    = re.compile(r"leak\s*:\s*(YES|NO)", re.IGNORECASE)
_LEAKED_RE  = re.compile(r"leaked\s*:\s*(.+?)(?:\n\s*rewrite\s*:|$)", re.IGNORECASE | re.DOTALL)
_REWRITE_RE = re.compile(r"rewrite\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def parse_oracle_audit(text: str):
    """Parses the auditor's reply into (leak: bool, leaked: str | None, rewrite: str | None)."""
    m = _LEAK_RE.search(text)
    leak = bool(m) and m.group(1).upper() == "YES"

    def grab(rx):
        g = rx.search(text)
        if not g:
            return None
        val = g.group(1).strip()
        return None if not val or val.upper() == "NONE" else val

    return leak, grab(_LEAKED_RE), grab(_REWRITE_RE)


def postprocess_audit_oracle(llm_dir: str, category: str,
                             result_name: str = "audit_oracle_batch_result.jsonl"):
    """
    Applies the oracle-audit verdicts to {category}/refined/questions_and_descriptions.json: for each
    (task, question) it fills oracle_leak (the leaked content, or None if clean) and oracle_rewrite
    (the auditor's tightened description, or None). Nothing is auto-applied; a human reviews the
    flags and decides whether to swap in a rewrite.
    """
    from batch_processing import extract_text
    refined_dir = EXPERIMENT / llm_dir / category / "refined"
    qd_path = refined_dir / "questions_and_descriptions.json"
    entries = json.load(open(qd_path, encoding="utf-8"))
    by_id = {e["task_id"]: e for e in entries}

    flagged = clean = skipped = 0
    with open(refined_dir / result_name, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            text  = extract_text(entry)
            parts = entry.get("custom_id", "").split("__")   # audit_oracle, humanevalcomm_{id}, {qkey}
            if text is None or len(parts) != 3:
                skipped += 1
                continue
            task_id, qkey = int(parts[1].split("_")[1]), parts[2]
            e = by_id.get(task_id)
            if e is None or qkey not in e["questions"]:
                skipped += 1
                continue
            leak, leaked, rewrite = parse_oracle_audit(text)
            q = e["questions"][qkey]
            q["oracle_leak"]    = leaked if leak else None
            q["oracle_rewrite"] = rewrite if leak else None
            flagged += leak
            clean   += (not leak)

    with open(qd_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print(f"{llm_dir}/{category}: oracle audit -> {flagged} flagged, {clean} clean, {skipped} skipped")
    return qd_path


if __name__ == "__main__":


    # --- Leak-check the oracle descriptions, then apply the flags ---
    # build_audit_oracle_batch("LLM1", "1c")
    # req = EXPERIMENT / "LLM1" / "1c" / "refined" / "audit_oracle_batch_request.jsonl"
    # batch = create_batch(req)
    # save_batch_meta(batch, req)
    # batch_id = json.load(open(EXPERIMENT / "LLM1" / "1c" / "refined" / "audit_oracle_batch_meta.json"))["batch_id"]
    # batch    = wait_for_batch(batch_id, poll=60)
    # save_results(batch, EXPERIMENT / "LLM1" / "1c" / "refined" / "audit_oracle_batch_result.jsonl")
    # postprocess_audit_oracle("LLM1", "1c")
    pass
