# Refinement Pipeline (HumanEvalComm)

This folder holds the clarifying-question refinement experiment on HumanEvalComm, in
[`refineHumanEvalComm/`](refineHumanEvalComm/).

## Overview

For one coder LLM and one HumanEvalComm category (e.g. `1c`), the pipeline:

1. **Baseline**: samples program candidates straight from the category's manipulated
   task description and scores them for incoherence and error.
2. **Questions**: for every task with mean baseline incoherence > 0, the coder asks its own binary
   clarifying questions (3 by default).
3. **Coder descriptions**: for each question, the coder writes two short clarifications, one
   assuming YES (`description1`) and one assuming NO (`description2`).
4. **Oracle description**: a fixed oracle model, which sees the reference solution and held-out
   tests, writes the true answer (`oracle_description`).
5. **Audit**: a fixed auditor model (a different vendor) checks each oracle answer for leakage
   beyond the question. It sets the `oracle_leak` flag and fills `oracle_rewrite` if necessary.
6. **Refined candidates**: the coder samples new programs from the *manipulated* task description with one
   clarification appended. The coder branches (`desc1`, `desc2`) and the oracle branch (`oracle`)
   are built separately, so the coder branches can run before the oracle answers are settled. If
   an audit flagged a leak and supplied a rewrite, the rewrite replaces the oracle answer.
7. **Score + aggregate**: the refined candidates are scored and averaged across runs. Error is only
   computed for the baseline and the `oracle` branch, because `desc1`/`desc2` encode a guessed
   answer, not the ground truth.

Every LLM step has the same three parts: **build** a request file (`*_request.jsonl`), **run** it
to get a result file (`*_result.jsonl`), then **postprocess** the results into the experiment
folder. Each step reads the previous step's output.

## Modules

| File | Role |
| --- | --- |
| `common.py` | Paths, dataset name, API backend, category map, and the model roster (`CODER_MODELS`, `ORACLE_MODEL`, `AUDITOR_MODEL`) |
| `build_batch.py` | Builds baseline and refined-candidate request files |
| `refine_descriptions.py` | Builds the questions, coder-description and oracle request files |
| `audit.py` | Builds the oracle-audit requests and applies the audit verdicts |
| `litellm_chat.py` | Runs a request file synchronously against a LiteLLM proxy (concurrent, rate-limited, resumable) |
| `batch_processing.py` | OpenRouter Batch API submission/retrieval, and all `postprocess_*` functions |
| `compute_stats.py` | Scores candidate files for incoherence and error using `difftrust` |
| `data_analysis.py` | Aggregates per-run scores into `aggregate.json` |
| `run.py` | Entry point: the full pipeline as a sequence of commented-out steps |

## Setup

1. Install the dependencies from the repository root, as in the main [README](../README.md):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Check that the dataset is present at `HumanEvalComm/.data/dataset-complete.pkl` or build a new datset from `HumanEvalComm/instance.py`.

3. Pick a backend in `common.py`:
   - `BASE = "litellm"` (default): set `API_BASE_LL` to your LiteLLM proxy URL.
   - `BASE = "openrouter"`: requests go through the OpenRouter Batch API.

4. Put the API key in a `.env` file at the repository root (loaded with `python-dotenv`) or export
   it:

   ```bash
   LITELLM_API_KEY=...       # for BASE = "litellm"
   OPENROUTER_API_KEY=...    # for BASE = "openrouter"
   ```

5. If needed, adjust the models in `common.py`. The keys of `CODER_MODELS` (e.g.
   `Qwen3CoderNext`) are used as experiment folder names.

## Running

The pipeline is driven from [`refineHumanEvalComm/run.py`](refineHumanEvalComm/run.py):

1. Set the coder and category at the top of the file:

   ```python
   llm, cat = "Qwen3CoderNext", "1c"
   provider = None   # optionally pin an OpenRouter provider (a name string, or a routing dict)
   ```

2. Uncomment **one step block** (build → run → postprocess) and run the script from the repository
   root:

   ```bash
   python refine/refineHumanEvalComm/run.py
   ```

3. Check the output, comment the block out again, then uncomment the next one. Run the steps in
   order:

   | # | Step | Functions |
   | --- | --- | --- |
   | 1 | Baseline | `build_baseline_batch` → `run_request_file` → `postprocess_baseline` → `score_phase` → `aggregate_phase` |
   | 2 | Questions | `build_questions_batch` → `run_request_file` → `postprocess_questions` |
   | 3 | Coder YES/NO descriptions | `build_descriptions_batch` → `run_request_file` → `postprocess_descriptions` |
   | 4 | Oracle description | `build_oracle_batch` → `run_request_file` → `postprocess_oracle` |
   | 5 | Oracle audit | `build_audit_oracle_batch` → `run_request_file` → `postprocess_audit_oracle` |
   | 6a | Refined candidates, coder branches | `build_refined_coder_candidates_batch` → `run_request_file` → `postprocess_refined_candidates` |
   | 6b | Refined candidates, oracle branch | `build_refined_oracle_candidates_batch` → `run_request_file` → `postprocess_refined_candidates` |
   | 6c | Score + aggregate | `score_phase(phase="refined")` → `aggregate_phase(phase="refined")` |

   Step 2 depends on the baseline `aggregate.json`, because only tasks with incoherence > 0 are
   refined. Between steps 5 and 6b, review the audit flags in `questions_and_descriptions.json`.

### Notes

- **Resumable.** `run_request_file` skips any `custom_id` already in the result file, and
  `compute_stats` skips files already in `stats.json`. To continue after an interruption or
  failed requests, run the same step again.
- **Truncated outputs.** Responses cut off by `max_tokens` are dropped, not written. To regenerate
  them, run the step again with a higher (or unset) `max_tokens`.
- **Rate limits.** Tune `max_workers` and `max_requests_per_min` in `run_request_file` to your
  proxy's limits.
- **Scoring workers.** The metric timeout is wall-clock, so keep `score_phase(..., workers=N)` at
  or below your core count to avoid spurious timeouts.
- **OpenRouter Batch API.** With `BASE = "openrouter"`, submit each request file with
  `create_batch` / `save_batch_meta` (or `submit_llm`), wait with `wait_for_batch`, write the
  results with `save_results`, then run the same `postprocess_*` function. See the `__main__`
  blocks of `batch_processing.py` and `refine_descriptions.py` for examples.

## Output layout

All output goes under `refineHumanEvalComm/<EXPERIMENT>/` (set by `EXPERIMENT` in `common.py`):

```
<llm>/<category>/
├── baseline_batch_request.jsonl / baseline_batch_result.jsonl(.gz)
├── baseline/
│   ├── run0 … run9/          # one candidate file per task + stats.json
│   └── aggregate.json        # per-task mean incoherence / error across runs
└── refined/
    ├── questions_and_descriptions.json   # questions, desc1/desc2, oracle answer, audit flags
    ├── *_batch_request.jsonl / *_batch_result.jsonl(.gz)   # one pair per LLM step
    ├── run0 … run9/          # one candidate file per (task, question, branch) + stats.json
    └── aggregate.json
```

Refined candidate files are named `humanevalcomm_<task_id>__<qkey>__<branch>`, where `branch` is
`desc1`, `desc2` or `oracle`.

### Encrypted reasoning removed

The committed `*_batch_result.jsonl` files are the raw API responses with the encrypted reasoning
removed. In `message.provider_specific_fields.reasoning_details`, the `data` field of every
`reasoning.encrypted` entry and the `signature` field of `reasoning.text` entries are deleted.
These fields are opaque, provider-encrypted messages that cannot be read or verified without the
provider. They made up about two thirds of the file size and do not compress. Everything else
is unchanged, including the candidate code (`message.content`), the readable reasoning summaries
and text, token usage and costs. The pipeline does not use the removed fields.

### Compressed result files

Some `*_batch_result.jsonl` files still exceed GitHub's 100 MB file size limit after the encrypted
reasoning is removed. Those are committed gzipped as `*_batch_result.jsonl.gz`, and the
uncompressed file is listed in `.gitignore`. The pipeline reads the uncompressed `.jsonl`, so after
cloning, decompress them in place (from the repository root):

```bash
find refine/refineHumanEvalComm/.HEC-experiment -name '*.jsonl.gz' -exec gunzip -k {} \;
```

To add a new large result file, run `gzip -k <file>`, commit the `.gz`, and add the uncompressed
file to `.gitignore`.
