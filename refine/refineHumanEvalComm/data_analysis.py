"""
Aggregates the per-run stats produced by compute_stats.score_phase into per-category summaries, and
combines the baseline and refined summaries into the analysis data files.

aggregate_phase writes {phase}/aggregate.json. build_baseline_data and build_complete_data read those
files and write, under .HEC-experiment/{llm_dir}/analysis/:
  baseline_data_{cat}.json  - list of every baseline (manipulated) task:
      {task_id, name, incoherence_list, mean_incoherence, error_list, mean_error}
  complete_data_{cat}.json  - list of refined tasks:
      {task_id, name,
       baseline: {incoherence_list, mean_incoherence, error_list, mean_error},
       refined:  {q1_desc1:{incoherence_list, mean_incoherence}, q1_desc2:{...},
                  q1_oracle:{incoherence_list, mean_incoherence, error_list, mean_error},
                  q2_..., q3_...}}
"""
import json
import statistics
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import EXPERIMENT

CODER_QS  = ["q1", "q2", "q3"]
MIN_VALID = 2     # a metric list needs at least this many non-null entries to be usable in MWU


def aggregate_phase(llm_dir: str, category: str, phase: str = "baseline"):
    """
    Reads every run{r}/stats.json under .HEC-experiment/{llm_dir}/{category}/{phase}/ and, per task,
    collects incoherence and error across runs plus their means.
    Writes the summary to {phase}/aggregate.json and returns it.

    llm_dir/category/phase: which phase directory to summarise
    returns:                list of per-task aggregate dicts
    """
    phase_dir = EXPERIMENT / llm_dir / category / phase
    run_dirs = sorted((d for d in phase_dir.iterdir() if d.is_dir() and d.name.startswith("run")),
                      key=lambda d: int(d.name[len("run"):]))

    per_run = {}
    for d in run_dirs:
        stats_path = d / "stats.json"
        if not stats_path.exists():
            continue
        with open(stats_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        per_run[int(d.name[len("run"):])] = {r["key"]: r for r in rows}

    runs = sorted(per_run)
    if not runs:
        print(f"{llm_dir}/{category}/{phase}: no run stats found (run score_phase first)")
        return []

    # Align across runs by candidate key (the file name), which is stable from run to run. For the
    # baseline that is one key per task; for the refined phase it is one key per (task, question,
    # branch) condition, so conditions of the same task do not collide.
    keys = sorted({k for run_rows in per_run.values() for k in run_rows})
    aggregated = []
    for key in keys:
        inc = [per_run[r].get(key, {}).get("incoherence") for r in runs]
        err = [per_run[r].get(key, {}).get("error") for r in runs]
        inc_valid = [v for v in inc if v is not None]
        err_valid = [v for v in err if v is not None]
        sample = next(per_run[r][key] for r in runs if key in per_run[r])
        aggregated.append({
            "key":              key,
            "task_id":          sample.get("task_id"),
            "name":             sample.get("name"),
            "variant":          sample.get("variant"),
            "runs":             runs,
            "incoherence_list": inc,
            "mean_incoherence": statistics.mean(inc_valid) if inc_valid else None,
            "error_list":       err,
            "mean_error":       statistics.mean(err_valid) if err_valid else None,
        })

    out_path = phase_dir / "aggregate.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    print(f"{llm_dir}/{category}/{phase}: aggregated {len(aggregated)} task(s) over runs {runs} -> {out_path}")
    return aggregated


def _branch(v): return v.split("__")[-1] if v and "__" in v else None
def _qkey(v):   return "__".join(v.split("__")[:-1]) if v and "__" in v else None
def _ok(lst):   return sum(1 for x in (lst or []) if x is not None) >= MIN_VALID


def _load_aggregate(llm_dir: str, category: str, phase: str):
    with open(EXPERIMENT / llm_dir / category / phase / "aggregate.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _write_analysis(llm_dir: str, name: str, data):
    out_dir = EXPERIMENT / llm_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    return out_path


def build_baseline_data(llm_dir: str, category: str):
    """
    Copies every baseline (manipulated) task from baseline/aggregate.json into
    analysis/baseline_data_{category}.json, keeping only the incoherence and error fields.

    llm_dir/category: which baseline to read (run aggregate_phase first)
    returns:          list of per-task baseline dicts
    """
    baseline_data = [{"task_id": x["task_id"], "name": x["name"],
                      "incoherence_list": x["incoherence_list"], "mean_incoherence": x["mean_incoherence"],
                      "error_list": x["error_list"], "mean_error": x["mean_error"]}
                     for x in _load_aggregate(llm_dir, category, "baseline")]
    out_path = _write_analysis(llm_dir, f"baseline_data_{category}.json", baseline_data)
    print(f"{llm_dir}/{category}: baseline_data={len(baseline_data)} tasks -> {out_path}")
    return baseline_data


def build_complete_data(llm_dir: str, category: str):
    """
    Combines baseline/aggregate.json and refined/aggregate.json into
    analysis/complete_data_{category}.json, one entry per task with its baseline and all refined
    branches. Only tasks with a valid, non-zero baseline incoherence and a valid baseline error, whose
    three coder questions all have complete desc1, desc2 and oracle branches, are included.

    llm_dir/category: which experiment to read (run aggregate_phase for both phases first)
    returns:          list of per-task dicts
    """
    base = {x["task_id"]: x for x in _load_aggregate(llm_dir, category, "baseline")}
    ref  = _load_aggregate(llm_dir, category, "refined")

    # gather refined branch rows per (task, qkey, branch)
    rows = {}
    for x in ref:
        rows.setdefault((x["task_id"], _qkey(x["variant"])), {})[_branch(x["variant"])] = x

    def desc_block(x):    return {"incoherence_list": x["incoherence_list"], "mean_incoherence": x["mean_incoherence"]}
    def oracle_block(x):  return {"incoherence_list": x["incoherence_list"], "mean_incoherence": x["mean_incoherence"],
                                  "error_list": x["error_list"], "mean_error": x["mean_error"]}

    def question_complete(bd):
        return (all(b in bd for b in ("desc1", "desc2", "oracle"))
                and _ok(bd["desc1"]["incoherence_list"]) and _ok(bd["desc2"]["incoherence_list"])
                and _ok(bd["oracle"]["error_list"]))

    complete_data = []
    task_ids = {t for (t, q) in rows}
    for tid in sorted(task_ids):
        b = base.get(tid)
        if b is None or b["mean_incoherence"] is None or b["mean_error"] is None \
           or b["mean_incoherence"] <= 0 \
           or not _ok(b["incoherence_list"]) or not _ok(b["error_list"]):
            continue   # complete_data = tasks with non-zero baseline incoherence
        # require all three coder questions complete
        if not all((tid, q) in rows and question_complete(rows[(tid, q)]) for q in CODER_QS):
            continue
        refined = {}
        for q in CODER_QS:
            bd = rows[(tid, q)]
            refined[f"{q}_desc1"]  = desc_block(bd["desc1"])
            refined[f"{q}_desc2"]  = desc_block(bd["desc2"])
            refined[f"{q}_oracle"] = oracle_block(bd["oracle"])
        complete_data.append({"task_id": tid, "name": b["name"],
                              "baseline": {"incoherence_list": b["incoherence_list"],
                                           "mean_incoherence": b["mean_incoherence"],
                                           "error_list": b["error_list"],
                                           "mean_error": b["mean_error"]},
                              "refined": refined})
    out_path = _write_analysis(llm_dir, f"complete_data_{category}.json", complete_data)
    print(f"{llm_dir}/{category}: complete_data={len(complete_data)} tasks -> {out_path}")
    return complete_data


if __name__ == "__main__":

    # --- Aggregate per-run stats into {phase}/aggregate.json ---
    # aggregate_phase("LLM1", "1c")                    # baseline
    # aggregate_phase("LLM1", "1c", phase="refined")   # refined

    # --- Combine the aggregates into analysis/{baseline,complete}_data_{cat}.json ---
    # build_baseline_data("LLM1", "1c")
    # build_complete_data("LLM1", "1c")
    pass
