from common import EXPERIMENT, CODER_MODELS
from build_batch import (build_baseline_batch,
                         build_refined_coder_candidates_batch,
                         build_refined_oracle_candidates_batch)
from refine_descriptions import build_questions_batch, build_descriptions_batch, build_oracle_batch
from audit import (build_audit_oracle_batch, postprocess_audit_oracle)
from batch_processing import (postprocess_baseline, postprocess_questions,
                              postprocess_descriptions, postprocess_oracle, postprocess_refined_candidates)
from compute_stats import score_phase
from data_analysis import aggregate_phase, build_baseline_data, build_complete_data
from litellm_chat import run_request_file

llm, cat = "Qwen3CoderNext", "1c"
provider = None # When using OpenRouter many models provide the option to select a specific provider

R = EXPERIMENT/llm/cat

if __name__ == "__main__":

    """
    Run sequentially using LiteLLM 
    """

    # --- Baseline rounds. Uncomment top-to-bottom: each round consumes the previous round's output. ---

    # 1. Baseline: build -> run -> postprocess -> score -> aggregate
    #build_baseline_batch(llm, CODER_MODELS[llm], categories=[cat], num_candidates=10, num_runs=10, provider=provider)
    #run_request_file(R/"baseline_batch_request.jsonl", max_workers=64, max_tokens=8192, timeout=120, keep_alive=True, max_requests_per_min=500)
    #postprocess_baseline(llm, [cat])
    #score_phase(llm, cat) 
    #aggregate_phase(llm, cat)

    # --- Refinement rounds. Uncomment top-to-bottom: each round consumes the previous round's output. ---

    # 2. Round 1 - clarifying questions (each coder LLM asks its own): build -> run -> postprocess

    #build_questions_batch(llm, cat, model=CODER_MODELS[llm], num_questions=3, provider=provider, temperature=None)
    #run_request_file(R/"refined"/"questions_batch_request.jsonl", max_workers=64, max_tokens=8192, timeout=120, keep_alive=True, max_requests_per_min=1000)
    #postprocess_questions(llm, cat)

    # 3. Round 2 - coder YES/NO descriptions: build -> run -> postprocess
    #build_descriptions_batch(llm, cat, model=CODER_MODELS[llm], provider=provider, temperature=None)
    #run_request_file(R/"refined"/"descriptions_batch_request.jsonl", max_workers=64, max_tokens=8192, timeout=120, keep_alive=True, max_requests_per_min=1000)
    #postprocess_descriptions(llm, cat)

    # 4. Round 3 - oracle true description (fixed ORACLE_MODEL): build -> run -> postprocess
    #build_oracle_batch(llm, cat)
    #run_request_file(R/"refined"/"oracle_batch_request.jsonl", max_workers=64, max_tokens=8192, timeout=120, keep_alive=True, max_requests_per_min=1000)
    #postprocess_oracle(llm, cat)

    # 5. Round 4 - Auditor (flag oracle answers that leak beyond the question; fills oracle_leak/
    #     oracle_rewrite for human review, applies nothing): build -> run -> postprocess
    #build_audit_oracle_batch(llm, cat)
    #run_request_file(R/"refined"/"audit_oracle_batch_request.jsonl", max_workers=64, max_tokens=8192, timeout=120, keep_alive=True, max_requests_per_min=1000)
    #postprocess_audit_oracle(llm, cat)

    # 6a. Round 5, coder branches - refined candidates from the coder YES/NO descriptions (desc1/desc2):
    #     build -> run -> postprocess. Independent of the oracle-description decision.
    #build_refined_coder_candidates_batch(llm, cat, model=CODER_MODELS[llm], num_candidates=10, num_runs=10, provider=provider)
    #run_request_file(R/"refined"/"refined_coder_candidates_batch_request.jsonl", max_workers=64, max_tokens=8192, timeout=120, keep_alive=True, max_requests_per_min=1000)
    #postprocess_refined_candidates(llm, cat, result_name="refined_coder_candidates_batch_result.jsonl")

    # 6b. Round 6, oracle branch - refined candidates from the oracle's true description:
    #     build -> run -> postprocess. Run once the oracle_description/oracle_rewrite decision is settled.
    #build_refined_oracle_candidates_batch(llm, cat, model=CODER_MODELS[llm], num_candidates=10, num_runs=10, provider=provider)
    #run_request_file(R/"refined"/"refined_oracle_candidates_batch_request.jsonl", max_workers=64, max_tokens=8192, timeout=120, keep_alive=True, max_requests_per_min=1000)
    #postprocess_refined_candidates(llm, cat, result_name="refined_oracle_candidates_batch_result.jsonl")

    # 6c. Score + aggregate once the branches you want are in (reads whichever branches are present):
    #score_phase(llm, cat, phase="refined", workers=2)
    #aggregate_phase(llm, cat, phase="refined")

    # 7. Analysis data - combine the baseline and refined aggregates into analysis/:
    #     baseline_data_{cat}.json (every baseline task) and complete_data_{cat}.json (tasks with
    #     non-zero baseline incoherence and all three questions complete in every branch).
    #build_baseline_data(llm, cat)
    #build_complete_data(llm, cat)