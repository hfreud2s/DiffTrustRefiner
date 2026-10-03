"""
Refinement phase for HumanEvalComm: clarifying questions and refined descriptions.

The refinement chain is: questions -> refined (yes/no) descriptions -> oracle (true) description ->
candidates from each description -> score. Each round consumes the previous round's results.

Step 5: question generation. Each coder LLM asks its own clarifying questions (per default)
about a category's (possibly ambiguous) description. build_questions_batch() writes the batch request file.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import CATEGORIES, EXPERIMENT, DATASET_NAME, ORACLE_MODEL, load_instances, spec_for, provider_obj
from batch_processing import create_batch, save_batch_meta, wait_for_batch, save_results


def needs_refinement(llm_dir: str, category: str, phase: str = "baseline"):
    """
    The task ids of a (LLM, category) that need clarifying questions: those whose baseline mean
    incoherence is above zero. Reads the cross-run summary {phase}/aggregate.json produced by
    data_analysis.aggregate_phase.

    returns: set of task ids with mean_incoherence > 0
    """
    aggregate_path = EXPERIMENT / llm_dir / category / phase / "aggregate.json"
    with open(aggregate_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {row["task_id"] for row in rows
            if row.get("mean_incoherence") is not None and row["mean_incoherence"] > 0}


def infer_model(llm_dir: str, category: str):
    """Reads the model originally used for the category, from its baseline request file."""
    req_path = EXPERIMENT / llm_dir / category / "baseline_batch_request.jsonl"
    with open(req_path, "r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    return first["body"]["model"]


def generate_binary_questions(description: str, num_questions: int = 3):
    """
    Builds the prompt that asks the model to identify genuinely underspecified behaviours in a task
    description as yes/no questions. Ported from the MBPP pipeline; takes the description of the
    category being refined (the ambiguous one for a manipulated category).
    """
    prompt = (
        f"You are analyzing a Python programming task to identify genuine implementation ambiguities.\n\n"
        f"Task specification:\n{description}\n\n"
        f"Identify {num_questions} binary question(s) about this specification where:\n"
        f"- Each question targets a specific behavior that is genuinely underspecified\n"
        f"- Each question can be answered with either YES or NO\n"
        f"- A YES answer and a NO answer would lead to observably different code (different outputs on at least one input)\n"
        f"- The question cannot be answered just by reading the specification carefully\n"
        f"- The question is about WHAT the function should return or do, not HOW to implement it\n\n"
        f"For example: Consider the specification \"Sort a given list of integers.\"\n"
        f"Good question: 'Should the list be sorted in increasing order?'\n\n"
        f"Bad question: 'How should the function handle edge cases?' (too vague, not binary)\n\n"
        f"Output format:\n"
        f"question 1: ...\nquestion 2: ...\nquestion 3: ...\n\n"
        f"Output only the questions, nothing else."
    )
    return prompt


def build_questions_batch(llm_dir: str, category: str, model: str = None,
                          num_questions: int = 3, dataset_name: str = DATASET_NAME,
                          provider=None, temperature: float = 0.0):
    """
    Writes the OpenRouter batch request that asks the LLM to generate clarifying questions for the
    tasks of (llm_dir, category) whose baseline incoherence is above zero.

    Output: .HEC-experiment/{llm_dir}/{category}/refined/questions_batch_request.jsonl
    One item per task, custom_id "questions__humanevalcomm_{task_id}", body is a chat request 
    whose prompt embeds that category's description.

    llm_dir/category: which (LLM, category) to build for
    model:            Question-generation model (defaults to the one the baseline batch used)
    num_questions:    binary questions to ask per task (default 3)
    returns:          the path written, or None if nothing needs refinement
    """
    variant = CATEGORIES[category]
    need    = needs_refinement(llm_dir, category)
    if not need:
        print(f"{llm_dir}/{category}: no task has incoherence > 0, nothing to refine")
        return None
    instances = load_instances(dataset_name)
    if model is None:
        model = infer_model(llm_dir, category)

    out_dir  = EXPERIMENT / llm_dir / category / "refined"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "questions_batch_request.jsonl"

    written = skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for task_id in sorted(need):
            inst = instances.get(task_id)
            spec = spec_for(inst, variant) if inst is not None else None
            if spec is None:
                skipped += 1
                continue
            content = generate_binary_questions(spec.description, num_questions)
            body = {"model": model, "messages": [{"role": "user", "content": content}]}
            if temperature is not None:
                body["temperature"] = temperature
            if provider is not None:
                body["provider"] = provider_obj(provider)
            f.write(json.dumps({
                "custom_id": f"questions__humanevalcomm_{task_id}",
                "body":      body,
            }) + "\n")
            written += 1
    print(f"{llm_dir}/{category}: {written} question request(s) -> {out_path} "
          f"({skipped} skipped; {len(need)} tasks need refinement)")
    return out_path


def generate_binary_descriptions(description: str, question: str):
    """
    Builds the prompt asking the coder for two short clarifications (answers) to the yes/no
    question, one for YES and one for NO. These are NOT rewritten specifications: the refined
    prompt is the ORIGINAL task with the clarification appended (see build_refined_candidates_batch),
    so the task's own text and examples are preserved and only this one question's answer is added.
    The two differ from each other in nothing but that answer.
    """
    return (
        f"Given this Python programming task:\n"
        f"{description}\n\n"
        f"And this yes/no question about it:\n"
        f"{question}\n\n"
        f"Write exactly two short clarifications that could be appended to the task, one assuming "
        f"the answer to the question is YES, one assuming NO. Each states ONLY what this one "
        f"question resolves. They differ from each other in nothing else.\n"
        f"Rules:\n"
        f"- State the resulting behavior neutrally (what the function should do). Do not say that any "
        f"part of the task is right or wrong and do not refer to 'the docstring' or 'the examples'. "
        f"Just state the behavior this answer implies.\n"
        f"- Resolve ONLY this question. Do not invent, assume, or resolve anything else the task "
        f"leaves open: not input constraints (lengths, non-emptiness, types), edge cases, or other "
        f"ambiguities.\n"
        f"- Keep it to one or two sentences. Describe behavior, not implementation steps. No code, "
        f"and do not add new examples.\n\n"
        f"Output format:\n"
        f"description 1: ...\ndescription 2: ...\n\n"
        f"Example: task \"Sort a list of integers.\", question \"Should the list be sorted in "
        f"ascending order?\"\n"
        f"description 1: The list should be sorted in ascending order.\n"
        f"description 2: The list should be sorted in descending order."
    )


def build_descriptions_batch(llm_dir: str, category: str, model: str = None,
                             dataset_name: str = DATASET_NAME, provider=None,
                             temperature: float = 0.0):
    """
    Writes the OpenRouter batch asking the coder LLM for the YES/NO refined descriptions of every
    question in {category}/refined/questions_and_descriptions.json, including any q_auditor question
    (it is scored like the rest). One request per (task, question); custom_id
    "descriptions__humanevalcomm_{task_id}__{qkey}". Output:
    {category}/refined/descriptions_batch_request.jsonl. Model defaults to the coder's own baseline
    slug. Post-process with batch_processing.postprocess_descriptions.
    """
    variant     = CATEGORIES[category]
    refined_dir = EXPERIMENT / llm_dir / category / "refined"
    with open(refined_dir / "questions_and_descriptions.json", "r", encoding="utf-8") as f:
        entries = json.load(f)
    instances = load_instances(dataset_name, {e["task_id"] for e in entries})
    if model is None:
        model = infer_model(llm_dir, category)

    out_path = refined_dir / "descriptions_batch_request.jsonl"
    written = skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            inst = instances.get(e["task_id"])
            spec = spec_for(inst, variant) if inst is not None else None
            if spec is None:
                skipped += 1
                continue
            for qkey, q in e["questions"].items():
                content = generate_binary_descriptions(spec.description, q["question"])
                body = {"model": model, "messages": [{"role": "user", "content": content}]}
                if temperature is not None:
                    body["temperature"] = temperature
                if provider is not None:
                    body["provider"] = provider_obj(provider)
                f.write(json.dumps({
                    "custom_id": f"descriptions__humanevalcomm_{e['task_id']}__{qkey}",
                    "body": body,
                }) + "\n")
                written += 1
    print(f"{llm_dir}/{category}: {written} description request(s) -> {out_path} "
          f"({skipped} task(s) skipped)")
    return out_path



def generate_oracle_prompt(description: str, question: str, code: str, test: str):
    """
    Builds the prompt asking the oracle for the ground-truth answer to THIS one clarifying question,
    as a short NEUTRAL clarification. It is NOT a rewritten specification: the refined prompt is the
    original task with this clarification appended (see build_refined_candidates_batch), so the
    task's own text and examples are preserved and only this question's answer is added. The
    reference solution and its held-out tests are shown ONLY so the oracle can determine the correct
    answer, and must not be revealed (checked later by auditor #2).

    description: the category's task description, as the coder saw it (examples included)
    question:    the clarifying question whose answer to embed
    code:        the reference solution (inst.code), used only to determine this question's answer
    test:        the held-out reference tests (inst.test), for the same purpose only
    """
    return (
        f"Given this Python programming task:\n"
        f"{description}\n\n"
        f"And this yes/no question about it:\n"
        f"{question}\n\n"
        f"Here is the reference solution, to use ONLY to determine the correct answer to this one "
        f"question:\n{code}\n\n"
        f"And its held-out test cases, for the same purpose only:\n{test}\n\n"
        f"Write a single short clarification that could be appended to the task: the correct answer "
        f"to this question, the way the reference solution actually behaves. It must:\n"
        f"- State the resulting behavior neutrally (what the function should do). Do not say which "
        f"part of the task is right or wrong and do not refer to 'the docstring' or 'the examples'; "
        f"just state the behavior. You may read the task's own examples to determine it.\n"
        f"- Resolve ONLY this question. Do not resolve, assume, or state anything else the task "
        f"leaves open: not other ambiguities, edge cases, input constraints, or types.\n"
        f"- Not reveal or restate the reference solution, and not reveal, add, or change any "
        f"held-out test case or its values. Keep it to one or two sentences, describing behavior, "
        f"not implementation steps. No code, and do not add new examples.\n\n"
        f"Output format:\n"
        f"description: ...\n\n"
        f"Example: task \"Sort a list of integers.\", question \"Should the list be sorted in "
        f"ascending order?\", and the reference solution sorts ascending.\n"
        f"description: The list should be sorted in ascending order.\n\n"
        f"Bad output: \"Based on the reference solution, sort in ascending order in place.\" It "
        f"mentions the reference and adds information (in-place) the question did not ask about."
    )


def build_oracle_batch(llm_dir: str, category: str, model: str = ORACLE_MODEL,
                       dataset_name: str = DATASET_NAME, provider=None,
                       temperature: float = 0.0):
    """
    Writes the OpenRouter batch asking the fixed oracle for the ground-truth description of every
    question in {category}/refined/questions_and_descriptions.json, including any q_auditor question.
    One request per (task, question); custom_id "oracle__humanevalcomm_{task_id}__{qkey}". Output:
    {category}/refined/oracle_batch_request.jsonl. Post-process with
    batch_processing.postprocess_oracle, which fills oracle_description.

    Unlike the coder rounds, the model is the fixed ORACLE_MODEL (the same simulated user across all
    LLMs and categories), not the coder's own slug.
    """
    variant     = CATEGORIES[category]
    refined_dir = EXPERIMENT / llm_dir / category / "refined"
    with open(refined_dir / "questions_and_descriptions.json", "r", encoding="utf-8") as f:
        entries = json.load(f)
    instances = load_instances(dataset_name, {e["task_id"] for e in entries})

    out_path = refined_dir / "oracle_batch_request.jsonl"
    written = skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            inst = instances.get(e["task_id"])
            spec = spec_for(inst, variant) if inst is not None else None
            if spec is None:
                skipped += 1
                continue
            for qkey, q in e["questions"].items():
                content = generate_oracle_prompt(spec.description, q["question"], inst.code, inst.test)
                body = {"model": model, "messages": [{"role": "user", "content": content}]}
                if temperature is not None:
                    body["temperature"] = temperature
                if provider is not None:
                    body["provider"] = provider_obj(provider)
                f.write(json.dumps({
                    "custom_id": f"oracle__humanevalcomm_{e['task_id']}__{qkey}",
                    "body": body,
                }) + "\n")
                written += 1
    print(f"{llm_dir}/{category}: {written} oracle request(s) -> {out_path} "
          f"({skipped} task(s) skipped)")
    return out_path


if __name__ == "__main__":

    # --- Which tasks need refinement (baseline mean incoherence > 0), the set every round below acts on ---
    # print(sorted(needs_refinement("LLM1", "1c")))

    # --- Round 1: build the clarifying-question batch, then submit it with batch_processing ---
    # build_questions_batch("LLM1", "1c", num_questions=3)

    # req = EXPERIMENT / "LLM1" / "1c" / "refined" / "questions_batch_request.jsonl"
    # batch = create_batch(req) 
    # save_batch_meta(batch, req)

    # --- Later: wait for a batch and save its results ---
    # batch_id = json.load(open(EXPERIMENT / "LLM1" / "1c" / "refined" / "questions_batch_meta.json"))["batch_id"]
    # batch    = wait_for_batch(batch_id, poll=60)
    # save_results(batch, EXPERIMENT / "LLM1" / "1c" / "refined" / "questions_batch_result.jsonl")
    # then in batch_processing.py: postprocess_questions("LLM1", "1c")
    # (optional) Round 2 = auditor #1 on the question set: see audit.py

    # --- Round 3: build the YES/NO refined descriptions, then submit and retrieve ---
    # build_descriptions_batch("LLM1", "1c")
    # req = EXPERIMENT / "LLM1" / "1c" / "refined" / "descriptions_batch_request.jsonl"
    # batch = create_batch(req)
    # save_batch_meta(batch, req)
    # batch_id = json.load(open(EXPERIMENT / "LLM1" / "1c" / "refined" / "descriptions_batch_meta.json"))["batch_id"]
    # batch    = wait_for_batch(batch_id, poll=60)
    # save_results(batch, EXPERIMENT / "LLM1" / "1c" / "refined" / "descriptions_batch_result.jsonl")
    # then in batch_processing.py: postprocess_descriptions("LLM1", "1c")

    # --- Round 4: build the oracle's ground-truth description, then submit and retrieve ---
    # build_oracle_batch("LLM1", "1c")   # uses the fixed ORACLE_MODEL
    # req = EXPERIMENT / "LLM1" / "1c" / "refined" / "oracle_batch_request.jsonl"
    # batch = create_batch(req)
    # save_batch_meta(batch, req)
    # batch_id = json.load(open(EXPERIMENT / "LLM1" / "1c" / "refined" / "oracle_batch_meta.json"))["batch_id"]
    # batch    = wait_for_batch(batch_id, poll=60)
    # save_results(batch, EXPERIMENT / "LLM1" / "1c" / "refined" / "oracle_batch_result.jsonl")
    # then in batch_processing.py: postprocess_oracle("LLM1", "1c")

    # --- Round 5 (refined candidates) is built in build_batch.py: build_refined_candidates_batch("LLM1", "1c"),
    #     then batch_processing.postprocess_refined_candidates, compute_stats.score_phase(phase="refined"),
    #     data_analysis.aggregate_phase(phase="refined") ---
    pass
