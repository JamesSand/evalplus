"""Dump a few evalplus prompts in chat-template format, mirroring train_example.json."""
import json
from transformers import AutoTokenizer
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.provider.utility import make_raw_chat_prompt, _MAGIC_SPLITTER_

INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the "
    "problem and passes corresponding tests:"
)

MODEL = "Qwen/Qwen3-1.7B"  # instruct, has chat_template
tok = AutoTokenizer.from_pretrained(MODEL)


def build_messages(task_prompt: str):
    """Re-create the {role, content} list that evalplus passes to apply_chat_template."""
    user_msg = (
        f"{INSTRUCTION_PREFIX}\n"
        f"```\n"
        f"{task_prompt.strip()}\n"
        f"```\n"
    )
    asst_msg = (
        f"{RESPONSE_PREFIX}\n"
        f"```python\n"
        f"{_MAGIC_SPLITTER_}\n"
        f"```\n"
    )
    return [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": asst_msg},
    ]


def render_final_prompt(task_prompt: str) -> str:
    """The literal string vllm receives (chat template applied + split at MAGIC)."""
    return make_raw_chat_prompt(task_prompt, INSTRUCTION_PREFIX, RESPONSE_PREFIX, tok)


def make_record(dataset_name, task_id, task):
    raw_prompt = task["prompt"]
    messages = build_messages(raw_prompt)
    final_prompt_to_llm = render_final_prompt(raw_prompt)
    return {
        "data_source": f"evalplus/{dataset_name}",
        "task_id": task_id,
        "entry_point": task["entry_point"],
        "prompt": messages,                       # chat template format (matches train_example.json)
        "raw_task_prompt": raw_prompt,            # original task["prompt"]
        "rendered_prompt_to_llm": final_prompt_to_llm,  # actual string sent to model after chat template + split
        "ability": "code",
        "reward_model": {
            "ground_truth": task["canonical_solution"],
            "style": "code_pass_at_1",
        },
        "extra_info": {"split": "test"},
    }


he = get_human_eval_plus()
mb = get_mbpp_plus()
he_ids = list(he.keys())[:2]
mb_ids = list(mb.keys())[:2]

records = []
for tid in he_ids:
    records.append(make_record("humaneval", tid, he[tid]))
for tid in mb_ids:
    records.append(make_record("mbpp", tid, mb[tid]))

out_path = "/tmp/evalplus_prompt_samples.json"
with open(out_path, "w") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(records)} samples to {out_path}")
for r in records:
    print(f"\n--- {r['task_id']} ({r['entry_point']}) ---")
    print("[user.content]:")
    print(r["prompt"][0]["content"])
    print("[assistant.content (with MAGIC placeholder)]:")
    print(r["prompt"][1]["content"])
    print("[final string sent to LLM (after split at MAGIC)]:")
    print(repr(r["rendered_prompt_to_llm"][:600]) + " ...")
