# evalplus 的 prompt 拼接流程

## 0. 评测结果（Qwen3-1.7B，greedy / temp=0 / max_new_tokens=4096 / max_model_len=8192，单卡 RTX A6000）

| 数据集 | Base 模型 (`Qwen/Qwen3-1.7B-Base`) | Instruct 模型 (`Qwen/Qwen3-1.7B`) | Δ (Instruct − Base) |
|---|---|---|---|
| HumanEval (base) | 0.482 | **0.726** | +0.244 |
| HumanEval+       | 0.402 | **0.652** | +0.250 |
| MBPP (base)      | **0.672** | 0.574 | −0.098 ⚠️ |
| MBPP+            | **0.579** | 0.492 | −0.087 ⚠️ |

**观察：**
- HumanEval 上 instruct 比 base 高约 24~25 个百分点，符合预期。
- **MBPP 上 base 模型反而比 instruct 高近 10 个百分点**。原因推测：MBPP 的 `task["prompt"]` 是 ` """ docstring + assert """ ` 格式，对 base 模型的续写范式天然友好；instruct 走 chat template + sanitize 流程引入了额外噪声。

**复现命令：**

```bash
# Base 模型（用 --force_base_prompt 走 direct completion）
CUDA_VISIBLE_DEVICES=1 python -m evalplus.evaluate \
    --dataset humaneval --model Qwen/Qwen3-1.7B-Base --backend vllm \
    --greedy --max_new_tokens 4096 --max_model_len 8192 --force_base_prompt

CUDA_VISIBLE_DEVICES=1 python -m evalplus.evaluate \
    --dataset mbpp --model Qwen/Qwen3-1.7B-Base --backend vllm \
    --greedy --max_new_tokens 4096 --max_model_len 8192 --force_base_prompt

# Instruct 模型（默认走 chat template）
CUDA_VISIBLE_DEVICES=2 python -m evalplus.evaluate \
    --dataset humaneval --model Qwen/Qwen3-1.7B --backend vllm \
    --greedy --max_new_tokens 4096 --max_model_len 8192

CUDA_VISIBLE_DEVICES=3 python -m evalplus.evaluate \
    --dataset mbpp --model Qwen/Qwen3-1.7B --backend vllm \
    --greedy --max_new_tokens 4096 --max_model_len 8192
```

> 注：`--max_new_tokens` 和 `--max_model_len` 是给 `evalplus` 仓库本地打的小补丁（vllm provider 原本写死 `max_model_len=2048`，且 `max_new_tokens` 没从 CLI 透传到 vllm）。

**结果文件位置：**
```
evalplus_results/humaneval/Qwen--Qwen3-1.7B-Base_vllm_temp_0.0.{jsonl, raw.jsonl, eval_results.json}
evalplus_results/humaneval/Qwen--Qwen3-1.7B_vllm_temp_0.0.{jsonl, raw.jsonl, eval_results.json}
evalplus_results/mbpp/Qwen--Qwen3-1.7B-Base_vllm_temp_0.0.{jsonl, raw.jsonl, eval_results.json}
evalplus_results/mbpp/Qwen--Qwen3-1.7B_vllm_temp_0.0.{jsonl, raw.jsonl, eval_results.json}
```

每组三份：`.jsonl`（sanitized samples）/ `.raw.jsonl`（原始模型输出）/ `.eval_results.json`（per-task pass/fail + 失败的 input）。

---

## 1. 总流程图

```
get_human_eval_plus / get_mbpp_plus
        │
        ▼
task["prompt"]  ─────────────►  codegen.py:71
        │                         prompt = task["prompt"].strip() + "\n"
        ▼
VllmDecoder.codegen (provider/vllm.py:50)
        │
        ▼
   ┌────is_direct_completion()────┐
   │                              │
  Yes (base / no chat_template     No (instruct, chat_template 存在)
       / --force_base_prompt)      │
   │                              ▼
   │                    make_raw_chat_prompt(...)
   │                    (provider/utility.py:26)
   │                              │
   │                       构造 [user, assistant] 两条消息
   │                       assistant 消息里写到 ```python\n + MAGIC + \n```
   │                              │
   │                       tokenizer.apply_chat_template(..., tokenize=False)
   │                              │
   │                       .split(MAGIC)[0]
   │                              │
   ▼                              ▼
 LLM.generate(prompt, max_tokens, eos=...)
```

## 2. 关键代码片段

**取原始任务 prompt** (`evalplus/codegen.py:71`)：
```python
prompt = task["prompt"].strip() + "\n"
outputs = model.codegen(prompt, do_sample=not greedy, num_samples=...)
```

**走哪条分支** (`evalplus/provider/vllm.py:47`)：
```python
def is_direct_completion(self) -> bool:
    return self.force_base_prompt or self.tokenizer.chat_template is None
```

**chat 模式拼接** (`evalplus/provider/utility.py:23-58`)：
```python
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

def make_raw_chat_prompt(task_prompt, instruction_prefix, response_prefix, tokenizer):
    if tokenizer.chat_template is None:
        return task_prompt

    task_prompt = f"""\
{instruction_prefix}
```
{task_prompt.strip()}
```
"""
    response = f"""\
{response_prefix}
```python
{_MAGIC_SPLITTER_}
```
"""
    return tokenizer.apply_chat_template(
        [{"role":"user","content":task_prompt},
         {"role":"assistant","content":response}],
        tokenize=False,
    ).split(_MAGIC_SPLITTER_)[0]
```

**默认 prefix** (`evalplus/codegen.py:222-223`)：
```
instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
response_prefix    = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"
```

## 3. 关键 trick：MAGIC splitter

evalplus 把 assistant 那条消息**预先**写成：
```
{response_prefix}
```python
{_MAGIC_}
```
```

`apply_chat_template` 渲染完整对话后，按 `_MAGIC_` 切一刀只留前半段，结果是「user 完整 + assistant 已经写到 ```` ```python\n ```` 这一行」的状态送给 vllm。

效果：
- 模型从 ```` ```python\n ```` 直接续写函数体
- 默认 EOS 加了 `\n```\n`，函数写完会自然停
- 相当于**手动跳过了 chat 模型的格式包装**，模型不需要再写 markdown 头/尾
- Qwen3 chat template 在 assistant 已有 content 时会自动渲染 `<think>\n\n</think>\n\n`（"已完成思考"占位），所以**也变相跳过了 thinking 阶段**

## 4. EOS 列表（停止 token）

`evalplus/provider/utility.py:4-19`：

```python
EOS = ["<|endoftext|>", "<|endofmask|>", "</s>",
       "\nif __name__", "\ndef main(", "\nprint("]

# direct-completion 模式额外加：
humaneval -> ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
mbpp      -> ['\n"""', "\nassert"]

# chat 模式额外加：
chat -> ["\n```\n"]   # 关闭 markdown 代码块
```

## 5. 用 Qwen3-1.7B (instruct) 实跑出来的 chat-template 样本

格式参考 `Lucky_RL/lucky_data/math_datasets/train_verl_math/train_example.json`。

文件已存到 `/tmp/evalplus_prompt_samples.json`，共 4 条（HumanEval×2 + MBPP×2）。

下面挑两个看：

### 5.1 HumanEval/0 — `has_close_elements`

**`prompt`（chat-template 格式，list of {role, content}）**：

```json
[
  {
    "role": "user",
    "content": "Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n```\n"
  },
  {
    "role": "assistant",
    "content": "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:\n```python\n-[[]]-this-is-really-our-highest-priority-[[]]-\n```\n"
  }
]
```

**渲染后真正喂给 LLM 的字符串**（Qwen3 chat template + split at MAGIC，注意 `<think>\n\n</think>\n\n` 是模板自动插入的占位）：

```
<|im_start|>user
Please provide a self-contained Python script that solves the following problem in a markdown code block:
```
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
```
<|im_end|>
<|im_start|>assistant
<think>

</think>

Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:
```python
```

模型从最后一行 ```` ```python ```` 之后**直接续写函数体**。

---

### 5.2 Mbpp/2 — `similar_elements`

**`prompt`**：

```json
[
  {
    "role": "user",
    "content": "Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```\n\"\"\"\nWrite a function to find the shared elements from the given two lists.\nassert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\n\"\"\"\n```\n"
  },
  {
    "role": "assistant",
    "content": "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:\n```python\n-[[]]-this-is-really-our-highest-priority-[[]]-\n```\n"
  }
]
```

**渲染后**：

```
<|im_start|>user
Please provide a self-contained Python script that solves the following problem in a markdown code block:
```
"""
Write a function to find the shared elements from the given two lists.
assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))
"""
```
<|im_end|>
<|im_start|>assistant
<think>

</think>

Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:
```python
```

注意 MBPP 的原始 `task["prompt"]` 长这样（不像 HumanEval 给函数签名）：
```
"""
Write a function to find the shared elements from the given two lists.
assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))
"""
```
docstring 里的 `assert` 同时充当**任务说明**和**第一个测试用例的 hint**（让模型知道函数名、参数形态、返回类型）。

## 6. Base 模型怎么走（`--force_base_prompt`）

base 模式下完全跳过 `make_raw_chat_prompt`，直接把 `task["prompt"]` 当作续写 prefix 喂给模型，没有任何 chat 包装。所以 Qwen3-1.7B-Base 看到的 HumanEval/0 prompt 就是：

```
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
```

模型直接续写函数体，靠 EOS（`\ndef `、`\nclass ` 等）截断。这就是为什么 MBPP 上 base 模型反而比 instruct 高 —— MBPP 的 prompt 自带 docstring + assert，对 base 模型的续写范式天然友好。

## 7. 完整 4 条样本

见 `/tmp/evalplus_prompt_samples.json`，每条包含：
- `data_source` — `evalplus/humaneval` 或 `evalplus/mbpp`
- `task_id` — 比如 `HumanEval/0`、`Mbpp/2`
- `entry_point` — 函数名（用于 sanitize 时定位）
- `prompt` — chat template 格式 list（`[{role: user, ...}, {role: assistant, ...}]`）
- `raw_task_prompt` — 原始 `task["prompt"]`（base 模式直接喂这个）
- `rendered_prompt_to_llm` — 走 chat template 后实际进 LLM 的字符串
- `reward_model.ground_truth` — `task["canonical_solution"]`（标准答案）
