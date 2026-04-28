---
name: Local evalplus patches for max_new_tokens / max_model_len
description: This evalplus checkout has been edited to expose --max_new_tokens and --max_model_len through the CLI; relevant when running or modifying its codegen pipeline.
type: project
---

The evalplus checkout at `/ssd2/zhizhou/workspace/rotation-project/evalplus` has been **locally patched** (not upstream):

- `evalplus/provider/vllm.py` — `max_model_len` is now a parameter (was hardcoded to `2048`).
- `evalplus/provider/__init__.py` — `make_model` now accepts and forwards `max_new_tokens` and `max_model_len` for the vllm backend.
- `evalplus/codegen.py` — `run_codegen` now accepts `max_new_tokens` and `max_model_len`, defaulting to `768` / `2048`.

**Why:** Need long-form generation (4k tokens) for evaluation runs; upstream defaults can't fit it.

**How to apply:** When running `python -m evalplus.evaluate`, pass `--max_new_tokens N --max_model_len M`. Without these flags, the run still works at the original defaults. If you `git pull` from upstream, these edits will conflict — re-apply or merge before running long-context evals.

Editable install lives in conda env `best176` (`/ssd2/zhizhou/miniconda3/envs/best176`).
