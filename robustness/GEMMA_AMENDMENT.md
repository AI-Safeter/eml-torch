# User-requested Gemma replication

On 2026-09-08, the user requested Gemma E2B instead of SmolLM2. The primary model roster is amended to Qwen3-1.7B, Qwen3-4B, and **Gemma 4 E2B IT** (`google/gemma-4-E2B-it`). Its cached official revision and backend versions are pinned in `gemma/model.json`. The initial conversational reference to Gemma 3n was corrected after inspecting the available Gemma 4 checkpoint.

This change was requested after SmolLM2's addition ordinary-generation results were observed. It is not a selection based on Gemma test performance: no Gemma outputs have been collected by this study at the time of this amendment. SmolLM2's 90 completed fits and all completed evaluation artifacts remain as a superseded, incomplete arm. Its remaining inference jobs were stopped at the user's request. Neither those stops nor the model change are counted as failed fits. Qwen evaluation continues.

Gemma receives all three operations, all existing frozen data and prompt cohorts, the same rank-32/width-32 EML and SiLU heads, all five seeds, the same fitting budget, controls, intervention strengths, and primary acceptance criteria. The amended primary comparison still has nine model/operation cells. It uses shared operand data across models, not independent dataset replications. No test-based adjustment of width, depth, loss, or acceptance thresholds is allowed.

Gemma requires a separate backend (PyTorch 2.9.0+cu128 and Transformers 5.16.1). The Qwen environment remains as originally frozen. Cross-family results therefore do not isolate a software-version effect. Within Gemma, every head and control uses the same backend, float32 precision, and disabled TF32. Native Gemma execution preserves its per-layer embeddings, normalization, attention, generation, and final logit soft cap. Original-neuron controls must use Gemma's native GELU activation; the parameter-matched learned SiLU comparator remains SiLU.

Before collection, verify native-versus-adapter outputs, standalone MLP outputs and input gradients, zero-delta restoration, and generation hooks on training/validation prompts only. Freeze the Gemma adapter and execution sources before fitting/evaluation. The final report must identify the amended roster and preserve the superseded SmolLM2 results with their incomplete status.

The official [Gemma 4 model card](https://huggingface.co/google/gemma-4-E2B-it) distinguishes effective parameter count from total stored parameters. Report actual loaded counts; do not label the full checkpoint as two billion stored parameters.
