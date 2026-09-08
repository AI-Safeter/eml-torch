# Archived V-cache compression pilot

[Results](RESULTS.md) and `evidence/` preserve the Gemma E2B comparison: EML did not meet the quality target, and packed 4-bit V quantization was the strongest observed compression baseline. This was a small pilot, not a demonstration of improved answer quality or serving speed.

The failed pilot's executable code has been retired. Its complete source and reproduction commands remain in [commit 51cb441](https://github.com/AI-Safeter/eml-torch/tree/51cb441c80ef6eef5b2fe4480b1c6b45ec896cf5/kv_cache). Source hashes in `evidence/freeze.json` refer to that version. Checkpoints and sampled activations remain outside Git in the sibling `emltorch-kv-runs` directory.

The [one-layer correction study](../kv_correction/README.md) separately tests whether EML can correct quantization error without a low-dimensional encoding bottleneck.

SQuAD questions and answer references are from [Rajpurkar et al.'s release](https://rajpurkar.github.io/SQuAD-explorer/), distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Dataset excerpts retain that attribution and license.
