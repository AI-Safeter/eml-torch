# Language preprocessing correction before held-out evaluation

The first 42-fit grid used raw document tokens without Gemma's BOS token. A
selection-only check on 16 documents measured native language loss of 9.764632
nats/token without BOS and 4.795804 with BOS. This is a preprocessing error,
not evidence about EML. No replacement gate or final outputs had been evaluated.

The corrected run appends no new document content: it prepends the tokenizer's
BOS to each existing 256-token document. Language loss now covers all 256 text
tokens. The protocol's document length means text tokens; model input length is
257 including BOS. Document identities, hostname split, arithmetic tasks,
architecture grid, seeds, training budgets, selection rule, and acceptance
thresholds remain fixed. Held-out labels and inputs were prepared previously,
but their model outputs remain unopened. They remain the confirmation sets.

Use a separate run root ending in `-bos`. Preserve every original run. Reuse
only arithmetic activations, whose native chat prompts already contain BOS;
recollect language train/selection activations and retrain all 42 candidates
from their prescribed initialization. Never reuse fitted weights from the
incorrectly tokenized run. `prepare_corrected.py` records the original data and
reused arithmetic activation hashes. The collection and training freezes bind
the corrected inputs. Native language scoring already consumes the stored
sequence, so no scorer or acceptance adjustment is needed.

Arithmetic mechanism experiments remain in the original root and are unaffected
by this document-only correction. Results from the first replacement grid are
diagnostic and cannot establish deployment quality.
