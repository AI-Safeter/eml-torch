# Division seed interventions with streamed prefix weights

`streamed_cohort.py` computes the fixed Gemma division seed intervention cohort
on a GPU with less available memory. It preserves all 30 fitted heads, the
original control, the data, and the native FP32 CUDA arithmetic. It does not
generate answers or replace the main study queue.

Sixteen unchanged prefix MLPs, at layers 17 through 32, retain their weights in
pinned CPU memory. Each native MLP forward transfers its weights to CUDA and
then restores the CPU parameter references. The validated prefix cache reuses
unchanged layer outputs. The intervention remains at layer 33.

On a full batch of 64 validation prompts, all 11,904 method/condition records
match the resident-weight evaluator byte for byte. These are not independent
examples. Peak allocated memory was 8.344 GiB, versus 11.719 GiB for the
reference; reserved memory was 8.838 versus 12.098 GiB. CUDA context memory is
additional. The two runs used different shared H100s, so their elapsed times
are not a speed comparison. Streaming adds host transfers and keeps the full
weights in memory; this is neither compression nor an EML serving speedup.

Reproduce the comparison from the repository root with the pinned Gemma Python:

```bash
CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" robustness/gemma/streamed_cohort.py \
  --engine native --validation-only --output /absolute/new-reference
CUDA_VISIBLE_DEVICES=1 "$GEMMA_PYTHON" robustness/gemma/streamed_cohort.py \
  --engine streamed --validation-only --reference /absolute/new-reference \
  --output /absolute/new-candidate
CUDA_VISIBLE_DEVICES=1 "$GEMMA_PYTHON" robustness/gemma/validate_streamed.py \
  --output /absolute/new-controls.json
CUDA_VISIBLE_DEVICES=1 "$GEMMA_PYTHON" robustness/gemma/streamed_cohort.py \
  --engine streamed --output /absolute/new-seed-interventions --publish
```

The publication command requires the committed execution freeze and a live
main queue. It publishes a complete cohort atomically and yields if the main
queue reaches the same intervention cohort. The main queue can generate seed
answers concurrently; it later skips the completed intervention file. Omit
`--publish` for independent reproduction into a private directory.

`streamed-validation.json` records the numerical comparison. The compressed
native trace is retained as `streamed-native-validation.json.gz`.
`streamed-controls.json` checks equivalence to the original seed-selection
code, stage ownership, and CPU parameter restoration after a failed CUDA
forward. The final study audit verifies this execution freeze and the complete
test traces independently.
