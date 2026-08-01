# Simple LLaDA router ablation scripts

Put these files in the same folder as `router_llada.py`:

- `ablation_test_common.py`
- `ablation_test_stage_a.py`
- `ablation_test_stage_b.py`
- `ablation_test_stage_c.py`
- `ablation_test_stage_d.py`
- `ablation_test_stage_e.py`
- `ablation_test_stage_f.py`

Each stage appends its results to:

```text
ablation_test_report.json
```

Run the stages in order:

```bash
python ablation_test_stage_a.py
python ablation_test_stage_b.py
python ablation_test_stage_c.py
python ablation_test_stage_d.py
python ablation_test_stage_e.py
python ablation_test_stage_f.py
```

Before running each file, edit the constants at the top:

- `FOLDER_DATA`
- `DEVICE`
- `H`
- `SIZE_BLOCK`
- `NUM_LAYERS`
- `NUM_EPOCHS`

After Stage B, copy the best loss and normalization into Stage C. After Stage C, copy the strongest feature sets into Stage D. Continue this process through Stage F.

Stage B contains 70 screening experiments. Keep `NUM_EPOCHS = 1` or `3` for the first test, then increase it for serious training.

The online sparse-decoding validation stage is not included because it requires your actual LLaDA sparse decoding function, benchmark runner, latency measurement, and refresh policy.
