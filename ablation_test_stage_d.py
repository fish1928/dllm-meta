"""
Stage D: compact factorial confirmation.

Edit the top feature sets, normalizations, and losses after Stages B and C.
"""

from ablation_test_common import (
    REPORT_PATH,
    estimate_balanced_pos_weight,
    reset_stage,
    run_experiment,
)


FOLDER_DATA = "stats_gsm8k"
DEVICE = "cuda:0"
H = 5
SIZE_BLOCK = 128
NUM_LAYERS = 32
NUM_EPOCHS = 10
STAGE = "stage_d"

# Replace these defaults with the three strongest Stage C feature sets.
TOP_FEATURE_SETS = {
    "attention_only": ["attn_last"],
    "attention_geometry": ["attn_last", "pos_delta", "mask_density"],
    "core_mixed": ["attn_last", "conf", "margin", "pos_delta", "mask_density"],
}

# Replace these with the two strongest Stage B normalization recipes.
TOP_NORMALIZATIONS = ["log_znorm", "rank"]

# Replace these with the two strongest Stage B losses.
TOP_LOSSES = ["decay_within_h", "uniform_within_h"]

SEEDS = [233, 234, 235]

reset_stage(STAGE, REPORT_PATH)

balanced_pos_weight = None
if "bce_balanced" in TOP_LOSSES:
    balanced_pos_weight = estimate_balanced_pos_weight(
        FOLDER_DATA,
        h=H,
        size_block=SIZE_BLOCK,
    )

for feature_set_name, feature_names in TOP_FEATURE_SETS.items():
    for normalization in TOP_NORMALIZATIONS:
        for loss_name in TOP_LOSSES:
            for seed in SEEDS:
                pos_weight = balanced_pos_weight if loss_name == "bce_balanced" else None
                experiment_name = (
                    f"{feature_set_name}__{normalization}__{loss_name}__seed_{seed}"
                )
                run_experiment(
                    stage=STAGE,
                    name=experiment_name,
                    folder_data=FOLDER_DATA,
                    feature_names=feature_names,
                    normalization=normalization,
                    loss_name=loss_name,
                    loss_pos_weight=pos_weight,
                    router_name="mlp",
                    router_kwargs={"dim_hidden": 64, "num_blocks_mlp": 2},
                    h=H,
                    size_block=SIZE_BLOCK,
                    device=DEVICE,
                    num_layers=NUM_LAYERS,
                    num_epochs=NUM_EPOCHS,
                    seed=seed,
                )
