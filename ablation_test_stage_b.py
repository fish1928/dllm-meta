"""Stage B: screen loss functions and normalization recipes."""

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
NUM_EPOCHS = 10  # Screening budget. Increase after the code is verified.
STAGE = "stage_b"

reset_stage(STAGE, REPORT_PATH)

# Two representative feature sets from the plan.
FEATURE_ANCHORS = {
    "attention_only": ["attn_last"],
    "attention_full": ["attn_all"],
    "stale_only": ["conf", "margin"],
    "easy_only": ["pos_delta", "mask_density"],
    "stale_mixed": ["attn_last", "conf", "margin"],
    "easy_mixed": ["attn_last", "pos_delta", "mask_density"],
    "easy_mixed_full": ["attn_all", "pos_delta", "mask_density"],
    "core_mixed": ["attn_last", "conf", "margin", "pos_delta", "mask_density"],
    "core_mixed_all": ["attn_all", "conf", "margin", "pos_delta", "mask_density"]
}

NORMALIZATIONS = [
    "raw",
    "znorm_row",
    "rank",
    "minmax_row",
    "znorm_global",
    "log_znorm",
    "softmax_attn",
]

BALANCED_POS_WEIGHT = estimate_balanced_pos_weight(
    FOLDER_DATA,
    h=H,
    size_block=SIZE_BLOCK,
)
print("balanced BCE pos_weight =", BALANCED_POS_WEIGHT)

LOSSES = [
    ("uniform_within_h", None),
    ("decay_within_h", None),
    ("bce_within_h", None),
    ("bce_balanced", BALANCED_POS_WEIGHT),
    ("plackett_luce", None),
]

for anchor_name, feature_names in FEATURE_ANCHORS.items():
    for normalization in NORMALIZATIONS:
        for loss_name, pos_weight in LOSSES:
            experiment_name = f"{anchor_name}__{normalization}__{loss_name}"
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
            )
