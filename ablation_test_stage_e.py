"""Stage E: router architecture and small capacity comparison."""

from ablation_test_common import REPORT_PATH, reset_stage, run_experiment


FOLDER_DATA = "stats_gsm8k"
DEVICE = "cuda:0"
H = 5
SIZE_BLOCK = 128
NUM_LAYERS = 32
NUM_EPOCHS = 10
STAGE = "stage_e"

# Replace these values with the winning Stage D configuration.
BEST_FEATURES = ["attn_last", "conf", "margin", "pos_delta", "mask_density"]
BEST_NORMALIZATION = "log_znorm"
BEST_LOSS = "decay_within_h"
BEST_LOSS_POS_WEIGHT = None

reset_stage(STAGE, REPORT_PATH)

ROUTERS = [
    ("linear", "linear", {}),
    ("mlp_h32_b2", "mlp", {"dim_hidden": 32, "num_blocks_mlp": 2}),
    ("mlp_h64_b1", "mlp", {"dim_hidden": 64, "num_blocks_mlp": 1}),
    ("mlp_h64_b2", "mlp", {"dim_hidden": 64, "num_blocks_mlp": 2}),
    ("mlp_h64_b3", "mlp", {"dim_hidden": 64, "num_blocks_mlp": 3}),
    ("mlp_h128_b2", "mlp", {"dim_hidden": 128, "num_blocks_mlp": 2}),
    (
        "set_attention",
        "set_attention",
        {"dim_model": 32, "num_heads": 1, "dim_hidden": 64},
    ),
]

for experiment_name, router_name, router_kwargs in ROUTERS:
    run_experiment(
        stage=STAGE,
        name=experiment_name,
        folder_data=FOLDER_DATA,
        feature_names=BEST_FEATURES,
        normalization=BEST_NORMALIZATION,
        loss_name=BEST_LOSS,
        loss_pos_weight=BEST_LOSS_POS_WEIGHT,
        router_name=router_name,
        router_kwargs=router_kwargs,
        h=H,
        size_block=SIZE_BLOCK,
        device=DEVICE,
        num_layers=NUM_LAYERS,
        num_epochs=NUM_EPOCHS,
    )
