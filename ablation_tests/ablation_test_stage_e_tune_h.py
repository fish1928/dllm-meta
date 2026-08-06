"""Stage E: router architecture and small capacity comparison."""
import os

from ablation_test_common import REPORT_PATH, reset_stage, run_experiment

REPORT_PATH = 'ablation_test_report_stage_e_tune_h.json'

FOLDER_DATA = os.environ['FOLDER_DATA']
DEVICE = os.environ.get('DEVICE','cuda:0')
SIZE_BLOCK = int(os.environ['SIZE_BLOCK'])
NUM_LAYERS = 32
NUM_EPOCHS = 10
STAGE = "stage_e"

# Replace these values with the winning Stage D configuration.
#BEST_FEATURES = ["attn_last", "pos_delta", "mask_density"]
#BEST_FEATURES = ["attn_last", "conf", "margin", "pos_delta", "mask_density"]
BEST_FEATURES = ["attn_last", "pos_delta", "mask_density"]
BEST_NORMALIZATION = "rank"
BEST_LOSS = "plackett_luce"
BEST_LOSS_POS_WEIGHT = None

reset_stage(STAGE, REPORT_PATH)

ROUTERS = [
    ("mlp_h64_b2", "mlp", 5, {"dim_hidden": 64, "num_blocks_mlp": 2}),
    ("mlp_h64_b2", "mlp", 7, {"dim_hidden": 64, "num_blocks_mlp": 2}),
    ("mlp_h64_b2", "mlp", 9, {"dim_hidden": 64, "num_blocks_mlp": 2}),
    ("mlp_h64_b2", "mlp", 11, {"dim_hidden": 64, "num_blocks_mlp": 2}),
    ("mlp_h64_b2", "mlp", 13, {"dim_hidden": 64, "num_blocks_mlp": 2}),
    ("mlp_h64_b2", "mlp", 15, {"dim_hidden": 64, "num_blocks_mlp": 2}),
]

for experiment_name, router_name, h, router_kwargs in ROUTERS:
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
        h=h,
        size_block=SIZE_BLOCK,
        device=DEVICE,
        num_layers=NUM_LAYERS,
        num_epochs=NUM_EPOCHS,
    )
