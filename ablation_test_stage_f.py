"""
Stage F: final repeated evaluation and checkpoint saving.

Replace the configuration below with the winner from Stage E.
"""

import os

from ablation_test_common import REPORT_PATH, reset_stage, run_experiment


FOLDER_DATA = "stats_gsm8k"
DEVICE = "cuda:0"
H = 5
SIZE_BLOCK = 128
NUM_LAYERS = 32
NUM_EPOCHS = 20
STAGE = "stage_f"
CHECKPOINT_FOLDER = "ablation_test_checkpoints"

FINAL_FEATURES = ["attn_last", "conf", "margin", "pos_delta", "mask_density"]
FINAL_NORMALIZATION = "log_znorm"
FINAL_LOSS = "decay_within_h"
FINAL_LOSS_POS_WEIGHT = None
FINAL_ROUTER = "mlp"
FINAL_ROUTER_KWARGS = {"dim_hidden": 64, "num_blocks_mlp": 2}
SEEDS = [233, 234, 235, 236, 237]

reset_stage(STAGE, REPORT_PATH)
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

for seed in SEEDS:
    checkpoint_path = os.path.join(
        CHECKPOINT_FOLDER,
        f"final_router_seed_{seed}.pt",
    )
    run_experiment(
        stage=STAGE,
        name=f"final_seed_{seed}",
        folder_data=FOLDER_DATA,
        feature_names=FINAL_FEATURES,
        normalization=FINAL_NORMALIZATION,
        loss_name=FINAL_LOSS,
        loss_pos_weight=FINAL_LOSS_POS_WEIGHT,
        router_name=FINAL_ROUTER,
        router_kwargs=FINAL_ROUTER_KWARGS,
        h=H,
        size_block=SIZE_BLOCK,
        device=DEVICE,
        num_layers=NUM_LAYERS,
        num_epochs=NUM_EPOCHS,
        seed=seed,
        evaluate_result_groups=True,
        checkpoint_path=checkpoint_path,
    )
