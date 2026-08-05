"""
Stage C: feature ablation.

After Stage B, edit BEST_NORMALIZATION and BEST_LOSS to match its result.
"""
import os
from ablation_test_common import REPORT_PATH, reset_stage, run_experiment


FOLDER_DATA = os.environ['FOLDER_DATA']
DEVICE = os.environ.get('DEVICE', 'cuda:0')
H = 5
SIZE_BLOCK = int(os.environ['SIZE_BLOCK'])
NUM_LAYERS = 32
NUM_EPOCHS = 5
STAGE = "stage_c"

# Edit these two values after reading the Stage B report.
BEST_NORMALIZATION = "rank"
BEST_LOSS = "decay_within_h"
BEST_LOSS_POS_WEIGHT = None

# Change this to attn_all if the attention-representation comparison favors it.
ATTENTION_BASE = "attn_last"

reset_stage(STAGE, REPORT_PATH)


def run_feature_set(name, feature_names):
    run_experiment(
        stage=STAGE,
        name=name,
        folder_data=FOLDER_DATA,
        feature_names=feature_names,
        normalization=BEST_NORMALIZATION,
        loss_name=BEST_LOSS,
        loss_pos_weight=BEST_LOSS_POS_WEIGHT,
        router_name="mlp",
        router_kwargs={"dim_hidden": 64, "num_blocks_mlp": 2},
        h=H,
        size_block=SIZE_BLOCK,
        device=DEVICE,
        num_layers=NUM_LAYERS,
        num_epochs=NUM_EPOCHS,
    )


# C1. Single-feature strength.
SINGLE_FEATURES = [
    "attn_last",
    "attn_all",
    "conf",
    "margin",
    "entropy",
    "pos_delta",
    "mask_density",
    "x0_stability",
]
for feature_name in SINGLE_FEATURES:
    run_feature_set(f"single__{feature_name}", [feature_name])

# C2. Attention representation.
run_feature_set("attention_repr__attn_last", ["attn_last"])
run_feature_set("attention_repr__attn_all", ["attn_all"])
run_feature_set("attention_repr__attn_all_plus_last", ["attn_all", "attn_last"])

# C3. Add one feature to the selected attention representation.
ADDITIONAL_FEATURES = [
    "conf",
    "margin",
    "entropy",
    "pos_delta",
    "mask_density",
    "x0_stability",
    "step_progress",
]
run_feature_set("add_one__attention_only", [ATTENTION_BASE])
for feature_name in ADDITIONAL_FEATURES:
    run_feature_set(
        f"add_one__{feature_name}",
        [ATTENTION_BASE, feature_name],
    )

# C4. Semantic feature groups.
STATE_FEATURES = ["conf", "margin", "entropy", "x0_stability"]
GEOMETRY_FEATURES = ["pos_delta", "mask_density"]

GROUPS = {
    "group__attention_only": [ATTENTION_BASE],
    "group__state_only": STATE_FEATURES,
    "group__geometry_only": GEOMETRY_FEATURES,
    "group__attention_state": [ATTENTION_BASE] + STATE_FEATURES,
    "group__attention_geometry": [ATTENTION_BASE] + GEOMETRY_FEATURES,
    "group__attention_state_geometry": [ATTENTION_BASE] + STATE_FEATURES + GEOMETRY_FEATURES,
    "group__all_plus_progress": [ATTENTION_BASE] + STATE_FEATURES + GEOMETRY_FEATURES + ["step_progress"],
}
for name, feature_names in GROUPS.items():
    run_feature_set(name, feature_names)

# C5. Leave-one-out from the full candidate set.
# Edit FULL_FEATURE_SET after reviewing C1-C4 if you want a smaller winning set.
FULL_FEATURE_SET = [
    ATTENTION_BASE,
    "conf",
    "margin",
    "entropy",
    "pos_delta",
    "mask_density",
    "x0_stability",
    "step_progress",
]
run_feature_set("leave_one_out__full", FULL_FEATURE_SET)
for removed_feature in FULL_FEATURE_SET:
    remaining = [f for f in FULL_FEATURE_SET if f != removed_feature]
    run_feature_set(f"leave_one_out__minus_{removed_feature}", remaining)
