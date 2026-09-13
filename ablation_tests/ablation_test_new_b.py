"""New ablation, stage B: router architecture and training horizon, over the
three benchmark-tail dataset groups, at the stage-A winning recipe.

Two sub-stages in one report:
  new_b_arch:    linear | mlp (3 capacities) | set_attention, plus mockup
                 reference rows (random floor, raw-attention baseline,
                 nearest-right heuristic)
  new_b_horizon: h in {3,5,7,9,11,13,15} on the reference architecture.
                 recall@5 is always evaluated regardless of h, so rows stay
                 comparable on one basis (plus recall@h_train per row).

Update BEST_* below (or via env) after reading the stage-A DB results.
Every experiment name carries all varying parameters -- the old
stage_e_tune_h reused one name for all horizons and silently kept only the
last record; do not repeat that.

Env: same as ablation_test_new_a (FOLDER_ORACLE, THREAD, NUM_BLOCKS, DEVICE,
NUM_LAYERS, NUM_EPOCHS, MAX_CONF_AGE, DATASET_FILTER, GRID_FILTER, RESET,
RERUN), plus BEST_FEATURES (comma list), BEST_NORMALIZATION, BEST_LOSS.
"""

import os

from ablation_test_common_new import (
    REPORT_PATH,
    reset_stage,
    resolve_datasets,
    run_experiment_multi,
)

FOLDER_ORACLE = os.environ.get('FOLDER_ORACLE', 'stats_oracle')
THREAD = os.environ.get('THREAD', 'llada_base')
NUM_BLOCKS = int(os.environ.get('NUM_BLOCKS', 1))
DEVICE = os.environ.get('DEVICE', 'cuda:0')
NUM_LAYERS = int(os.environ.get('NUM_LAYERS', 32))    # dream: 28
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 10))
MAX_CONF_AGE = int(os.environ.get('MAX_CONF_AGE', 16))
DATASET_FILTER = os.environ.get('DATASET_FILTER', '')
GRID_FILTER = os.environ.get('GRID_FILTER', '')

# ---- stage-A winners (edit after querying the DB; deployable rows only:
# normalization in (rank, softmax_attn), no fresh conf) ----
BEST_FEATURES = os.environ.get('BEST_FEATURES', 'attn_last,pos_delta,mask_density').split(',')
BEST_NORMALIZATION = os.environ.get('BEST_NORMALIZATION', 'softmax_attn')
BEST_LOSS = os.environ.get('BEST_LOSS', 'plackett_luce')

H_DEFAULT = 5

DATASET_GROUPS = {
    'gsm8k': ['gsm8k'],
    'ifeval_followbench': ['ifeval', 'followbench'],
    'mix5': ['gsm8k', 'minerva_math', 'bbh', 'humaneval', 'truthfulqa_gen'],
}

# name -> (router_name, router_kwargs, feature_names_override)
ARCHITECTURES = {
    'linear':          ('linear', {}, None),
    'mlp_d32_b2':      ('mlp', {'dim_hidden': 32, 'num_blocks_mlp': 2}, None),
    'mlp_d64_b2':      ('mlp', {'dim_hidden': 64, 'num_blocks_mlp': 2}, None),    # reference
    'mlp_d128_b3':     ('mlp', {'dim_hidden': 128, 'num_blocks_mlp': 3}, None),
    'set_attention':   ('set_attention', {'dim_model': 32, 'num_heads': 1, 'dim_hidden': 64}, None),
    # reference rows (untrained): floor / raw-signal baselines
    'mockup_random':   ('mockup_random', {}, None),
    'mockup_raw_attn': ('mockup_raw', {}, ['attn_last']),
    # nearest_right reads Feature_pos_delta's signed dim and needs it FIRST
    'mockup_nearest_right': ('mockup_nearest_right', {},
                             ['pos_delta'] + [f for f in BEST_FEATURES if f != 'pos_delta']),
}

HORIZONS = [3, 5, 7, 9, 11, 13, 15]
ROUTER_HORIZON = ('mlp', {'dim_hidden': 64, 'num_blocks_mlp': 2})

if os.environ.get('RESET', '') == '1':
    reset_stage('new_b_arch', REPORT_PATH)
    reset_stage('new_b_horizon', REPORT_PATH)

for name_group, names_task in DATASET_GROUPS.items():
    if DATASET_FILTER and name_group != DATASET_FILTER:
        continue

    datasets = resolve_datasets(names_task, FOLDER_ORACLE, THREAD, NUM_BLOCKS)
    if not datasets:
        print(f'[warn] dataset group {name_group}: no oracle folders found, skipped')
        continue

    '''architecture sweep (fixed h)'''
    for name_arch, (router_name, router_kwargs, features_override) in ARCHITECTURES.items():
        name = f'arch-{name_arch}__h{H_DEFAULT}__data-{name_group}'
        if GRID_FILTER and GRID_FILTER not in name:
            continue
        run_experiment_multi(
            stage='new_b_arch',
            name=name,
            dataset_group=name_group,
            datasets=datasets,
            feature_names=features_override or BEST_FEATURES,
            normalization=BEST_NORMALIZATION,
            loss_name=BEST_LOSS,
            router_name=router_name,
            router_kwargs=router_kwargs,
            h=H_DEFAULT,
            device=DEVICE,
            num_layers=NUM_LAYERS,
            num_epochs=NUM_EPOCHS,
            max_conf_age=MAX_CONF_AGE,
        )

    '''horizon sweep (fixed reference architecture)'''
    for h in HORIZONS:
        name = f'h{h}__mlp_d64_b2__data-{name_group}'
        if GRID_FILTER and GRID_FILTER not in name:
            continue
        run_experiment_multi(
            stage='new_b_horizon',
            name=name,
            dataset_group=name_group,
            datasets=datasets,
            feature_names=BEST_FEATURES,
            normalization=BEST_NORMALIZATION,
            loss_name=BEST_LOSS,
            router_name=ROUTER_HORIZON[0],
            router_kwargs=ROUTER_HORIZON[1],
            h=h,
            device=DEVICE,
            num_layers=NUM_LAYERS,
            num_epochs=NUM_EPOCHS,
            max_conf_age=MAX_CONF_AGE,
        )
