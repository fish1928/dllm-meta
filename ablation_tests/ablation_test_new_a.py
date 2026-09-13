"""New ablation, stage A: feature combination x normalization x loss, over the
three benchmark-tail dataset groups.

Grid axes:
  features:  curated anchors over {attn_all, attn_last, conf, conf_aged,
             margin, pos_delta, mask_density} -- conf (fresh, LEAKY: offline
             ceiling only) and conf_aged (deployable) appear as paired anchors
             so the fresh-vs-aged gap is measurable per configuration
  norms:     raw, znorm_row, rank, minmax_row, znorm_global, log_znorm,
             softmax_attn      (only rank/softmax_attn are deploy-ready)
  losses:    uniform_within_h, decay_within_h, bce_within_h, bce_balanced,
             plackett_luce
  datasets:  gsm8k | ifeval_followbench | mix5 (gsm8k, minerva_math, bbh,
             humaneval, truthfulqa_gen)      -- oracle tails from stage 2

Full grid is ~1100 runs; RESUME is on by default (finished experiments are
skipped), so it can be phased with the filters below or split across GPUs.

Env:
  FOLDER_ORACLE   stats root (default stats_oracle)
  THREAD          oracle thread (default llada_base; dream threads: set
                  NUM_LAYERS=28)
  NUM_BLOCKS      oracle collection num_blocks suffix (default 1)
  DEVICE, NUM_LAYERS, NUM_EPOCHS, H, MAX_CONF_AGE
  DATASET_FILTER  run only this dataset group
  GRID_FILTER     substring filter on experiment names
  RESET=1         wipe this stage's records first;  RERUN=1  ignore resume

  python ablation_test_new_a.py                      # everything
  DATASET_FILTER=gsm8k GRID_FILTER=plackett_luce python ablation_test_new_a.py
"""

import os

from ablation_test_common_new import (
    LOSSES_ALL,
    NORMALIZATIONS_ALL,
    REPORT_PATH,
    estimate_balanced_pos_weight_multi,
    reset_stage,
    resolve_datasets,
    run_experiment_multi,
)

STAGE = 'new_a'

FOLDER_ORACLE = os.environ.get('FOLDER_ORACLE', 'stats_oracle')
THREAD = os.environ.get('THREAD', 'llada_base')
NUM_BLOCKS = int(os.environ.get('NUM_BLOCKS', 1))
DEVICE = os.environ.get('DEVICE', 'cuda:0')
NUM_LAYERS = int(os.environ.get('NUM_LAYERS', 32))    # dream: 28
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 10))
H = int(os.environ.get('H', 5))
MAX_CONF_AGE = int(os.environ.get('MAX_CONF_AGE', 16))
DATASET_FILTER = os.environ.get('DATASET_FILTER', '')
GRID_FILTER = os.environ.get('GRID_FILTER', '')

DATASET_GROUPS = {
    'gsm8k': ['gsm8k'],
    'ifeval_followbench': ['ifeval', 'followbench'],
    'mix5': ['gsm8k', 'minerva_math', 'bbh', 'humaneval', 'truthfulqa_gen'],
}

# curated anchors: attention choice x conf treatment x geometry/margin.
# 'geo' = pos_delta + mask_density (the incumbent llada-base winner is
# attn_last_geo with softmax_attn + plackett_luce).
FEATURE_ANCHORS = {
    'attn_last':                  ['attn_last'],
    'attn_all':                   ['attn_all'],
    'attn_last_geo':              ['attn_last', 'pos_delta', 'mask_density'],
    'attn_all_geo':               ['attn_all', 'pos_delta', 'mask_density'],
    'attn_last_geo_conf':         ['attn_last', 'pos_delta', 'mask_density', 'conf'],
    'attn_last_geo_conf_aged':    ['attn_last', 'pos_delta', 'mask_density', 'conf_aged'],
    'attn_last_geo_margin':       ['attn_last', 'pos_delta', 'mask_density', 'margin'],
    'attn_last_conf_aged_margin': ['attn_last', 'conf_aged', 'margin'],
    'no_attn_aged':               ['conf_aged', 'margin', 'pos_delta', 'mask_density'],
    'full_fresh':                 ['attn_last', 'conf', 'margin', 'pos_delta', 'mask_density'],
    'full_aged':                  ['attn_last', 'conf_aged', 'margin', 'pos_delta', 'mask_density'],
}

if os.environ.get('RESET', '') == '1':
    reset_stage(STAGE, REPORT_PATH)

count_total = 0
for name_group, names_task in DATASET_GROUPS.items():
    if DATASET_FILTER and name_group != DATASET_FILTER:
        continue

    datasets = resolve_datasets(names_task, FOLDER_ORACLE, THREAD, NUM_BLOCKS)
    if not datasets:
        print(f'[warn] dataset group {name_group}: no oracle folders found, skipped')
        continue

    balanced_pos_weight = None
    if 'bce_balanced' in LOSSES_ALL:
        balanced_pos_weight = estimate_balanced_pos_weight_multi(datasets, h=H)
        print(f'[{name_group}] balanced BCE pos_weight = {balanced_pos_weight:.2f}')

    for name_anchor, feature_names in FEATURE_ANCHORS.items():
        for normalization in NORMALIZATIONS_ALL:
            for loss_name in LOSSES_ALL:
                name = f'{name_anchor}__{normalization}__{loss_name}__data-{name_group}'
                if GRID_FILTER and GRID_FILTER not in name:
                    continue
                count_total += 1

                run_experiment_multi(
                    stage=STAGE,
                    name=name,
                    dataset_group=name_group,
                    datasets=datasets,
                    feature_names=feature_names,
                    normalization=normalization,
                    loss_name=loss_name,
                    loss_pos_weight=balanced_pos_weight if loss_name == 'bce_balanced' else None,
                    router_name='mlp',
                    router_kwargs={'dim_hidden': 64, 'num_blocks_mlp': 2},
                    h=H,
                    device=DEVICE,
                    num_layers=NUM_LAYERS,
                    num_epochs=NUM_EPOCHS,
                    max_conf_age=MAX_CONF_AGE,
                )

print(f'\n[new_a] grid walked: {count_total} experiments (finished ones were skipped)')
