"""AGE-CHANNEL experiment: is (stale value + its exact age) the right way to
feed conf/margin to the router?

Motivation: the __deployable (no conf/margin) bundle loses ~0.3 recall on bbh
vs the conf/margin bundle -- logits-derived signals are necessary. But fresh
values leak (they don't exist at deployment) and uniformly-aged values throw
away information the runner actually HAS: online, every conf/margin table
entry's age is known exactly (steps since that position was last probed or
refreshed). So the deployment-faithful input is the PAIR (stale value, age).

Two arms, both trained with the CURRENT BEST recipe (softmax_attn +
plackett_luce + set_attention d32/h1/hid64, mix_no_ifeval group):

  aged_with_age  attn_last + pos_delta + mask_density
                 + (conf value, conf age) + (margin value, margin age)
                 values randomly aged up to MAX_CONF_AGE; the SAME sampled age
                 is fed as a second channel per signal (random-age = train-time
                 augmentation over the whole age range; online we feed the true
                 table value and its true age)

  policy_pair    attn_last + pos_delta + mask_density
                 + conf_policy + margin_policy
                 deterministic refresh-clock aging, NO age input -- the same
                 feature set as stage-A 'full_all_policy', retrained here so
                 both arms share epochs/seed and are directly comparable

Records land in stage 'age_channel' of the SAME per-thread fast report, so
build_report_html.py shows them as an extra section and resume works as usual.

Usage:
  FOLDER_TRAIN=stats_train THREAD=llada_base DEVICE=cuda:0 \
      nohup python -u ablation_tests/ablation_test_age_channel.py > age.log 2>&1 &
Env: FOLDER_TRAIN, THREAD, NUM_BLOCKS (1), DEVICE, NUM_LAYERS (32; dream 28),
     NUM_EPOCHS (10 -- matches the stage-A table for cross-reading),
     H (5), MAX_CONF_AGE (16), REPORT_PATH (defaults like ablation_test_fast).

NOTE offline-only for now: 'conf_aged_age'/'margin_aged_age' need an online
margin table plus per-position age tracking in the snapshot before an e2e run
(build_online_x extension); this experiment decides whether that wiring is
worth writing.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation_test_common_new import (
    find_record,
    parse_summ,
    resolve_datasets,
    run_experiment_multi,
)

FOLDER_TRAIN = os.environ.get('FOLDER_TRAIN', 'stats_train')
THREAD = os.environ.get('THREAD', 'llada_base')

_report_default = f'ablation_test_report_fast_{THREAD}.json'
if THREAD == 'llada_base' and not os.path.exists(_report_default) \
        and os.path.exists('ablation_test_report_fast.json'):
    _report_default = 'ablation_test_report_fast.json'
REPORT_PATH = os.environ.get('REPORT_PATH', _report_default)

NUM_BLOCKS = int(os.environ.get('NUM_BLOCKS', 1))
DEVICE = os.environ.get('DEVICE', 'cuda:0')
NUM_LAYERS = int(os.environ.get('NUM_LAYERS', 32))
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 10))    # match the stage-A table
H = int(os.environ.get('H', 5))
MAX_CONF_AGE = int(os.environ.get('MAX_CONF_AGE', 16))

GROUP = 'mix_no_ifeval'
TASKS = ['gsm8k', 'minerva_math', 'bbh', 'humaneval', 'truthfulqa_gen']

RECIPE = {
    'normalization': 'softmax_attn',
    'loss_name': 'plackett_luce',
    'router_name': 'set_attention',
    'router_kwargs': {'dim_model': 32, 'num_heads': 1, 'dim_hidden': 64},
}

ARMS = {
    'aged_with_age': ['attn_last', 'pos_delta', 'mask_density',
                      'conf_aged_age', 'margin_aged_age'],
    'policy_pair':   ['attn_last', 'pos_delta', 'mask_density',
                      'conf_policy', 'margin_policy'],
}

STAGE = 'age_channel'


def main():
    datasets = resolve_datasets(TASKS, FOLDER_TRAIN, THREAD, NUM_BLOCKS)
    assert datasets, f'no train folders for group {GROUP} under {FOLDER_TRAIN}'
    print(f'group {GROUP}: {[(n, s) for n, _, s in datasets]}  '
          f'(epochs={NUM_EPOCHS}, h={H}, max_age={MAX_CONF_AGE})')

    for name, feats in ARMS.items():
        run_experiment_multi(stage=STAGE, name=name, feature_names=feats,
            dataset_group=GROUP, datasets=datasets, device=DEVICE,
            num_layers=NUM_LAYERS, num_epochs=NUM_EPOCHS,
            max_conf_age=MAX_CONF_AGE, h=H, report_path=REPORT_PATH, **RECIPE)
    # end

    print('\n===== age_channel results (recall@5, all / per task) =====')
    for name in ARMS:
        record = find_record(STAGE, name, REPORT_PATH)
        if not record or record.get('error') or not record.get('metrics'):
            print(f'  {name:16s} FAILED')
            continue
        # end
        metrics = record['metrics']
        parts = [f"all={metrics.get('all', {}).get('recall@5', '--')}"]
        for task in TASKS:
            parts.append(f"{task}={metrics.get(f'ds_{task}', {}).get('recall@5', '--')}")
        # end
        print(f'  {name:16s} ' + '  '.join(parts))
    # end

    values = {}
    for name in ARMS:
        record = find_record(STAGE, name, REPORT_PATH)
        parsed = parse_summ(((record or {}).get('metrics') or {}).get('all', {}).get('recall@5'))
        values[name] = parsed[0] if parsed else None
    # end
    if all(v is not None for v in values.values()):
        delta = values['aged_with_age'] - values['policy_pair']
        print(f'\n[verdict] age channel {"WINS" if delta > 0 else "loses"} by '
              f'{delta:+.3f} recall@5 over the policy pair '
              '(check the bbh column above -- that is where the clean bundle collapsed)')
        if delta > 0.01:
            print('[next] worth wiring online: margin table + per-position age tracking '
                  'in the snapshot, then the deciding e2e pair on bbh + gsm8k')
        # end
    # end
# end


if __name__ == '__main__':
    main()
# end
