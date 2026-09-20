"""Train the THREE conf/margin e2e bundles for the deciding comparison.

Arms (fixed recipe: softmax_attn + plackett_luce + set_attention d32/h1/hid64,
trained on mix_no_ifeval, EPOCHS_FINAL epochs):

  cm_clean    attn_last + pos_delta + mask_density          (no conf/margin)
  cm_policy   + conf_policy + margin_policy                 (deterministic
              refresh-clock aging at training time)
  cm_aged     + conf_aged + margin_aged                     (random-age
              augmentation at training time)
  cm_age      + conf_aged_age + margin_aged_age             (the AGED router:
              randomly aged values PLUS the true age as an input channel;
              deployment feeds (snapshot.conf/margin, snapshot.age) -- the
              runner tracks exact per-position staleness)

cm_clean/cm_policy/cm_aged deploy identically: the runner feeds the LIVE
stale conf/margin tables (snapshot.conf / snapshot.margin) -- those arms
differ only in how staleness was simulated during training. cm_age
additionally consumes the true age table at inference (spec features
conf_age/margin_age). Bundles land in FOLDER_BUNDLES
(routers_e2e) as <THREAD>__cm_<arm>.pt/.json, ready for run_llada_semi_mlp[_v2]
via path_router. Resume-safe: an arm whose .pt and .json already exist is
skipped (RERUN=1 to force).

Usage:
  FOLDER_TRAIN=stats_train THREAD=llada_base DEVICE=cuda:0 \
      nohup python -u ablation_tests/train_e2e_confmargin.py > train_cm.log 2>&1 &
Env: FOLDER_TRAIN, THREAD, NUM_BLOCKS (1), DEVICE, NUM_LAYERS (32; dream 28),
     EPOCHS_FINAL (20), H (5), MAX_CONF_AGE (16), FOLDER_BUNDLES (routers_e2e).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ablation_test_common_new import build_features, build_loss, resolve_datasets
from router_llada import FactoryRouter, RouterTrainer, build_geometry
from router_deploy import save_router_bundle

FOLDER_TRAIN = os.environ.get('FOLDER_TRAIN', 'stats_train')
THREAD = os.environ.get('THREAD', 'llada_base')
NUM_BLOCKS = int(os.environ.get('NUM_BLOCKS', 1))
DEVICE = os.environ.get('DEVICE', 'cuda:0')
NUM_LAYERS = int(os.environ.get('NUM_LAYERS', 32))
EPOCHS_FINAL = int(os.environ.get('EPOCHS_FINAL', 20))
H = int(os.environ.get('H', 5))
MAX_CONF_AGE = int(os.environ.get('MAX_CONF_AGE', 16))
FOLDER_BUNDLES = os.environ.get('FOLDER_BUNDLES', 'routers_e2e')

TASKS = ['gsm8k', 'minerva_math', 'bbh', 'humaneval', 'truthfulqa_gen']
GROUP = 'mix_no_ifeval'

RECIPE = {
    'normalization': 'softmax_attn',
    'loss': 'plackett_luce',
    'router_name': 'set_attention',
    'router_kwargs': {'dim_model': 32, 'num_heads': 1, 'dim_hidden': 64},
}

ARMS = {
    'cm_clean':  ['attn_last', 'pos_delta', 'mask_density'],
    'cm_policy': ['attn_last', 'pos_delta', 'mask_density', 'conf_policy', 'margin_policy'],
    'cm_aged':   ['attn_last', 'pos_delta', 'mask_density', 'conf_aged', 'margin_aged'],
    'cm_age':    ['attn_last', 'pos_delta', 'mask_density', 'conf_aged_age', 'margin_aged_age'],
}

MAP_SPEC = {'conf_policy': 'conf', 'conf_aged': 'conf',
            'margin_policy': 'margin', 'margin_aged': 'margin',
            'conf_aged_age': 'conf_age', 'margin_aged_age': 'margin_age'}


def train_arm(name_arm, features_variant, datasets):
    path_pt = os.path.join(FOLDER_BUNDLES, f'{THREAD}__{name_arm}.pt')
    if os.environ.get('RERUN', '') != '1' \
            and os.path.exists(path_pt) and os.path.exists(path_pt[:-3] + '.json'):
        print(f'[{name_arm}] bundle exists, skipping (RERUN=1 to force)')
        return
    # end

    torch.manual_seed(233)
    trainers = [RouterTrainer(folder, h=H, size_block=sb, device=DEVICE, seed=233)
                for _, folder, sb in datasets]
    feature_lists = [build_features(features_variant, folder, RECIPE['normalization'],
                                    NUM_LAYERS, MAX_CONF_AGE) for _, folder, _ in datasets]

    router = FactoryRouter.create(RECIPE['router_name'], **RECIPE['router_kwargs'])
    router.register_features(*feature_lists[0])
    router = router.to(DEVICE)
    for trainer, features in zip(trainers, feature_lists):
        trainer.router = router
        for feature in features:
            if hasattr(feature, 'fit') and hasattr(feature, 'fitted') and not feature.fitted():
                feature.fit(list(trainer._list_blocks(trainer.ids_train)), trainer.size_block)
            # end
        # end
    # end

    loss = build_loss(RECIPE['loss'])
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3, weight_decay=1e-4)
    router.train()
    for id_epoch in range(EPOCHS_FINAL):
        losses_epoch = []
        for trainer, features in zip(trainers, feature_lists):
            router.features = features
            for x, order in trainer._iter_blocks(trainer.ids_train):
                gap, cand_mask = build_geometry(order.cpu(), trainer.size_block)
                optimizer.zero_grad(set_to_none=True)
                loss_value = loss(router(x), gap.to(DEVICE), cand_mask.to(DEVICE), H)
                loss_value.backward()
                optimizer.step()
                losses_epoch.append(float(loss_value.item()))
            # end
        # end
        print(f'  [{name_arm}] epoch {id_epoch + 1}/{EPOCHS_FINAL}: '
              f'loss {sum(losses_epoch) / len(losses_epoch):.4f}', flush=True)
    # end

    router.eval()
    recalls = {}
    with torch.no_grad():
        for name_task, folder_eval, sb_eval in datasets:
            trainer_eval = RouterTrainer(folder_eval, h=H, size_block=sb_eval,
                                         device=DEVICE, seed=233)
            trainer_eval.router = router
            router.features = build_features(features_variant, folder_eval,
                                             RECIPE['normalization'], NUM_LAYERS, MAX_CONF_AGE)
            recalls[name_task] = trainer_eval.evaluate(hs=[H])[f'recall@{H}']
        # end
    # end

    features_spec = [MAP_SPEC.get(f, f) for f in features_variant]
    has_conf = any(f.startswith('conf') for f in features_variant)
    has_margin = any(f.startswith('margin') for f in features_variant)
    spec = {
        'features': features_spec,
        'conf_mode': 'aged' if has_conf else 'none',
        'margin_mode': 'aged' if has_margin else 'none',
        'train_aging': ('policy' if 'conf_policy' in features_variant else
                        'random_with_age' if 'conf_aged_age' in features_variant else
                        'random' if 'conf_aged' in features_variant else 'none'),
        'normalization': RECIPE['normalization'],
        'softmax_temperature': 1.0,
        'mask_density_window': 3,
        'loss': RECIPE['loss'],
        'dataset': GROUP,
        'router_name': RECIPE['router_name'],
        'router_kwargs': dict(RECIPE['router_kwargs']),
        'num_layers': NUM_LAYERS,
        'h': H,
        'max_conf_age': MAX_CONF_AGE,
        'seed': 233,
        'num_epochs': EPOCHS_FINAL,
        'recall_eval_no_ifeval': recalls,
        'thread': THREAD,
        'arm': name_arm,
    }
    spec['dim_in'] = router.dim_in
    router.features = feature_lists[0]
    save_router_bundle(router, spec, path_pt)
    print(f'[{name_arm}] saved {path_pt} recall_eval_no_ifeval={recalls}', flush=True)
# end


def main():
    datasets = resolve_datasets(TASKS, FOLDER_TRAIN, THREAD, NUM_BLOCKS)
    assert datasets, f'no train folders for group {GROUP} under {FOLDER_TRAIN}'
    print(f'group {GROUP}: {[(n, s) for n, _, s in datasets]}  '
          f'(epochs={EPOCHS_FINAL}, h={H}, max_age={MAX_CONF_AGE})')
    os.makedirs(FOLDER_BUNDLES, exist_ok=True)

    for name_arm, feats in ARMS.items():
        train_arm(name_arm, feats, datasets)
    # end

    print('\n=== all bundles ===')
    for name_arm in ARMS:
        path_json = os.path.join(FOLDER_BUNDLES, f'{THREAD}__{name_arm}.json')
        if os.path.exists(path_json):
            with open(path_json) as file:
                spec = json.load(file)
            print(f'  {name_arm}: {spec["features"]} '
                  f'(train_aging={spec.get("train_aging")}) '
                  f'recall={spec.get("recall_eval_no_ifeval")}')
        # end
    # end
    print('\nnext: DEVICE=cuda:0 LIMIT=150 bash run_bench_confmargin.bash')
# end


if __name__ == '__main__':
    main()
# end
