"""FAST ablation: anchored coordinate descent instead of a Cartesian grid.

One self-driving run per thread finds the best feature combination, loss,
normalization, architecture and training horizon in ~30 short trainings, then
retrains the winner and saves deployable bundles. Designed for the endgame:
resume-safe (rerun continues), one report JSON (works with the existing
sqlite importer), winners printed and saved to a summary file for the paper.

Design: start from the incumbent (the llada-base winner of the earlier full
grid: attn_last+geo / softmax_attn / plackett_luce / mlp d64 / h=5) and sweep
ONE dimension per stage on the MIX dataset group, carrying winners forward:
  fast_feat -> fast_norm -> fast_loss -> fast_arch -> fast_h -> fast_final
Leak guards baked into winner selection (everything is still MEASURED):
  - anchors containing fresh 'conf' are recorded but ineligible to win
  - non-deployable normalizations (anything but rank/softmax_attn) are
    recorded but ineligible (router_deploy.build_online_x cannot run them)
  - mockup routers are floors, never winners
  - 'margin' may win offline but is NOT deployable online (no online margin
    in build_online_x) -- a winning set containing margin trains and saves,
    with a loud warning to extend build_online_x before e2e
Final bundles: conf_aged maps to spec feature 'conf' with conf_mode='aged'
(the run_train_mlp convention -- deployment feeds the live stale conf).

Usage (llada-base):
  FOLDER_TRAIN=stats_train THREAD=llada_base DEVICE=cuda:0 \
      python ablation_tests/ablation_test_fast.py
Env: FOLDER_TRAIN (stats_train root of head-split TRAIN folders), THREAD,
     NUM_BLOCKS (default 1), DEVICE, NUM_LAYERS (32; dream 28), NUM_EPOCHS
     (10), EPOCHS_FINAL (20), H (5), MAX_CONF_AGE (16), REPORT_PATH,
     FOLDER_BUNDLES (routers_final).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from ablation_test_common_new import (
    NORMALIZATIONS_ALL,
    NORMALIZATIONS_DEPLOYABLE,
    LOSSES_ALL,
    build_features,
    build_loss,
    estimate_balanced_pos_weight_multi,
    find_record,
    parse_summ,
    resolve_datasets,
    run_experiment_multi,
)
from router_llada import FactoryRouter, RouterTrainer, build_geometry
from router_deploy import save_router_bundle, spec_dim_in

REPORT_PATH = os.environ.get('REPORT_PATH', 'ablation_test_report_fast.json')
FOLDER_TRAIN = os.environ.get('FOLDER_TRAIN', 'stats_train')
THREAD = os.environ.get('THREAD', 'llada_base')
NUM_BLOCKS = int(os.environ.get('NUM_BLOCKS', 1))
DEVICE = os.environ.get('DEVICE', 'cuda:0')
NUM_LAYERS = int(os.environ.get('NUM_LAYERS', 32))
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 10))
EPOCHS_FINAL = int(os.environ.get('EPOCHS_FINAL', 20))
H_DEFAULT = int(os.environ.get('H', 5))
MAX_CONF_AGE = int(os.environ.get('MAX_CONF_AGE', 16))
FOLDER_BUNDLES = os.environ.get('FOLDER_BUNDLES', 'routers_final')

# TRAINING-data versions (final stage trains one bundle per group); the mix
# deliberately excludes ifeval. EVALUATION is always on GROUP_EVAL -- every
# task except ifeval -- so the three bundles are scored on identical holdout
# samples (and the ifeval-trained bundle's number IS the cross-domain result).
DATASET_GROUPS = {
    'mix_no_ifeval': ['gsm8k', 'minerva_math', 'bbh', 'humaneval', 'truthfulqa_gen'],
    'gsm8k': ['gsm8k'],
    'ifeval': ['ifeval'],
}
GROUP_SEARCH = 'mix_no_ifeval'    # coordinate descent trains+evaluates here
GROUP_EVAL = 'mix_no_ifeval'      # final bundles all evaluate on this group's holdouts

INCUMBENT = {
    'features': ['attn_last', 'pos_delta', 'mask_density'],
    'normalization': 'softmax_attn',
    'loss': 'plackett_luce',
    'router_name': 'mlp',
    'router_kwargs': {'dim_hidden': 64, 'num_blocks_mlp': 2},
    'h': H_DEFAULT,
}

FEATURE_ANCHORS = {
    'attn_last':                  ['attn_last'],
    'attn_all':                   ['attn_all'],
    'attn_last_geo':              ['attn_last', 'pos_delta', 'mask_density'],
    'attn_all_geo':               ['attn_all', 'pos_delta', 'mask_density'],
    'attn_last_geo_conf':         ['attn_last', 'pos_delta', 'mask_density', 'conf'],
    'attn_last_geo_conf_aged':    ['attn_last', 'pos_delta', 'mask_density', 'conf_aged'],
    'attn_last_geo_conf_policy':  ['attn_last', 'pos_delta', 'mask_density', 'conf_policy'],
    'attn_last_geo_margin':       ['attn_last', 'pos_delta', 'mask_density', 'margin'],
    'attn_last_conf_aged_margin': ['attn_last', 'conf_aged', 'margin'],
    'no_attn_aged':               ['conf_aged', 'margin', 'pos_delta', 'mask_density'],
    'full_fresh':                 ['attn_last', 'conf', 'margin', 'pos_delta', 'mask_density'],
    'full_aged':                  ['attn_last', 'conf_aged', 'margin', 'pos_delta', 'mask_density'],
}

ARCHITECTURES = {
    'linear':        ('linear', {}),
    'mlp_d32_b2':    ('mlp', {'dim_hidden': 32, 'num_blocks_mlp': 2}),
    'mlp_d64_b2':    ('mlp', {'dim_hidden': 64, 'num_blocks_mlp': 2}),
    'mlp_d128_b3':   ('mlp', {'dim_hidden': 128, 'num_blocks_mlp': 3}),
    'set_attention': ('set_attention', {'dim_model': 32, 'num_heads': 1, 'dim_hidden': 64}),
    'mockup_random': ('mockup_random', {}),    # floor, never a winner
}

HORIZONS = [3, 5, 8, 12, 16]

DEPLOYABLE_ONLINE_FEATURES = {'attn_last', 'attn_all', 'pos_delta', 'mask_density', 'conf', 'conf_aged'}


STAGES = [
    ('fast_feat',  'A features (12 runs)'),
    ('fast_norm',  'B normalization (7 runs)'),
    ('fast_loss',  'C loss (5 runs)'),
    ('fast_arch',  'D architecture (6 runs)'),
    ('fast_h',     'E horizon (5 runs)'),
    ('fast_final', 'F final training + bundles (3-6 trainings)'),
]


def print_remaining(done_key):
    idx = [k for k, _ in STAGES].index(done_key)
    remaining = [label for _, label in STAGES[idx + 1:]]
    print(f'[progress] finished {STAGES[idx][1]}; remaining: '
          + (' -> '.join(remaining) if remaining else 'NONE, all done'))
# end


def recall5_of(stage, name):
    record = find_record(stage, name, REPORT_PATH)
    if not record or record.get('error') or not record.get('metrics'):
        return None
    parsed = parse_summ(record['metrics'].get('all', {}).get('recall@5'))
    return parsed[0] if parsed else None
# end


def pick_best(stage, candidates, eligible):
    """candidates: {name: payload}; eligible(name, payload) -> bool.
    Returns (best_name, best_recall); prints the stage table."""
    rows = []
    for name, payload in candidates.items():
        r5 = recall5_of(stage, name)
        rows.append((name, r5, eligible(name, payload)))
    # end
    rows.sort(key=lambda r: (-1 if r[1] is None else r[1]), reverse=True)
    print(f'\n===== {stage} results =====')
    for name, r5, ok in rows:
        marker = '  ' if ok else '  [ineligible]'
        print(f'  {name:32s} recall@5={r5 if r5 is not None else "FAILED"}{marker}')
    # end
    for name, r5, ok in rows:
        if ok and r5 is not None:
            return name, r5
        # end
    # end
    raise RuntimeError(f'{stage}: no eligible finished run')
# end


def main():
    datasets_search = resolve_datasets(DATASET_GROUPS[GROUP_SEARCH], FOLDER_TRAIN, THREAD, NUM_BLOCKS)
    assert datasets_search, f'no train folders for group {GROUP_SEARCH} under {FOLDER_TRAIN}'
    print(f'search group {GROUP_SEARCH}: {[(n, s) for n, _, s in datasets_search]}')

    common = dict(dataset_group=GROUP_SEARCH, datasets=datasets_search, device=DEVICE,
                  num_layers=NUM_LAYERS, num_epochs=NUM_EPOCHS, max_conf_age=MAX_CONF_AGE,
                  report_path=REPORT_PATH)
    winner = dict(INCUMBENT)

    '''stage A: features (fixed: incumbent norm/loss/arch/h)'''
    for name, feats in tqdm(list(FEATURE_ANCHORS.items()), desc='stage A features', disable=None):
        run_experiment_multi(stage='fast_feat', name=name, feature_names=feats,
            normalization=winner['normalization'], loss_name=winner['loss'],
            router_name=winner['router_name'], router_kwargs=winner['router_kwargs'],
            h=winner['h'], **common)
    # end
    # CONF POLICY: offline recall is only trustworthy between recipes of equal
    # deployment fidelity. Fresh conf is banned outright (leak). conf_aged /
    # conf_policy anchors may win ONLY if they beat the best conf-FREE anchor
    # by CONF_MARGIN (default 0.02) -- offline gains smaller than that have
    # historically NOT survived e2e. Whenever the winner carries a conf
    # variant, the final stage ALSO trains the best conf-free recipe so a
    # cheap e2e run makes the actual call.
    conf_margin = float(os.environ.get('CONF_MARGIN', 0.02))
    best_free, r5_free = pick_best('fast_feat', FEATURE_ANCHORS,
        eligible=lambda n, feats: not any(f.startswith('conf') for f in feats))
    best_any, r5_any = pick_best('fast_feat', FEATURE_ANCHORS,
        eligible=lambda n, feats: 'conf' not in feats)    # fresh conf = leak
    if best_any != best_free and r5_any is not None and r5_free is not None \
            and r5_any >= r5_free + conf_margin:
        best_feat, r5 = best_any, r5_any
        print(f'[conf] {best_any} beats conf-free {best_free} by >= {conf_margin} offline '
              f'({r5_any} vs {r5_free}); e2e must confirm -- dual bundles will be saved')
    else:
        best_feat, r5 = best_free, r5_free
        if best_any != best_free:
            print(f'[conf] {best_any} offline gain over {best_free} < {conf_margin} '
                  f'({r5_any} vs {r5_free}): not trusted, conf-free recipe carried forward')
        # end
    # end
    winner['features'] = FEATURE_ANCHORS[best_feat]
    winner_features_free = FEATURE_ANCHORS[best_free]
    print(f'--> features: {best_feat} {winner["features"]} (recall@5 {r5})')
    print_remaining('fast_feat')

    '''stage B: normalization'''
    for norm in tqdm(NORMALIZATIONS_ALL, desc='stage B norms', disable=None):
        run_experiment_multi(stage='fast_norm', name=f'norm-{norm}',
            feature_names=winner['features'], normalization=norm, loss_name=winner['loss'],
            router_name=winner['router_name'], router_kwargs=winner['router_kwargs'],
            h=winner['h'], **common)
    # end
    best_norm, r5 = pick_best('fast_norm', {f'norm-{n}': n for n in NORMALIZATIONS_ALL},
        eligible=lambda name, norm: norm in NORMALIZATIONS_DEPLOYABLE)
    winner['normalization'] = best_norm.replace('norm-', '')
    print(f'--> normalization: {winner["normalization"]} (recall@5 {r5})')
    print_remaining('fast_norm')

    '''stage C: loss'''
    pos_weight = estimate_balanced_pos_weight_multi(datasets_search, h=winner['h'])
    for loss in tqdm(LOSSES_ALL, desc='stage C losses', disable=None):
        run_experiment_multi(stage='fast_loss', name=f'loss-{loss}',
            feature_names=winner['features'], normalization=winner['normalization'],
            loss_name=loss, loss_pos_weight=pos_weight if loss == 'bce_balanced' else None,
            router_name=winner['router_name'], router_kwargs=winner['router_kwargs'],
            h=winner['h'], **common)
    # end
    best_loss, r5 = pick_best('fast_loss', {f'loss-{l}': l for l in LOSSES_ALL},
        eligible=lambda name, loss: True)
    winner['loss'] = best_loss.replace('loss-', '')
    winner['loss_pos_weight'] = pos_weight if winner['loss'] == 'bce_balanced' else None
    print(f'--> loss: {winner["loss"]} (recall@5 {r5})')
    print_remaining('fast_loss')

    '''stage D: architecture'''
    for name, (rname, rkw) in tqdm(list(ARCHITECTURES.items()), desc='stage D archs', disable=None):
        run_experiment_multi(stage='fast_arch', name=f'arch-{name}',
            feature_names=winner['features'], normalization=winner['normalization'],
            loss_name=winner['loss'], loss_pos_weight=winner.get('loss_pos_weight'),
            router_name=rname, router_kwargs=rkw, h=winner['h'], **common)
    # end
    best_arch, r5 = pick_best('fast_arch', {f'arch-{n}': n for n in ARCHITECTURES},
        eligible=lambda name, arch: not arch.startswith('mockup'))
    winner['router_name'], winner['router_kwargs'] = ARCHITECTURES[best_arch.replace('arch-', '')]
    print(f'--> architecture: {best_arch} (recall@5 {r5})')
    print_remaining('fast_arch')

    '''stage E: horizon (recall@5 stays the comparison basis across h)'''
    for h in tqdm(HORIZONS, desc='stage E horizons', disable=None):
        run_experiment_multi(stage='fast_h', name=f'h{h}',
            feature_names=winner['features'], normalization=winner['normalization'],
            loss_name=winner['loss'], loss_pos_weight=winner.get('loss_pos_weight'),
            router_name=winner['router_name'], router_kwargs=winner['router_kwargs'],
            h=h, **common)
    # end
    best_h, r5 = pick_best('fast_h', {f'h{h}': h for h in HORIZONS},
        eligible=lambda name, h: True)
    winner['h'] = int(best_h[1:])
    print(f'--> horizon: h={winner["h"]} (recall@5 {r5})')
    print_remaining('fast_h')

    '''stage F: retrain per dataset group, save deployable bundles. If the
    winner carries a conf variant, ALSO train the best conf-free recipe --
    offline recall cannot settle the conf axis, so both bundle sets are saved
    and a cheap e2e run makes the final call (command printed at the end).'''
    os.makedirs(FOLDER_BUNDLES, exist_ok=True)
    summary = {'thread': THREAD, 'winner': {k: v for k, v in winner.items()}, 'bundles': {}}

    variants = [('', winner['features'])]
    has_conf_variant = any(f.startswith('conf') for f in winner['features'])
    if has_conf_variant:
        variants.append(('__noconf', winner_features_free))
    # end

    jobs_final = [(s, f) for s, f in variants]
    bar_final = tqdm(total=len(jobs_final) * len(DATASET_GROUPS), desc='stage F final trainings', disable=None)
    for suffix, features_variant in jobs_final:
        features_spec = ['conf' if f in ('conf_aged', 'conf_policy') else f for f in features_variant]
        conf_mode = 'aged' if any(f in ('conf_aged', 'conf_policy') for f in features_variant) else \
                    ('fresh' if 'conf' in features_variant else 'none')
        if not set(features_variant) <= DEPLOYABLE_ONLINE_FEATURES:
            print('[WARNING] features not computable online yet: '
                  f'{set(features_variant) - DEPLOYABLE_ONLINE_FEATURES} '
                  '-- extend router_deploy before e2e with this bundle')
        # end

        for name_group, names_task in DATASET_GROUPS.items():
            datasets = resolve_datasets(names_task, FOLDER_TRAIN, THREAD, NUM_BLOCKS)
            if not datasets:
                print(f'[warn] final: group {name_group} has no folders, skipped')
                bar_final.update(1)
                continue
            # end

            torch.manual_seed(233)
            trainers = [RouterTrainer(folder, h=winner['h'], size_block=sb, device=DEVICE, seed=233)
                        for _, folder, sb in datasets]
            feature_lists = [build_features(features_variant, folder, winner['normalization'],
                                            NUM_LAYERS, MAX_CONF_AGE) for _, folder, _ in datasets]
            router = FactoryRouter.create(winner['router_name'], **winner['router_kwargs'])
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
            loss = build_loss(winner['loss'], pos_weight=winner.get('loss_pos_weight'))
            optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3, weight_decay=1e-4)
            router.train()
            bar_epochs = tqdm(range(EPOCHS_FINAL), desc=f'  train {name_group}{suffix}', leave=False, disable=None)
            for id_epoch in bar_epochs:
                losses_epoch = []
                for trainer, features in zip(trainers, feature_lists):
                    router.features = features
                    for x, order in trainer._iter_blocks(trainer.ids_train):
                        gap, cand_mask = build_geometry(order.cpu(), trainer.size_block)
                        optimizer.zero_grad(set_to_none=True)
                        loss_value = loss(router(x), gap.to(DEVICE), cand_mask.to(DEVICE), winner['h'])
                        loss_value.backward()
                        optimizer.step()
                        losses_epoch.append(float(loss_value.item()))
                    # end
                # end
                loss_mean = sum(losses_epoch) / len(losses_epoch)
                bar_epochs.set_postfix(loss=f'{loss_mean:.4f}')
                print(f'  [final {name_group}{suffix}] epoch {id_epoch + 1}/{EPOCHS_FINAL}: '
                      f'loss {loss_mean:.4f}', flush=True)
            # end
            router.eval()
            datasets_eval = resolve_datasets(DATASET_GROUPS[GROUP_EVAL], FOLDER_TRAIN, THREAD, NUM_BLOCKS)
            recalls = {}
            with torch.no_grad():
                for name_task, folder_eval, sb_eval in datasets_eval:
                    trainer_eval = RouterTrainer(folder_eval, h=winner['h'], size_block=sb_eval,
                                                 device=DEVICE, seed=233)
                    trainer_eval.router = router
                    router.features = build_features(features_variant, folder_eval,
                                                     winner['normalization'], NUM_LAYERS, MAX_CONF_AGE)
                    recalls[name_task] = trainer_eval.evaluate(hs=[winner['h']])[f'recall@{winner["h"]}']
                # end
            # end

            spec = {
                'features': features_spec,
                'conf_mode': conf_mode,
                'normalization': winner['normalization'],
                'softmax_temperature': 1.0,
                'mask_density_window': 3,
                'loss': winner['loss'],
                'dataset': name_group,
                'router_name': winner['router_name'],
                'router_kwargs': dict(winner['router_kwargs']),
                'num_layers': NUM_LAYERS,
                'h': winner['h'],
                'max_conf_age': MAX_CONF_AGE,
                'seed': 233,
                'num_epochs': EPOCHS_FINAL,
                'recall_eval_no_ifeval': recalls,
                'thread': THREAD,
            }
            spec['dim_in'] = spec_dim_in(spec)
            router.features = feature_lists[0]
            path_pt = os.path.join(FOLDER_BUNDLES, f'{THREAD}__{name_group}{suffix}.pt')
            save_router_bundle(router, spec, path_pt)
            summary['bundles'][name_group + suffix] = {
                'path': path_pt, 'features': features_variant, 'recall_eval_no_ifeval': recalls}
            print(f'[final] {name_group}{suffix}: saved {path_pt} recall_eval_no_ifeval={recalls}')
            bar_final.update(1)
        # end
    # end

    bar_final.close()
    print_remaining('fast_final')
    path_summary = os.path.join(FOLDER_BUNDLES, f'{THREAD}__ablation_summary.json')
    with open(path_summary, 'w') as file:
        json.dump(summary, file, indent=2)
    # end
    print(f'\n=== DONE. Winner chain and bundles in {path_summary} ===')
    print(json.dumps(summary['winner'], indent=2))
    if has_conf_variant:
        print('\n[conf] offline could not settle the conf axis: run the deciding e2e pair, e.g.')
        print(f'  ...run_benchmark_main.py --tasks gsm8k --limit 200 ... '
              f'path_router={FOLDER_BUNDLES}/{THREAD}__mix_no_ifeval.pt')
        print(f'  ...same command with path_router={FOLDER_BUNDLES}/{THREAD}__mix_no_ifeval__noconf.pt')
    # end
# end


if __name__ == '__main__':
    main()
# end
