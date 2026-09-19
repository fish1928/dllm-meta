"""Shared scaffolding for the NEW ablation suite (ablation_test_new_a / _new_b).

Differences vs the old ablation_test_common.py, all deliberate:

1. MULTI-DATASET training: an experiment trains ONE router over a LIST of
   oracle stats folders (the benchmark-tail collections from stage 2), using
   the shared-router / feature-swap pattern of run_train_mlp.py: features are
   bound to a folder, so per dataset the feature list is rebound while the
   router weights are shared. size_block is inferred per folder from the stats
   filenames, so 256- and 512-token collections mix freely (row-wise
   normalizations keep the router length-independent).

2. conf vs conf_aged are SEPARATE feature names. 'conf' is the fresh oracle
   confidence -- it looks strong in offline recall but the advantage is
   fresh-logit information that does NOT survive deployment (established by
   the distillation ablation). 'conf_aged' (random staleness up to
   MAX_CONF_AGE, mirroring refresh_interval) is the deployable proxy. Keep
   both in the grid so the gap is measured, but pick winners from conf_aged /
   no-conf rows (the DB views encode this).

3. Normalization follows the DEPLOY convention of run_train_mlp/router_deploy:
   'rank' and 'softmax_attn' apply percentile-rank to EVERY non-attention
   feature (the old suite kept pos_delta/mask_density raw -- a train/deploy
   mismatch). The remaining recipes (raw, znorm_row, minmax_row, znorm_global,
   log_znorm) are exploratory: build_online_x must be extended before one of
   them can go end-to-end.

4. RESUME by default: an experiment whose record already exists with metrics
   and no error is skipped (RERUN=1 to force). reset_stage is opt-in.

Report shape is the same stage-keyed JSON as before; metrics carry an
aggregated 'all' group plus one 'ds_<task>' group per dataset, which the
sqlite importer ingests as separate result_groups.
"""

import json
import os
import re
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from router_llada import (
    FactoryLoss,
    FactoryRouter,
    Feature_attn_all,
    Feature_attn_last,
    Feature_conf,
    Feature_log_scaled,
    Feature_margin,
    Feature_mask_density,
    Feature_minmax_row,
    Feature_pos_delta,
    Feature_rank_normed,
    Feature_softmax_row,
    Feature_znormed_global,
    Feature_znormed_row,
    FeatureBase,
    RouterTrainer,
    build_geometry,
    load_stat,
    sanitize,
)


REPORT_PATH = os.environ.get('REPORT_PATH', 'ablation_test_report_new.json')

FEATURES_ATTENTION = {'attn_last', 'attn_all'}

NORMALIZATIONS_DEPLOYABLE = ('rank', 'softmax_attn')
NORMALIZATIONS_ALL = ('raw', 'znorm_row', 'rank', 'minmax_row', 'znorm_global',
                      'log_znorm', 'softmax_attn')

LOSSES_ALL = ('uniform_within_h', 'decay_within_h', 'bce_within_h',
              'bce_balanced', 'plackett_luce')


class Feature_conf_policy_aged(FeatureBase):
    # conf[t - (t mod kr), p]: the DETERMINISTIC staleness of deployment under
    # a gen-refresh clock of kr steps (age = steps since the last full block
    # refresh) -- unlike random-age augmentation, this replays the exact age
    # structure the deployed conf table has, making offline recall honest for
    # the conf axis.
    def __init__(self, folder_data, kr=16):
        super().__init__(folder_data)
        self.kr = int(kr)
    # end

    def dim(self):
        return 1
    # end

    def load_block(self, id_sample, pos_base, size_block):
        conf = sanitize(load_stat(self._folder_base(id_sample), 'conf', pos_base, size_block))
        T = conf.shape[0]
        source_row = (torch.arange(T) // self.kr) * self.kr    # last refresh step
        return conf.gather(dim=0, index=source_row.unsqueeze(-1).expand(T, conf.shape[1])).unsqueeze(-1)
    # end
# end


class Feature_conf_random_aged(FeatureBase):
    # conf[t - age, p] with age ~ U{0..min(t, max_age)}, resampled every load
    # (each epoch re-iterates blocks -> fresh ages, i.e. random-age
    # augmentation). Mirrors run_train_mlp.Feature_conf_random_aged.
    def __init__(self, folder_data, max_age=16):
        super().__init__(folder_data)
        self.max_age = int(max_age)

    def dim(self):
        return 1

    def load_block(self, id_sample, pos_base, size_block):
        conf = sanitize(load_stat(self._folder_base(id_sample), 'conf', pos_base, size_block))
        T = conf.shape[0]

        max_age_per_row = torch.arange(T).clamp(max=self.max_age)
        ages = torch.floor(torch.rand(T, conf.shape[1]) * (max_age_per_row[:, None].float() + 1.0)).long()
        source_row = (torch.arange(T)[:, None] - ages).clamp(min=0)

        return conf.gather(dim=0, index=source_row).unsqueeze(-1)


# ---------------------------------------------------------------------------
# report I/O (resume-aware)
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _load_report(report_path: str) -> Dict[str, Any]:
    if not os.path.exists(report_path):
        return {}
    with open(report_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def reset_stage(stage: str, report_path: str = REPORT_PATH) -> None:
    report = _load_report(report_path)
    report[stage] = []
    with open(report_path, 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)


def find_record(stage: str, name: str, report_path: str = REPORT_PATH) -> Optional[Dict[str, Any]]:
    for record in _load_report(report_path).get(stage, []):
        if record.get('name') == name:
            return record
    return None


def already_done(stage: str, name: str, report_path: str = REPORT_PATH) -> bool:
    record = find_record(stage, name, report_path)
    return bool(record and record.get('metrics') and not record.get('error'))


def save_result(stage: str, name: str, config: Dict[str, Any],
                metrics: Optional[Dict[str, Any]] = None, error: Optional[str] = None,
                report_path: str = REPORT_PATH) -> None:
    report = _load_report(report_path)
    records = report.setdefault(stage, [])
    record = {'name': name, 'config': _jsonable(config),
              'metrics': _jsonable(metrics), 'error': error}
    for index, old in enumerate(records):
        if old.get('name') == name:
            records[index] = record
            break
    else:
        records.append(record)
    with open(report_path, 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)


# ---------------------------------------------------------------------------
# datasets (oracle stats folders from run_collect_oracle.bash)
# ---------------------------------------------------------------------------

def infer_size_block(folder_data: str) -> int:
    """Read the block width from the stats filenames of the first sample."""
    ids = sorted(f for f in os.listdir(folder_data) if f.isdigit())
    assert ids, f'no sample folders in {folder_data}'
    folder_sample = os.path.join(folder_data, ids[0])
    for filename in os.listdir(folder_sample):
        if filename.startswith('unmask_') and filename.endswith('.pt'):
            start, end = filename[len('unmask_'):-len('.pt')].split('_')
            return int(end) - int(start)
    raise RuntimeError(f'no unmask_<s>_<e>.pt in {folder_sample}')


def resolve_datasets(names_task: Sequence[str], folder_root: str, thread: str,
                     num_blocks: int = 1) -> List[Tuple[str, str, int]]:
    """(task, folder, size_block) per existing collection; missing ones warn+drop."""
    datasets = []
    for name_task in names_task:
        folder = os.path.join(folder_root, f'{thread}_{name_task}_b{num_blocks}')
        if not os.path.isdir(folder):
            print(f'[warn] oracle folder missing, dropped from group: {folder}')
            continue
        datasets.append((name_task, folder, infer_size_block(folder)))
    return datasets


# ---------------------------------------------------------------------------
# features / normalization / loss
# ---------------------------------------------------------------------------

def make_feature(name: str, folder_data: str, num_layers: int, max_conf_age: int):
    if name == 'attn_last':
        return Feature_attn_last(folder_data)
    if name == 'attn_all':
        return Feature_attn_all(folder_data, num_layers=num_layers)
    if name == 'conf':
        return Feature_conf(folder_data)
    if name == 'conf_aged':
        return Feature_conf_random_aged(folder_data, max_age=max_conf_age)
    if name == 'conf_policy':
        return Feature_conf_policy_aged(folder_data, kr=max_conf_age)
    if name == 'margin':
        return Feature_margin(folder_data)
    if name == 'pos_delta':
        return Feature_pos_delta(folder_data)
    if name == 'mask_density':
        return Feature_mask_density(folder_data)
    raise ValueError(f'Unknown feature: {name}')


def normalize_feature(name: str, feature, normalization: str):
    """DEPLOY convention for rank/softmax_attn (matches run_train_mlp and
    router_deploy.build_online_x: non-attention features get percentile rank);
    the other recipes are exploratory (offline only until build_online_x
    learns them)."""
    if normalization == 'rank':
        return Feature_rank_normed(feature)
    if normalization == 'softmax_attn':
        if name in FEATURES_ATTENTION:
            return Feature_softmax_row(feature, temperature=1.0)
        return Feature_rank_normed(feature)
    if normalization == 'raw':
        return feature
    if normalization == 'znorm_row':
        return Feature_znormed_row(feature)
    if normalization == 'minmax_row':
        return Feature_minmax_row(feature)
    if normalization == 'znorm_global':
        return Feature_znormed_global(feature)
    if normalization == 'log_znorm':
        if name in FEATURES_ATTENTION:
            return Feature_znormed_row(Feature_log_scaled(feature))
        return Feature_znormed_row(feature)
    raise ValueError(f'Unknown normalization: {normalization}')


def build_features(feature_names: Sequence[str], folder_data: str, normalization: str,
                   num_layers: int, max_conf_age: int):
    features = []
    for name in feature_names:
        feature = make_feature(name, folder_data, num_layers, max_conf_age)
        features.append(normalize_feature(name, feature, normalization))
    return features


def build_loss(loss_name: str, pos_weight: Optional[float] = None):
    if loss_name == 'bce_balanced':
        if pos_weight is None:
            raise ValueError('bce_balanced requires pos_weight')
        return FactoryLoss.create('bce_within_h', pos_weight=pos_weight)
    return FactoryLoss.create(loss_name)


def estimate_balanced_pos_weight_multi(datasets: Sequence[Tuple[str, str, int]], h: int,
                                       holdout: float = 0.2, filter_result: str = 'all',
                                       seed: int = 233) -> float:
    """Positive/negative candidate counts aggregated over every dataset's train split."""
    positives = 0
    negatives = 0
    for _, folder_data, size_block in datasets:
        splitter = RouterTrainer(folder_data, h=h, size_block=size_block, device='cpu',
                                 holdout=holdout, filter_result=filter_result, seed=seed)
        for id_sample, pos_base in splitter._list_blocks(splitter.ids_train):
            folder_base = os.path.join(folder_data, str(id_sample))
            unmask = load_stat(folder_base, 'unmask', pos_base, size_block)
            order = unmask.squeeze(-1).long() - pos_base
            gap, cand_mask = build_geometry(order, size_block)
            positive_mask = cand_mask & (gap >= 1) & (gap <= h)
            positives += int(positive_mask.sum())
            negatives += int((cand_mask & ~positive_mask).sum())
    if positives == 0:
        raise RuntimeError('No positive labels while estimating pos_weight')
    return negatives / positives


# ---------------------------------------------------------------------------
# metric aggregation ("0.123 (n=45)" strings from attn_order_eval.summ)
# ---------------------------------------------------------------------------

PATTERN_SUMM = re.compile(r'^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\(n=(\d+)\)\s*$')


def parse_summ(value: Any) -> Optional[Tuple[float, int]]:
    if isinstance(value, str):
        match = PATTERN_SUMM.fullmatch(value)
        if match:
            return float(match.group(1)), int(match.group(2))
    return None


def aggregate_reports(reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """n-weighted mean per metric across per-dataset evaluate() reports,
    emitted in the same 'x (n=y)' format the importer parses."""
    aggregated: Dict[str, Any] = {}
    keys = set().union(*(r.keys() for r in reports))
    for key in sorted(keys):
        if key == 'router':
            continue
        if key == 'n_blocks':
            aggregated[key] = sum(int(r.get(key, 0)) for r in reports)
            continue
        parsed = [parse_summ(r[key]) for r in reports if key in r]
        parsed = [p for p in parsed if p is not None]
        if not parsed:
            continue
        n_total = sum(n for _, n in parsed)
        mean = sum(v * n for v, n in parsed) / max(n_total, 1)
        aggregated[key] = '{:.3f} (n={})'.format(mean, n_total)
    return aggregated


# ---------------------------------------------------------------------------
# the multi-dataset experiment
# ---------------------------------------------------------------------------

def run_experiment_multi(
    *,
    stage: str,
    name: str,
    dataset_group: str,
    datasets: Sequence[Tuple[str, str, int]],    # (task, folder, size_block)
    feature_names: Sequence[str],
    normalization: str,
    loss_name: str,
    loss_pos_weight: Optional[float] = None,
    router_name: str = 'mlp',
    router_kwargs: Optional[Dict[str, Any]] = None,
    h: int = 5,
    device: str = 'cuda:0',
    num_layers: int = 32,
    num_epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    holdout: float = 0.2,
    filter_result: str = 'all',
    seed: int = 233,
    max_conf_age: int = 16,
    report_path: str = REPORT_PATH,
    skip_existing: bool = True,
):
    router_kwargs = router_kwargs or {}
    config = {
        'dataset_group': dataset_group,
        'datasets': [nm for nm, _, _ in datasets],
        'dataset_folders': [folder for _, folder, _ in datasets],
        'size_blocks': [sb for _, _, sb in datasets],
        'features': list(feature_names),
        'normalization': normalization,
        'loss': loss_name,
        'loss_pos_weight': loss_pos_weight,
        'router': router_name,
        'router_kwargs': router_kwargs,
        'h': h,
        'device': device,
        'num_layers': num_layers,
        'num_epochs': num_epochs,
        'lr': lr,
        'weight_decay': weight_decay,
        'holdout': holdout,
        'filter_result': filter_result,
        'seed': seed,
        'max_conf_age': max_conf_age,
    }

    if not datasets:
        save_result(stage, name, config, error='dataset group resolved to zero folders',
                    report_path=report_path)
        print(f'[{stage}] {name}: SKIPPED, no datasets')
        return None

    if skip_existing and os.environ.get('RERUN', '') != '1' and already_done(stage, name, report_path):
        print(f'[{stage}] {name}: already done, skipping (RERUN=1 to force)')
        return None

    print(f'\n[{stage}] {name}')
    try:
        torch.manual_seed(seed)

        trainers = [
            RouterTrainer(folder, h=h, size_block=size_block, device=device,
                          lr=lr, weight_decay=weight_decay, holdout=holdout,
                          filter_result=filter_result, seed=seed)
            for _, folder, size_block in datasets
        ]
        feature_lists = [
            build_features(feature_names, folder, normalization, num_layers, max_conf_age)
            for _, folder, _ in datasets
        ]

        router = FactoryRouter.create(router_name, **router_kwargs)
        router.register_features(*feature_lists[0])
        router = router.to(device)

        # dataset-fitted normalizers (znorm_global): each dataset's feature
        # list is fitted on that dataset's OWN train split, mirroring
        # RouterTrainer.register_router
        for trainer, features in zip(trainers, feature_lists):
            trainer.router = router
            for feature in features:
                if hasattr(feature, 'fit') and hasattr(feature, 'fitted') and not feature.fitted():
                    feature.fit(list(trainer._list_blocks(trainer.ids_train)), trainer.size_block)

        loss = build_loss(loss_name, pos_weight=loss_pos_weight)

        if router.trainable():
            optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=weight_decay)
            router.train()
            for id_epoch in range(num_epochs):
                losses = []
                for trainer, features in zip(trainers, feature_lists):
                    router.features = features    # rebind data source, weights unchanged
                    for x, order in trainer._iter_blocks(trainer.ids_train):
                        gap, cand_mask = build_geometry(order.cpu(), trainer.size_block)
                        gap, cand_mask = gap.to(device), cand_mask.to(device)

                        optimizer.zero_grad(set_to_none=True)
                        loss_value = loss(router(x), gap, cand_mask, h)
                        loss_value.backward()
                        optimizer.step()
                        losses.append(float(loss_value.item()))
                print(f'  epoch {id_epoch}: loss {sum(losses) / len(losses):.4f} over {len(losses)} blocks')

        router.eval()
        hs = sorted({3, 5, 10, h, 2 * h})    # recall@5 always present for cross-h comparison
        metrics: Dict[str, Any] = {}
        reports_ds = []
        with torch.no_grad():
            for (name_task, _, _), trainer, features in zip(datasets, trainers, feature_lists):
                router.features = features
                report_ds = trainer.evaluate(hs=hs)
                metrics[f'ds_{name_task}'] = report_ds
                reports_ds.append(report_ds)

        metrics['all'] = aggregate_reports(reports_ds)
        metrics['all']['router'] = router.describe()

        save_result(stage, name, config, metrics=metrics, report_path=report_path)
        print(json.dumps(_jsonable(metrics['all']), indent=2))
        return metrics

    except Exception:
        error = traceback.format_exc()
        save_result(stage, name, config, error=error, report_path=report_path)
        print(error)
        return None
