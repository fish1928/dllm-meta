#################################################
# Router (future-unmask-position selector) framework.
#
# Composable pieces, each a class:
#   - Feature_*:  load one input signal per block from a stats folder
#                 (run_collect_metrics_llada.py layout), shape (T, L, d)
#   - Router_*:   score candidates, shape (T, L); mockup routers are
#                 non-trainable baselines
#   - Loss_*:     ranking losses over (scores, gap, cand_mask, h)
#   - FactoryFeature / FactoryRouter / FactoryLoss: create by name
#   - RouterTrainer: sample-folder splits, result filtering, training loop,
#                 evaluation via attn_order_eval (recall@h / pr_auc / ndcg)
#
# Usage sketch:
#   router = FactoryRouter.create('mlp', dim_hidden=64).register_features(
#       Feature_attn_last(folder_data), Feature_margin(folder_data))
#   trainer = RouterTrainer(folder_data, h=5, device='cuda:0')
#   trainer.register_router(router).register_loss(Loss_uniform_within_h())
#   trainer.train(num_epochs=10)
#   print(trainer.evaluate())
#
# Conventions:
#   - one unmask per step (T steps, block length L, typically T == L)
#   - gap[t, p] = steps until position p unmasks after step t; candidates gap > 0
#   - sentinel values (-inf) in stored metrics are sanitized to 0.0 on load;
#     candidate masking is done from the unmask order, not from sentinels
#################################################

import json
import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from attn_order_eval import ScoreOrderEval, summ
from tools_debug import jprint

NEG_INF = torch.finfo(torch.float32).min


'''---------------- data helpers ----------------'''


def load_stat(folder_base, name, pos_base, size_block):
    path = os.path.join(folder_base, f'{name}_{pos_base}_{pos_base + size_block}.pt')
    return torch.load(path, map_location='cpu')
# end


def sanitize(value):    # sentinel / non-finite -> 0.0; candidacy comes from the unmask order
    value = value.float()
    bad = ~torch.isfinite(value) | (value <= -1e30)    # collector's sentinel is finfo.min, which IS finite
    return torch.where(bad, torch.zeros_like(value), value)
# end


def build_geometry(order, size_block):    # order (T,) block-local -> gap (T, L), cand_mask (T, L)
    T = order.shape[0]
    step_of = torch.full((size_block,), -1, dtype=torch.long)
    step_of[order] = torch.arange(T)

    gap = step_of.view(1, -1) - torch.arange(T).view(T, 1)    # (T, L)
    cand_mask = gap > 0
    return gap, cand_mask
# end


'''---------------- features ----------------'''


class FeatureBase(ABC):

    def __init__(self, folder_data):
        self.folder_data = folder_data
    # end

    def get_name(self):
        return self.__class__.__name__
    # end

    @abstractmethod
    def dim(self):
        raise NotImplementedError
    # end

    @abstractmethod
    def load_block(self, id_sample, pos_base, size_block):    # -> (T, L, dim) fp32
        raise NotImplementedError
    # end

    def _folder_base(self, id_sample):
        return os.path.join(self.folder_data, str(id_sample))
    # end

    def _order_local(self, id_sample, pos_base, size_block):
        unmask = load_stat(self._folder_base(id_sample), 'unmask', pos_base, size_block)
        return unmask.squeeze(-1).long() - pos_base
    # end
# end


class Feature_conf(FeatureBase):
    def dim(self):
        return 1
    # end

    def load_block(self, id_sample, pos_base, size_block):
        value = load_stat(self._folder_base(id_sample), 'conf', pos_base, size_block)
        return sanitize(value).unsqueeze(-1)
    # end
# end


class Feature_margin(FeatureBase):
    def dim(self):
        return 1
    # end

    def load_block(self, id_sample, pos_base, size_block):
        value = load_stat(self._folder_base(id_sample), 'margin', pos_base, size_block)
        return sanitize(value).unsqueeze(-1)
    # end
# end


class Feature_entropy(FeatureBase):
    def dim(self):
        return 1
    # end

    def load_block(self, id_sample, pos_base, size_block):
        value = load_stat(self._folder_base(id_sample), 'entropy', pos_base, size_block)
        return sanitize(value).unsqueeze(-1)
    # end
# end


class Feature_attn_last(FeatureBase):
    def dim(self):
        return 1
    # end

    def load_block(self, id_sample, pos_base, size_block):
        attn = load_stat(self._folder_base(id_sample), 'attn', pos_base, size_block)    # (T, layers, 1, L)
        return sanitize(attn.squeeze(-2)[:, -1, :]).unsqueeze(-1)
    # end
# end


class Feature_attn_all(FeatureBase):
    def __init__(self, folder_data, num_layers=32):
        super().__init__(folder_data)
        self.num_layers = num_layers
    # end

    def dim(self):
        return self.num_layers
    # end

    def load_block(self, id_sample, pos_base, size_block):
        attn = load_stat(self._folder_base(id_sample), 'attn', pos_base, size_block)    # (T, layers, 1, L)
        attn = attn.squeeze(-2)
        assert attn.shape[1] == self.num_layers,\
            f'num_layers mismatch: file has {attn.shape[1]}, feature configured {self.num_layers}'
        return sanitize(attn).permute(0, 2, 1)    # (T, L, layers)
    # end
# end


class Feature_pos_delta(FeatureBase):
    # dim 0: signed (p - just_unmasked) / L; dim 1: |p - just_unmasked| / L
    def dim(self):
        return 2
    # end

    def load_block(self, id_sample, pos_base, size_block):
        order = self._order_local(id_sample, pos_base, size_block)    # (T,)
        positions = torch.arange(size_block).view(1, -1)
        delta = (positions - order.view(-1, 1)).float() / size_block    # (T, L)
        return torch.stack([delta, delta.abs()], dim=-1)
    # end
# end


class Feature_step_progress(FeatureBase):
    def dim(self):
        return 1
    # end

    def load_block(self, id_sample, pos_base, size_block):
        order = self._order_local(id_sample, pos_base, size_block)
        T = order.shape[0]
        progress = (torch.arange(T).float() / max(T - 1, 1)).view(-1, 1, 1)
        return progress.expand(T, size_block, 1).clone()
    # end
# end


class Feature_mask_density(FeatureBase):
    # fraction of still-masked positions within +-window around each candidate
    def __init__(self, folder_data, window=3):
        super().__init__(folder_data)
        self.window = window
    # end

    def dim(self):
        return 1
    # end

    def load_block(self, id_sample, pos_base, size_block):
        order = self._order_local(id_sample, pos_base, size_block)
        gap, cand_mask = build_geometry(order, size_block)
        masked = cand_mask.float()    # (T, L) still-masked after step t

        kernel = torch.ones(1, 1, 2 * self.window + 1) / (2 * self.window + 1)
        density = F.conv1d(masked.unsqueeze(1), kernel, padding=self.window).squeeze(1)
        return density.unsqueeze(-1)
    # end
# end


class Feature_x0_stability(FeatureBase):
    # 1.0 if the candidate's argmax token is unchanged from the previous step; row 0 = 0.0
    def dim(self):
        return 1
    # end

    def load_block(self, id_sample, pos_base, size_block):
        x0 = load_stat(self._folder_base(id_sample), 'x0', pos_base, size_block)    # (T, L) long
        stable = torch.zeros(x0.shape, dtype=torch.float32)
        stable[1:] = (x0[1:] == x0[:-1]).float()
        return stable.unsqueeze(-1)
    # end
# end


'''---------------- feature normalization wrappers ----------------

Row-wise wrappers (rank / znorm_row / minmax_row / softmax_row) compute their
statistics over the CANDIDATES of each step only (still-masked positions) and
zero out non-candidates -- so a candidate's normalized value does not drift as
the block fills up. Candidacy is known from the live mask state at inference,
so all row-wise wrappers stay exactly reproducible at deployment.
Feature_znormed_global is dataset-fitted (over candidate entries): the trainer
auto-fits it on the train split and its statistics are persisted inside the
router checkpoint.
'''


class FeatureWrapperBase(FeatureBase):

    def __init__(self, feature_inner):
        super().__init__(feature_inner.folder_data)
        self.feature_inner = feature_inner
    # end

    def dim(self):
        return self.feature_inner.dim()
    # end

    def get_name(self):
        return f'{self._tag()}({self.feature_inner.get_name()})'
    # end

    def _tag(self):
        return self.__class__.__name__.replace('Feature_', '')
    # end

    def _cand_mask_3d(self, id_sample, pos_base, size_block):    # (T, L, 1) bool
        order = self._order_local(id_sample, pos_base, size_block)
        _, cand_mask = build_geometry(order, size_block)
        return cand_mask.unsqueeze(-1)
    # end
# end


class Feature_rank_normed(FeatureWrapperBase):
    """
    Replace each feature value with its percentile rank among candidates.

    Tied values receive their average rank.
    Non-candidates are set to 0.

    Examples:
        [0.2, 0.2, 0.8] -> [0.25, 0.25, 1.0]
        [0.5, 0.5, 0.5] -> [0.5, 0.5, 0.5]
    """

    def load_block(self, id_sample, pos_base, size_block):
        x = self.feature_inner.load_block(
            id_sample,
            pos_base,
            size_block,
        )  # (T, L, d)

        cand = self._cand_mask_3d(
            id_sample,
            pos_base,
            size_block,
        ).expand_as(x)  # (T, L, d)

        # Compare every candidate i against every candidate j.
        #
        # x_i: (T, L, 1, d)
        # x_j: (T, 1, L, d)
        x_i = x.unsqueeze(2)
        x_j = x.unsqueeze(1)

        # Whether comparison position j is a valid candidate.
        cand_j = cand.unsqueeze(1)

        # Number of candidate values strictly below x_i.
        num_lower = ((x_j < x_i) & cand_j).sum(dim=2)

        # Number of candidate values equal to x_i, including x_i itself.
        num_equal = ((x_j == x_i) & cand_j).sum(dim=2)

        # Average rank for tied values:
        #
        # rank range occupied by the tie group:
        #   num_lower, ..., num_lower + num_equal - 1
        #
        # average:
        #   num_lower + (num_equal - 1) / 2
        average_rank = (
            num_lower.float()
            + 0.5 * (num_equal.float() - 1.0)
        )

        # Number of candidates per row and feature dimension.
        num_candidates = cand.sum(dim=1).float()  # (T, d)

        # Map ranks from [0, n - 1] into [0, 1].
        denominator = (num_candidates - 1.0).clamp(min=1.0)
        rank_normed = average_rank / denominator.unsqueeze(1)

        # A row containing one candidate receives rank 0.
        # Non-candidates are always zero.
        return rank_normed.masked_fill(~cand, 0.0)
    # end
# end


class Feature_znormed_row(FeatureWrapperBase):
    # standardize each dim over the row's candidates; non-candidates -> 0
    def load_block(self, id_sample, pos_base, size_block):
        x = self.feature_inner.load_block(id_sample, pos_base, size_block)
        cand = self._cand_mask_3d(id_sample, pos_base, size_block).expand_as(x).float()

        n = cand.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (x * cand).sum(dim=1, keepdim=True) / n
        var = ((x - mean).pow(2) * cand).sum(dim=1, keepdim=True) / n
        out = (x - mean) / (var.sqrt() + 1e-6)
        return out * cand
    # end
# end


class Feature_minmax_row(FeatureWrapperBase):
    # scale each dim over the row's candidates to [0, 1]; non-candidates -> 0
    def load_block(self, id_sample, pos_base, size_block):
        x = self.feature_inner.load_block(id_sample, pos_base, size_block)
        cand = self._cand_mask_3d(id_sample, pos_base, size_block).expand_as(x)

        x_min = x.masked_fill(~cand, float('inf')).amin(dim=1, keepdim=True)
        x_max = x.masked_fill(~cand, float('-inf')).amax(dim=1, keepdim=True)
        out = (x - x_min) / (x_max - x_min + 1e-6)
        return sanitize(out).masked_fill(~cand, 0.0)    # rows without candidates -> 0
    # end
# end


class Feature_softmax_row(FeatureWrapperBase):
    # softmax each dim over the row's candidates; non-candidates -> 0
    def __init__(self, feature_inner, temperature=1.0):
        super().__init__(feature_inner)
        self.temperature = temperature
    # end

    def load_block(self, id_sample, pos_base, size_block):
        x = self.feature_inner.load_block(id_sample, pos_base, size_block)
        cand = self._cand_mask_3d(id_sample, pos_base, size_block).expand_as(x)

        out = torch.softmax((x / self.temperature).masked_fill(~cand, NEG_INF), dim=1)
        return torch.nan_to_num(out, nan=0.0).masked_fill(~cand, 0.0)
    # end
# end


class Feature_log_scaled(FeatureWrapperBase):
    # log(x + eps) for heavily right-skewed non-negative signals (e.g. attention);
    # non-candidates -> 0 so the negative log values never leak into set statistics.
    # compose with a row normalizer on top, e.g. Feature_znormed_row(Feature_log_scaled(...))
    def __init__(self, feature_inner, eps=1e-6):
        super().__init__(feature_inner)
        self.eps = eps
    # end

    def load_block(self, id_sample, pos_base, size_block):
        x = self.feature_inner.load_block(id_sample, pos_base, size_block)
        cand = self._cand_mask_3d(id_sample, pos_base, size_block).expand_as(x)
        return torch.log(x.clamp(min=0.0) + self.eps).masked_fill(~cand, 0.0)
    # end
# end


class Feature_znormed_global(FeatureWrapperBase):
    # dataset-fitted per-dim standardization over CANDIDATE entries; the trainer
    # fits it on the train split and the statistics are saved with the checkpoint
    def __init__(self, feature_inner):
        super().__init__(feature_inner)
        self.mean = None
        self.std = None
    # end

    def fitted(self):
        return self.mean is not None
    # end

    def fit(self, blocks, size_block):    # blocks: iterable of (id_sample, pos_base)
        xs = []
        for id_sample, pos_base in blocks:
            x = self.feature_inner.load_block(id_sample, pos_base, size_block)
            cand = self._cand_mask_3d(id_sample, pos_base, size_block).expand_as(x)
            xs.append(x[cand].reshape(-1, self.dim()))    # (t, l) candidate entries keep d-contiguity
        # end
        x_all = torch.cat(xs, dim=0)
        self.mean = x_all.mean(dim=0)
        self.std = x_all.std(dim=0) + 1e-6
        return self
    # end

    def get_state(self):
        return {'mean': self.mean, 'std': self.std}
    # end

    def set_state(self, state):
        self.mean, self.std = state['mean'], state['std']
    # end

    def load_block(self, id_sample, pos_base, size_block):
        assert self.fitted(), 'Feature_znormed_global must be fitted (the trainer fits it on register_router)'
        x = self.feature_inner.load_block(id_sample, pos_base, size_block)
        cand = self._cand_mask_3d(id_sample, pos_base, size_block).expand_as(x)
        return ((x - self.mean) / self.std).masked_fill(~cand, 0.0)
    # end
# end


'''---------------- routers ----------------'''


class RouterBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = []
        self.dim_in = 0
    # end

    def register_features(self, *features):
        self.features = list(features)
        self.dim_in = sum(f.dim() for f in features)
        self._build()
        return self
    # end

    def _build(self):    # called after features are known; trainable routers build modules here
        pass
    # end

    def trainable(self):
        return any(p.requires_grad for p in self.parameters())
    # end

    def build_block_x(self, id_sample, pos_base, size_block):    # -> (T, L, dim_in)
        assert self.features, 'call register_features first'
        return torch.cat([f.load_block(id_sample, pos_base, size_block) for f in self.features], dim=-1)
    # end

    def forward(self, x):    # (T, L, dim_in) -> scores (T, L)
        raise NotImplementedError
    # end

    def describe(self):
        return {
            'router': self.__class__.__name__,
            'features': [f.get_name() for f in self.features],
            'dim_in': self.dim_in,
        }
    # end

    def save_checkpoint(self, path):
        features_state = [f.get_state() if hasattr(f, 'get_state') else None for f in self.features]
        torch.save({
            'state_dict': self.state_dict(),
            'describe': self.describe(),
            'features_state': features_state,
        }, path)
    # end

    def load_checkpoint(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(state['state_dict'])

        for feature, feature_state in zip(self.features, state.get('features_state', [])):
            if feature_state is not None and hasattr(feature, 'set_state'):
                feature.set_state(feature_state)
            # end
        # end
        return self
    # end
# end


class Router_linear(RouterBase):    # logistic-regression-strength baseline
    def _build(self):
        self.head = nn.Linear(self.dim_in, 1)
    # end

    def forward(self, x):
        return self.head(x).squeeze(-1)
    # end
# end


class GatedMLPBlock(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, bias=True):
        super().__init__()
        self.project_gate = nn.Linear(dim_in, dim_hidden, bias=bias)
        self.project_up = nn.Linear(dim_in, dim_hidden, bias=bias)
        self.project_down = nn.Linear(dim_hidden, dim_out, bias=bias)
        self.activation = nn.SiLU()
    # end

    def forward(self, x):
        return self.project_down(self.activation(self.project_gate(x)) * self.project_up(x))
    # end
# end


class Router_mlp(RouterBase):    # pointwise gated MLP (the legacy SimpleMLP shape, fp32)
    def __init__(self, dim_hidden=64, num_blocks_mlp=2):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.num_blocks_mlp = num_blocks_mlp
    # end

    def _build(self):
        blocks = []
        dim_current = self.dim_in
        for _ in range(self.num_blocks_mlp - 1):
            blocks.append(GatedMLPBlock(dim_current, self.dim_hidden, self.dim_hidden))
            dim_current = self.dim_hidden
        # end
        blocks.append(GatedMLPBlock(dim_current, self.dim_hidden, 1))
        self.blocks = nn.Sequential(*blocks)
    # end

    def forward(self, x):
        return self.blocks(x).squeeze(-1)
    # end
# end


class Router_set_attention(RouterBase):
    # 1-layer self-attention across the candidate set, then an MLP head:
    # lets scores depend on how a candidate compares to the others
    def __init__(self, dim_model=32, num_heads=1, dim_hidden=64):
        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_hidden = dim_hidden
    # end

    def _build(self):
        self.embed = nn.Linear(self.dim_in, self.dim_model)
        self.attention = nn.MultiheadAttention(self.dim_model, self.num_heads, batch_first=True)
        self.norm = nn.LayerNorm(self.dim_model)
        self.head = GatedMLPBlock(self.dim_model, self.dim_hidden, 1)
    # end

    def forward(self, x):    # rows (steps) are the batch; positions are the set
        e = self.embed(x)
        a, _ = self.attention(e, e, e, need_weights=False)
        e = self.norm(e + a)
        return self.head(e).squeeze(-1)
    # end
# end


class Router_mockup_random(RouterBase):
    def __init__(self, seed=233):
        super().__init__()
        self.generator = torch.Generator().manual_seed(seed)
    # end

    def trainable(self):
        return False
    # end

    def forward(self, x):
        return torch.rand(x.shape[:-1], generator=self.generator, device='cpu').to(x.device)
    # end
# end


class Router_mockup_raw_feature(RouterBase):
    # score = dim 0 of the first registered feature (raw attention / conf / margin baselines)
    def trainable(self):
        return False
    # end

    def forward(self, x):
        return x[..., 0]
    # end
# end


class Router_mockup_nearest_right(RouterBase):
    # requires Feature_pos_delta registered FIRST (reads its signed dim);
    # right-side candidates ranked by proximity, then left-side by proximity
    def trainable(self):
        return False
    # end

    def forward(self, x):
        delta = x[..., 0]    # signed (p - just_unmasked) / L
        return torch.where(delta > 0, -delta, -(2.0 + delta.abs()))
    # end
# end


'''---------------- losses ----------------'''


class LossBase(ABC):
    @abstractmethod
    def __call__(self, scores, gap, cand_mask, h):    # -> scalar
        raise NotImplementedError
    # end
# end


def _masked_log_softmax(scores, mask):
    return F.log_softmax(scores.masked_fill(~mask, NEG_INF), dim=-1)
# end


def _soft_ce(scores, cand_mask, y):    # y >= 0, rows normalized by caller; skips empty rows
    logp = _masked_log_softmax(scores, cand_mask)
    loss_rows = -torch.where(y > 0, y * logp, torch.zeros_like(y)).sum(-1)
    row_valid = y.sum(-1) > 0
    assert bool(row_valid.any()), 'no row has a positive target'
    return loss_rows[row_valid].mean()
# end


class Loss_uniform_within_h(LossBase):
    # listwise CE against a uniform target over the next-h positions (matches recall@h)
    def __call__(self, scores, gap, cand_mask, h):
        pos = (gap >= 1) & (gap <= h)
        y = pos.float()
        y = y / y.sum(-1, keepdim=True).clamp(min=1.0)
        return _soft_ce(scores, cand_mask, y)
    # end
# end


class Loss_decay_within_h(LossBase):
    # legacy target: weight h+1-gap on the next-h positions (sooner = heavier)
    def __call__(self, scores, gap, cand_mask, h):
        pos = (gap >= 1) & (gap <= h)
        y = (h + 1 - gap).float() * pos.float()
        y = y / y.sum(-1, keepdim=True).clamp(min=1.0)
        return _soft_ce(scores, cand_mask, y)
    # end
# end


class Loss_bce_within_h(LossBase):
    # per-candidate binary "will unmask within h steps"; pos_weight counters class imbalance
    def __init__(self, pos_weight=None):
        self.pos_weight = pos_weight
    # end

    def __call__(self, scores, gap, cand_mask, h):
        pos = ((gap >= 1) & (gap <= h)).float()
        scores_flat = scores[cand_mask]
        target_flat = pos[cand_mask]

        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = torch.tensor(self.pos_weight, device=scores.device)
        # end
        return F.binary_cross_entropy_with_logits(scores_flat, target_flat, pos_weight=pos_weight)
    # end
# end


class Loss_plackett_luce(LossBase):
    # ListMLE over the true order of the next h unmasks (sequential softmax)
    def __call__(self, scores, gap, cand_mask, h):
        losses = []
        for j in range(1, h + 1):
            remaining = cand_mask & (gap >= j)
            target = gap == j
            row_valid = target.any(-1) & (remaining.sum(-1) > 1)
            if not bool(row_valid.any()):
                continue
            # end
            logp = _masked_log_softmax(scores, remaining)
            ll = torch.where(target, logp, torch.zeros_like(logp)).sum(-1)
            losses.append(-ll[row_valid])
        # end
        assert losses, 'no valid (row, j) pairs'
        return torch.cat(losses).mean()
    # end
# end


'''---------------- factories ----------------'''


class FactoryFeature:
    _REGISTRY = {
        'conf': Feature_conf,
        'margin': Feature_margin,
        'entropy': Feature_entropy,
        'attn_last': Feature_attn_last,
        'attn_all': Feature_attn_all,
        'pos_delta': Feature_pos_delta,
        'step_progress': Feature_step_progress,
        'mask_density': Feature_mask_density,
        'x0_stability': Feature_x0_stability,
    }

    _REGISTRY_WRAPPER = {
        'rank': Feature_rank_normed,
        'znorm_row': Feature_znormed_row,
        'minmax_row': Feature_minmax_row,
        'softmax_row': Feature_softmax_row,
        'log': Feature_log_scaled,
        'znorm_global': Feature_znormed_global,
    }

    @classmethod
    def create(cls, name, folder_data, **kwargs):
        return cls._REGISTRY[name](folder_data, **kwargs)
    # end

    @classmethod
    def wrap(cls, name, feature_inner, **kwargs):
        return cls._REGISTRY_WRAPPER[name](feature_inner, **kwargs)
    # end
# end


class FactoryRouter:
    _REGISTRY = {
        'linear': Router_linear,
        'mlp': Router_mlp,
        'set_attention': Router_set_attention,
        'mockup_random': Router_mockup_random,
        'mockup_raw': Router_mockup_raw_feature,
        'mockup_nearest_right': Router_mockup_nearest_right,
    }

    @classmethod
    def create(cls, name, **kwargs):
        return cls._REGISTRY[name](**kwargs)
    # end
# end


class FactoryLoss:
    _REGISTRY = {
        'uniform_within_h': Loss_uniform_within_h,
        'decay_within_h': Loss_decay_within_h,
        'bce_within_h': Loss_bce_within_h,
        'plackett_luce': Loss_plackett_luce,
    }

    @classmethod
    def create(cls, name, **kwargs):
        return cls._REGISTRY[name](**kwargs)
    # end
# end


'''---------------- trainer ----------------'''


class RouterTrainer:

    def __init__(self, folder_data, h=5, size_block=64, device='cpu',
                 lr=1e-3, weight_decay=1e-4, holdout=0.2, filter_result='all', seed=233):
        self.folder_data = folder_data
        self.h = h
        self.size_block = size_block
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.holdout = holdout
        self.filter_result = filter_result    # 'all' | 'pass' | 'fail'
        self.seed = seed

        self.router = None
        self.loss = None

        self.ids_train, self.ids_eval = self._split_samples()
    # end

    def register_router(self, router):
        self.router = router.to(self.device)

        # dataset-fitted normalizers are fitted on the TRAIN split only
        for feature in router.features:
            if hasattr(feature, 'fit') and hasattr(feature, 'fitted') and not feature.fitted():
                feature.fit(list(self._list_blocks(self.ids_train)), self.size_block)
                jprint(f'fitted {feature.get_name()} on {len(self.ids_train)} train samples')
            # end
        # end
        return self
    # end

    def register_loss(self, loss):
        self.loss = loss() if isinstance(loss, type) else loss
        return self
    # end

    def _read_result(self, id_sample):
        path = os.path.join(self.folder_data, str(id_sample), 'generated.json')
        if not os.path.exists(path):
            return 'unknown'
        # end
        with open(path, 'r') as file:
            return json.load(file).get('result', 'unknown')
        # end
    # end

    def _split_samples(self):
        ids_all = sorted(int(f) for f in os.listdir(self.folder_data) if f.isdigit())

        if self.filter_result in ('pass', 'fail'):
            ids_all = [i for i in ids_all if self._read_result(i) == self.filter_result]
        # end
        assert ids_all, f'no sample folders (filter_result={self.filter_result})'

        n_eval = max(1, int(len(ids_all) * self.holdout))
        return ids_all[:-n_eval], ids_all[-n_eval:]    # tail holdout, consistent with the mockup split
    # end

    def _list_blocks(self, ids_sample):    # yields (id_sample, pos_base)
        for id_sample in ids_sample:
            folder_base = os.path.join(self.folder_data, str(id_sample))
            with open(os.path.join(folder_base, '.pos_root'), 'r') as file:
                pos_root = int(file.read())
            # end
            num_blk = len([f for f in os.listdir(folder_base) if f.startswith('unmask_') and f.endswith('.pt')])

            for id_blk in range(num_blk):
                yield id_sample, pos_root + id_blk * self.size_block
            # end
        # end
    # end

    def _iter_blocks(self, ids_sample):
        for id_sample, pos_base in self._list_blocks(ids_sample):
            folder_base = os.path.join(self.folder_data, str(id_sample))
            x = self.router.build_block_x(id_sample, pos_base, self.size_block).to(self.device)
            unmask = load_stat(folder_base, 'unmask', pos_base, self.size_block)
            order = unmask.squeeze(-1).long() - pos_base
            yield x, order.to(self.device)
        # end
    # end

    def train(self, num_epochs=10, log_every=1):
        assert self.router is not None and self.loss is not None

        if not self.router.trainable():
            jprint(f'{self.router.__class__.__name__} is a mockup; nothing to train')
            return self
        # end

        torch.manual_seed(self.seed)
        optimizer = torch.optim.AdamW(self.router.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for id_epoch in range(num_epochs):
            losses = []
            for x, order in self._iter_blocks(self.ids_train):
                gap, cand_mask = build_geometry(order.cpu(), self.size_block)
                gap, cand_mask = gap.to(self.device), cand_mask.to(self.device)

                optimizer.zero_grad()
                scores = self.router(x)
                loss = self.loss(scores, gap, cand_mask, self.h)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            # end

            if id_epoch % log_every == 0:
                jprint(f'epoch {id_epoch}: loss {sum(losses) / len(losses):.4f} over {len(losses)} blocks')
            # end
        # end
        return self
    # end

    @torch.no_grad()
    def evaluate(self, hs=None, ids_sample=None):
        assert self.router is not None
        hs = hs or [3, self.h, 2 * self.h]
        ids_sample = ids_sample if ids_sample is not None else self.ids_eval

        values = {h: [] for h in hs}
        values_ap = []
        values_ndgc = []
        for x, order in self._iter_blocks(ids_sample):
            scores = self.router(x)
            evaluator = ScoreOrderEval(scores.cpu(), order.cpu())
            for h in hs:
                values[h].append(evaluator.recall_at_h(h))
            # end
            values_ap.append(evaluator.pr_auc(self.h))
            values_ndgc.append(evaluator.ndcg_at_h(self.h))
        # end

        report = {f'recall@{h}': summ(torch.cat(values[h])) for h in hs}
        report[f'pr_auc@{self.h}'] = summ(torch.cat(values_ap))
        report[f'ndgc@{self.h}'] = summ(torch.cat(values_ndgc))
        report['n_blocks'] = len(values_ap)
        report['router'] = self.router.describe()
        return report
    # end
# end
