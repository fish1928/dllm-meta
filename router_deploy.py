#################################################
# Router deployment adapter.
#
# Bridges offline training (router_llada features from stats folders) and
# online inference (run_llada_semi_cached_mlp building the SAME features from
# live cache state). Train/deploy parity is the whole point: both sides share
# percentile_rank_masked / sanitize and identical feature math, and a saved
# router travels as a BUNDLE = <name>.pt (weights) + <name>.json (spec).
#
# Spec fields:
#   features: list of ['attn_last' | 'attn_all' | 'pos_delta' | 'mask_density' | 'conf']
#   normalization: 'rank' | 'softmax_attn'   (softmax_attn: attention features get
#                  masked row-softmax, all other features get candidate rank)
#   conf_mode: 'fresh' | 'aged' | 'none'     (training-side treatment; deployment
#                  always feeds the live snapshot conf, stale by policy)
#   router_name / router_kwargs / dim_in / num_layers / mask_density_window /
#   softmax_temperature / h / loss / dataset ...
#################################################

import json
import os

import torch
import torch.nn.functional as F

from router_llada import (
    FactoryRouter,
    FeatureBase,
    percentile_rank_masked,
    sanitize,
)

NEG_INF = torch.finfo(torch.float32).min

FEATURES_ATTENTION = {'attn_last', 'attn_all'}


def feature_dim(name, num_layers):
    return {
        'attn_last': 1,
        'attn_all': num_layers,
        'pos_delta': 2,
        'mask_density': 1,
        'conf': 1,
    }[name]
# end


def spec_dim_in(spec):
    return sum(feature_dim(name, spec['num_layers']) for name in spec['features'])
# end


'''---------------- bundle save / load ----------------'''


class Feature_deploy_stub(FeatureBase):
    # dimension carrier for rebuilding a router at deployment; never loads blocks
    def __init__(self, dim_in):
        super().__init__(folder_data=None)
        self._dim = int(dim_in)
    # end

    def dim(self):
        return self._dim
    # end

    def load_block(self, id_sample, pos_base, size_block):
        raise RuntimeError('deploy stub cannot load stats blocks; use build_online_x')
    # end
# end


def save_router_bundle(router, spec, path_pt):
    router.save_checkpoint(path_pt)
    with open(path_pt[:-3] + '.json', 'w') as file:
        json.dump(spec, file, indent=2)
    # end
# end


def load_router_bundle(path_pt, device='cpu'):
    with open(path_pt[:-3] + '.json', 'r') as file:
        spec = json.load(file)
    # end

    router = FactoryRouter.create(
        spec['router_name'],
        **spec.get('router_kwargs', {}),
    ).register_features(Feature_deploy_stub(spec['dim_in']))

    router.load_checkpoint(path_pt)
    router = router.to(device).eval()
    return router, spec
# end


'''---------------- online feature construction ----------------'''


def masked_softmax_row(x, cand_3d, temperature):
    # mirrors Feature_softmax_row: masked softmax over positions, zeros elsewhere
    out = torch.softmax((x / temperature).masked_fill(~cand_3d, NEG_INF), dim=1)
    return torch.nan_to_num(out, nan=0.0).masked_fill(~cand_3d, 0.0)
# end


def build_online_x(spec, attn_rows_all, conf_block, mask_still, pos_last):
    """
    Build the router input for ONE sparse step, block-local.

    attn_rows_all: (num_layers, L) head-averaged attention rows of the just-
                   unmasked token (aggregated over k when num_unmask > 1)
    conf_block:    (L,) live snapshot confidence (stale by refresh policy)
    mask_still:    (L,) bool, still-masked positions (the candidates)
    pos_last:      0-dim long tensor, block-local position of the last unmask

    Returns (1, L, dim_in), same layout as offline build_block_x row t.
    """
    L = mask_still.shape[0]
    device = mask_still.device
    cand = mask_still.view(1, L)
    cand_3d = cand.unsqueeze(-1)

    temperature = float(spec.get('softmax_temperature', 1.0))
    window = int(spec.get('mask_density_window', 3))

    parts = []
    for name in spec['features']:
        if name == 'attn_last':
            raw = attn_rows_all[-1].view(1, L, 1)
        elif name == 'attn_all':
            raw = attn_rows_all.t().contiguous().view(1, L, -1)
        elif name == 'conf':
            raw = conf_block.view(1, L, 1)
        elif name == 'pos_delta':
            positions = torch.arange(L, device=device)
            delta = (positions - pos_last).float() / L
            raw = torch.stack([delta, delta.abs()], dim=-1).view(1, L, 2)
        elif name == 'mask_density':
            # ones-kernel then divide, matching Feature_mask_density bit-for-bit
            kernel = torch.ones(1, 1, 2 * window + 1, device=device)
            count = F.conv1d(mask_still.float().view(1, 1, L), kernel, padding=window)
            raw = (count / (2 * window + 1)).view(1, L, 1)
        else:
            raise ValueError(f'unknown online feature: {name}')
        # end

        raw = sanitize(raw)

        if spec['normalization'] == 'softmax_attn' and name in FEATURES_ATTENTION:
            parts.append(masked_softmax_row(raw, cand_3d.expand_as(raw), temperature))
        else:
            parts.append(percentile_rank_masked(raw, cand))
        # end
    # end

    return torch.cat(parts, dim=-1)    # (1, L, dim_in)
# end


@torch.no_grad()
def select_topk_candidates(router, spec, attn_rows_all, conf_block, mask_still, pos_last, k):
    """Score candidates online and return the block-local indices of the top k."""
    x = build_online_x(spec, attn_rows_all, conf_block, mask_still, pos_last)
    scores = router(x).view(-1)
    scores = scores.masked_fill(~mask_still, NEG_INF)
    return scores.topk(k).indices
# end
