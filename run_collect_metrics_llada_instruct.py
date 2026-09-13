#################################################
# Oracle collection, llada-instruct thread (full denoising, no cache).
#
# GSAI-ML/LLaDA-8B-Instruct, NATIVE block diffusion under FULL-CANVAS
# semantics -- matching run_llada_instruct.py: the whole prompt+gen canvas is
# forwarded every step (future blocks visible as masks), and conf outside the
# current block is forced to -inf before the unmask decision. Chat template is
# always applied (add --use_official_gsm8k_prompt for the OpenCompass 4-shot
# CoT gsm8k recipe; pair it with a 0-shot gsm8k mockup CSV).
#
# num_blocks is the investigation knob for this thread: sweep several values
# (e.g. 32 = official block 8 at len_target 256; 1 = pure diffusion).
# The attn plugin cannot infer the current block from dense canvas queries, so
# it is forced per block via set_id_block_forced (reset on exit).
#
# Usage:
#   python run_collect_metrics_llada_instruct.py \
#       --path_mockup benchmark_mockup/mockup_gsm8k_0shot_p10.csv \
#       --folder_output stats_oracle/llada_instruct_gsm8k_b32 \
#       --len_target 256 --num_blocks 32 --use_official_gsm8k_prompt
#################################################

import os

import torch
import torch.nn.functional as F

from components_llada import SimpleLogitsSnapshot, Stats
from tools_llada import BlockDiffusionQuotaHelper
from plugins_llada import CacheAttnPlugin_Enabled
from modeling_llada_yukai_06 import LLaDAModelLM
from collect_metrics_common import NAMES_STATS, OracleCollectorBase, build_parser, main_collect


class OracleCollector(OracleCollectorBase):

    use_chat_template = True

    @torch.no_grad()
    def collect_one(self, x, len_prompt, folder_stats):
        args = self.args
        id_mask = args.id_mask
        size_block = self.size_block

        neg_inf = torch.finfo(torch.float32).min

        # official generate() semantics: the whole canvas is in context every
        # step; x IS the canvas (prompt + all mask blocks)
        len_full = len_prompt + args.num_blocks * size_block
        assert x.shape[1] == len_full
        idx_canvas = torch.arange(len_full, dtype=torch.long, device=x.device)
        shape_target = (x.shape[0], len_full, -1)

        try:
            for id_block in range(args.num_blocks):
                block_start = len_prompt + id_block * size_block
                block_end = block_start + size_block
                mask_mask_blk = x[:, block_start:block_end] == id_mask

                idx_block = torch.arange(block_start, block_end, dtype=torch.long, device=x.device)
                idx_block_2d = idx_block.unsqueeze(0)
                quota_helper = BlockDiffusionQuotaHelper(mask_mask_blk, self.step_per_block)

                # dense canvas queries carry every block's rows; tell the attn
                # plugin which block is being decoded
                CacheAttnPlugin_Enabled.set_id_block_forced(id_block)

                # future-block masks are legitimate candidates to the sorter but
                # must never be unmasked before their block starts
                mask_outside = torch.ones((1, len_full), dtype=torch.bool, device=x.device)
                mask_outside[:, block_start:block_end] = False

                stats = Stats(block_start, block_end, names=NAMES_STATS)

                for step in range(self.step_per_block):
                    logits = self.model(x, idx_current=idx_canvas, shape_target=shape_target).logits

                    # metrics on the current block only (canvas rows == global positions)
                    logits_blk = logits[:, idx_block].float()
                    mask_blk = (x[:, block_start:block_end] == id_mask).squeeze(0)
                    sentinel = torch.tensor(neg_inf, device=x.device)

                    logp_blk = F.log_softmax(logits_blk, dim=-1)
                    p_blk = logp_blk.exp()

                    # margin = p(top1) - p(top2), fresh, pre-decision
                    top2 = p_blk.topk(2, dim=-1).values.squeeze(0)    # (size_block, 2)
                    margin_blk = torch.where(mask_blk, top2[:, 0] - top2[:, 1], sentinel)
                    stats.margin.add(block_start + step, margin_blk.cpu())

                    # full-vocab predictive entropy per candidate
                    entropy_blk = -(p_blk * logp_blk).sum(dim=-1).squeeze(0)    # (size_block,)
                    entropy_blk = torch.where(mask_blk, entropy_blk, sentinel)
                    stats.entropy.add(block_start + step, entropy_blk.cpu())

                    # argmax token per candidate (for offline argmax-stability features)
                    stats.x0.add(block_start + step, logits_blk.argmax(dim=-1).squeeze(0).cpu())

                    # confidence via the standard snapshot path (canvas-sized)
                    snapshot = SimpleLogitsSnapshot(x, x, id_mask)
                    snapshot.update_x0_(idx_block_2d, logits_blk)
                    conf_snapshot = snapshot.transform_logits(self.collector, logits_blk, idx_transform=idx_block_2d)
                    stats.conf.add(block_start + step, conf_snapshot.squeeze(0)[block_start:block_end].cpu())

                    # unmask decision, restricted to the current block
                    conf_snapshot = conf_snapshot.masked_fill(mask_outside, neg_inf)
                    idx_sorted_by_conf = self.sorter.argsort(conf_snapshot, snapshot)
                    num_unmask = quota_helper.get_quota(step)
                    idx_transform = idx_sorted_by_conf[:, :num_unmask]
                    snapshot.materialize_by_idx_(idx_transform, conf_snapshot)

                    # attention rows of the just-unmasked token: all layers, block-local coordinates
                    attn_all = self.plugin_cache_attn.collect_attn_from_all_blocks(self.model)   # (num_layers, size_block, size_block)
                    idx_local = idx_transform.squeeze(0) - block_start
                    assert bool((idx_local >= 0).all() and (idx_local < size_block).all()),\
                        f'unmask position outside current block: {idx_transform.tolist()}'
                    stats.attn.add(block_start + step, attn_all[:, idx_local, :].cpu())

                    stats.unmask.add(block_start + step, idx_transform.squeeze(0).cpu())

                    # token id written at the unmask position (unmask + token = full trajectory)
                    token_written = torch.gather(snapshot.x0, 1, idx_transform).squeeze(0)
                    stats.token.add(block_start + step, token_written.cpu())

                    snapshot.update_this(1, idx_transform, x0=x)
                # end for step

                os.makedirs(folder_stats, exist_ok=True)
                stats.stack_and_save_all(folder_stats)
            # end for block
        finally:
            CacheAttnPlugin_Enabled.set_id_block_forced(None)
        # end

        with open(os.path.join(folder_stats, '.pos_root'), 'w+') as file:
            file.write(str(len_prompt))
        # end

        return len_full
    # end
# end


if __name__ == '__main__':
    parser = build_parser(id_model='GSAI-ML/LLaDA-8B-Instruct', id_mask=126336)
    parser.add_argument('--use_official_gsm8k_prompt', action='store_true',
                        help='rebuild the OpenCompass 4-shot CoT gsm8k prompt '
                             '(use a 0-shot gsm8k mockup CSV with this)')
    main_collect(OracleCollector, LLaDAModelLM, parser)
# end
