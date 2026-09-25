import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from tools_debug import jprint


class DiffusionQuotaHelper(ABC):
    @abstractmethod
    def get_quota(self, step_current):
        pass
    # end
# end

class BlockDiffusionQuotaHelper(DiffusionQuotaHelper):
    def __init__(self, block_mask_index: torch.Tensor, steps_per_block: int) -> torch.Tensor:
        device = block_mask_index.device
        dtype = torch.long

        total = block_mask_index.sum(dim=1)                  # (B,)
        base  = torch.div(total, steps_per_block, rounding_mode='floor')  # (B,)
        rem   = total - base * steps_per_block                         # (B,)

        # Start with base for all steps
        num_transfer_tokens = base.unsqueeze(1).expand(-1, steps_per_block).to(dtype)  # (B, steps)

        # Add +1 to the first `rem[b]` steps for each batch b — without tensor slicing
        cols = torch.arange(steps_per_block, device=device).unsqueeze(0)               # (1, steps)
        add_mask = cols < rem.unsqueeze(1)                                   # (B, steps)

        # keep on CPU: one sync here instead of a hidden device->host sync per step in get_quota
        self.num_transfer_tokens = (num_transfer_tokens + add_mask.to(dtype)).cpu()       # (B, steps)
    # end

    def get_quota(self, step_current):
        quota_current = self.num_transfer_tokens[:, step_current]

        if quota_current.dim() == 2 and quota_current.size(1) == 1:
            quota_current = quota_current.squeeze(1)
        # end

        return int(quota_current.max())    # python int; batch size 1 in this pipeline
    # end

    def get_quota_max(self):
        return int(self.num_transfer_tokens.max())
    # end
# end


class ConfKSorter:

    def argsort(self, conf_all):
        idx_sorted = torch.argsort(conf_all, dim=1, descending=True)
        return idx_sorted
    # end
# end

class RandomKSorter(ConfKSorter):
    def argsort(self, confidence, snapshot):

        confidence = torch.where(
                snapshot.mask_mask,
                torch.rand(confidence.shape[0], confidence.shape[1], device=confidence.device),
                confidence
            )

        return super().argsort(confidence)

    # end
# end


class TopKSorter(ConfKSorter):
    def argsort(self, confidence, snapshot):
        return super().argsort(confidence)
    # end
# end


class ConfCollectorInterface(ABC):
    @abstractmethod
    def get_index(self, snapshot, idx=None):
        pass
    # end
# end

class TruthCollector(ConfCollectorInterface):
    def get_index(self, snapshot, idx=None):
        index = snapshot.y

        if idx is not None:
            index = torch.gather(index, 1, idx)
        # end

        return index.unsqueeze(-1)
    # end
# end


class MaxCollector(ConfCollectorInterface):
    def get_index(self, snapshot, idx=None):
        index = snapshot.x0

        if idx is not None:
            index = torch.gather(index, 1, idx)
        # end
        #         
        return index.unsqueeze(-1)
    # end
# end



class LogitsTransformer:
    def transform_logits(self, logits, collector):
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = collector.gather_x0_p(p, self)
        return x0_p
    # end
# end


class PPLCalculator:
    def cal(self, probs_all, mask_target=None, eps=1e-12):
        if mask_target is None:
            mask_target = slice(None)
        # end

        probs_collected = probs_all[mask_target].reshape(-1)  # [B * K]

        mean_prob = probs_collected.mean(dim=-1)  # [B]

        nll_collected = -torch.log(probs_collected + eps)   # [B, K]
        nll_per = nll_collected.mean(dim=-1)                 # [B]
        ppl_per = torch.exp(nll_per)                        # [B]

        return ppl_per.item(), mean_prob.item()
    # end
# end


class RefreshIdxHelper:
    TYPE_HIDDEN = {
        'k':'_k_previous',
        'v':'_v_previous'
    }

    def __init__(self, dict_filename_to_list_idx_sorted, type_hidden_str, size_block, randomed=False):
        self.dict_filename_to_list_idx_sorted = dict_filename_to_list_idx_sorted
        self.type_hidden=RefreshIdxHelper.TYPE_HIDDEN[type_hidden_str]
        self.size_block = size_block
        self.randomed = randomed
    # end

    def set_budget(self, budget):
        self.budget = budget
        return self
    # end

    def set_sample_id(self, id_sample):
        self.id_sample = id_sample
        return self
    # end

    def set_randomed(self, randomed):
        self.randomed = randomed
    # end

    def get_refresh_idx(self, x, id_step, id_block, return_sorted=True, id_step_global=None):   # id_step_global is special case
        id_sample = self.id_sample
        budget = self.budget
        size_block = self.size_block

        if id_step_global is None:  # this is used for special case
            id_step_global = id_step + id_block * size_block
        # end

        randomed = self.randomed

        filename = f'batch_{id_sample}{self.type_hidden}.pt'
        list_step_list_idx_sorted = self.dict_filename_to_list_idx_sorted[filename]

        assert list_step_list_idx_sorted[id_step_global]['step'] == id_step,\
            f'{list_step_list_idx_sorted[id_step_global]["step"]} == {id_step}'

        list_idx_sorted = list_step_list_idx_sorted[id_step_global]['idx']

        if budget < 1.0:
            budget = int(len(list_idx_sorted) * budget) or 1
        # end

        list_idx_sorted = torch.tensor(list_idx_sorted, dtype=torch.long, device=x.device)

        if randomed:
            idxs_list_idx_rand = torch.randperm(list_idx_sorted.shape[0])
            list_idx_sorted = list_idx_sorted[idxs_list_idx_rand]
        # end

        result = list_idx_sorted[:budget]

        return torch.sort(result)[0] if return_sorted else result
    # end
# end


'''shared runner helpers (used by the four baseline runners:
   run_llada_semi / run_dream_semi / run_llada_instruct / run_dream_instruct)'''

def collect_ids_stop(tokenizer):
    # generation-terminator ids for instruct checkpoints: the tokenizer eos plus
    # the chat-turn terminators some templates use instead of eos (Dream/Qwen
    # emits <|im_end|>). Missing tokens map to unk/None and are skipped.
    ids_stop = set()
    if tokenizer.eos_token_id is not None:
        ids_stop.add(tokenizer.eos_token_id)
    # end

    for token_stop in ('<|im_end|>', '<|endoftext|>', '<|eot_id|>'):
        id_token = tokenizer.convert_tokens_to_ids(token_stop)
        if id_token is not None and id_token >= 0 and id_token != tokenizer.unk_token_id:
            ids_stop.add(id_token)
        # end
    # end

    return ids_stop
# end


def truncate_text_at_stop(tokenizer, ids_generated, ids_stop, words_stop):
    # instruct SFT EOS-fills the canvas tail, and an EOS in an EARLIER block
    # leaves later blocks' junk in the answer (fatal for last-number answer
    # extraction) -> cut at the first stop id across the WHOLE generated region
    # at ids level, then decode and apply the harness stop words.
    has_done = False

    tensor_stop = torch.tensor(sorted(ids_stop), dtype=ids_generated.dtype, device=ids_generated.device)
    hits_stop = torch.isin(ids_generated, tensor_stop).nonzero()
    if hits_stop.numel() > 0:
        ids_generated = ids_generated[:hits_stop[0, 0]]
        has_done = True
    # end

    text = tokenizer.decode(ids_generated, skip_special_tokens=True)
    for word_stop in words_stop:
        if word_stop in text:
            text = text.split(word_stop)[0]
            has_done = True
        # end
    # end

    return text, has_done
# end


'''---------------- d2Cache scoring helpers ----------------
Ported near-verbatim from the official d2Cache repo (src/utils/common.py,
arXiv 2509.23094) so the in-framework reimplementation scores candidates
bit-comparably. certainty_density: Gaussian-smoothed fraction of KNOWN tokens
around each position (their boundary rule: left of the canvas counts as known;
right counts as known only if the last position is known). nucleus_select:
smallest score-mass set exceeding top_p, at least min_k.'''


def certainty_density(mask, sigma):
    # mask: (B, L) bool, True = known (decoded) token in the GEN region
    assert sigma > 0
    B, L = mask.shape
    device = mask.device
    float_mask = mask.float()

    padded_mask = F.pad(float_mask, (L, L), 'constant', 1.0)
    padded_mask[mask[:, -1] == False, 2 * L:] = 0.0

    extended_L = 3 * L
    padded_len = 2 * extended_L

    dist = torch.cat((
        torch.arange(extended_L, device=device),
        torch.arange(-extended_L, 0, device=device),
    ))
    kernel_fft = torch.fft.fft(torch.exp(-(dist ** 2) / (2 * sigma ** 2)), n=padded_len)

    weighted_sum_ext = torch.fft.ifft(
        torch.fft.fft(F.pad(padded_mask, (0, extended_L)), n=padded_len) * kernel_fft,
        n=padded_len).real
    kernel_sum_ext = torch.fft.ifft(
        torch.fft.fft(torch.ones(B, extended_L * 2, device=device), n=padded_len) * kernel_fft,
        n=padded_len).real

    weighted_sum = weighted_sum_ext[..., L:2 * L]
    kernel_sum_at_pos = kernel_sum_ext[..., L:2 * L].clamp_min(1e-8)
    return weighted_sum / kernel_sum_at_pos
# end


def nucleus_select(scores, top_p, min_k=1, mask=None):
    # scores: (B, L) non-negative; returns (B, L) bool selection
    scores = torch.where(mask, scores, torch.zeros_like(scores)) if mask is not None else scores

    probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-9)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus_mask = cumulative_probs <= top_p

    k = min(min_k, scores.shape[-1])
    top_k_mask = torch.arange(nucleus_mask.shape[-1], device=nucleus_mask.device) < k

    combined_mask = nucleus_mask | top_k_mask
    if mask is not None:
        combined_mask &= torch.gather(mask, 1, sorted_indices)
    # end

    return torch.zeros_like(scores, dtype=torch.bool).scatter_(
        dim=1, index=sorted_indices, src=combined_mask)
# end


def inflate_selection(q_mask, inflate_w):
    # d2Cache gap inflation: if two selected positions are within inflate_w of
    # each other, select everything between them (their eval default is 0=off)
    if inflate_w <= 0:
        return q_mask
    # end
    B, T = q_mask.shape
    device = q_mask.device
    arange_t = torch.arange(T, device=device).expand(B, -1)

    masked_next = torch.where(q_mask, arange_t, T)
    next_sel = torch.flip(torch.cummin(torch.flip(masked_next, dims=[-1]), dim=-1).values, dims=[-1])
    dist_next = next_sel - arange_t

    masked_prev = torch.where(q_mask, arange_t, -1)
    prev_sel = torch.cummax(masked_prev, dim=-1).values
    dist_prev = arange_t - prev_sel

    gap_len = dist_next + dist_prev
    return q_mask | ((gap_len <= inflate_w) & (prev_sel >= 0) & (next_sel < T))
# end


class RunnerReport:
    # per-sample wall-clock report shared by all runners, so baseline and router
    # runs produce directly comparable json artifacts; rewritten per sample
    # (crash-safe), no-op when config.path_report is unset
    def __init__(self):
        self.rows = []
    # end

    def add_and_dump(self, config, len_prompt, has_done, duration_s):
        path_report = getattr(config, 'path_report', None)
        if not path_report:
            return
        # end

        # peak GPU memory (GiB) since process start: allocated = tensors held
        # (weights + activations + caches), reserved = allocator footprint
        # (closer to nvidia-smi). None on CPU; never fails the report.
        mem_alloc_gib = mem_reserved_gib = None
        try:
            import torch
            if torch.cuda.is_available():
                mem_alloc_gib = round(torch.cuda.max_memory_allocated() / 2**30, 3)
                mem_reserved_gib = round(torch.cuda.max_memory_reserved() / 2**30, 3)
            # end
        except Exception:
            pass
        # end

        self.rows.append({
            'id_sample': len(self.rows),
            'len_prompt': len_prompt,
            'has_done': has_done,
            'duration_s': round(duration_s, 4),
            'mem_alloc_gib': mem_alloc_gib,
            'mem_reserved_gib': mem_reserved_gib,
        })

        import json
        with open(path_report, 'w') as file:
            json.dump({
                'path_router': getattr(config, 'path_router', None),
                'num_samples': len(self.rows),
                'duration_total_s': round(sum(row['duration_s'] for row in self.rows), 2),
                'rows': self.rows,
            }, file, indent=2)
        # end
    # end
# end