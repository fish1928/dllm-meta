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
        self.num_transfer_tokens = num_transfer_tokens + add_mask.to(dtype)       # (B, steps)
    # end

    def get_quota(self, step_current):
        quota_current = self.num_transfer_tokens[:, step_current]

        if quota_current.dim() == 2 and quota_current.size(1) == 1:
            quota_current = quota_current.squeeze(1)
        # end

        return quota_current
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
    def get_index(self, snapshot):
        pass
    # end
# end

class TruthCollector(ConfCollectorInterface):
    def get_index(self, snapshot):
        index = snapshot.y.unsqueeze(-1)
        return index
    # end
# end


class MaxCollector(ConfCollectorInterface):
    def get_index(self, snapshot):
        index = snapshot.x0.unsqueeze(-1)
        return index
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
            f'{list_step_list_idx_sorted[id_step_global]['step']} == {id_step}'

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