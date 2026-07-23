import torch
import torch.nn.functional as F
import os

from tools_debug import jprint

class SimpleLogitsSnapshot:

    def __init__(self, x, y, id_mask, x0=None, conf=None):
        self.id_mask = id_mask

        self.x = x
        self.y = y

        assert x0 is None or x0.shape[1] == self.x.shape[1]
        assert conf is None or conf.shape[1] == self.x.shape[1]

        if x0 is None:
            self.x0 = torch.zeros(self.x.shape, dtype=torch.long, device=self.x.device)
        else:
            self.x0 = x0
        # end

        self.p_finalized = torch.zeros(self.x.shape, dtype=torch.float32, device=self.x.device)

        if conf is None:
            self.conf = torch.zeros(self.x.shape, dtype=torch.float32, device=self.x.device)
        else:
            self.conf = conf
        # end
    # end

    def get_x(self):
        return self.x
    # end

    def get_y(self):
        return self.y
    # end

    def get_p_finalized(self):
        return self.p_finalized
    # end


    def materialize_by_idx_(self, idx, conf):

        x0_target = torch.gather(self.x0, dim=-1, index=idx)
        conf_target = torch.gather(conf, dim=-1, index=idx)
        self.x.scatter_(1, idx, x0_target)
        self.p_finalized.scatter_(1, idx, conf_target)
    # end

    # logits rows must be aligned with idx_transform positions (same order)
    def update_logits_(self, idx_transform, logits):
        assert idx_transform.dim() == 2, "idx_transform.dim(): {} == 2 false".format(idx_transform.dim())

        x0 = torch.argmax(logits, dim=-1)
        self.x0.scatter_(1, idx_transform, x0)
        return idx_transform
    # end


    # logits rows must be aligned with idx_transform positions (same order);
    # call update_logits_ first so collector reads fresh x0 at those positions
    def transform_logits(self, collector, logits, idx_transform=None):
        index_p_transform = collector.get_index(self, idx_transform)
        p_transformed = F.softmax(logits.float(), dim=-1)

        x0_p_transformed = torch.gather(p_transformed, dim=-1, index=index_p_transform).squeeze(-1)

        if idx_transform is not None:
            self.conf.scatter_(1, idx_transform, x0_p_transformed)
        else:
            self.conf = x0_p_transformed
        # end

        neg_inf = torch.tensor(
            torch.finfo(self.conf.dtype).min,
            device=self.conf.device,
            dtype=self.conf.dtype
        )

        mask_mask = self.x == self.id_mask

        return torch.where(mask_mask, self.conf, neg_inf)
    # end

    def update_this(self, dim, idx_src, idx_tgt=None, **kwargs):

        if idx_tgt is None:
            idx_transform = idx_src
        else:
            idx_tgt=idx_tgt.unsqueeze(0)
            
            idx_transform = torch.gather(idx_tgt, dim=-1, index=idx_src)
        # end

        for k, v in kwargs.items(): # k is a local property name, v is the target to scatter
            v.scatter_(dim, idx_transform, torch.gather(getattr(self, k), dim=dim, index=idx_src))
        # end

        return self
    # end


    # def get_margin_p(self, idx_a=0, idx_b=1):
    #     logits = logits.to(torch.float64)                            # match the float64 softmax convention; chunk over T if memory-bound
    #     mask_mask = self.x == self.id_mask

    #     lse = torch.logsumexp(logits, dim=-1)                        # [T, L]  log-partition (full vocab scan)
    #     top2 = logits.topk(2, dim=-1).values                        # [T, L, 2]  rank 0 = largest logit
    #     p1 = (top2[..., idx_a] - lse).exp()                             # [T, L]  top-1 prob
    #     p2 = (top2[..., idx_b] - lse).exp()                             # [T, L]  top-2 prob
    #     margin_p = p1 - p2

    #     neg_inf = torch.tensor(torch.finfo(logits.dtype).min, device=logits.device, dtype=logits.dtype)
    #     margin_p = torch.where(mask_mask.squeeze(0), margin_p.squeeze(0), neg_inf)
    #     return margin_p
    # # end

    # def get_margin_p(self, idx_a=0, idx_b=1):
    #     p = F.softmax(self.logits.to(torch.float64), dim=-1)
    #     idx_sorted = torch.argsort(p, dim=-1, descending=True)        # [N, V]

    #     a = torch.gather(p, -1, idx_sorted[:, idx_a:idx_a+1])         # [N, 1]  keep dim
    #     b = torch.gather(p, -1, idx_sorted[:, idx_b:idx_b+1])         # [N, 1]
    #     return (a - b).squeeze(-1)
    # # end

    # def update_logits_(self, idx_transform, logits):
    #     B, L, H = logits.shape
    #     assert idx_transform.dim() == 2, "idx_transform.dim(): {} == 2 false".format(idx_transform.dim())
        
    #     idx_logits = idx_transform.view(B,-1,1).expand(B, -1, H)

    #     self.logits.scatter_(1, idx_logits, logits)
    #     x0 = torch.argmax(logits, dim=-1)
    #     self.x0.scatter_(1, idx_transform, x0)
    # # end

    # def transform_logits_(self, collector):

    #     logits_transform = self.logits
    #     p = F.softmax(logits_transform.to(torch.float64), dim=-1)

    #     index_p_all = collector.get_index(self)

    #     x0_p = torch.gather(p, dim=-1, index=index_p_all).squeeze(-1)

    #     neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)

    #     mask_mask = self.x == self.id_mask
    #     self.conf = torch.where(mask_mask, x0_p, neg_inf)  # (B, L)   # so only the masked part has confidence

    #     return self.conf
    # # end

# end

'''For RunModelAndCollectStats'''

class IndexedElementList:
    def __init__(self, idx_start, idx_end, name=None):
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.indexed_elements = [None] * (idx_end - idx_start)
        self.name = name
    # end

    def add(self, idx_relative, element):
        self.indexed_elements[idx_relative - self.idx_start] = element
    # end

    def get(self, idx_relative):
        return self.indexed_elements[idx_relative - self.idx_start]
    # end

    def has_empty(self):
        for indexed_element in self.indexed_elements:
            if indexed_element is None:
                return True
            # end
        # end

        return False
    # end

    def stack_and_save(self, path_to_save):
        stacked_elements = torch.stack(self.indexed_elements)
        torch.save(stacked_elements, os.path.join(path_to_save, f'{self.name}_{self.idx_start}_{self.idx_end}.pt'))
    # end
# end


class Stats:
    _STATS = ('margin', 'conf', 'attn', 'unmask')

    def __init__(self, idx_start, idx_end, margin_idx_a=0, margin_idx_b=1):
        for name in self._STATS:
            setattr(self, name, IndexedElementList(idx_start, idx_end, name=name))
        # end

        self.margin.idx_a = margin_idx_a
        self.margin.idx_b = margin_idx_b
    # end

    def stack_and_save_all(self, path_to_save):
        for name_stat in self._STATS:
            statlist = getattr(self, name_stat)
            if not statlist.has_empty():
                statlist.stack_and_save(path_to_save)
            # end
        # end
    # end
# end

