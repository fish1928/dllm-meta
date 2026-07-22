import torch

class AttnOrderEval:

    def _build_geometry(self):
        T, L = self.T, self.L
        dev = self.attn.device

        step_of = torch.full((L,), -1, dtype=torch.long, device=dev)   # [L] step at which a position is unmasked, -1 = never
        step_of[self.order] = torch.arange(T, device=dev)

        t_idx = torch.arange(T, device=dev).view(T, 1)
        future_gap = step_of.view(1, L) - t_idx                        # [T, L]  gap>0 -> future candidate, gap=1 -> next
        cand_mask = future_gap > 0                                     # [T, L]  still-masked at step t

        neg_inf = torch.finfo(self.attn.dtype).min
        S = self.attn.masked_fill(~cand_mask, neg_inf)                 # [T, L]  attention restricted to future candidates

        n_cand = (T - 1) - torch.arange(T, device=dev)                 # [T]  number of still-masked tokens after step t
        return future_gap, cand_mask, S, n_cand
    # end

    def __init__(self, attn_rows, order):                             # attn_rows [T, L], order [T] long
        assert attn_rows.dim() == 2, "attn_rows.dim(): {} == 2 false".format(attn_rows.dim())
        assert order.dim() == 1, "order.dim(): {} == 1 false".format(order.dim())

        self.attn = attn_rows.to(torch.float64)
        self.order = order.to(torch.long)

        self.T, self.L = self.attn.shape

        self.future_gap, self.cand_mask, self.S, self.n_cand = self._build_geometry()
    # end

    def get_attn(self):
        return self.attn
    # end

    def get_order(self):
        return self.order
    # end

    def get_future_gap(self):
        return self.future_gap
    # end

    def get_S(self):
        return self.S
    # end

    def get_n_cand(self):
        return self.n_cand
    # end

    def _nan_invalid(self, value, valid):                            # [T], [T] bool -> [T] with nan where invalid
        nan = torch.full_like(value, float("nan"))
        return torch.where(valid, value, nan)
    # end

    def recall_at_h(self, h):   # of the next-h soonest tokens, how many land in q's top-h attended; per-step [T], use .nanmean()
        rel = (self.future_gap >= 1) & (self.future_gap <= h)        # [T, L]  soonest-h == relevant
        toph = self.S.topk(h, dim=-1).indices                       # [T, h]  predicted top-h candidates
        hit = rel.gather(-1, toph).double().sum(-1)                 # [T]
        recall = hit / h                                            # [T]

        valid = self.n_cand >= h                                    # need h relevant tokens to exist
        return self._nan_invalid(recall, valid)
    # end

    def pr_auc(self, h):   # average precision, positives = next-h soonest, scored by attention; per-step [T], use .nanmean()
        idx_sorted = self.S.argsort(dim=-1, descending=True)        # [T, L]  non-candidates (-inf) sink to the end
        gap_sorted = self.future_gap.gather(-1, idx_sorted)         # [T, L]

        is_cand = gap_sorted > 0                                    # [T, L]  a retrieved candidate at this rank
        is_pos = (gap_sorted >= 1) & (gap_sorted <= h)              # [T, L]  a soonest-h positive

        tp = is_pos.cumsum(-1).double()                            # [T, L]
        retrieved = is_cand.cumsum(-1).double().clamp_min(1.0)     # [T, L]  rank among candidates only
        precision = tp / retrieved                                # [T, L]

        P = is_pos.double().sum(-1)                                # [T]  == min(h, n_cand)
        ap = (precision * is_pos.double()).sum(-1) / P.clamp_min(1.0)   # [T]

        valid = self.n_cand > h                                    # need >=1 positive and >=1 negative
        return self._nan_invalid(ap, valid)
    # end

    def ndcg_at_h(self, H, gain="linear"):   # graded by soonness, attention ranks the candidates; per-step [T], use .nanmean()
        dev = self.attn.device

        base = (H - (self.future_gap - 1)).clamp(min=0).double()    # [T, L]  gap=1 -> H, gap=H -> 1, gap>H -> 0
        if gain == "exp":
            base = torch.exp2(base) - 1.0
        # end
        grade = base * self.cand_mask.double()                     # [T, L]  zero out non-candidates

        disc = 1.0 / torch.log2(torch.arange(H, device=dev).double() + 2.0)   # [H]
        pred = self.S.topk(H, dim=-1).indices                      # [T, H]  predicted ranking
        dcg = (grade.gather(-1, pred) * disc).sum(-1)              # [T]

        ideal = grade.topk(H, dim=-1).values                       # [T, H]  ideal ranking
        idcg = (ideal * disc).sum(-1)                              # [T]

        ndcg = dcg / idcg.clamp_min(torch.finfo(torch.float64).tiny)   # [T]
        valid = (self.n_cand >= 2) & (idcg > 0)
        return self._nan_invalid(ndcg, valid)
    # end
# end

class MarginOrderEval(AttnOrderEval):

    def __init__(self, margin, order):                              # margin [T, L] (already preprocessed, p1 - p2), order [T] long
        assert margin.dim() == 2, "margin.dim(): {} == 2 false".format(margin.dim())

        super().__init__(margin, order)                             # reuse geometry + recall_at_h / pr_auc / ndcg_at_h
    # end

    def get_margin(self):
        return self.attn   # the base stores the ranking score here (generic [T, L] score slot)
    # end
# end


class ConfidenceOrderEval(AttnOrderEval):

    def __init__(self, confidence, order):                          # confidence [T, L] (already preprocessed), order [T] long
        assert confidence.dim() == 2, "confidence.dim(): {} == 2 false".format(confidence.dim())

        super().__init__(confidence, order)                         # reuse geometry + recall_at_h / pr_auc / ndcg_at_h
    # end

    def get_confidence(self):
        return self.attn   # the base stores the ranking score here (generic [T, L] score slot)
    # end
# end



class ScoreOrderEval(AttnOrderEval):

    def __init__(self, score, order):
        super().__init__(score, order)
    # end

    def get_score(self):
        return self.score
    # end
# end


def summ(x):
    return "{:.3f} (n={})".format(x.nanmean().item(), int((~x.isnan()).sum()))
# end
