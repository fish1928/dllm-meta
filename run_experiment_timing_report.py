#################################################
# Aggregate the timing campaign (run_experiment_timing_8gpu.bash):
# per (method, thread, gen length): measured s/doc (mean over the 5 docs,
# generation only) + ANALYTIC TFLOPs/doc + effective TFLOP/s.
#
# Model dims are read from the HF config of each checkpoint (exact GQA/FFN/
# vocab); a fallback table is used offline. FLOPs formulas (documented
# approximations, same conventions as run_experiment_6_horizontal_equal_quality):
#   per-row per-layer: QKVO = 2 d (2d + 2 d_kv), gated FFN = 6 d ff,
#   attention = 4 T_ctx d; LM head = 2 d V per logits row; dllm-cache cached
#   rows pay the V-projection (2 d d_kv); d2cache rollout extras at expected
#   size p*T. 'ours' on llada_instruct (block-structured runner) is an
#   approximation: Kr ticks refresh the CURRENT 32-block, adaptive steps h+1
#   rows, prompt ticks P rows.
#
# Usage:  python run_experiment_timing_report.py [--folder results_experiment_timing]
#################################################

import argparse
import glob
import json
import math
import os

THREADS = ('llada_base', 'llada_instruct', 'dream_base', 'dream_instruct')
METHODS = ('dense', 'ours', 'dllmcache', 'fastdllm', 'd2cache')
LENS = (256, 512)
KP, KR, H = 64, 8, 8    # must match the campaign's env defaults

ID_MODEL = {
    'llada_base': 'GSAI-ML/LLaDA-8B-Base', 'llada_instruct': 'GSAI-ML/LLaDA-8B-Instruct',
    'dream_base': 'Dream-org/Dream-v0-Base-7B', 'dream_instruct': 'Dream-org/Dream-v0-Instruct-7B',
}

DIMS_FALLBACK = {    # d, d_kv, ff, layers, vocab
    'llada': dict(d=4096, d_kv=4096, ff=12288, layers=32, vocab=126464),
    'dream': dict(d=3584, d_kv=512, ff=18944, layers=28, vocab=152064),
}


def model_dims(thread):
    family = 'llada' if 'llada' in thread else 'dream'
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(ID_MODEL[thread], trust_remote_code=True)
        d = config.hidden_size
        n_heads = config.num_attention_heads
        n_kv = getattr(config, 'num_key_value_heads', None) or n_heads
        d_kv = (d // n_heads) * n_kv
        ff = getattr(config, 'intermediate_size', None) or getattr(config, 'mlp_hidden_size')
        layers = getattr(config, 'num_hidden_layers', None) or getattr(config, 'n_layers')
        vocab = config.vocab_size
        return dict(d=d, d_kv=d_kv, ff=ff, layers=layers, vocab=vocab)
    except Exception:
        return DIMS_FALLBACK[family]
# end


def _mk(dims):
    d, d_kv, ff = dims['d'], dims['d_kv'], dims['ff']
    row_fixed = 2 * d * (2 * d + 2 * d_kv) + 6 * d * ff    # QKVO + gated FFN
    def fl_rows(rows, t_ctx):
        return rows * dims['layers'] * (row_fixed + 4 * t_ctx * d)
    def fl_head(rows):
        return rows * 2 * d * dims['vocab']
    def fl_vrow():
        return 2 * d * d_kv
    return fl_rows, fl_head, fl_vrow
# end


def _count_ticks(steps, every, include_zero):
    if not every:
        return 1 if include_zero else 0
    ticks = [s for s in range(steps) if s % every == 0]
    return len(ticks) if include_zero else len(ticks) - 1
# end


def flops_doc(method, thread, dims, P, G):
    fl_rows, fl_head, fl_vrow = _mk(dims)
    S, T, L = G, P + G, dims['layers']

    if method == 'dense':
        return S * (fl_rows(T, T) + fl_head(T))

    if method == 'ours':
        n_kp = _count_ticks(S, KP, include_zero=False)
        if thread == 'llada_instruct':    # block runner: Kr refreshes the CURRENT 32-block
            size_bk = 32
            n_blocks = G // size_bk
            n_kr = n_blocks * _count_ticks(size_bk, KR, include_zero=True)
            rows_kr = size_bk + 1
        else:
            n_kr = _count_ticks(S, KR, include_zero=True)
            rows_kr = G + 1
        n_adapt = S - n_kr
        total = fl_rows(P, P)
        total += n_kr * (fl_rows(rows_kr, T) + fl_head(rows_kr))
        total += n_adapt * (fl_rows(H + 1, T) + fl_head(H + 1))
        total += n_kp * fl_rows(P, T)
        return total

    if method == 'dllmcache':
        n_all = _count_ticks(S, KP, include_zero=True)
        n_resp = max(0, _count_ticks(S, KR, include_zero=True) - n_all)
        n_adapt = S - n_all - n_resp
        rows_sel = max(1, int(0.25 * G))
        extra_sel = _mk(dims)[0](1, T) // L - fl_vrow()    # full row minus its V-proj
        total = n_all * (fl_rows(T, T) + fl_head(T))
        total += n_resp * (fl_rows(G, T) + P * L * fl_vrow() + fl_head(T))
        total += n_adapt * (L * (T * fl_vrow() + rows_sel * extra_sel) + fl_head(T))
        return total

    if method == 'fastdllm':
        size_bk = 32
        n_blocks = G // size_bk
        per_block = fl_rows(T, T) + fl_head(T) \
                    + (size_bk - 1) * (fl_rows(size_bk, T) + fl_head(size_bk))
        return n_blocks * per_block

    if method == 'd2cache':
        rows = 32 + 1 + 0.1 * T
        return fl_rows(T, T) + fl_head(T) + S * (fl_rows(rows, T) + fl_head(rows))

    raise ValueError(method)
# end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', default='results_experiment_timing')
    config = parser.parse_args()

    dims_of = {thread: model_dims(thread) for thread in THREADS}

    print(f'\n===== timing + analytic TFLOPs ({config.folder}) =====')
    print(f'  {"method":10s} {"thread":15s} {"gen":>4s} {"P avg":>6s} '
          f'{"s/doc":>8s} {"TF/doc":>8s} {"TF/s":>7s} {"x dense":>8s} {"mem GiB":>8s}')

    rows_csv = ['method,thread,gen_len,len_prompt_avg,s_doc,tflops_doc,tflops_per_s,'
                'speedup_vs_dense,mem_alloc_gib,mem_reserved_gib']
    s_dense = {}
    for gen in LENS:
        for method in METHODS:
            for thread in THREADS:
                tag = f'{method}__{thread}__g{gen}'
                path = os.path.join(config.folder, f'{tag}__runner.json')
                if not os.path.exists(path):
                    print(f'  {method:10s} {thread:15s} {gen:>4d}   (missing)')
                    continue
                report = json.load(open(path))
                rows = report['rows']
                s_doc = report['duration_total_s'] / len(rows)
                P = sum(r['len_prompt'] for r in rows) / len(rows)
                tflops = flops_doc(method, thread, dims_of[thread], P, gen) / 1e12
                tfs = tflops / s_doc
                if method == 'dense':
                    s_dense[(thread, gen)] = s_doc
                speedup = s_dense.get((thread, gen), float('nan')) / s_doc
                # peak over the run (rows carry the running peak; last is max)
                mem_alloc = max((r.get('mem_alloc_gib') or 0) for r in rows) or None
                mem_res = max((r.get('mem_reserved_gib') or 0) for r in rows) or None
                text_mem = f'{mem_alloc:>8.2f}' if mem_alloc else '      --'
                print(f'  {method:10s} {thread:15s} {gen:>4d} {P:>6.0f} '
                      f'{s_doc:>8.2f} {tflops:>8.1f} {tfs:>7.1f} {speedup:>7.2f}x {text_mem}')
                rows_csv.append(f'{method},{thread},{gen},{P:.0f},{s_doc:.2f},'
                                f'{tflops:.1f},{tfs:.1f},{speedup:.2f},'
                                f'{mem_alloc if mem_alloc else ""},{mem_res if mem_res else ""}')
            # end
        # end
    # end

    path_csv = os.path.join(config.folder, 'timing_summary.csv')
    with open(path_csv, 'w') as file:
        file.write('\n'.join(rows_csv) + '\n')
    print(f'\ncsv -> {path_csv}')
    print('note: speedup vs dense assumes the dense row of the same (thread, gen) exists')
# end


if __name__ == '__main__':
    main()
