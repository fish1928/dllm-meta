#################################################
# Bisection tool: is our modified modeling file (modeling_llada_yukai_06)
# numerically equivalent to the official HF LLaDA model on real weights?
#
# Loads the two implementations SEQUENTIALLY (fits on one GPU), runs the same
# token ids through both, and compares logits. Also prints the prompt-encoding
# diff (add_special_tokens True vs False) to settle the BOS question.
#
# Usage:
#   python check_modeling_parity.py --id_model GSAI-ML/LLaDA-8B-Instruct --device cuda:0
#################################################

import argparse

import torch
from transformers import AutoModel, AutoTokenizer

from modeling_llada_yukai_06 import LLaDAModelLM
from constants_llada import DTYPE_EVAL, ID_MASK


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_model', type=str, default='GSAI-ML/LLaDA-8B-Instruct')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--len_mask', type=int, default=32, help='mask tokens appended to the prompt')
    return parser.parse_args()
# end


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.id_model, trust_remote_code=True)

    # ---- BOS / special-token check ----
    text = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': "Question: What is 2+2?\nLet's think step by step\nAnswer:"}],
        add_generation_prompt=True, tokenize=False)
    ids_true = tokenizer(text, add_special_tokens=True)['input_ids']
    ids_false = tokenizer(text, add_special_tokens=False)['input_ids']
    print(f'[bos-check] add_special_tokens=True : len={len(ids_true)}, head={ids_true[:5]}')
    print(f'[bos-check] add_special_tokens=False: len={len(ids_false)}, head={ids_false[:5]}')
    print(f'[bos-check] identical: {ids_true == ids_false}  '
          f'(if not, the harness [False] is missing tokens the official path [True] has)')
    print(f'[bos-check] bos_token_id={tokenizer.bos_token_id}, eos_token_id={tokenizer.eos_token_id}')

    # ---- fixed test input: templated prompt + masked tail, mimicking generation state ----
    ids_input = ids_true + [ID_MASK] * args.len_mask
    x = torch.tensor(ids_input, dtype=torch.long).view(1, -1).to(args.device)
    print(f'[input] total length {x.shape[1]} ({len(ids_true)} prompt + {args.len_mask} mask)')

    # ---- official model ----
    model_a = AutoModel.from_pretrained(args.id_model, trust_remote_code=True,
                                        torch_dtype=DTYPE_EVAL).eval().to(args.device)
    with torch.no_grad():
        logits_a = model_a(x).logits.float().cpu()
    # end
    del model_a
    torch.cuda.empty_cache()

    # ---- our modeling (dense path: full-window idx, plugins not registered -> need disabled stubs) ----
    from plugins_llada import (SaveKVPreviousPlugin_Disabled, CachePastKVPlugin_Disabled,
                               CacheAttnPlugin_Disabled, CacheVOPlugin_Disabled)
    model_b = LLaDAModelLM.from_pretrained(args.id_model, trust_remote_code=True,
                                           torch_dtype=DTYPE_EVAL).eval().to(args.device)
    model_b.fill_plugin(CachePastKVPlugin_Disabled)
    model_b.fill_plugin(SaveKVPreviousPlugin_Disabled)
    model_b.fill_plugin(CacheAttnPlugin_Disabled)
    model_b.fill_plugin(CacheVOPlugin_Disabled)

    idx_current = torch.arange(x.shape[1], dtype=torch.long, device=args.device)
    with torch.no_grad():
        logits_b = model_b(x, idx_current=idx_current,
                           shape_target=(1, x.shape[1], -1)).logits.float().cpu()
    # end

    # ---- comparison ----
    diff = (logits_a - logits_b).abs()
    agree_top1 = (logits_a.argmax(-1) == logits_b.argmax(-1)).float().mean().item()
    print(f'\n[logits] max|diff|  = {diff.max().item():.6f}')
    print(f'[logits] mean|diff| = {diff.mean().item():.6f}')
    print(f'[logits] top-1 agreement over positions = {agree_top1:.4f}')

    diff_mask = diff[0, len(ids_true):]
    print(f'[logits] masked-region max|diff| = {diff_mask.max().item():.6f} '
          f'(these positions drive unmask decisions)')

    if agree_top1 > 0.999 and diff.max().item() < 0.5:
        print('\nVERDICT: modeling files are equivalent (bf16 noise); '
              'the harness gap is prompt/tokenization/loop, NOT the model code.')
    else:
        print('\nVERDICT: modeling files DIVERGE -> debug modeling_llada_yukai_06 '
              '(prime suspects: rotary refactor, input embedding scaling, final norm).')
    # end
# end


if __name__ == '__main__':
    main()
# end
