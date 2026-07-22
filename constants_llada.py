import torch

DTYPE_EVAL = torch.bfloat16
TEXT_MASK = '<|mdm_mask|>'
ID_MASK = 126336
NAME_MLP = 'models_mlp/mlp_attn_gsm8k_64.pt'
# NAME_MLP = 'mlp_attn_gsm8k_64.pt'   # 0.125 on instruction in ifeval
# NAME_MLP = 'mlp_attn.pt'  # -> 0000 on ifeval
NAME_MLP3 = 'mlp_3_gsm8k_64.pt'
NAME_MLP_FULLATTN = 'mlp_attn_full_gsm8k_64.pt'
