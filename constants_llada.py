import torch

DTYPE_EVAL = torch.bfloat16
TEXT_MASK = '<|mdm_mask|>'
ID_MASK = 126336            # LLaDA '<|mdm_mask|>'
ID_MASK_DREAM = 151666      # Dream '<|mask|>' -- pass as id_mask in model_args for dream runs
NAME_MLP = 'models_mlp/mlp_attn_gsm8k_64.pt'
# NAME_MLP = 'mlp_attn_gsm8k_64.pt'   # 0.125 on instruction in ifeval
# NAME_MLP = 'mlp_attn.pt'  # -> 0000 on ifeval
NAME_MLP3 = 'mlp_3_gsm8k_64.pt'
NAME_MLP_FULLATTN = 'mlp_attn_full_gsm8k_64.pt'
