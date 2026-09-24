#################################################
# Fast-dLLM CACHE-ONLY reimplementation -- Dream runner. Identical to
# run_llada_fastdllm (DualCache + uniform greedy decode, no parallel decoding)
# except the dream logits shift: the token at position p is predicted by the
# output row at p-1, so block-only forwards query [block_start-1, block_end)
# and read logits[:, :-1]; the full-canvas block-start forward reads
# logits[:, idx_block - 1]. Same convention as run_dream_d2cache.
#
# Threads: dream_base (plain prompts) and dream_instruct
# (truncate_at_eos=True via model_args).
#
# model_args example (block 32 at len 256):
#   ...,runner=run_dream_fastdllm,num_blocks=8,num_unmask_per_step=1,...
#################################################

from run_llada_fastdllm import RunModel as RunModelLLaDA


class RunModel(RunModelLLaDA):

    DREAM_SHIFT = True
# end
