#################################################
# dLLM-Cache reimplementation -- Dream runner. Identical to
# run_llada_dllm_cache except the dream logits shift: the token at position p
# is predicted by the output row at p-1, so
#   - decode reads logits[:, idx_gen - 1] (full window is forwarded every
#     step, so the shifted rows always have fresh-or-cached outputs), and
#   - the VO plugin's prompt/response boundary moves back one row
#     (set_prompt_length(len_prompt - 1), set_response_length(size_block + 1)):
#     row len_prompt-1 predicts the FIRST response token, so it must be in the
#     adaptively-updated region, not frozen with the prompt between Kp ticks.
#
# Requires the VO hooks in modeling_dream_yukai (select_hidden /
# load_merge_and_update_hidden in DreamDecoderLayerYukai, RoPE selected
# in-attention by the per-layer idx_current).
#
# Threads: dream_base (plain prompts) and dream_instruct (truncate_at_eos=True
# via model_args). num_blocks=1, size_batch=1.
#
# model_args example (v-rate 0.25, Kp 96, Kr 16):
#   ...,runner=run_dream_dllm_cache,dllmc_v_rate=0.25,
#   step_refresh_remainder=16,step_refresh_remainder_prompt=96,...
#################################################

from run_llada_dllm_cache import RunModel as RunModelLLaDA


class RunModel(RunModelLLaDA):

    DREAM_SHIFT = True
# end
