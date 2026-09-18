#################################################
# d2Cache reimplementation -- Dream runner. Identical to run_llada_d2cache
# except the dream logits shift: the token at position p is predicted by the
# output row at p-1, so candidate logits are read from the (candidate - 1)
# rows, which are queried alongside the candidates ([extras | cand-1 | cand]).
# Their repo handles the same shift (response_mask pad + last-prompt-token
# selection); querying the shifted rows directly is our framework's standard
# pattern and keeps their KV fresh as a side effect.
#
# Threads: dream_base (plain prompts) and dream_instruct (use_chat_template
# + truncate_at_eos=True via model_args). num_blocks=1 (blockless by design).
#################################################

from run_llada_d2cache import RunModel as RunModelLLaDA


class RunModel(RunModelLLaDA):

    DREAM_SHIFT = True
# end
