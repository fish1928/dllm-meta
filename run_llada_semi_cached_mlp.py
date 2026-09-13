#################################################
# SUPERSEDED by run_llada_semi_mlp.py (llada-base thread of the four-runner
# split; instruct went to run_llada_instruct_mlp.py). Kept as an alias so old
# commands with runner=run_llada_semi_cached_mlp keep working.
# NOTE the truncate_at_eos flag is gone: base runs never needed it, and the
# instruct runners hard-enable EOS truncation instead.
#################################################

from run_llada_semi_mlp import RunModel
