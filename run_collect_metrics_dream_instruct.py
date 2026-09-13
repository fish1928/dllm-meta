#################################################
# Oracle collection, dream-instruct thread (full denoising, no cache).
#
# Dream-org/Dream-v0-Instruct-7B. The decoding loop is IDENTICAL to dream-base
# (growing window + dream shift; num_blocks=1 -> full canvas natively, matching
# the official Dream recipe and run_dream_instruct.py), so the collector is
# inherited from run_collect_metrics_dream_base. The thread differences are
# prompt-side and post-side, both handled by the shared scaffolding:
#   - chat template applied to every prompt (Qwen-style)
#   - answer checking cuts at eos AND <|im_end|> (collect_ids_stop)
#
# Usage:
#   python run_collect_metrics_dream_instruct.py \
#       --path_mockup benchmark_mockup/mockup_gsm8k_0shot_p10.csv \
#       --folder_output stats_oracle/dream_instruct_gsm8k_b1 --len_target 256 --num_blocks 1
#################################################

from modeling_dream_yukai import DreamModelLM
from collect_metrics_common import build_parser, main_collect
from run_collect_metrics_dream_base import OracleCollector as OracleCollectorDreamBase


class OracleCollector(OracleCollectorDreamBase):

    use_chat_template = True
# end


if __name__ == '__main__':
    main_collect(
        OracleCollector,
        DreamModelLM,
        build_parser(id_model='Dream-org/Dream-v0-Instruct-7B', id_mask=151666),
    )
# end
