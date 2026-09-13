#################################################
# SUPERSEDED by the four thread collectors:
#   run_collect_metrics_llada_base.py / run_collect_metrics_llada_instruct.py /
#   run_collect_metrics_dream_base.py / run_collect_metrics_dream_instruct.py
# (shared scaffolding in collect_metrics_common.py).
#
# Kept as an alias delegating to the llada-base collector so old commands keep
# working. The old --use_chat_template flag is gone: chat-template collection
# is the llada-instruct thread now (run_collect_metrics_llada_instruct.py,
# which also decodes full-canvas as that checkpoint requires).
#################################################

from run_collect_metrics_llada_base import OracleCollector
from modeling_llada_yukai_06 import LLaDAModelLM
from collect_metrics_common import build_parser, main_collect


if __name__ == '__main__':
    main_collect(
        OracleCollector,
        LLaDAModelLM,
        build_parser(id_model='GSAI-ML/LLaDA-8B-Base', id_mask=126336),
    )
# end
