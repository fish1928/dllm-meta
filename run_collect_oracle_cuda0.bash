THREAD=llada_instruct DEVICE=cuda:0 LIMIT=500 PERCENT=1 NUM_BLOCKS_LIST=32 bash run_collect_oracle.bash || echo "command failed, continuing"
THREAD=llada_instruct DEVICE=cuda:0 LIMIT=500 PERCENT=1 NUM_BLOCKS_LIST=16 bash run_collect_oracle.bash || echo "command failed, continuing"
THREAD=llada_instruct DEVICE=cuda:0 LIMIT=500 PERCENT=1 NUM_BLOCKS_LIST=8 bash run_collect_oracle.bash || echo "command failed, continuing"
