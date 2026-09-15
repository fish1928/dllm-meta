THREAD=llada_base DEVICE=cuda:1 bash run_collect_oracle.bash || echo "command failed, continuing"
THREAD=dream_base DEVICE=cuda:1 bash run_collect_oracle.bash || echo "command failed, continuing"
THREAD=dream_instruct DEVICE=cuda:1 bash run_collect_oracle.bash || echo "command failed, continuing"
