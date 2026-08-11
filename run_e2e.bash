LIMIT=1000 DEVICE='cuda:0' nohup bash run_e2e_gsm8k.sh > run_e2e_gsm8k.log &
LIMIT=200 DEVICE='cuda:1' nohup bash run_e2e_ifeval.sh  > run_e2e_ifeval.log &