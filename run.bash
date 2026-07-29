# k=4,h=5
accelerate launch --num_processes=1 run_benchmark_main.py --tasks gsm8k --limit 1 --model test --batch_size 1 --num_fewshot 5 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=128,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:0',step_refresh_remainder=5,select_only_in_h=True,runner=run_model_semi_cached_mlp,h=5  || echo "command failed, continuing"

# dream (mask id 151666, dream runner + dream modeling are auto-selected via id_model / runner)
accelerate launch --num_processes=1 run_benchmark_main.py --tasks gsm8k --model test --batch_size 1 --num_fewshot 5 --device 'cuda:0' --model_args id_model='Dream-org/Dream-v0-Base-7B',size_batch=1,len_target=128,num_blocks=1,num_unmask_per_step=1,id_mask=151666,step_refresh_remainder=5,select_only_in_h=True,runner=run_dream_semi_cached_mlp,h=5  || echo "command failed, continuing"
