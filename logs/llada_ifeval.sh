# k=4,h=5
accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --model test --batch_size 1 --num_fewshot 5 --device 'cuda:0' --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=256,num_blocks=1,num_unmask_per_step=1,id_mask=126336,step_refresh_remainder=5,select_only_in_h=True,runner=run_llada_semi_cached_mlp,h=5  || echo "command failed, continuing"

# k=8,h=5
accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --model test --batch_size 1 --num_fewshot 5 --device 'cuda:0' --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=256,num_blocks=1,num_unmask_per_step=1,id_mask=126336,step_refresh_remainder=9,select_only_in_h=True,runner=run_llada_semi_cached_mlp,h=5  || echo "command failed, continuing"

# k=12,h=5
accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --model test --batch_size 1 --num_fewshot 5 --device 'cuda:0' --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=256,num_blocks=1,num_unmask_per_step=1,id_mask=126336,step_refresh_remainder=13,select_only_in_h=True,runner=run_llada_semi_cached_mlp,h=5

# k=16,h=5
accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --model test --batch_size 1 --num_fewshot 5 --device 'cuda:0' --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=256,num_blocks=1,num_unmask_per_step=1,id_mask=126336,step_refresh_remainder=17,select_only_in_h=True,runner=run_llada_semi_cached_mlp,h=5  || echo "command failed, continuing"

# k=24,h=5
accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --model test --batch_size 1 --num_fewshot 5 --device 'cuda:0' --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=256,num_blocks=1,num_unmask_per_step=1,id_mask=126336,step_refresh_remainder=25,select_only_in_h=True,runner=run_llada_semi_cached_mlp,h=5  || echo "command failed, continuing"

# k=33,h=5
accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --model test --batch_size 1 --num_fewshot 5 --device 'cuda:0' --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=256,num_blocks=1,num_unmask_per_step=1,id_mask=126336,step_refresh_remainder=33,select_only_in_h=True,runner=run_llada_semi_cached_mlp,h=5  || echo "command failed, continuing"
 
# k=16,h=8
accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --model test --batch_size 1 --num_fewshot 5 --device 'cuda:0' --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=256,num_blocks=1,num_unmask_per_step=1,id_mask=126336,step_refresh_remainder=17,select_only_in_h=True,runner=run_llada_semi_cached_mlp,h=8 || echo "command failed, continuing"

# k=16,h=12
accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --model test --batch_size 1 --num_fewshot 5 --device 'cuda:0' --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=256,num_blocks=1,num_unmask_per_step=1,id_mask=126336,step_refresh_remainder=17,select_only_in_h=True,runner=run_llada_semi_cached_mlp,h=12 || echo "command failed, continuing"

# k=16,h=16
accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --model test --batch_size 1 --num_fewshot 5 --device 'cuda:0' --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=256,num_blocks=1,num_unmask_per_step=1,id_mask=126336,step_refresh_remainder=17,select_only_in_h=True,runner=run_llada_semi_cached_mlp,h=16 || echo "command failed, continuing"
