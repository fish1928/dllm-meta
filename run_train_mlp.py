from router_llada import NEG_INF,\
    Feature_attn_last, Feature_conf,\
    Feature_rank_normed, Feature_znormed_row, Feature_log_scaled,\
    Router_mlp, Loss_decay_within_h,\
    FactoryFeature, FactoryRouter, FactoryLoss,\
    RouterTrainer

folder_data = 'stats_gsm8k'


router = FactoryRouter\
    .create('mlp', dim_hidden=64)\
    .register_features(
        Feature_rank_normed(Feature_log_scaled(Feature_attn_last(folder_data))),
        Feature_rank_normed(Feature_conf(folder_data))
    )
# end


trainer = RouterTrainer(folder_data, h=5, device='cuda:0',size_block=128)
trainer.register_router(router).register_loss(Loss_decay_within_h())
trainer.train(num_epochs=1)

print(trainer.evaluate())