"""Stage A: sanity checks and simple baselines."""
import os

from ablation_test_common import (
    Feature_negated,
    REPORT_PATH,
    reset_stage,
    run_experiment,
)
from router_llada import (
    Feature_attn_last,
    Feature_conf,
    Feature_entropy,
    Feature_margin,
    Feature_pos_delta,
    FactoryLoss,
    FactoryRouter,
    RouterTrainer,
)


FOLDER_DATA = os.environ['FOLDER_DATA']
DEVICE = "cuda:0"
H = 5
SIZE_BLOCK = int(os.environ['SIZE_BLOCK'])
NUM_EPOCHS = 5
STAGE = "stage_a"

reset_stage(STAGE, REPORT_PATH)


def run_mockup(name, router, feature, feature_name):
    trainer = RouterTrainer(FOLDER_DATA, h=H, device=DEVICE, size_block=SIZE_BLOCK)
    trainer.register_router(router.register_features(feature))
    trainer.register_loss(FactoryLoss.create("uniform_within_h"))
    trainer.train(num_epochs=1)
    metrics = {"all": trainer.evaluate()}

    # Reuse the common report writer through a one-line trainable-style record.
    from ablation_test_common import save_result

    save_result(
        STAGE,
        name,
        {
            "router": router.__class__.__name__,
            "features": [feature_name],
            "h": H,
            "size_block": SIZE_BLOCK,
        },
        metrics=metrics,
    )
    print(name, metrics)


# 1. Random lower bound.
run_mockup(
    "random",
    FactoryRouter.create("mockup_random", seed=233),
    Feature_attn_last(FOLDER_DATA),
    "attn_last_dummy_input",
)

# 2. Raw single-feature baselines.
run_mockup(
    "raw_attn_last",
    FactoryRouter.create("mockup_raw"),
    Feature_attn_last(FOLDER_DATA),
    "attn_last",
)
run_mockup(
    "raw_conf",
    FactoryRouter.create("mockup_raw"),
    Feature_conf(FOLDER_DATA),
    "conf",
)
run_mockup(
    "raw_margin",
    FactoryRouter.create("mockup_raw"),
    Feature_margin(FOLDER_DATA),
    "margin",
)
run_mockup(
    "negative_raw_entropy",
    FactoryRouter.create("mockup_raw"),
    Feature_negated(Feature_entropy(FOLDER_DATA)),
    "-entropy",
)

# 3. Simple positional heuristic.
run_mockup(
    "nearest_right",
    FactoryRouter.create("mockup_nearest_right"),
    Feature_pos_delta(FOLDER_DATA),
    "pos_delta",
)

# 4. Trainable linear baseline using the core features.
run_experiment(
    stage=STAGE,
    name="linear_core_features",
    folder_data=FOLDER_DATA,
    feature_names=["attn_last", "conf", "margin", "pos_delta", "mask_density"],
    normalization="raw",
    loss_name="uniform_within_h",
    router_name="linear",
    h=H,
    size_block=SIZE_BLOCK,
    device=DEVICE,
    num_epochs=NUM_EPOCHS,
)
