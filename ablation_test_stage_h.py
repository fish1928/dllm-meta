"""
Stage H: training-horizon ablation.

Train one router for each horizon from 3 through 15, while evaluating every
trained router with the same fixed evaluation horizon: 5.

The report is appended to:
    ablation_test_report.json

Edit only the configuration section below to match the winner from Stage E.
"""
import os
import json
import traceback

import torch

from attn_order_eval import ScoreOrderEval, summ
from router_llada import FactoryRouter, RouterTrainer

from ablation_test_common import (
    REPORT_PATH,
    build_features,
    build_loss,
    estimate_balanced_pos_weight,
    reset_stage,
    save_result,
)

REPORT_PATH = 'ablation_test_report_stage_h.json'

# ---------------------------------------------------------------------------
# Fixed experiment configuration
# Replace these values with the winning configuration from Stage E.
# ---------------------------------------------------------------------------

FOLDER_DATA = os.environ['FOLDER_DATA']
DEVICE = os.environ.get('DEVICE', 'cuda:0')
SIZE_BLOCK = int(os.environ['SIZE_BLOCK'])
NUM_LAYERS = 32
NUM_EPOCHS = 10

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HOLDOUT = 0.2
FILTER_RESULT = "all"
SEED = 233

STAGE = "stage_h"

FIXED_FEATURES = [
    "attn_last",
    "pos_delta",
    "mask_density",
]

FIXED_NORMALIZATION = "rank"
FIXED_LOSS = "plackett_luce"

FIXED_ROUTER = "mlp"
FIXED_ROUTER_KWARGS = {
    "dim_hidden": 64,
    "num_blocks_mlp": 2,
}

# Train separate models with h = 3, 4, ..., 15.
TRAIN_HORIZONS = range(3, 16)

# Evaluate every trained model with exactly the same horizon.
EVALUATION_HORIZON = 5


def evaluate_at_fixed_h(trainer, h=5):
    """
    Evaluate recall, PR-AUC, and NDCG at one fixed horizon.

    This avoids RouterTrainer.evaluate() using trainer.h for PR-AUC/NDCG,
    because trainer.h changes across the training-horizon experiments.
    """
    trainer.router.eval()

    recall_values = []
    pr_auc_values = []
    ndcg_values = []

    for x, order in trainer._iter_blocks(trainer.ids_eval):
        scores = trainer.router(x)
        evaluator = ScoreOrderEval(scores.cpu(), order.cpu())

        recall_values.append(evaluator.recall_at_h(h))
        pr_auc_values.append(evaluator.pr_auc(h))

        # Support either spelling used by different local evaluator versions.
        if hasattr(evaluator, "ndcg"):
            ndcg_values.append(evaluator.ndcg(h))
        elif hasattr(evaluator, "ndgc"):
            ndcg_values.append(evaluator.ndgc(h))
        elif hasattr(evaluator, "ndcg_at_h"):
            ndcg_values.append(evaluator.ndcg_at_h(h))
        else:
            raise AttributeError(
                "ScoreOrderEval does not provide ndcg(h), ndgc(h), "
                "or ndcg_at_h(h)."
            )

    report = {
        f"recall@{h}": summ(torch.cat(recall_values)),
        f"pr_auc@{h}": summ(torch.cat(pr_auc_values)),
        f"ndcg@{h}": summ(torch.cat(ndcg_values)),
        "n_blocks": len(recall_values),
        "router": trainer.router.describe(),
    }
    return report


reset_stage(STAGE, REPORT_PATH)

for train_horizon in TRAIN_HORIZONS:
    experiment_name = f"train_h_{train_horizon}__eval_h_{EVALUATION_HORIZON}"

    # Balanced BCE needs a different class weight for each training horizon.
    loss_pos_weight = None
    if FIXED_LOSS == "bce_balanced":
        loss_pos_weight = estimate_balanced_pos_weight(
            FOLDER_DATA,
            h=train_horizon,
            size_block=SIZE_BLOCK,
            holdout=HOLDOUT,
            filter_result=FILTER_RESULT,
            seed=SEED,
        )

    config = {
        "folder_data": FOLDER_DATA,
        "features": FIXED_FEATURES,
        "normalization": FIXED_NORMALIZATION,
        "loss": FIXED_LOSS,
        "loss_pos_weight": loss_pos_weight,
        "router": FIXED_ROUTER,
        "router_kwargs": FIXED_ROUTER_KWARGS,
        "h": train_horizon,
        "train_horizon": train_horizon,
        "evaluation_horizon": EVALUATION_HORIZON,
        "size_block": SIZE_BLOCK,
        "device": DEVICE,
        "num_layers": NUM_LAYERS,
        "num_epochs": NUM_EPOCHS,
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "holdout": HOLDOUT,
        "filter_result": FILTER_RESULT,
        "seed": SEED,
    }

    print(
        f"\n[{STAGE}] {experiment_name}: "
        f"train h={train_horizon}, evaluate h={EVALUATION_HORIZON}"
    )

    try:
        # Keep model initialization identical across horizon experiments.
        torch.manual_seed(SEED)

        features = build_features(
            FIXED_FEATURES,
            FOLDER_DATA,
            FIXED_NORMALIZATION,
            num_layers=NUM_LAYERS,
        )

        router = FactoryRouter.create(
            FIXED_ROUTER,
            **FIXED_ROUTER_KWARGS,
        ).register_features(*features)

        loss = build_loss(
            FIXED_LOSS,
            pos_weight=loss_pos_weight,
        )

        trainer = RouterTrainer(
            FOLDER_DATA,
            h=train_horizon,
            size_block=SIZE_BLOCK,
            device=DEVICE,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            holdout=HOLDOUT,
            filter_result=FILTER_RESULT,
            seed=SEED,
        )

        trainer.register_router(router).register_loss(loss)
        trainer.train(num_epochs=NUM_EPOCHS)

        with torch.no_grad():
            evaluation = evaluate_at_fixed_h(
                trainer,
                h=EVALUATION_HORIZON,
            )

        metrics = {
            "all": evaluation,
        }

        save_result(
            stage=STAGE,
            name=experiment_name,
            config=config,
            metrics=metrics,
            report_path=REPORT_PATH,
        )

        print(json.dumps(metrics, indent=2))

    except Exception:
        error = traceback.format_exc()

        save_result(
            stage=STAGE,
            name=experiment_name,
            config=config,
            error=error,
            report_path=REPORT_PATH,
        )

        print(error)
