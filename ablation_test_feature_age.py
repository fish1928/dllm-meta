"""
Evaluate confidence and margin as they become stale.

Age definition
--------------
For decoding row t and feature age a:

    aged_feature[t, p] = fresh_feature[t - a, p]

The first a rows of each block are skipped because no t-a feature exists.

The script reports three evaluations for every age:
1. raw_conf:         confidence used directly as the ranking score
2. raw_margin:       margin used directly as the ranking score
3. mlp_conf_margin:  one MLP trained with fresh rank-normalized confidence
                     and margin, then frozen and evaluated with aged inputs

This is a controlled uniform-age experiment. In real sparse decoding, each
position can have a different age because queried positions are refreshed.

Results are written to:
    ablation_test_report.json
under:
    stage_feature_age
"""

import json
import os
import traceback

import torch

from attn_order_eval import ScoreOrderEval, summ
from router_llada import (
    Feature_conf,
    Feature_margin,
    Feature_rank_normed,
    FactoryLoss,
    FactoryRouter,
    RouterTrainer,
    load_stat,
    sanitize,
)

from ablation_test_common import (
    REPORT_PATH,
    reset_stage,
    save_result,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FOLDER_DATA = os.environ['FOLDER_DATA']
DEVICE = os.environ.get("DEVICE", "cuda:0")
SIZE_BLOCK = int(os.environ['SIZE_BLOCK'])

TRAIN_HORIZON = 5
EVALUATION_HORIZON = 5

NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HOLDOUT = 0.2
FILTER_RESULT = "all"
SEED = 233

# Evaluate age 0, 1, ..., 15.
MAX_AGE = 15
AGES = range(0, MAX_AGE + 1)

STAGE = "stage_feature_age"

ROUTER_KWARGS = {
    "dim_hidden": 64,
    "num_blocks_mlp": 2,
}

LOSS_NAME = "plackett_luce"
NORMALIZATION = "rank"


# ---------------------------------------------------------------------------
# Aged feature definitions
# ---------------------------------------------------------------------------

class FeatureAgedMixin:
    """
    Shift a stored per-step feature backward by `age` rows.

    Rows earlier than `age` are filled from row 0, but those rows are excluded
    from evaluation.
    """

    def __init__(self, folder_data, age=0):
        super().__init__(folder_data)
        if age < 0:
            raise ValueError("age must be non-negative")
        self.age = int(age)

    def get_name(self):
        return f"age_{self.age}({super().get_name()})"

    def _apply_age(self, value):
        if self.age == 0:
            return value

        if self.age >= value.shape[0]:
            raise ValueError(
                f"age={self.age} must be smaller than T={value.shape[0]}"
            )

        aged = torch.empty_like(value)

        # These rows are skipped during evaluation.
        aged[:self.age] = value[0:1].expand_as(aged[:self.age])

        # At current row t, use the feature last observed at row t-age.
        aged[self.age:] = value[:-self.age]
        return aged


class Feature_aged_conf(FeatureAgedMixin, Feature_conf):
    def load_block(self, id_sample, pos_base, size_block):
        value = load_stat(
            self._folder_base(id_sample),
            "conf",
            pos_base,
            size_block,
        )
        return sanitize(self._apply_age(value)).unsqueeze(-1)


class Feature_aged_margin(FeatureAgedMixin, Feature_margin):
    def load_block(self, id_sample, pos_base, size_block):
        value = load_stat(
            self._folder_base(id_sample),
            "margin",
            pos_base,
            size_block,
        )
        return sanitize(self._apply_age(value)).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_ndcg(evaluator, h):
    """Support the method names used by different local evaluator versions."""
    if hasattr(evaluator, "ndcg"):
        return evaluator.ndcg(h)
    if hasattr(evaluator, "ndgc"):
        return evaluator.ndgc(h)
    if hasattr(evaluator, "ndcg_at_h"):
        return evaluator.ndcg_at_h(h)

    raise AttributeError(
        "ScoreOrderEval must provide ndcg(h), ndgc(h), or ndcg_at_h(h)"
    )


def summarize_evaluation(recall_values, pr_auc_values, ndcg_values, h, n_blocks):
    if not recall_values:
        raise RuntimeError("No valid blocks were available for evaluation")

    return {
        f"recall@{h}": summ(torch.cat(recall_values)),
        f"pr_auc@{h}": summ(torch.cat(pr_auc_values)),
        f"ndcg@{h}": summ(torch.cat(ndcg_values)),
        "n_blocks": n_blocks,
    }


@torch.no_grad()
def evaluate_raw_feature_by_age(trainer, feature_name, age, h=5):
    """
    Evaluate raw confidence or raw margin at one fixed age.

    Higher confidence and higher margin both receive higher ranking scores.
    """
    if feature_name == "conf":
        feature = Feature_aged_conf(FOLDER_DATA, age=age)
    elif feature_name == "margin":
        feature = Feature_aged_margin(FOLDER_DATA, age=age)
    else:
        raise ValueError(f"Unsupported raw feature: {feature_name}")

    recall_values = []
    pr_auc_values = []
    ndcg_values = []
    n_blocks = 0

    for id_sample, pos_base in trainer._list_blocks(trainer.ids_eval):
        folder_base = os.path.join(FOLDER_DATA, str(id_sample))

        # Shape: (T, L, 1) -> (T, L)
        scores = feature.load_block(
            id_sample,
            pos_base,
            SIZE_BLOCK,
        ).squeeze(-1)

        unmask = load_stat(
            folder_base,
            "unmask",
            pos_base,
            SIZE_BLOCK,
        )
        order = unmask.squeeze(-1).long() - pos_base

        # Re-index the trajectory so row 0 corresponds to original row `age`.
        scores = scores[age:]
        order = order[age:]

        if order.numel() <= h:
            continue

        evaluator = ScoreOrderEval(scores.cpu(), order.cpu())
        recall_values.append(evaluator.recall_at_h(h))
        pr_auc_values.append(evaluator.pr_auc(h))
        ndcg_values.append(evaluate_ndcg(evaluator, h))
        n_blocks += 1

    return summarize_evaluation(
        recall_values,
        pr_auc_values,
        ndcg_values,
        h,
        n_blocks,
    )


@torch.no_grad()
def evaluate_frozen_mlp_by_age(trainer, age, h=5):
    """
    Evaluate the fresh-trained MLP with confidence and margin both aged by
    the same number of steps.

    Rank normalization is recomputed over the current candidate set, matching
    deployment behavior.
    """
    trainer.router.eval()

    feature_conf = Feature_rank_normed(
        Feature_aged_conf(FOLDER_DATA, age=age)
    )
    feature_margin = Feature_rank_normed(
        Feature_aged_margin(FOLDER_DATA, age=age)
    )

    recall_values = []
    pr_auc_values = []
    ndcg_values = []
    n_blocks = 0

    for id_sample, pos_base in trainer._list_blocks(trainer.ids_eval):
        folder_base = os.path.join(FOLDER_DATA, str(id_sample))

        x = torch.cat(
            [
                feature_conf.load_block(
                    id_sample,
                    pos_base,
                    SIZE_BLOCK,
                ),
                feature_margin.load_block(
                    id_sample,
                    pos_base,
                    SIZE_BLOCK,
                ),
            ],
            dim=-1,
        ).to(DEVICE)

        unmask = load_stat(
            folder_base,
            "unmask",
            pos_base,
            SIZE_BLOCK,
        )
        order = unmask.squeeze(-1).long() - pos_base

        scores = trainer.router(x)

        # Skip rows that have no valid t-age observation.
        scores = scores[age:]
        order = order[age:]

        if order.numel() <= h:
            continue

        evaluator = ScoreOrderEval(scores.cpu(), order.cpu())
        recall_values.append(evaluator.recall_at_h(h))
        pr_auc_values.append(evaluator.pr_auc(h))
        ndcg_values.append(evaluate_ndcg(evaluator, h))
        n_blocks += 1

    report = summarize_evaluation(
        recall_values,
        pr_auc_values,
        ndcg_values,
        h,
        n_blocks,
    )
    report["router"] = trainer.router.describe()
    return report


# ---------------------------------------------------------------------------
# Train one fresh model
# ---------------------------------------------------------------------------

reset_stage(STAGE, REPORT_PATH)

torch.manual_seed(SEED)

fresh_router = FactoryRouter.create(
    "mlp",
    **ROUTER_KWARGS,
).register_features(
    Feature_rank_normed(Feature_conf(FOLDER_DATA)),
    Feature_rank_normed(Feature_margin(FOLDER_DATA)),
)

fresh_loss = FactoryLoss.create(LOSS_NAME)

trainer = RouterTrainer(
    FOLDER_DATA,
    h=TRAIN_HORIZON,
    size_block=SIZE_BLOCK,
    device=DEVICE,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    holdout=HOLDOUT,
    filter_result=FILTER_RESULT,
    seed=SEED,
)

trainer.register_router(fresh_router).register_loss(fresh_loss)
trainer.train(num_epochs=NUM_EPOCHS)


# ---------------------------------------------------------------------------
# Evaluate each age
# ---------------------------------------------------------------------------

for age in AGES:
    print(f"\n[{STAGE}] evaluating age={age}")

    common_config = {
        "folder_data": FOLDER_DATA,
        "age": age,
        "train_horizon": TRAIN_HORIZON,
        "evaluation_horizon": EVALUATION_HORIZON,
        "size_block": SIZE_BLOCK,
        "device": DEVICE,
        "num_epochs": NUM_EPOCHS,
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "holdout": HOLDOUT,
        "filter_result": FILTER_RESULT,
        "seed": SEED,
    }

    experiments = [
        (
            f"raw_conf__age_{age}",
            {
                **common_config,
                "features": ["conf"],
                "normalization": "raw",
                "loss": None,
                "router": "mockup_raw",
                "router_kwargs": {},
            },
            lambda: evaluate_raw_feature_by_age(
                trainer,
                feature_name="conf",
                age=age,
                h=EVALUATION_HORIZON,
            ),
        ),
        (
            f"raw_margin__age_{age}",
            {
                **common_config,
                "features": ["margin"],
                "normalization": "raw",
                "loss": None,
                "router": "mockup_raw",
                "router_kwargs": {},
            },
            lambda: evaluate_raw_feature_by_age(
                trainer,
                feature_name="margin",
                age=age,
                h=EVALUATION_HORIZON,
            ),
        ),
        (
            f"mlp_conf_margin__age_{age}",
            {
                **common_config,
                "features": ["conf", "margin"],
                "normalization": NORMALIZATION,
                "loss": LOSS_NAME,
                "router": "mlp",
                "router_kwargs": ROUTER_KWARGS,
                "trained_with_feature_age": 0,
                "evaluated_with_feature_age": age,
            },
            lambda: evaluate_frozen_mlp_by_age(
                trainer,
                age=age,
                h=EVALUATION_HORIZON,
            ),
        ),
    ]

    for experiment_name, config, evaluate_fn in experiments:
        try:
            metrics = {
                "all": evaluate_fn(),
            }

            save_result(
                stage=STAGE,
                name=experiment_name,
                config=config,
                metrics=metrics,
                report_path=REPORT_PATH,
            )

            print(f"\n{experiment_name}")
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
