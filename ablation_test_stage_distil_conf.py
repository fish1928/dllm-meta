"""
Stage: confidence distillation for the LLaDA query router.

The router uses only inference-time features:
    attn_last + pos_delta + mask_density

Fresh full-denoising confidence is used only as an auxiliary TRAINING target.
It is never registered as a router feature and is not needed during inference.

Total loss:
    L_total = L_plackett_luce + lambda_distill * L_conf_distill

L_conf_distill is a listwise cross-entropy between:
    teacher distribution: fresh-confidence ranking over current candidates
    student distribution: router scores over current candidates

Expected local modules:
    router_llada_v2.py
    ablation_test_common.py
    attn_order_eval.py
"""

import json
import os
import traceback
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

from attn_order_eval import ScoreOrderEval, summ
from router_llada import (
    Feature_attn_last,
    Feature_mask_density,
    Feature_pos_delta,
    Feature_rank_normed,
    FactoryRouter,
    Loss_plackett_luce,
    RouterTrainer,
    build_geometry,
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
DEVICE = os.environ.get('DEVICE', 'cuda:0')
SIZE_BLOCK = int(os.environ['SIZE_BLOCK'])

TRAIN_HORIZON = 5
EVALUATION_HORIZON = 5

NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HOLDOUT = 0.2
FILTER_RESULT = "all"
SEED = 233

STAGE = "stage_conf_distill"
CHECKPOINT_DIR = "checkpoints"

ROUTER_KWARGS = {
    "dim_hidden": 64,
    "num_blocks_mlp": 2,
}

# lambda=0 is the ordinary Plackett-Luce baseline.
DISTILL_WEIGHTS = [
    0.0,
    0.05,
    0.10,
    0.25,
    0.50,
    1.00,
]

# Lower temperature makes the teacher more concentrated on the
# highest-confidence candidates.
DISTILL_TEMPERATURE = 0.5

# Supported: "rank", "raw", "log".
# "rank" is recommended because the router is evaluated as a ranking model.
TEACHER_TRANSFORM = "rank"

NEG_INF = torch.finfo(torch.float32).min


# ---------------------------------------------------------------------------
# Router features
# ---------------------------------------------------------------------------

def build_router_features(folder_data: str):
    """Build the fixed inference-time feature combination."""
    return [
        Feature_rank_normed(Feature_attn_last(folder_data)),
        Feature_rank_normed(Feature_pos_delta(folder_data)),
        Feature_rank_normed(Feature_mask_density(folder_data)),
    ]


# ---------------------------------------------------------------------------
# Confidence teacher loss
# ---------------------------------------------------------------------------

def candidate_percentile_rank(
    values: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Tie-aware percentile rank among candidates.

    Args:
        values:         (T, L)
        candidate_mask: (T, L) bool

    Returns:
        (T, L), candidate values in [0, 1], non-candidates equal to zero.
    """
    if values.shape != candidate_mask.shape:
        raise ValueError(
            f"values shape {values.shape} does not match "
            f"candidate mask shape {candidate_mask.shape}"
        )

    output = torch.zeros_like(values, dtype=torch.float32)

    for row_index in range(values.shape[0]):
        mask_row = candidate_mask[row_index]
        row_values = values[row_index, mask_row]
        n = int(row_values.numel())

        if n == 0:
            continue
        if n == 1:
            output[row_index, mask_row] = 1.0
            continue

        smaller = (
            row_values[:, None] > row_values[None, :]
        ).sum(dim=1)
        equal = (
            row_values[:, None] == row_values[None, :]
        ).sum(dim=1)

        average_rank = (
            smaller.float() + 0.5 * (equal.float() - 1.0)
        )
        output[row_index, mask_row] = average_rank / float(n - 1)

    return output


class LossConfidenceDistill:
    """
    Listwise soft-target distillation from fresh confidence.

    Fresh confidence constructs the training target only. It never becomes
    part of the router input tensor.
    """

    def __init__(
        self,
        temperature: float = 0.10,
        teacher_transform: str = "rank",
        eps: float = 1e-8,
    ):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if teacher_transform not in {"rank", "raw", "log"}:
            raise ValueError(
                "teacher_transform must be 'rank', 'raw', or 'log'"
            )

        self.temperature = float(temperature)
        self.teacher_transform = teacher_transform
        self.eps = float(eps)

    def _teacher_scores(
        self,
        confidence: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        confidence = sanitize(confidence)

        if self.teacher_transform == "rank":
            return candidate_percentile_rank(confidence, candidate_mask)
        if self.teacher_transform == "log":
            return torch.log(confidence.clamp(min=self.eps))
        return confidence

    def __call__(
        self,
        router_scores: torch.Tensor,
        confidence: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        if router_scores.shape != confidence.shape:
            raise ValueError(
                f"router score shape {router_scores.shape} does not match "
                f"confidence shape {confidence.shape}"
            )

        teacher_scores = self._teacher_scores(
            confidence,
            candidate_mask,
        )

        teacher_prob = torch.softmax(
            (teacher_scores / self.temperature).masked_fill(
                ~candidate_mask,
                NEG_INF,
            ),
            dim=-1,
        )

        student_log_prob = F.log_softmax(
            router_scores.masked_fill(~candidate_mask, NEG_INF),
            dim=-1,
        )

        row_valid = candidate_mask.sum(dim=-1) > 1
        if not bool(row_valid.any()):
            raise RuntimeError("No row contains at least two candidates")

        loss_rows = -torch.where(
            candidate_mask,
            teacher_prob * student_log_prob,
            torch.zeros_like(student_log_prob),
        ).sum(dim=-1)

        return loss_rows[row_valid].mean()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def evaluate_ndcg(
    evaluator: ScoreOrderEval,
    horizon: int,
) -> torch.Tensor:
    """Support local evaluator naming differences."""
    if hasattr(evaluator, "ndcg"):
        return evaluator.ndcg(horizon)
    if hasattr(evaluator, "ndgc"):
        return evaluator.ndgc(horizon)
    if hasattr(evaluator, "ndcg_at_h"):
        return evaluator.ndcg_at_h(horizon)

    raise AttributeError(
        "ScoreOrderEval must provide ndcg(h), ndgc(h), or ndcg_at_h(h)"
    )


class ConfidenceDistillTrainer(RouterTrainer):
    """RouterTrainer extension that loads fresh confidence during training."""

    def __init__(
        self,
        *args,
        distill_weight: float,
        distill_temperature: float,
        teacher_transform: str,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if distill_weight < 0:
            raise ValueError("distill_weight must be non-negative")

        self.distill_weight = float(distill_weight)
        self.sequence_loss = Loss_plackett_luce()
        self.distill_loss = LossConfidenceDistill(
            temperature=distill_temperature,
            teacher_transform=teacher_transform,
        )
        self.last_train_report: Dict[str, float] = {}

    def _iter_blocks_with_confidence(
        self,
        ids_sample: Iterable[int],
    ):
        """
        Yield inference-time inputs plus fresh confidence as a teacher target.

        x:          (T, L, d)
        order:      (T,)
        confidence: (T, L)
        """
        for id_sample, pos_base in self._list_blocks(ids_sample):
            folder_base = os.path.join(
                self.folder_data,
                str(id_sample),
            )

            x = self.router.build_block_x(
                id_sample,
                pos_base,
                self.size_block,
            ).to(self.device)

            unmask = load_stat(
                folder_base,
                "unmask",
                pos_base,
                self.size_block,
            )
            order = (
                unmask.squeeze(-1).long() - pos_base
            ).to(self.device)

            confidence = sanitize(
                load_stat(
                    folder_base,
                    "conf",
                    pos_base,
                    self.size_block,
                )
            ).to(self.device)

            yield x, order, confidence

    def train(
        self,
        num_epochs: int = 10,
        log_every: int = 1,
    ):
        if self.router is None:
            raise RuntimeError("register_router() must be called first")
        if not self.router.trainable():
            raise RuntimeError(
                "Confidence distillation requires a trainable router"
            )

        torch.manual_seed(self.seed)

        optimizer = torch.optim.AdamW(
            self.router.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.router.train()

        for epoch in range(num_epochs):
            total_values: List[float] = []
            sequence_values: List[float] = []
            distill_values: List[float] = []

            for x, order, confidence in self._iter_blocks_with_confidence(
                self.ids_train
            ):
                gap, candidate_mask = build_geometry(
                    order.cpu(),
                    self.size_block,
                )
                gap = gap.to(self.device)
                candidate_mask = candidate_mask.to(self.device)

                optimizer.zero_grad(set_to_none=True)

                scores = self.router(x)

                loss_sequence = self.sequence_loss(
                    scores,
                    gap,
                    candidate_mask,
                    self.h,
                )

                if self.distill_weight > 0:
                    loss_distill = self.distill_loss(
                        scores,
                        confidence,
                        candidate_mask,
                    )
                else:
                    loss_distill = scores.new_zeros(())

                loss_total = (
                    loss_sequence
                    + self.distill_weight * loss_distill
                )

                loss_total.backward()
                optimizer.step()

                total_values.append(float(loss_total.item()))
                sequence_values.append(float(loss_sequence.item()))
                distill_values.append(float(loss_distill.item()))

            if not total_values:
                raise RuntimeError("No training blocks were found")

            self.last_train_report = {
                "epoch": epoch,
                "loss_total": sum(total_values) / len(total_values),
                "loss_plackett_luce": (
                    sum(sequence_values) / len(sequence_values)
                ),
                "loss_conf_distill": (
                    sum(distill_values) / len(distill_values)
                ),
            }

            if epoch % log_every == 0:
                print(
                    f"epoch {epoch}: "
                    f"total={self.last_train_report['loss_total']:.6f}, "
                    f"pl={self.last_train_report['loss_plackett_luce']:.6f}, "
                    f"distill={self.last_train_report['loss_conf_distill']:.6f}"
                )

        return self

    @torch.no_grad()
    def evaluate_fixed_h(
        self,
        horizon: int = 5,
        ids_sample: Optional[List[int]] = None,
    ) -> Dict:
        """
        Evaluate using only registered router features.

        Fresh confidence is not loaded or used here.
        """
        if self.router is None:
            raise RuntimeError("register_router() must be called first")

        ids_sample = (
            ids_sample if ids_sample is not None else self.ids_eval
        )

        self.router.eval()

        recall_values = []
        pr_auc_values = []
        ndcg_values = []

        for x, order in self._iter_blocks(ids_sample):
            scores = self.router(x)
            evaluator = ScoreOrderEval(scores.cpu(), order.cpu())

            recall_values.append(evaluator.recall_at_h(horizon))
            pr_auc_values.append(evaluator.pr_auc(horizon))
            ndcg_values.append(evaluate_ndcg(evaluator, horizon))

        if not recall_values:
            raise RuntimeError("No evaluation blocks were found")

        return {
            f"recall@{horizon}": summ(torch.cat(recall_values)),
            f"pr_auc@{horizon}": summ(torch.cat(pr_auc_values)),
            f"ndcg@{horizon}": summ(torch.cat(ndcg_values)),
            "n_blocks": len(recall_values),
            "router": self.router.describe(),
        }

    @torch.no_grad()
    def evaluate_teacher_alignment(
        self,
        ids_sample: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Diagnostic: how often max fresh confidence equals the next oracle unmask.
        """
        ids_sample = (
            ids_sample if ids_sample is not None else self.ids_eval
        )

        matches = 0
        valid_rows = 0

        for _, order, confidence in self._iter_blocks_with_confidence(
            ids_sample
        ):
            gap, candidate_mask = build_geometry(
                order.cpu(),
                self.size_block,
            )
            gap = gap.to(self.device)
            candidate_mask = candidate_mask.to(self.device)

            row_valid = (
                (gap == 1).any(dim=-1)
                & candidate_mask.any(dim=-1)
            )
            if not bool(row_valid.any()):
                continue

            teacher_choice = confidence.masked_fill(
                ~candidate_mask,
                NEG_INF,
            ).argmax(dim=-1)

            oracle_choice = (gap == 1).float().argmax(dim=-1)

            matches += int(
                (
                    teacher_choice[row_valid]
                    == oracle_choice[row_valid]
                ).sum().item()
            )
            valid_rows += int(row_valid.sum().item())

        return {
            "teacher_top1_match_rate": matches / max(valid_rows, 1),
            "teacher_top1_matches": matches,
            "teacher_top1_rows": valid_rows,
        }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(distill_weight: float):
    experiment_name = (
        "plackett_luce_only"
        if distill_weight == 0
        else f"pl_plus_conf_distill_lambda_{distill_weight:g}"
    )

    config = {
        "folder_data": FOLDER_DATA,
        "features": [
            "rank(attn_last)",
            "rank(pos_delta)",
            "rank(mask_density)",
        ],
        "fresh_confidence_is_router_input": False,
        "fresh_confidence_training_target": distill_weight > 0,
        "loss": {
            "sequence": "plackett_luce",
            "distillation": "candidate_listwise_cross_entropy",
            "total": (
                "plackett_luce + lambda_distill * confidence_distillation"
            ),
            "lambda_distill": distill_weight,
            "teacher_transform": TEACHER_TRANSFORM,
            "teacher_temperature": DISTILL_TEMPERATURE,
        },
        "router": "mlp",
        "router_kwargs": ROUTER_KWARGS,
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

    print(f"\n[{STAGE}] {experiment_name}")

    try:
        torch.manual_seed(SEED)

        router = FactoryRouter.create(
            "mlp",
            **ROUTER_KWARGS,
        ).register_features(*build_router_features(FOLDER_DATA))

        trainer = ConfidenceDistillTrainer(
            FOLDER_DATA,
            h=TRAIN_HORIZON,
            size_block=SIZE_BLOCK,
            device=DEVICE,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            holdout=HOLDOUT,
            filter_result=FILTER_RESULT,
            seed=SEED,
            distill_weight=distill_weight,
            distill_temperature=DISTILL_TEMPERATURE,
            teacher_transform=TEACHER_TRANSFORM,
        )

        trainer.register_router(router)
        trainer.train(num_epochs=NUM_EPOCHS)

        metrics = {
            "all": trainer.evaluate_fixed_h(
                horizon=EVALUATION_HORIZON
            ),
            "teacher_diagnostic": trainer.evaluate_teacher_alignment(),
            "final_train_loss": trainer.last_train_report,
        }

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"{STAGE}__{experiment_name}.pt",
        )
        trainer.router.save_checkpoint(checkpoint_path)
        metrics["checkpoint"] = checkpoint_path

        save_result(
            stage=STAGE,
            name=experiment_name,
            config=config,
            metrics=metrics,
            report_path=REPORT_PATH,
        )

        print(json.dumps(metrics, indent=2))
        return trainer, metrics

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
        return None, None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def verify_distillation_loss():
    """Aligned router scores should have lower loss than reversed scores."""
    confidence = torch.tensor(
        [[0.90, 0.70, 0.20, 0.00]],
        dtype=torch.float32,
    )
    candidate_mask = torch.tensor(
        [[True, True, True, False]]
    )

    scores_aligned = torch.tensor(
        [[3.0, 2.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    scores_reversed = torch.tensor(
        [[0.0, 2.0, 3.0, 0.0]],
        dtype=torch.float32,
    )

    loss_fn = LossConfidenceDistill(
        temperature=DISTILL_TEMPERATURE,
        teacher_transform=TEACHER_TRANSFORM,
    )

    aligned = float(
        loss_fn(
            scores_aligned,
            confidence,
            candidate_mask,
        ).item()
    )
    reversed_value = float(
        loss_fn(
            scores_reversed,
            confidence,
            candidate_mask,
        ).item()
    )

    if not aligned < reversed_value:
        raise AssertionError(
            "Distillation self-test failed: "
            f"aligned={aligned}, reversed={reversed_value}"
        )

    print(
        "Distillation self-test passed:",
        {
            "aligned_loss": round(aligned, 6),
            "reversed_loss": round(reversed_value, 6),
        },
    )


if __name__ == "__main__":
    verify_distillation_loss()
    reset_stage(STAGE, REPORT_PATH)

    for weight in DISTILL_WEIGHTS:
        run_experiment(distill_weight=weight)
