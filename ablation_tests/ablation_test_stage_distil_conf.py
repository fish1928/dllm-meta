"""
Stage: confidence distillation for the LLaDA query router.

The router uses only inference-time features:
    attn_last + pos_delta + mask_density

Fresh full-denoising confidence is used only as an auxiliary TRAINING target.
It is never registered as a router feature and is not needed during inference.

Training schedule (per experiment):
    stage 1 (optional): distill-only epochs   L = L_conf_distill
    stage 2 (optional): joint epochs          L = L_plackett_luce + lambda * L_conf_distill

The distill-only arm measures the DISTILLATION CEILING: how much of the fresh-
confidence ranking is expressible from inference-time features at all.

L_conf_distill is a listwise cross-entropy between:
    teacher distribution: softmax(teacher_transform(conf) / T) over candidates
                          (transform 'log' => teacher proportional to conf^(1/T))
    student distribution: router scores over current candidates
The reported 'kl' component is CE minus teacher entropy (floor 0).

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
    percentile_rank_masked,
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

# With teacher_transform="log":  teacher_prob ∝ conf^(1/T)
#   T = 1.0  -> teacher IS the (normalized) confidence distribution
#   T = 0.5  -> conf squared (sharper)
# NOTE: with "rank" the scores live in [0, 1], so a meaningful temperature must
# be on the order of the rank spacing (~1/n ≈ 0.02) -- T=0.5 over ranks makes
# the teacher nearly uniform, which turns the distill term into an
# anti-sharpening regularizer (the loss then RISES as Plackett-Luce sharpens
# the student). "log" is scale-correct by construction.
DISTILL_TEMPERATURE = 0.5

# Supported: "rank", "raw", "log".
TEACHER_TRANSFORM = "log"

# Each experiment: optional distill-only stage, then optional joint stage.
# The optimizer (and Adam state) restarts at the stage boundary.
EXPERIMENTS = [
    {"name": "pl_only",              "distill_weight": 0.0,  "epochs_distill_only": 0,          "epochs_joint": NUM_EPOCHS},
    {"name": "distill_only_ceiling", "distill_weight": 1.0,  "epochs_distill_only": NUM_EPOCHS, "epochs_joint": 0},
    {"name": "joint_lambda_0.1",     "distill_weight": 0.1,  "epochs_distill_only": 0,          "epochs_joint": NUM_EPOCHS},
    {"name": "joint_lambda_0.5",     "distill_weight": 0.5,  "epochs_distill_only": 0,          "epochs_joint": NUM_EPOCHS},
    {"name": "two_stage_lambda_0.25", "distill_weight": 0.25, "epochs_distill_only": NUM_EPOCHS // 2, "epochs_joint": NUM_EPOCHS},
]

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
            return percentile_rank_masked(confidence, candidate_mask)
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

        # KL = CE - H(teacher): same gradient (teacher fixed), but floor 0, so
        # "distance from the teacher" is readable at a glance in the logs
        teacher_log_prob = torch.where(
            teacher_prob > 0,
            teacher_prob.clamp(min=1e-12).log(),
            torch.zeros_like(teacher_prob),
        )
        entropy_rows = -(teacher_prob * teacher_log_prob).sum(dim=-1)

        loss = loss_rows[row_valid].mean()
        self.last_components = {
            "ce": float(loss.detach().item()),
            "kl": float((loss_rows - entropy_rows)[row_valid].mean().detach().item()),
            "teacher_entropy": float(entropy_rows[row_valid].mean().detach().item()),
        }

        return loss


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
        num_epochs_distill_only: int = 0,
        num_epochs_joint: int = 10,
        log_every: int = 1,
    ):
        """
        Two-stage schedule:
            stage 'distill': L = L_conf_distill                    (teacher only)
            stage 'joint':   L = L_pl + distill_weight * L_distill

        Either stage may be empty. Call once per stage from outside if an
        evaluation between stages is wanted (the optimizer restarts per call).
        """
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

        num_epochs = num_epochs_distill_only + num_epochs_joint

        for epoch in range(num_epochs):
            stage_distill_only = epoch < num_epochs_distill_only
            name_stage = "distill" if stage_distill_only else "joint"

            total_values: List[float] = []
            sequence_values: List[float] = []
            distill_values: List[float] = []
            kl_values: List[float] = []

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

                need_distill = stage_distill_only or self.distill_weight > 0
                if need_distill:
                    loss_distill = self.distill_loss(
                        scores,
                        confidence,
                        candidate_mask,
                    )
                    kl_values.append(
                        self.distill_loss.last_components["kl"]
                    )
                else:
                    loss_distill = scores.new_zeros(())

                if stage_distill_only:
                    loss_total = loss_distill
                else:
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
                "stage": name_stage,
                "loss_total": sum(total_values) / len(total_values),
                "loss_plackett_luce": (
                    sum(sequence_values) / len(sequence_values)
                ),
                "loss_conf_distill": (
                    sum(distill_values) / len(distill_values)
                ),
                "distill_kl": (
                    sum(kl_values) / len(kl_values)
                    if kl_values else None
                ),
            }

            if epoch % log_every == 0:
                kl_report = self.last_train_report["distill_kl"]
                print(
                    f"epoch {epoch} [{name_stage}]: "
                    f"total={self.last_train_report['loss_total']:.6f}, "
                    f"pl={self.last_train_report['loss_plackett_luce']:.6f}, "
                    f"distill_ce={self.last_train_report['loss_conf_distill']:.6f}, "
                    f"distill_kl={kl_report if kl_report is None else round(kl_report, 6)}"
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

def run_experiment(experiment: Dict):
    experiment_name = experiment["name"]
    distill_weight = float(experiment["distill_weight"])
    epochs_distill_only = int(experiment["epochs_distill_only"])
    epochs_joint = int(experiment["epochs_joint"])

    config = {
        "folder_data": FOLDER_DATA,
        "features": [
            "rank(attn_last)",
            "rank(pos_delta)",
            "rank(mask_density)",
        ],
        "fresh_confidence_is_router_input": False,
        "fresh_confidence_training_target": (
            distill_weight > 0 or epochs_distill_only > 0
        ),
        "loss": {
            "sequence": "plackett_luce",
            "distillation": "candidate_listwise_cross_entropy",
            "schedule": {
                "epochs_distill_only": epochs_distill_only,
                "epochs_joint": epochs_joint,
            },
            "total_joint_stage": (
                "plackett_luce + lambda_distill * confidence_distillation"
            ),
            "lambda_distill": distill_weight,
            "teacher_transform": TEACHER_TRANSFORM,
            "teacher_temperature": DISTILL_TEMPERATURE,
            "teacher_semantics": "prob ∝ conf^(1/T) for transform='log'",
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

        metrics = {}

        if epochs_distill_only > 0:
            trainer.train(
                num_epochs_distill_only=epochs_distill_only,
                num_epochs_joint=0,
            )
            # distillation ceiling: how much of the teacher's ranking the
            # inference-time features can express, before any label training
            metrics["after_distill_stage"] = trainer.evaluate_fixed_h(
                horizon=EVALUATION_HORIZON
            )
            trainer.router.train()
        # end

        if epochs_joint > 0:
            trainer.train(
                num_epochs_distill_only=0,
                num_epochs_joint=epochs_joint,
            )
        # end

        metrics["all"] = trainer.evaluate_fixed_h(
            horizon=EVALUATION_HORIZON
        )
        metrics["teacher_diagnostic"] = trainer.evaluate_teacher_alignment()
        metrics["final_train_loss"] = trainer.last_train_report

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


def verify_teacher_sharpness():
    """
    The teacher must be meaningfully sharper than uniform; a near-uniform
    teacher turns the distill term into an anti-sharpening regularizer
    (this is exactly what T=0.5 over [0,1] rank scores produced).
    """
    n = 50
    confidence = torch.rand(1, n)
    candidate_mask = torch.ones(1, n, dtype=torch.bool)

    loss_fn = LossConfidenceDistill(
        temperature=DISTILL_TEMPERATURE,
        teacher_transform=TEACHER_TRANSFORM,
    )
    teacher_scores = loss_fn._teacher_scores(confidence, candidate_mask)
    teacher_prob = torch.softmax(
        (teacher_scores / loss_fn.temperature).masked_fill(
            ~candidate_mask, NEG_INF
        ),
        dim=-1,
    )

    entropy = -(teacher_prob * teacher_prob.clamp(min=1e-12).log()).sum()
    entropy_uniform = torch.log(torch.tensor(float(n)))
    ratio = float(entropy / entropy_uniform)

    if ratio > 0.97:
        raise AssertionError(
            "Teacher is nearly uniform "
            f"(entropy ratio {ratio:.4f}); lower the temperature or use "
            "teacher_transform='log'"
        )

    print(
        "Teacher sharpness self-test passed:",
        {"entropy_ratio_vs_uniform": round(ratio, 4)},
    )


if __name__ == "__main__":
    verify_distillation_loss()
    verify_teacher_sharpness()
    reset_stage(STAGE, REPORT_PATH)

    for experiment in EXPERIMENTS:
        run_experiment(experiment)
