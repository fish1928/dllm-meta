"""
Stage I v2: random-age confidence training with slow age discount.

Key changes from v1
-------------------
1. A new position-specific age matrix is sampled for every training epoch.
2. Confidence is retrieved from the corresponding earlier denoising step:

       aged_conf[t, p] = fresh_conf[t - age[t, p], p]

3. A configurable slow exponential discount is applied after candidate-rank
   normalization:

       discounted_conf[t, p]
           = ranked_aged_conf[t, p] * exp(-AGE_DECAY_RATE * age[t, p])

4. The script compares:
   - attention + geometry baseline;
   - aged confidence with age metadata and no discount;
   - aged confidence with age metadata and slow discount.

Age sampling
------------
For zero-based row t:

    age[t, p] ~ UniformInteger(0, min(MAX_CONF_AGE, t))

Therefore, at human-readable step 8 (zero-based row 7), valid ages are 0..7.

The random age matrix:
- is position-specific;
- is re-randomized once per training epoch;
- is deterministic for a fixed seed and epoch;
- is shared by the confidence feature and age metadata;
- uses the same evaluation draws across all comparable experiments.

This is random-age augmentation. It does not simulate a fully coherent refresh
history in which each candidate's age increments and resets after querying.

Results are appended under:
    stage_random_age_v2

in:
    ablation_test_report.json
"""

import json
import math
import os
import traceback
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from attn_order_eval import ScoreOrderEval, summ
from router_llada import (
    FeatureBase,
    FactoryLoss,
    FactoryRouter,
    RouterTrainer,
    build_geometry,
    load_stat,
    sanitize,
)

from ablation_test_common import (
    REPORT_PATH,
    build_features,
    reset_stage,
    save_result,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FOLDER_DATA = os.environ['FOLDER_DATA']
DEVICE = os.environ.get('DEVICE', 'cuda:0')
SIZE_BLOCK = int(os.environ['SIZE_BLOCK'])
NUM_LAYERS = 32

TRAIN_HORIZON = 5
EVALUATION_HORIZON = 5

NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HOLDOUT = 0.2
FILTER_RESULT = "all"
SEED = 233

FULL_REFRESH_INTERVAL = 16

# Inclusive largest sampled age. With this setting, ages are 0..16 once
# enough trajectory rows are available.
MAX_CONF_AGE = 16

# Slow exponential decay:
#
#   weight(age) = exp(-AGE_DECAY_RATE * age)
#
# With 0.10:
#   age 0  -> 1.000
#   age 1  -> 0.905
#   age 5  -> 0.607
#   age 10 -> 0.368
#   age 16 -> 0.202
#
# Reduce this value for even slower decay.
AGE_DECAY_RATE = 0.10

# Average metrics over multiple deterministic random-age evaluation draws.
NUM_EVAL_AGE_DRAWS = 5

STAGE = "stage_random_age_v2"

BASE_FEATURES = [
    "attn_last",
    "pos_delta",
    "mask_density",
]
BASE_NORMALIZATION = "rank"

LOSS_NAME = "plackett_luce"
ROUTER_NAME = "mlp"
ROUTER_KWARGS = {
    "dim_hidden": 64,
    "num_blocks_mlp": 2,
}

CHECKPOINT_DIR = "checkpoints"


# Both aged-confidence models receive identical age metadata. Their only
# difference is whether the confidence channel is manually discounted.
EXPERIMENTS = [
    {
        "name": "base_attn_geo",
        "use_random_aged_conf": False,
        "use_age_metadata": False,
        "confidence_decay_rate": None,
    },
    {
        "name": "random_aged_conf_with_age_no_discount",
        "use_random_aged_conf": True,
        "use_age_metadata": True,
        "confidence_decay_rate": 0.0,
    },
    {
        "name": "random_aged_conf_with_age_slow_decay",
        "use_random_aged_conf": True,
        "use_age_metadata": True,
        "confidence_decay_rate": AGE_DECAY_RATE,
    },
]


# ---------------------------------------------------------------------------
# Random age generation
# ---------------------------------------------------------------------------

class RandomAgeProvider:
    """
    Generate one age matrix shared by all age-aware features in one block.

    A context is identified by:
        mode:    "train" or "eval"
        draw_id: training epoch or evaluation draw

    Calling set_context() clears the block cache. Within one context, repeated
    feature loads for the same block receive exactly the same age matrix.
    """

    def __init__(self, max_age: int, seed: int):
        if max_age < 0:
            raise ValueError("max_age must be non-negative")

        self.max_age = int(max_age)
        self.seed = int(seed)

        self.mode = "train"
        self.draw_id = 0
        self._cache: Dict[Tuple, torch.Tensor] = {}

    def set_context(self, mode: str, draw_id: int) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(f"Unsupported age-sampling mode: {mode}")

        self.mode = mode
        self.draw_id = int(draw_id)

        # This is what guarantees that every epoch/draw creates new samples.
        self._cache.clear()

    def _block_seed(self, id_sample: int, pos_base: int) -> int:
        """
        Stable seed independent of Python's randomized hash implementation.
        """
        mode_code = 17 if self.mode == "train" else 43

        value = (
            self.seed
            + mode_code * 1_000_003
            + self.draw_id * 10_000_019
            + int(id_sample) * 1_000_033
            + int(pos_base) * 97_409
        )

        return value % (2**63 - 1)

    def get(
        self,
        id_sample: int,
        pos_base: int,
        num_steps: int,
        num_positions: int,
    ) -> torch.Tensor:
        """
        Return a position-specific age matrix with shape (T, L).

        At row t:
            age[t, p] is sampled uniformly from
            {0, ..., min(max_age, t)}.
        """
        key = (
            self.mode,
            self.draw_id,
            int(id_sample),
            int(pos_base),
            int(num_steps),
            int(num_positions),
        )

        cached = self._cache.get(key)
        if cached is not None:
            return cached

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._block_seed(id_sample, pos_base))

        max_age_per_row = torch.arange(num_steps, dtype=torch.long)
        max_age_per_row = max_age_per_row.clamp(max=self.max_age)

        uniform = torch.rand(
            num_steps,
            num_positions,
            generator=generator,
        )

        ages = torch.floor(
            uniform * (max_age_per_row[:, None].float() + 1.0)
        ).long()

        row = torch.arange(num_steps, dtype=torch.long)[:, None]

        assert bool((ages >= 0).all())
        assert bool((ages <= self.max_age).all())
        assert bool((ages <= row).all())

        self._cache[key] = ages
        return ages


def gather_by_age(
    fresh_value: torch.Tensor,
    ages: torch.Tensor,
) -> torch.Tensor:
    """
    Gather fresh_value[t - age[t,p], p] while keeping position p fixed.

    Args:
        fresh_value: shape (T, L)
        ages:       shape (T, L), with 0 <= ages[t,p] <= t

    Returns:
        aged_value: shape (T, L)
    """
    if fresh_value.ndim != 2:
        raise ValueError(
            f"fresh_value must have shape (T, L), got {fresh_value.shape}"
        )

    if ages.shape != fresh_value.shape:
        raise ValueError(
            f"ages shape {ages.shape} does not match "
            f"value shape {fresh_value.shape}"
        )

    num_steps = fresh_value.shape[0]
    current_row = torch.arange(num_steps, dtype=torch.long)[:, None]
    source_row = current_row - ages

    if bool((source_row < 0).any()):
        raise ValueError("A sampled age requested a row before row 0")

    return fresh_value.gather(dim=0, index=source_row)


def exponential_age_weight(
    ages: torch.Tensor,
    decay_rate: float,
) -> torch.Tensor:
    """
    Return exp(-decay_rate * age).

    decay_rate = 0 gives no manual discount.
    """
    if decay_rate < 0:
        raise ValueError("decay_rate must be non-negative")

    return torch.exp(-float(decay_rate) * ages.float())


# ---------------------------------------------------------------------------
# Candidate-wise normalization
# ---------------------------------------------------------------------------

def candidate_percentile_rank(
    values: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Tie-aware percentile rank among current candidates only.

    Non-candidate outputs are zero and are ignored by the ranking loss.
    """
    if values.shape != candidate_mask.shape:
        raise ValueError(
            "values and candidate_mask must have the same shape"
        )

    ranked = torch.zeros_like(values, dtype=torch.float32)

    for row_index in range(values.shape[0]):
        mask_row = candidate_mask[row_index]
        candidate_values = values[row_index, mask_row]
        num_candidates = int(candidate_values.numel())

        if num_candidates <= 1:
            continue

        smaller = (
            candidate_values[:, None]
            > candidate_values[None, :]
        ).sum(dim=1)

        equal = (
            candidate_values[:, None]
            == candidate_values[None, :]
        ).sum(dim=1)

        average_rank = (
            smaller.float()
            + 0.5 * (equal.float() - 1.0)
        )

        ranked[row_index, mask_row] = (
            average_rank / float(num_candidates - 1)
        )

    return ranked


# ---------------------------------------------------------------------------
# Age-aware features
# ---------------------------------------------------------------------------

class Feature_random_aged_conf_rank_discounted(FeatureBase):
    """
    Randomly aged, candidate-rank-normalized confidence with optional discount.

    Processing order:
        fresh confidence
        -> lookup at previous step using sampled age
        -> rank normalization among current candidates
        -> multiply by exp(-decay_rate * age)

    The feature is not rank-normalized again after discounting.
    """

    def __init__(
        self,
        folder_data: str,
        age_provider: RandomAgeProvider,
        decay_rate: float,
    ):
        super().__init__(folder_data)

        if decay_rate < 0:
            raise ValueError("decay_rate must be non-negative")

        self.age_provider = age_provider
        self.decay_rate = float(decay_rate)

    def dim(self):
        return 1

    def get_name(self):
        return (
            "random_aged_rank_discounted("
            f"conf,lambda={self.decay_rate:g})"
        )

    def load_block(self, id_sample, pos_base, size_block):
        fresh_conf = load_stat(
            self._folder_base(id_sample),
            "conf",
            pos_base,
            size_block,
        )
        fresh_conf = sanitize(fresh_conf)

        if fresh_conf.ndim != 2:
            raise ValueError(
                f"Expected confidence shape (T, L), "
                f"got {fresh_conf.shape}"
            )

        num_steps, num_positions = fresh_conf.shape

        ages = self.age_provider.get(
            id_sample=id_sample,
            pos_base=pos_base,
            num_steps=num_steps,
            num_positions=num_positions,
        )

        aged_conf = gather_by_age(
            fresh_conf,
            ages,
        )

        order = self._order_local(
            id_sample,
            pos_base,
            size_block,
        )
        _, candidate_mask = build_geometry(
            order,
            size_block,
        )

        ranked_conf = candidate_percentile_rank(
            aged_conf,
            candidate_mask,
        )

        age_weight = exponential_age_weight(
            ages,
            self.decay_rate,
        )

        discounted_conf = ranked_conf * age_weight

        return discounted_conf.unsqueeze(-1)


class Feature_random_age_metadata(FeatureBase):
    """
    Age channels:
        0. age / MAX_CONF_AGE
        1. age == 0
        2. age == 1

    The explicit age channels allow the MLP to learn a correction beyond the
    manually selected exponential discount.
    """

    def __init__(
        self,
        folder_data: str,
        age_provider: RandomAgeProvider,
    ):
        super().__init__(folder_data)
        self.age_provider = age_provider

    def dim(self):
        return 3

    def get_name(self):
        return "random_conf_age(age_norm,is_age_0,is_age_1)"

    def load_block(self, id_sample, pos_base, size_block):
        conf = load_stat(
            self._folder_base(id_sample),
            "conf",
            pos_base,
            size_block,
        )

        if conf.ndim != 2:
            raise ValueError(
                f"Expected confidence shape (T, L), got {conf.shape}"
            )

        num_steps, num_positions = conf.shape

        ages = self.age_provider.get(
            id_sample=id_sample,
            pos_base=pos_base,
            num_steps=num_steps,
            num_positions=num_positions,
        )

        denominator = max(self.age_provider.max_age, 1)

        age_norm = ages.float() / float(denominator)
        is_age_zero = (ages == 0).float()
        is_age_one = (ages == 1).float()

        return torch.stack(
            [
                age_norm,
                is_age_zero,
                is_age_one,
            ],
            dim=-1,
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def evaluate_ndcg(evaluator, h):
    if hasattr(evaluator, "ndcg"):
        return evaluator.ndcg(h)

    if hasattr(evaluator, "ndgc"):
        return evaluator.ndgc(h)

    if hasattr(evaluator, "ndcg_at_h"):
        return evaluator.ndcg_at_h(h)

    raise AttributeError(
        "ScoreOrderEval must provide ndcg(h), "
        "ndgc(h), or ndcg_at_h(h)"
    )


class RandomAgeRouterTrainer(RouterTrainer):

    def __init__(
        self,
        *args,
        age_provider: RandomAgeProvider,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.age_provider = age_provider

    def train(self, num_epochs=10, log_every=1):
        assert self.router is not None
        assert self.loss is not None

        if not self.router.trainable():
            print(
                f"{self.router.__class__.__name__} is a mockup; "
                "nothing to train"
            )
            return self

        torch.manual_seed(self.seed)

        optimizer = torch.optim.AdamW(
            self.router.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.router.train()

        for epoch in range(num_epochs):
            # Critical v2 behavior:
            # draw_id changes with epoch and the cache is cleared, so every
            # epoch gets a new age matrix and performs new confidence lookups.
            self.age_provider.set_context(
                mode="train",
                draw_id=epoch,
            )

            losses: List[float] = []

            for x, order in tqdm(
                self._iter_blocks(self.ids_train),
                desc=f"epoch {epoch}",
            ):
                gap, candidate_mask = build_geometry(
                    order.cpu(),
                    self.size_block,
                )

                gap = gap.to(self.device)
                candidate_mask = candidate_mask.to(self.device)

                optimizer.zero_grad(set_to_none=True)

                scores = self.router(x)

                loss = self.loss(
                    scores,
                    gap,
                    candidate_mask,
                    self.h,
                )

                loss.backward()
                optimizer.step()

                losses.append(float(loss.item()))

            if epoch % log_every == 0:
                mean_loss = sum(losses) / max(len(losses), 1)

                print(
                    f"epoch {epoch}: "
                    f"loss {mean_loss:.6f} "
                    f"over {len(losses)} blocks"
                )

        return self

    @torch.no_grad()
    def evaluate_random_ages(
        self,
        h: int,
        num_draws: int,
        ids_sample: Optional[List[int]] = None,
    ):
        assert self.router is not None

        ids_sample = (
            ids_sample
            if ids_sample is not None
            else self.ids_eval
        )

        self.router.eval()

        recall_values = []
        pr_auc_values = []
        ndcg_values = []
        num_blocks = 0

        for draw_id in range(num_draws):
            # Each evaluation draw uses a new age matrix. Since all experiments
            # share seed and draw_id, their evaluation ages are comparable.
            self.age_provider.set_context(
                mode="eval",
                draw_id=draw_id,
            )

            for x, order in self._iter_blocks(ids_sample):
                scores = self.router(x)

                evaluator = ScoreOrderEval(
                    scores.cpu(),
                    order.cpu(),
                )

                recall_values.append(
                    evaluator.recall_at_h(h)
                )
                pr_auc_values.append(
                    evaluator.pr_auc(h)
                )
                ndcg_values.append(
                    evaluate_ndcg(evaluator, h)
                )

                num_blocks += 1

        if not recall_values:
            raise RuntimeError(
                "No evaluation blocks were available"
            )

        return {
            f"recall@{h}": summ(
                torch.cat(recall_values)
            ),
            f"pr_auc@{h}": summ(
                torch.cat(pr_auc_values)
            ),
            f"ndcg@{h}": summ(
                torch.cat(ndcg_values)
            ),
            "n_blocks": num_blocks,
            "num_age_draws": num_draws,
            "router": self.router.describe(),
        }


# ---------------------------------------------------------------------------
# Experiment construction
# ---------------------------------------------------------------------------

def build_experiment_features(
    *,
    folder_data: str,
    age_provider: RandomAgeProvider,
    use_random_aged_conf: bool,
    use_age_metadata: bool,
    confidence_decay_rate: Optional[float],
):
    features = []

    if use_random_aged_conf:
        if confidence_decay_rate is None:
            raise ValueError(
                "Aged confidence requires confidence_decay_rate"
            )

        features.append(
            Feature_random_aged_conf_rank_discounted(
                folder_data,
                age_provider,
                decay_rate=confidence_decay_rate,
            )
        )

    if use_age_metadata:
        if not use_random_aged_conf:
            raise ValueError(
                "Age metadata requires aged confidence"
            )

        features.append(
            Feature_random_age_metadata(
                folder_data,
                age_provider,
            )
        )

    features.extend(
        build_features(
            BASE_FEATURES,
            folder_data,
            BASE_NORMALIZATION,
            num_layers=NUM_LAYERS,
        )
    )

    return features


def run_experiment(experiment):
    experiment_name = experiment["name"]

    age_provider = RandomAgeProvider(
        max_age=MAX_CONF_AGE,
        seed=SEED,
    )

    config = {
        "folder_data": FOLDER_DATA,
        "features": {
            "base": BASE_FEATURES,
            "use_random_aged_conf": (
                experiment["use_random_aged_conf"]
            ),
            "use_age_metadata": (
                experiment["use_age_metadata"]
            ),
        },
        "base_normalization": BASE_NORMALIZATION,
        "aged_conf_normalization": "candidate_rank",
        "confidence_decay": {
            "type": "exp",
            "rate": experiment["confidence_decay_rate"],
            "formula": "exp(-rate * age)",
            "applied_after_rank_normalization": True,
        },
        "age_metadata": [
            "age_norm",
            "is_age_zero",
            "is_age_one",
        ],
        "loss": LOSS_NAME,
        "router": ROUTER_NAME,
        "router_kwargs": ROUTER_KWARGS,
        "train_horizon": TRAIN_HORIZON,
        "evaluation_horizon": EVALUATION_HORIZON,
        "full_refresh_interval": FULL_REFRESH_INTERVAL,
        "max_conf_age": MAX_CONF_AGE,
        "age_sampling": (
            "independent_uniform_integer_"
            "0_to_min(max_conf_age,current_zero_based_row)"
        ),
        "resample_train_ages_each_epoch": True,
        "num_eval_age_draws": NUM_EVAL_AGE_DRAWS,
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

    print(f"\n[{STAGE}] {experiment_name}")

    try:
        torch.manual_seed(SEED)

        features = build_experiment_features(
            folder_data=FOLDER_DATA,
            age_provider=age_provider,
            use_random_aged_conf=(
                experiment["use_random_aged_conf"]
            ),
            use_age_metadata=(
                experiment["use_age_metadata"]
            ),
            confidence_decay_rate=(
                experiment["confidence_decay_rate"]
            ),
        )

        router = FactoryRouter.create(
            ROUTER_NAME,
            **ROUTER_KWARGS,
        ).register_features(*features)

        loss = FactoryLoss.create(LOSS_NAME)

        trainer = RandomAgeRouterTrainer(
            FOLDER_DATA,
            h=TRAIN_HORIZON,
            size_block=SIZE_BLOCK,
            device=DEVICE,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            holdout=HOLDOUT,
            filter_result=FILTER_RESULT,
            seed=SEED,
            age_provider=age_provider,
        )

        trainer.register_router(router)
        trainer.register_loss(loss)
        trainer.train(num_epochs=NUM_EPOCHS)

        metrics = {
            "random_age": trainer.evaluate_random_ages(
                h=EVALUATION_HORIZON,
                num_draws=NUM_EVAL_AGE_DRAWS,
            )
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
# Self-tests
# ---------------------------------------------------------------------------

def verify_age_lookup_logic():
    fresh = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ]
    )

    ages = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 1],
            [2, 1, 0],
            [1, 3, 2],
        ]
    )

    actual = gather_by_age(
        fresh,
        ages,
    )

    expected = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [0.0, 11.0, 2.0],
            [0.0, 11.0, 22.0],
            [20.0, 1.0, 12.0],
        ]
    )

    if not torch.equal(actual, expected):
        raise AssertionError(
            "Age lookup self-test failed.\n"
            f"Expected:\n{expected}\n"
            f"Actual:\n{actual}"
        )

    print("Age lookup self-test passed.")


def verify_epoch_resampling():
    provider = RandomAgeProvider(
        max_age=16,
        seed=SEED,
    )

    provider.set_context(
        mode="train",
        draw_id=0,
    )
    age_epoch_0 = provider.get(
        id_sample=7,
        pos_base=0,
        num_steps=32,
        num_positions=64,
    ).clone()

    provider.set_context(
        mode="train",
        draw_id=1,
    )
    age_epoch_1 = provider.get(
        id_sample=7,
        pos_base=0,
        num_steps=32,
        num_positions=64,
    ).clone()

    if torch.equal(age_epoch_0, age_epoch_1):
        raise AssertionError(
            "Epoch age resampling self-test failed: "
            "epoch 0 and epoch 1 produced identical age matrices"
        )

    print("Per-epoch age resampling self-test passed.")


def verify_decay_logic():
    ages = torch.tensor(
        [0, 1, 5, 10, 16],
        dtype=torch.long,
    )

    weights = exponential_age_weight(
        ages,
        AGE_DECAY_RATE,
    )

    expected = torch.exp(
        -AGE_DECAY_RATE * ages.float()
    )

    if not torch.allclose(weights, expected):
        raise AssertionError(
            "Decay self-test failed"
        )

    if not math.isclose(
        float(weights[0]),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise AssertionError(
            "Age-zero decay weight must equal 1"
        )

    print(
        "Decay self-test passed:",
        {
            int(age): round(float(weight), 4)
            for age, weight in zip(ages, weights)
        },
    )


if __name__ == "__main__":
    verify_age_lookup_logic()
    verify_epoch_resampling()
    verify_decay_logic()

    reset_stage(
        STAGE,
        REPORT_PATH,
    )

    for experiment in EXPERIMENTS:
        run_experiment(experiment)
