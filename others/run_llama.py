#!/usr/bin/env python3
"""
Evaluate a Hugging Face Llama-style causal language model on GSM8K through
EleutherAI lm-evaluation-harness.

The custom LM wrapper intentionally processes requests one by one
(batch_size=1) and uses the Hugging Face KV cache during generation.

Example:
    CUDA_VISIBLE_DEVICES=1 python run_llama_gsm8k.py \
        --model_id meta-llama/Llama-2-7b-hf \
        --device cuda:0 \
        --max_new_tokens 256 \
        --limit 100

When CUDA_VISIBLE_DEVICES=1 is set, physical GPU 1 appears inside Python as
cuda:0. To keep both GPUs visible and address the second one directly, omit
CUDA_VISIBLE_DEVICES and pass --device cuda:1.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm

from lm_eval import evaluator
from lm_eval.api.model import LM
from lm_eval.utils import make_table


_DTYPE_MAP: dict[str, torch.dtype | str] = {
    "auto": "auto",
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class Llama7BHarness(LM):
    """Minimal lm_eval wrapper for generation-based tasks such as GSM8K."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        revision: str | None = None,
        max_length: int = 4096,
        default_max_gen_toks: int = 256,
    ) -> None:
        super().__init__()

        if dtype not in _DTYPE_MAP:
            raise ValueError(
                f"Unsupported dtype={dtype!r}. "
                f"Choose from {sorted(_DTYPE_MAP)}."
            )
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested {device}, but torch.cuda.is_available() is False."
            )

        self._device = torch.device(device)
        self._max_length = int(max_length)
        self._default_max_gen_toks = int(default_max_gen_toks)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )

        # Llama tokenizers commonly have no pad token. Reusing EOS is standard
        # for decoder-only inference.
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            torch_dtype=_DTYPE_MAP[dtype],
            low_cpu_mem_usage=True,
        )
        self.model.to(self._device)
        self.model.eval()

        # Explicitly enable the autoregressive KV cache.
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def batch_size(self) -> int:
        # This wrapper deliberately evaluates one request at a time.
        return 1

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._default_max_gen_toks

    @property
    def eot_token_id(self) -> int:
        token_id = self.tokenizer.eos_token_id
        if token_id is None:
            raise ValueError("Tokenizer has no eos_token_id.")
        return int(token_id)

    def tok_encode(self, string: str, **_: Any) -> list[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: Iterable[int], **_: Any) -> str:
        return self.tokenizer.decode(list(tokens), skip_special_tokens=True)

    @staticmethod
    def _request_args(request: Any) -> tuple[Any, ...]:
        """Support current Instance objects and older tuple-like requests."""
        if hasattr(request, "args"):
            return tuple(request.args)
        if isinstance(request, tuple):
            return request
        raise TypeError(f"Unsupported lm_eval request type: {type(request)!r}")

    @staticmethod
    def _truncate_at_stop(text: str, until: list[str]) -> str:
        earliest: int | None = None
        for stop in until:
            if not stop:
                continue
            index = text.find(stop)
            if index >= 0 and (earliest is None or index < earliest):
                earliest = index
        return text if earliest is None else text[:earliest]

    def generate_until(self, requests: list[Any]) -> list[str]:
        """
        Run one request at a time.

        Each lm_eval request is expected to contain:
            (context: str, generation_kwargs: dict)
        """
        outputs: list[str] = []

        for request in tqdm(requests):
            context, request_gen_kwargs = self._request_args(request)
            gen_kwargs = copy.deepcopy(request_gen_kwargs or {})

            until = gen_kwargs.pop("until", [])
            if isinstance(until, str):
                until = [until]
            else:
                until = list(until)

            # lm_eval normally uses max_gen_toks. Accept max_new_tokens too.
            max_new_tokens = int(
                gen_kwargs.pop(
                    "max_gen_toks",
                    gen_kwargs.pop(
                        "max_new_tokens",
                        self._default_max_gen_toks,
                    ),
                )
            )

            do_sample = bool(gen_kwargs.pop("do_sample", False))

            # Keep only common Hugging Face generation arguments. This avoids
            # passing lm_eval-only keys to transformers.generate().
            allowed_keys = {
                "temperature",
                "top_p",
                "top_k",
                "num_beams",
                "repetition_penalty",
                "length_penalty",
                "no_repeat_ngram_size",
            }
            hf_gen_kwargs = {
                key: value
                for key, value in gen_kwargs.items()
                if key in allowed_keys
            }

            # Sampling-only arguments can produce warnings with greedy decoding.
            if not do_sample:
                hf_gen_kwargs.pop("temperature", None)
                hf_gen_kwargs.pop("top_p", None)
                hf_gen_kwargs.pop("top_k", None)

            encoded = self.tokenizer(
                context,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=self._max_length,
            )
            encoded = {
                key: value.to(self._device)
                for key, value in encoded.items()
            }
            prompt_length = encoded["input_ids"].shape[1]

            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    use_cache=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **hf_gen_kwargs,
                )

            new_token_ids = generated[0, prompt_length:]
            text = self.tokenizer.decode(
                new_token_ids,
                skip_special_tokens=True,
            )
            outputs.append(self._truncate_at_stop(text, until))

        return outputs

    def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
        """
        GSM8K does not call this method. It is defined so the class satisfies
        the lm_eval LM interface.

        Implement exact continuation scoring before using this wrapper for
        MMLU, ARC-C, HellaSwag, or other likelihood-based tasks.
        """
        raise NotImplementedError(
            "This GSM8K wrapper implements generate_until only. "
            "loglikelihood is required for multiple-choice tasks."
        )

    def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
        """
        GSM8K does not call this method. Perplexity tasks require an
        implementation of rolling log-likelihood.
        """
        raise NotImplementedError(
            "loglikelihood_rolling is not needed for GSM8K."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        required=True,
        help="Hugging Face model ID or local model directory.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device, for example cuda:0 or cuda:1.",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(_DTYPE_MAP),
        default="bfloat16",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum prompt/context length.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum generated tokens for each GSM8K answer.",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help=(
            "Override GSM8K few-shot count. Omit this argument to use the "
            "task configuration's default."
        ),
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Examples to evaluate; an integer count or lm_eval fraction.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("gsm8k_llama7b_results.json"),
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
    )
    parser.add_argument(
        "--revision",
        default=None,
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="Include per-example prompts and outputs in the result object.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lm = Llama7BHarness(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        max_length=args.max_length,
        default_max_gen_toks=args.max_new_tokens,
    )

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=["gsm8k"],
        num_fewshot=args.num_fewshot,
        batch_size=1,
        limit=args.limit,
        log_samples=args.log_samples,
        # In lm_eval, gen_kwargs is typically a comma-separated string.
        gen_kwargs=f"max_gen_toks={args.max_new_tokens},do_sample=False",
    )

    if results is None:
        raise RuntimeError("lm_eval returned no result.")

    print(make_table(results))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False, default=str)

    print(f"\nSaved results to: {args.output_path}")


if __name__ == "__main__":
    main()
