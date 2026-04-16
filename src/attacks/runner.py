from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any

import torch

from src.attacks.base import AttackConfig, AttackOutput
from src.attacks.losses import per_image_segmentation_accuracy
from src.attacks.bim import BIMAttack
from src.attacks.cospgd import CosPGDAttack
from src.attacks.dag import DAGAttack
from src.attacks.fgsm import FGSMAttack
from src.attacks.fspgd import FSPGDAttack
from src.attacks.pgd import PGDAttack
from src.attacks.rppgd import RPPGDAttack
from src.attacks.sea import SEAAttack
from src.attacks.segpgd import SegPGDAttack
from src.attacks.transfer import (
    DI2FGSMAttack,
    MIFGSMAttack,
    NIDITIFGSMAttack,
    NIFGSMAttack,
    TASSAttack,
    TIFGSMAttack,
)
from src.attacks.transegpgd import TranSegPGDAttack
from src.models.base import SegmentationModelAdapter
from src.models.sparse import use_attack_backward_mode


ATTACKS = {
    "fgsm": FGSMAttack,
    "pgd": PGDAttack,
    "rppgd": RPPGDAttack,
    "rp-pgd": RPPGDAttack,
    "cospgd": CosPGDAttack,
    "bim": BIMAttack,
    "segpgd": SegPGDAttack,
    "sea": SEAAttack,
    "mi-fgsm": MIFGSMAttack,
    "mifgsm": MIFGSMAttack,
    "ni-fgsm": NIFGSMAttack,
    "nifgsm": NIFGSMAttack,
    "di2-fgsm": DI2FGSMAttack,
    "di²-fgsm": DI2FGSMAttack,
    "di_fgsm": DI2FGSMAttack,
    "ti-fgsm": TIFGSMAttack,
    "tifgsm": TIFGSMAttack,
    "ni+di+ti": NIDITIFGSMAttack,
    "ni-di-ti": NIDITIFGSMAttack,
    "niditi": NIDITIFGSMAttack,
    "dag": DAGAttack,
    "tass": TASSAttack,
    "transegpgd": TranSegPGDAttack,
    "fspgd": FSPGDAttack,
}


def init_attack_runtime_aggregate(config: AttackConfig) -> dict[str, Any]:
    return {
        "num_batches": 0,
        "num_samples": 0,
        "sparse_modules_configured": 0,
        "best_mean_score_sum": 0.0,
        "selected_restart_histogram": [0 for _ in range(config.num_restarts)],
        "restart_score_sum_by_restart": [0.0 for _ in range(config.num_restarts)],
        "restart_score_sample_count_by_restart": [0 for _ in range(config.num_restarts)],
    }


def update_attack_runtime_aggregate(
    aggregate: dict[str, Any],
    metadata: dict[str, Any],
    *,
    batch_size: int,
) -> None:
    aggregate["num_batches"] += 1
    aggregate["num_samples"] += int(batch_size)
    aggregate["sparse_modules_configured"] = max(
        int(aggregate["sparse_modules_configured"]),
        int(metadata.get("sparse_modules_configured", 0)),
    )
    if "best_mean_score" in metadata:
        aggregate["best_mean_score_sum"] += float(metadata["best_mean_score"]) * float(batch_size)

    selected_restart_histogram = metadata.get("selected_restart_histogram")
    if isinstance(selected_restart_histogram, list):
        counts = aggregate["selected_restart_histogram"]
        for restart_index, count in enumerate(selected_restart_histogram):
            if restart_index < len(counts):
                counts[restart_index] += int(count)

    restart_summaries = metadata.get("restart_summaries")
    if isinstance(restart_summaries, list):
        restart_score_sum_by_restart = aggregate["restart_score_sum_by_restart"]
        restart_score_sample_count_by_restart = aggregate["restart_score_sample_count_by_restart"]
        for restart_summary in restart_summaries:
            if not isinstance(restart_summary, dict):
                continue
            restart_index = int(restart_summary.get("restart_index", -1))
            if restart_index < 0 or restart_index >= len(restart_score_sum_by_restart):
                continue
            restart_score_sum_by_restart[restart_index] += float(restart_summary.get("mean_score", 0.0)) * float(batch_size)
            restart_score_sample_count_by_restart[restart_index] += int(batch_size)


def finalize_attack_runtime_aggregate(aggregate: dict[str, Any]) -> dict[str, Any]:
    num_samples = int(aggregate["num_samples"])
    if num_samples <= 0:
        return {
            "sparse_modules_configured": 0,
            "runtime_batches_aggregated": 0,
            "runtime_samples_aggregated": 0,
            "best_mean_score": 0.0,
            "selected_restart_histogram": [],
            "selected_restart_fraction": [],
            "restart_mean_score_by_restart": [],
        }

    selected_restart_histogram = [int(count) for count in aggregate["selected_restart_histogram"]]
    restart_mean_score_by_restart: list[float | None] = []
    for score_sum, sample_count in zip(
        aggregate["restart_score_sum_by_restart"],
        aggregate["restart_score_sample_count_by_restart"],
        strict=True,
    ):
        if int(sample_count) <= 0:
            restart_mean_score_by_restart.append(None)
        else:
            restart_mean_score_by_restart.append(float(score_sum) / float(sample_count))

    return {
        "sparse_modules_configured": int(aggregate["sparse_modules_configured"]),
        "runtime_batches_aggregated": int(aggregate["num_batches"]),
        "runtime_samples_aggregated": num_samples,
        "best_mean_score": float(aggregate["best_mean_score_sum"]) / float(num_samples),
        "selected_restart_histogram": selected_restart_histogram,
        "selected_restart_fraction": [float(count) / float(num_samples) for count in selected_restart_histogram],
        "restart_mean_score_by_restart": restart_mean_score_by_restart,
    }


@dataclass(slots=True)
class AttackRunner:
    model: SegmentationModelAdapter
    last_run_metadata: dict[str, Any] | None = None

    def _score_adversarial_images(self, adversarial_images: torch.Tensor, targets: torch.Tensor, ignore_index: int | None) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model.logits(adversarial_images)
            return per_image_segmentation_accuracy(logits=logits, targets=targets, ignore_index=ignore_index)

    def _protocol_metadata(
        self,
        config: AttackConfig,
        *,
        sparse_modules_configured: int,
        best_scores: torch.Tensor,
        selected_restart_indices: torch.Tensor,
        restart_summaries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        selected_counts = torch.bincount(selected_restart_indices, minlength=config.num_restarts).cpu().tolist()
        metadata = {
            **config.protocol_metadata(),
            "sparse_modules_configured": int(sparse_modules_configured),
            "best_mean_score": float(best_scores.mean().detach().cpu().item()),
            "selected_restart_histogram": selected_counts,
        }
        if config.num_restarts > 1:
            metadata["selected_restart_indices"] = selected_restart_indices.cpu().tolist()
            metadata["restart_summaries"] = restart_summaries
        return metadata

    def run(self, config: AttackConfig, images: torch.Tensor, targets: torch.Tensor) -> AttackOutput:
        attack_name = config.name.lower()
        if attack_name not in ATTACKS:
            available = ", ".join(sorted(ATTACKS))
            raise KeyError(f"Unknown attack '{config.name}'. Available: {available}")

        attack = ATTACKS[attack_name](self.model, config)
        raw_model = getattr(self.model, "model", None)
        context = (
            use_attack_backward_mode(raw_model, config.attack_backward_mode)
            if isinstance(raw_model, torch.nn.Module)
            else nullcontext(0)
        )

        with context as sparse_modules_configured:
            best_output: AttackOutput | None = None
            best_scores: torch.Tensor | None = None
            selected_restart_indices = torch.zeros(images.shape[0], device=images.device, dtype=torch.long)
            restart_summaries: list[dict[str, Any]] = []

            for restart_index in range(config.num_restarts):
                output = attack.run(images, targets)
                sample_scores = self._score_adversarial_images(
                    adversarial_images=output.adversarial_images,
                    targets=targets,
                    ignore_index=config.ignore_index,
                )
                restart_summary = {
                    "restart_index": restart_index,
                    "mean_score": float(sample_scores.mean().detach().cpu().item()),
                    "metadata": dict(output.metadata),
                }
                restart_summaries.append(restart_summary)

                if best_output is None or best_scores is None:
                    best_output = output
                    best_scores = sample_scores
                    continue

                if config.targeted:
                    update_mask = sample_scores >= best_scores
                    best_scores = torch.maximum(best_scores, sample_scores)
                else:
                    update_mask = sample_scores <= best_scores
                    best_scores = torch.minimum(best_scores, sample_scores)

                if update_mask.any():
                    best_output.adversarial_images[update_mask] = output.adversarial_images[update_mask]
                    best_output.perturbation[update_mask] = output.perturbation[update_mask]
                    selected_restart_indices[update_mask] = restart_index

            if best_output is None or best_scores is None:
                raise RuntimeError("Attack runner failed to produce an attack output.")

            runtime_metadata = self._protocol_metadata(
                config,
                sparse_modules_configured=int(sparse_modules_configured),
                best_scores=best_scores,
                selected_restart_indices=selected_restart_indices,
                restart_summaries=restart_summaries,
            )
            final_metadata = dict(best_output.metadata)
            final_metadata.update(runtime_metadata)
            self.last_run_metadata = final_metadata
            return AttackOutput(
                adversarial_images=best_output.adversarial_images.detach(),
                perturbation=best_output.perturbation.detach(),
                metadata=final_metadata,
            )
