from __future__ import annotations

from dataclasses import dataclass

import torch

from src.attacks.base import AttackConfig, AttackOutput
from src.attacks.bim import BIMAttack
from src.attacks.fgsm import FGSMAttack
from src.attacks.pgd import PGDAttack
from src.models.base import SegmentationModelAdapter


ATTACKS = {
    "fgsm": FGSMAttack,
    "pgd": PGDAttack,
    "bim": BIMAttack,
}


@dataclass(slots=True)
class AttackRunner:
    model: SegmentationModelAdapter

    def run(self, config: AttackConfig, images: torch.Tensor, targets: torch.Tensor) -> AttackOutput:
        attack_name = config.name.lower()
        if attack_name not in ATTACKS:
            available = ", ".join(sorted(ATTACKS))
            raise KeyError(f"Unknown attack '{config.name}'. Available: {available}")
        attack = ATTACKS[attack_name](self.model, config)
        return attack.run(images, targets)
