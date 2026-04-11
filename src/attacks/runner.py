from __future__ import annotations

from dataclasses import dataclass

import torch

from src.attacks.base import AttackConfig, AttackOutput
from src.attacks.bim import BIMAttack
from src.attacks.cospgd import CosPGDAttack
from src.attacks.dag import DAGAttack
from src.attacks.fgsm import FGSMAttack
from src.attacks.fspgd import FSPGDAttack
from src.attacks.pgd import PGDAttack
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


ATTACKS = {
    "fgsm": FGSMAttack,
    "pgd": PGDAttack,
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
