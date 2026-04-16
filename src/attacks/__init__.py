"""Adversarial attack scaffolding for segmentation models."""

from src.attacks.base import AttackConfig, AttackOutput, SegmentationAttack
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
from src.attacks.runner import AttackRunner

__all__ = [
    "AttackConfig",
    "AttackOutput",
    "SegmentationAttack",
    "AttackRunner",
    "FGSMAttack",
    "PGDAttack",
    "RPPGDAttack",
    "CosPGDAttack",
    "BIMAttack",
    "SegPGDAttack",
    "SEAAttack",
    "MIFGSMAttack",
    "NIFGSMAttack",
    "DI2FGSMAttack",
    "TIFGSMAttack",
    "NIDITIFGSMAttack",
    "DAGAttack",
    "TASSAttack",
    "TranSegPGDAttack",
    "FSPGDAttack",
]
