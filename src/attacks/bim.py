from __future__ import annotations

from src.attacks.pgd import PGDAttack


class BIMAttack(PGDAttack):
    """Alias placeholder for iterative FGSM-style attacks."""

    attack_name = "bim"
