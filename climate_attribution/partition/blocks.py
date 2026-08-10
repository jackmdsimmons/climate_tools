"""
Step 2 of the three-step attribution model: give each subset its own driver set,
then decompose the portfolio as the sum of its blocks.

    M_p,t = sum_k ( sum_{j in P_k} D_1,j * ... * D_Nk,j )

The number of drivers may differ per block. Divested and newly bought holdings
typically get one driver -- their whole contribution -- while survivors carry the
full chain of weight, sector, selection and intensity drivers. Because LMDI is
additive within each block and the blocks are disjoint, the effects still sum
exactly to the portfolio's change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..core.lmdi import RESIDUAL_TOL, Decomposition, lmdi_decompose

EFFECT_SEPARATOR = " :: "


@dataclass(frozen=True)
class Block:
    """One disjoint subset of the portfolio, with the drivers that apply to it."""

    name: str
    drivers: Mapping[str, tuple[Sequence[float], Sequence[float]]]
    labels: Sequence[str] | None = None


@dataclass(frozen=True)
class PartitionedDecomposition:
    """Result of decomposing a partitioned portfolio."""

    by_block: dict[str, Decomposition]

    @property
    def effects(self) -> dict[str, float]:
        """Flat effects, keyed 'block :: driver', in block then driver order."""
        return {
            f"{block}{EFFECT_SEPARATOR}{driver}": value
            for block, result in self.by_block.items()
            for driver, value in result.effects.items()
        }

    @property
    def total_t0(self) -> float:
        return sum(r.total_t0 for r in self.by_block.values())

    @property
    def total_t1(self) -> float:
        return sum(r.total_t1 for r in self.by_block.values())

    @property
    def delta(self) -> float:
        return self.total_t1 - self.total_t0

    @property
    def explained(self) -> float:
        return sum(r.explained for r in self.by_block.values())

    @property
    def residual(self) -> float:
        return self.delta - self.explained

    def check_additivity(self, tol: float = RESIDUAL_TOL) -> None:
        """Raise if the effects fail to reconstruct the portfolio's change."""
        scale = max(1.0, abs(self.delta), abs(self.total_t0))
        if abs(self.residual) > tol * scale:
            raise AssertionError(
                f"partitioned additivity violated: delta={self.delta!r}, "
                f"explained={self.explained!r}, residual={self.residual!r}"
            )

    def waterfall(self) -> list[tuple[str, float]]:
        """
        Steps for a waterfall chart: opening value, one step per effect, close.

        Mirrors the exhibits in Bouchet (2025), e.g. portfolio t0 -> divested
        contribution -> reallocation -> sector allocation -> stock selection ->
        emissions intensity -> portfolio t1.
        """
        steps: list[tuple[str, float]] = [("Portfolio t0", self.total_t0)]
        steps.extend(self.effects.items())
        steps.append(("Portfolio t1", self.total_t1))
        return steps


def decompose_blocks(blocks: Iterable[Block]) -> PartitionedDecomposition:
    """
    Decompose each block independently and combine the results.

    Raises:
        ValueError: if two blocks share a name, or an instrument appears in more
            than one block -- the subsets must be disjoint for the effects to add
            up to the portfolio's change.
    """
    results: dict[str, Decomposition] = {}
    seen_labels: dict[str, str] = {}

    for block in blocks:
        if block.name in results:
            raise ValueError(f"duplicate block name {block.name!r}")

        result = lmdi_decompose(block.drivers, labels=block.labels)

        if block.labels is not None:
            for label in result.labels:
                if label in seen_labels:
                    raise ValueError(
                        f"instrument {label!r} appears in both block "
                        f"{seen_labels[label]!r} and block {block.name!r}; "
                        "blocks must be disjoint"
                    )
                seen_labels[label] = block.name

        results[block.name] = result

    if not results:
        raise ValueError("at least one block is required")

    combined = PartitionedDecomposition(by_block=results)
    combined.check_additivity()
    return combined
