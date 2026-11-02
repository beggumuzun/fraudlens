"""Synthetic transaction data: customers, merchants, normal behaviour and fraud scenarios."""

from fraudlens.data.generator import (
    COLUMNS,
    GeneratorConfig,
    generate_transactions,
    iter_transactions,
    summarise,
)

__all__ = [
    "COLUMNS",
    "GeneratorConfig",
    "generate_transactions",
    "iter_transactions",
    "summarise",
]
