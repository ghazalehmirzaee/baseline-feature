# models/graph_modules/config.py
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class GraphConfig:
    """Configuration for graph construction."""

    # Thresholds for relationship tiers
    direct_threshold: int = 10
    limited_threshold: int = 3

    # Confidence parameters
    gamma: float = 0.8
    confidence_high: float = 0.8
    confidence_med: float = 0.6
    confidence_low: float = 0.4

    # Area weight parameters
    area_weight_factor: float = 0.1
    min_area_ratio: float = 0.1

    # Graph construction parameters
    normalize_weights: bool = True
    add_self_loops: bool = True

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'GraphConfig':
        """Create config from dictionary."""
        graph_config = config.get('graph', {})
        return cls(
            direct_threshold=graph_config.get('direct_threshold', 10),
            limited_threshold=graph_config.get('limited_threshold', 3),
            gamma=graph_config.get('gamma', 0.8),
            confidence_high=graph_config.get('confidence_high', 0.8),
            confidence_med=graph_config.get('confidence_med', 0.6),
            confidence_low=graph_config.get('confidence_low', 0.4),
            area_weight_factor=graph_config.get('area_weight_factor', 0.1),
            min_area_ratio=graph_config.get('min_area_ratio', 0.1),
            normalize_weights=graph_config.get('normalize_weights', True),
            add_self_loops=graph_config.get('add_self_loops', True)
        )

