from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ViralScoreBreakdown:
    vph_score: float
    views_score: float
    recency_score: float
    engagement_score: float
    total_score: float


def _normalize(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return max(0.0, min(1.0, value / max_value))


def compute_viral_score(
    vph_1h: float,
    views: int,
    hours_since_publish: float,
    likes_per_view: float,
) -> ViralScoreBreakdown:
    # Heuristic weights
    vph_component = _normalize(vph_1h, 5000.0)
    views_component = _normalize(views, 500000.0)
    recency_component = 1.0 - _normalize(hours_since_publish, 168.0)
    engagement_component = _normalize(likes_per_view, 0.08)

    total = (
        vph_component * 0.45
        + views_component * 0.2
        + recency_component * 0.2
        + engagement_component * 0.15
    )
    return ViralScoreBreakdown(
        vph_score=round(vph_component * 100, 2),
        views_score=round(views_component * 100, 2),
        recency_score=round(recency_component * 100, 2),
        engagement_score=round(engagement_component * 100, 2),
        total_score=round(total * 100, 2),
    )
