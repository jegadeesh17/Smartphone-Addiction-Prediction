"""Behavioral metric calculations, digital hygiene rules, and clinical intervention generators."""

from typing import List
from src.schemas import BehavioralMetrics, BehavioralProfileInput

# Canonical clinical recommendation strings
RECOMMENDATION_SLEEP_PROTECTION = (
    "Sleep Protection: Screen time exceeds sleep duration. "
    "Enforce a 60-minute pre-bed digital curfew."
)
RECOMMENDATION_RECREATION_AUDIT = (
    "Recreation Audit: Over 60% of device time is social media/gaming. "
    "Establish focused non-digital hobbies."
)
RECOMMENDATION_NOTIFICATION_HYGIENE = (
    "Notification Hygiene: High daily unlock frequency (>120 opens/day) or alert volume (>180/day). "
    "Batch non-essential notifications into designated hourly slots."
)
RECOMMENDATION_WEEKEND_DISCONNECT = (
    "Weekend Disconnect: Surge of >2.5 hours on weekends suggests compensatory bingeing. "
    "Schedule structured offline weekend activities."
)


def compute_metrics(p: BehavioralProfileInput) -> BehavioralMetrics:
    """Computes derived behavioral ratios and habit fragmentation indicators from input profile.

    Adheres strictly to SPEC AC-1.4, PAR-2, and ARCHITECTURE ratio definitions.
    Uses epsilon 1e-5 to prevent division by zero while preserving floating-point precision.
    """
    eps = 1e-5

    screen_to_sleep = p.daily_screen_time_hours / (p.sleep_hours + eps)
    recreational_hours = p.social_media_hours + p.gaming_hours
    recreational_share = recreational_hours / (p.daily_screen_time_hours + eps)

    if p.daily_screen_time_hours == 0.0:
        avg_unlock = 0.0
    else:
        avg_unlock = (p.daily_screen_time_hours * 60.0) / (p.app_opens_per_day + eps)

    weekend_surge = p.weekend_screen_time - p.daily_screen_time_hours
    total_accounted = p.daily_screen_time_hours + p.work_study_hours + p.sleep_hours
    exceeds_budget = bool(total_accounted > 24.0 or (p.daily_screen_time_hours + p.sleep_hours > 24.0))

    return BehavioralMetrics(
        screen_to_sleep_ratio=round(screen_to_sleep, 4),
        recreational_share=round(recreational_share, 4),
        avg_unlock_minutes=round(avg_unlock, 4),
        weekend_surge_hours=round(weekend_surge, 4),
        total_accounted_hours=round(total_accounted, 4),
        exceeds_daily_budget=exceeds_budget,
    )


def generate_clinical_interventions(
    p: BehavioralProfileInput,
    metrics: BehavioralMetrics,
) -> List[str]:
    """Evaluates behavioral metrics against clinical hygiene trigger thresholds.

    Triggers:
    - Sleep Protection: screen_to_sleep_ratio > 1.2
    - Recreation Audit: recreational_share > 0.60
    - Notification Hygiene: app_opens_per_day > 120 or notifications_per_day > 180
    - Weekend Disconnect: weekend_surge_hours > 2.5
    """
    interventions: List[str] = []

    if metrics.screen_to_sleep_ratio > 1.2:
        interventions.append(RECOMMENDATION_SLEEP_PROTECTION)

    if metrics.recreational_share > 0.60:
        interventions.append(RECOMMENDATION_RECREATION_AUDIT)

    if p.app_opens_per_day > 120 or p.notifications_per_day > 180:
        interventions.append(RECOMMENDATION_NOTIFICATION_HYGIENE)

    if metrics.weekend_surge_hours > 2.5:
        interventions.append(RECOMMENDATION_WEEKEND_DISCONNECT)

    return interventions


def generate_recommendations(
    profile: BehavioralProfileInput,
    metrics: BehavioralMetrics,
    probability: float = 0.0,
) -> List[str]:
    """Compatibility wrapper matching the ARCHITECTURE.md signature.

    Evaluates behavioral ratios and probability to return prioritized habit advice.
    """
    return generate_clinical_interventions(profile, metrics)


def classify_severity(probability: float) -> str:
    """Classifies risk probability into 3-tier severity ('High', 'Moderate', 'Healthy') per SPEC AC-1.1 and PAR-4."""
    if probability >= 0.70:
        return "High"
    elif probability >= 0.40:
        return "Moderate"
    else:
        return "Healthy"


def classify_risk_tier(probability: float) -> str:
    """Classifies risk probability into uppercase risk tier ('HIGH', 'MODERATE', 'LOW')."""
    if probability >= 0.70:
        return "HIGH"
    elif probability >= 0.40:
        return "MODERATE"
    else:
        return "LOW"


def classify_clinical_spectrum(probability: float) -> str:
    """Classifies risk probability into the 3-tier clinical spectrum per ADR-0006."""
    if probability >= 0.70:
        return "Elevated Risk Tier"
    elif probability >= 0.40:
        return "Compensatory Usage Pattern"
    else:
        return "Balanced Habit Profile"


def get_authoritative_status_label(
    is_addicted: bool,
    probability: float,
    threshold: float = 0.50,
) -> str:
    """Generates the single authoritative status pill label consolidating decision and risk tier (ADR-0006)."""
    tier = classify_risk_tier(probability)
    if is_addicted:
        return f"ADDICTION DETECTED • {tier} RISK"
    else:
        return f"HEALTHY PATTERN • {tier} RISK"
