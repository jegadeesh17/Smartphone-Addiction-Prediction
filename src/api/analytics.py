"""Cohort Analytics API router and population distribution endpoints.

Exposes demographic cohort slicing, 2D screen-sleep joint density matrix,
and personal quantile benchmark calculation (Milestone 2, Journey 2, AC-2.1 to AC-2.5).
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.cohort_service import ALLOWED_DIMENSIONS, get_cohort_service
from src.config import Settings, get_settings
from src.inference import get_inference_engine
from src.schemas import (
    BehavioralMetrics,
    BenchmarkOverlayRequest,
    BenchmarkOverlayResponse,
    CohortDistributionResponse,
    CohortFilterParams,
    HabitOptimizeRequest,
    HabitOptimizeResponse,
    ScreenSleepMatrixResponse,
    WhatIfRequest,
    WhatIfResponse,
)

router = APIRouter(prefix="/api/analytics", tags=["Cohort Analytics"])


def _extract_ratios(metrics: BehavioralMetrics) -> dict[str, float]:
    """Extract standard and legacy ratio aliases matching AC-1.4 and AC-3.1."""
    return {
        "screen_to_sleep_ratio": metrics.screen_to_sleep_ratio,
        "screen_to_sleep": metrics.screen_to_sleep_ratio,
        "recreational_share": metrics.recreational_share,
        "recreational_to_screen": metrics.recreational_share,
        "avg_unlock_minutes": metrics.avg_unlock_minutes,
        "weekend_surge_hours": metrics.weekend_surge_hours,
        "weekend_diff": metrics.weekend_surge_hours,
        "total_accounted_hours": metrics.total_accounted_hours,
    }


@router.get(
    "/cohorts",
    response_model=CohortDistributionResponse,
    summary="Get population cohort distributions and metric breakdowns",
)
def get_cohorts(
    dimension: str = Query(
        default="age_bracket",
        description="Primary demographic segmentation dimension ('age_bracket', 'gender', 'stress_level', 'academic_work_impact')",
    ),
    filter_gender: Optional[str] = Query(
        default=None,
        description="Optional filter conditioning on participant gender ('Female', 'Male', 'Other')",
    ),
    filter_stress: Optional[str] = Query(
        default=None,
        description="Optional filter conditioning on chronic stress level ('Low', 'Medium', 'High')",
    ),
    settings: Settings = Depends(get_settings),
) -> CohortDistributionResponse:
    """Retrieve pre-aggregated demographic cohort breakdown and percentiles (AC-2.1, AC-2.2).

    Validates dimension parameters with explicit HTTP 422 error on invalid inputs (AC-2.5).
    """
    if dimension not in ALLOWED_DIMENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid dimension '{dimension}'. Permissible dimensions are: {list(ALLOWED_DIMENSIONS)}",
        )

    try:
        params = CohortFilterParams(
            dimension=dimension,  # type: ignore[arg-type]
            filter_gender=filter_gender,
            filter_stress=filter_stress,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    service = get_cohort_service(cohort_summary_path=settings.COHORT_CACHE_PATH)
    return service.get_cohorts(params)


@router.get(
    "/distributions/screen-sleep-matrix",
    response_model=ScreenSleepMatrixResponse,
    summary="Get 2D screen time vs sleep duration joint density grid",
)
def get_screen_sleep_matrix(
    settings: Settings = Depends(get_settings),
) -> ScreenSleepMatrixResponse:
    """Retrieve 2D joint density and addiction prevalence grid (AC-2.3)."""
    service = get_cohort_service(cohort_summary_path=settings.COHORT_CACHE_PATH)
    return service.get_screen_sleep_matrix()


@router.post(
    "/distributions/benchmark-overlay",
    response_model=BenchmarkOverlayResponse,
    summary="Compute personal quantile benchmark overlay",
)
def compute_benchmark_overlay(
    body: BenchmarkOverlayRequest,
    settings: Settings = Depends(get_settings),
) -> BenchmarkOverlayResponse:
    """Compute quantile percentiles against 691k population curves (AC-2.4)."""
    service = get_cohort_service(cohort_summary_path=settings.COHORT_CACHE_PATH)
    return service.compute_benchmark_overlay(body)


@router.post(
    "/what-if",
    response_model=WhatIfResponse,
    summary="Evaluate counterfactual What-If behavioral simulation",
)
def simulate_what_if(
    request: WhatIfRequest,
    settings: Settings = Depends(get_settings),
) -> WhatIfResponse:
    """Evaluate counterfactual What-If simulation deltas against baseline profile (AC-3.1).

    Applies lifestyle modifications with physiological boundary clamping,
    evaluates simulated risk probability and risk delta, and returns updated ratios.
    """
    engine = get_inference_engine(
        use_mock=settings.USE_MOCK_MODEL,
        model_path=settings.MODEL_PATH,
        priors_path=settings.PRIORS_PATH,
    )

    baseline = request.baseline_profile
    base_res = engine.predict(baseline)

    # 1. Recreational modifications
    sim_social = max(0.0, min(24.0, baseline.social_media_hours + request.delta_social_media_hours))
    sim_gaming = max(0.0, min(24.0, baseline.gaming_hours + request.delta_gaming_hours))

    # 2. Daily screen time adjustment
    if request.delta_daily_screen_time_hours != 0.0:
        sim_screen = baseline.daily_screen_time_hours + request.delta_daily_screen_time_hours
    else:
        # Proportionally adjust screen time if recreational deltas changed
        rec_delta = (sim_social - baseline.social_media_hours) + (sim_gaming - baseline.gaming_hours)
        sim_screen = baseline.daily_screen_time_hours + rec_delta

    # Clamping daily screen time within physiological limits
    sim_screen = max(0.0, min(24.0, sim_screen))
    if sim_screen < (sim_social + sim_gaming):
        sim_screen = min(24.0, sim_social + sim_gaming)

    # 3. Weekend screen time adjustment
    if request.delta_weekend_screen_time != 0.0:
        sim_weekend = baseline.weekend_screen_time + request.delta_weekend_screen_time
    else:
        sim_weekend = baseline.weekend_screen_time
    sim_weekend = max(0.0, min(24.0, sim_weekend))

    # 4. Sleep hours clamped to [1.0, 18.0]
    sim_sleep = max(1.0, min(18.0, baseline.sleep_hours + request.delta_sleep_hours))

    # 5. App opens and notifications clamped to [0, 500]
    sim_opens = max(0, min(500, int(baseline.app_opens_per_day + request.delta_app_opens_per_day)))
    sim_notifs = max(0, min(500, int(baseline.notifications_per_day + request.delta_notifications_per_day)))

    # Construct simulated profile
    sim_profile = baseline.model_copy(
        update={
            "daily_screen_time_hours": round(sim_screen, 2),
            "social_media_hours": round(sim_social, 2),
            "gaming_hours": round(sim_gaming, 2),
            "weekend_screen_time": round(sim_weekend, 2),
            "sleep_hours": round(sim_sleep, 2),
            "app_opens_per_day": sim_opens,
            "notifications_per_day": sim_notifs,
        }
    )

    sim_res = engine.predict(sim_profile)
    risk_delta = round(sim_res.probability - base_res.probability, 4)

    return WhatIfResponse(
        baseline_probability=base_res.probability,
        simulated_probability=sim_res.probability,
        risk_delta=risk_delta,
        baseline_classification=base_res.classification,
        simulated_classification=sim_res.classification,
        baseline_status_label=base_res.status_label,
        simulated_status_label=sim_res.status_label,
        baseline_ratios=_extract_ratios(base_res.ratios),
        simulated_ratios=_extract_ratios(sim_res.ratios),
        simulated_profile=sim_profile,
        interventions=sim_res.interventions,
    )


@router.post(
    "/what-if/optimize",
    response_model=HabitOptimizeResponse,
    summary="Compute optimal habit modification pathway",
)
def optimize_habits(
    request: HabitOptimizeRequest,
    settings: Settings = Depends(get_settings),
) -> HabitOptimizeResponse:
    """Compute minimal habit modifications to lower addiction risk below threshold tau (AC-3.2).

    If baseline risk is already below threshold tau, returns 0 reductions.
    Otherwise, executes an iterative optimization search incrementally reducing screen time
    and interactions while increasing sleep duration until target threshold is achieved.
    """
    engine = get_inference_engine(
        use_mock=settings.USE_MOCK_MODEL,
        model_path=settings.MODEL_PATH,
        priors_path=settings.PRIORS_PATH,
    )

    baseline = request.baseline_profile
    base_res = engine.predict(baseline)

    tau = request.target_threshold if request.target_threshold is not None else baseline.decision_threshold
    target_prob = tau

    if request.target_risk_tier:
        tier_upper = request.target_risk_tier.strip().upper()
        if "LOW" in tier_upper:
            target_prob = min(tau, 0.39)
        elif "MOD" in tier_upper:
            target_prob = min(tau, 0.69)

    # Baseline already within target threshold
    if base_res.probability < target_prob:
        pathway = [
            f"Current habits are already within healthy bounds: baseline risk probability {base_res.probability:.1%} "
            f"is below target threshold {tau:.2f} ({base_res.status_label}). "
            "Maintain your current balanced digital routines."
        ]
        return HabitOptimizeResponse(
            baseline_probability=base_res.probability,
            target_threshold=round(tau, 2),
            target_screen_time_reduction_hours=0.0,
            target_sleep_increase_hours=0.0,
            target_app_opens_reduction=0,
            target_notifications_reduction=0,
            projected_probability=base_res.probability,
            projected_classification=base_res.classification,
            projected_status_label=base_res.status_label,
            achievable=True,
            recommended_pathway=pathway,
            optimized_profile=baseline,
        )

    s0 = baseline.daily_screen_time_hours
    sl0 = baseline.sleep_hours
    o0 = baseline.app_opens_per_day
    n0 = baseline.notifications_per_day
    sm0 = baseline.social_media_hours
    g0 = baseline.gaming_hours
    w0 = baseline.weekend_screen_time

    best_cand = baseline
    best_res = base_res
    best_s_red = 0.0
    best_sl_inc = 0.0
    best_o_red = 0
    best_n_red = 0

    achieved = False

    for step in range(1, 26):
        s_red = round(min(step * 0.25, max(0.0, s0 - 2.0)), 2)
        sl_inc = round(min(step * 0.25, max(0.0, 8.5 - sl0)), 2)
        o_red = int(min(step * 10, max(0, o0 - 20)))
        n_red = int(min(step * 15, max(0, n0 - 20)))

        rec_tot = sm0 + g0
        if rec_tot > 0:
            sm_red = round(min(sm0, s_red * (sm0 / rec_tot)), 2)
            g_red = round(min(g0, s_red * (g0 / rec_tot)), 2)
        else:
            sm_red = 0.0
            g_red = 0.0
        w_red = round(min(s_red, max(0.0, w0 - 2.0)), 2)

        cand = baseline.model_copy(
            update={
                "daily_screen_time_hours": round(s0 - s_red, 2),
                "social_media_hours": round(sm0 - sm_red, 2),
                "gaming_hours": round(g0 - g_red, 2),
                "weekend_screen_time": round(w0 - w_red, 2),
                "sleep_hours": round(sl0 + sl_inc, 2),
                "app_opens_per_day": int(o0 - o_red),
                "notifications_per_day": int(n0 - n_red),
            }
        )
        res = engine.predict(cand)
        if res.probability < best_res.probability:
            best_res = res
            best_cand = cand
            best_s_red = s_red
            best_sl_inc = sl_inc
            best_o_red = o_red
            best_n_red = n_red

        if res.probability < target_prob:
            achieved = True
            break

    # Build actionable pathway messages
    pathway: list[str] = []
    if best_s_red > 0:
        opt_s = round(s0 - best_s_red, 2)
        pathway.append(
            f"Daily Screen Time: Reduce weekday screen time by {best_s_red:.2f}h "
            f"(from {s0:.1f}h to {opt_s:.1f}h/day), focusing on social media and gaming reduction."
        )
    if best_sl_inc > 0:
        opt_sl = round(sl0 + best_sl_inc, 2)
        pathway.append(
            f"Sleep Restoration: Increase daily sleep duration by {best_sl_inc:.2f}h "
            f"(from {sl0:.1f}h to {opt_sl:.1f}h/night) by enforcing a 60-minute pre-bed digital curfew."
        )
    if best_o_red > 0 or best_n_red > 0:
        pathway.append(
            f"Interaction Hygiene: Reduce app opens by {best_o_red} unlocks/day "
            f"and batch notifications to eliminate {best_n_red} alerts/day."
        )
    if (w0 - (baseline.weekend_screen_time - best_s_red)) > 0:
        opt_w = round(max(2.0, w0 - best_s_red), 2)
        pathway.append(
            f"Weekend Balance: Curtail weekend screen time from {w0:.1f}h to {opt_w:.1f}h/day "
            "to prevent compensatory weekend bingeing."
        )

    if achieved:
        pathway.append(
            f"Projected Outcome: Addiction risk decreases from {base_res.probability:.1%} to {best_res.probability:.1%} "
            f"({best_res.status_label}), successfully achieving target threshold of {tau:.2f}."
        )
    else:
        pathway.append(
            f"Projected Outcome: Aggressive habit modifications reduce risk from {base_res.probability:.1%} to {best_res.probability:.1%} "
            f"({best_res.status_label}), but target threshold {tau:.2f} may require additional clinical support."
        )

    return HabitOptimizeResponse(
        baseline_probability=base_res.probability,
        target_threshold=round(tau, 2),
        target_screen_time_reduction_hours=best_s_red,
        target_sleep_increase_hours=best_sl_inc,
        target_app_opens_reduction=best_o_red,
        target_notifications_reduction=best_n_red,
        projected_probability=best_res.probability,
        projected_classification=best_res.classification,
        projected_status_label=best_res.status_label,
        achievable=achieved,
        recommended_pathway=pathway,
        optimized_profile=best_cand,
    )

