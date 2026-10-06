"""Unit tests for behavioral metrics, clinical interventions, and status helpers in src/recommendations.py."""

import pytest

from src.schemas import BehavioralProfileInput
from src.recommendations import (
    RECOMMENDATION_NOTIFICATION_HYGIENE,
    RECOMMENDATION_RECREATION_AUDIT,
    RECOMMENDATION_SLEEP_PROTECTION,
    RECOMMENDATION_WEEKEND_DISCONNECT,
    classify_clinical_spectrum,
    classify_risk_tier,
    classify_severity,
    compute_metrics,
    generate_clinical_interventions,
    generate_recommendations,
    get_authoritative_status_label,
)


class TestComputeMetrics:
    """Validation test suite for compute_metrics formula precision and edge cases."""

    def test_ratio_calculation_precision_ac_1_4(self) -> None:
        """Verify exact ratio math matches SPEC AC-1.4:

        screen_to_sleep: 8.0 / 6.0 approx 1.3333 +- 0.001
        recreational_to_screen: (3.0 + 1.0) / 8.0 = 0.5000 +- 0.001
        avg_unlock_minutes: (8.0 * 60) / 80 = 6.0000 +- 0.001
        weekend_diff: 11.0 - 8.0 = 3.0000 +- 0.001
        """
        profile = BehavioralProfileInput(
            daily_screen_time_hours=8.0,
            sleep_hours=6.0,
            social_media_hours=3.0,
            gaming_hours=1.0,
            app_opens_per_day=80,
            weekend_screen_time=11.0,
            work_study_hours=2.0,
        )

        metrics = compute_metrics(profile)

        # Exact ratio checks
        assert abs(metrics.screen_to_sleep_ratio - 1.3333) < 0.001
        assert abs(metrics.screen_to_sleep - 1.3333) < 0.001

        assert abs(metrics.recreational_share - 0.5000) < 0.001
        assert abs(metrics.recreational_to_screen - 0.5000) < 0.001

        assert abs(metrics.avg_unlock_minutes - 6.0000) < 0.001

        assert abs(metrics.weekend_surge_hours - 3.0000) < 0.001
        assert abs(metrics.weekend_diff - 3.0000) < 0.001

        assert metrics.total_accounted_hours == 16.0
        assert metrics.exceeds_daily_budget is False

    def test_zero_screen_time_edge_case(self) -> None:
        """Verify no ZeroDivisionError and clean zeros when daily screen time is zero."""
        profile = BehavioralProfileInput(
            daily_screen_time_hours=0.0,
            social_media_hours=0.0,
            gaming_hours=0.0,
            work_study_hours=4.0,
            weekend_screen_time=0.0,
            sleep_hours=8.0,
            app_opens_per_day=0,
            notifications_per_day=0,
        )
        metrics = compute_metrics(profile)
        assert metrics.screen_to_sleep_ratio == 0.0
        assert metrics.recreational_share == 0.0
        assert metrics.avg_unlock_minutes == 0.0
        assert metrics.weekend_surge_hours == 0.0
        assert metrics.total_accounted_hours == 12.0
        assert metrics.exceeds_daily_budget is False

    def test_exceeds_daily_budget_flag(self) -> None:
        """Verify exceeds_daily_budget flag triggers when screen + work + sleep > 24h."""
        profile = BehavioralProfileInput(
            daily_screen_time_hours=12.0,
            work_study_hours=8.0,
            sleep_hours=6.0,
        )
        metrics = compute_metrics(profile)
        assert metrics.total_accounted_hours == 26.0
        assert metrics.exceeds_daily_budget is True


class TestClinicalInterventions:
    """Validation test suite for digital hygiene recommendation triggers."""

    def test_all_four_triggers_fire_ac_1_5(self) -> None:
        """Verify all 4 triggers fire simultaneously per SPEC AC-1.5:

        screen_to_sleep = 1.35 (> 1.2)
        recreational_to_screen = 0.65 (> 0.60)
        app_opens_per_day = 135 (> 120)
        weekend_diff = 2.8 (> 2.5)
        """
        # Configure input where metrics breach all 4 thresholds
        # screen=8.1, sleep=6.0 -> 8.1 / 6.0 = 1.35 (> 1.2)
        # social=4.0, gaming=1.265, screen=8.1 -> 5.265 / 8.1 = 0.65 (> 0.60)
        # opens=135 (> 120)
        # weekend=10.9, screen=8.1 -> weekend_diff = 2.8 (> 2.5)
        profile = BehavioralProfileInput(
            daily_screen_time_hours=8.1,
            sleep_hours=6.0,
            social_media_hours=4.0,
            gaming_hours=1.265,
            app_opens_per_day=135,
            notifications_per_day=150,
            weekend_screen_time=10.9,
            work_study_hours=2.0,
        )
        metrics = compute_metrics(profile)
        interventions = generate_clinical_interventions(profile, metrics)

        assert len(interventions) == 4
        assert any("Sleep Protection" in rec for rec in interventions)
        assert any("Recreation Audit" in rec for rec in interventions)
        assert any("Notification Hygiene" in rec for rec in interventions)
        assert any("Weekend Disconnect" in rec for rec in interventions)

    def test_sleep_protection_trigger_only(self) -> None:
        """Verify Sleep Protection fires if and only if screen_to_sleep_ratio > 1.2."""
        # Breached: screen 7.5, sleep 5.0 -> 1.5 > 1.2
        p_breach = BehavioralProfileInput(
            daily_screen_time_hours=7.5,
            sleep_hours=5.0,
            social_media_hours=1.0,
            gaming_hours=0.5,
            app_opens_per_day=50,
            notifications_per_day=50,
            weekend_screen_time=8.0,
        )
        m_breach = compute_metrics(p_breach)
        recs = generate_clinical_interventions(p_breach, m_breach)
        assert len(recs) == 1
        assert "Sleep Protection" in recs[0]
        assert recs[0] == RECOMMENDATION_SLEEP_PROTECTION

        # Boundary: ratio exactly 1.2 does not fire
        p_exact = BehavioralProfileInput(
            daily_screen_time_hours=6.0,
            sleep_hours=5.0,  # 6.0 / 5.0 = 1.20
            social_media_hours=1.0,
            gaming_hours=0.5,
            app_opens_per_day=50,
            notifications_per_day=50,
            weekend_screen_time=7.0,
        )
        m_exact = compute_metrics(p_exact)
        assert generate_clinical_interventions(p_exact, m_exact) == []

    def test_recreation_audit_trigger_only(self) -> None:
        """Verify Recreation Audit fires if and only if recreational_share > 0.60."""
        # Breached: social 3.5, gaming 1.5, screen 7.0 -> 5.0 / 7.0 = 0.714 > 0.60
        p_breach = BehavioralProfileInput(
            daily_screen_time_hours=7.0,
            sleep_hours=8.0,  # ratio 7/8 = 0.875 < 1.2
            social_media_hours=3.5,
            gaming_hours=1.5,
            app_opens_per_day=60,
            notifications_per_day=80,
            weekend_screen_time=8.0,  # diff 1.0 < 2.5
        )
        m_breach = compute_metrics(p_breach)
        recs = generate_clinical_interventions(p_breach, m_breach)
        assert len(recs) == 1
        assert "Recreation Audit" in recs[0]
        assert recs[0] == RECOMMENDATION_RECREATION_AUDIT

        # Boundary: share exactly 0.60 does not fire
        p_exact = BehavioralProfileInput(
            daily_screen_time_hours=10.0,
            sleep_hours=9.0,
            social_media_hours=4.0,
            gaming_hours=2.0,  # 6.0 / 10.0 = 0.60
            app_opens_per_day=60,
            notifications_per_day=80,
            weekend_screen_time=11.0,
        )
        m_exact = compute_metrics(p_exact)
        assert generate_clinical_interventions(p_exact, m_exact) == []

    def test_notification_hygiene_dual_triggers(self) -> None:
        """Verify Notification Hygiene fires on either app_opens > 120 OR notifications > 180."""
        # Test 1: opens > 120 (125 opens, 100 notifications)
        p_opens = BehavioralProfileInput(
            daily_screen_time_hours=6.0,
            sleep_hours=8.0,
            social_media_hours=1.0,
            gaming_hours=0.5,
            app_opens_per_day=125,
            notifications_per_day=100,
            weekend_screen_time=7.0,
        )
        recs_opens = generate_clinical_interventions(p_opens, compute_metrics(p_opens))
        assert len(recs_opens) == 1
        assert "Notification Hygiene" in recs_opens[0]
        assert recs_opens[0] == RECOMMENDATION_NOTIFICATION_HYGIENE

        # Test 2: notifications > 180 (80 opens, 190 notifications)
        p_notifs = BehavioralProfileInput(
            daily_screen_time_hours=6.0,
            sleep_hours=8.0,
            social_media_hours=1.0,
            gaming_hours=0.5,
            app_opens_per_day=80,
            notifications_per_day=190,
            weekend_screen_time=7.0,
        )
        recs_notifs = generate_clinical_interventions(p_notifs, compute_metrics(p_notifs))
        assert len(recs_notifs) == 1
        assert "Notification Hygiene" in recs_notifs[0]

        # Boundary: opens == 120 and notifs == 180 does not fire
        p_exact = BehavioralProfileInput(
            daily_screen_time_hours=6.0,
            sleep_hours=8.0,
            social_media_hours=1.0,
            gaming_hours=0.5,
            app_opens_per_day=120,
            notifications_per_day=180,
            weekend_screen_time=7.0,
        )
        assert generate_clinical_interventions(p_exact, compute_metrics(p_exact)) == []

    def test_weekend_disconnect_trigger_only(self) -> None:
        """Verify Weekend Disconnect fires if and only if weekend_surge_hours > 2.5."""
        # Breached: weekend 9.6, weekday 7.0 -> diff 2.6 > 2.5
        p_breach = BehavioralProfileInput(
            daily_screen_time_hours=7.0,
            sleep_hours=8.0,
            social_media_hours=1.5,
            gaming_hours=0.5,
            app_opens_per_day=50,
            notifications_per_day=60,
            weekend_screen_time=9.6,
        )
        m_breach = compute_metrics(p_breach)
        recs = generate_clinical_interventions(p_breach, m_breach)
        assert len(recs) == 1
        assert "Weekend Disconnect" in recs[0]
        assert recs[0] == RECOMMENDATION_WEEKEND_DISCONNECT

        # Boundary: diff exactly 2.5 does not fire
        p_exact = BehavioralProfileInput(
            daily_screen_time_hours=7.0,
            sleep_hours=8.0,
            social_media_hours=1.5,
            gaming_hours=0.5,
            app_opens_per_day=50,
            notifications_per_day=60,
            weekend_screen_time=9.5,  # diff 2.5
        )
        m_exact = compute_metrics(p_exact)
        assert generate_clinical_interventions(p_exact, m_exact) == []

    def test_zero_interventions_for_healthy_profile(self) -> None:
        """Verify no interventions generated for a balanced habit profile."""
        p_healthy = BehavioralProfileInput(
            daily_screen_time_hours=4.0,
            sleep_hours=8.0,
            social_media_hours=1.0,
            gaming_hours=0.5,
            app_opens_per_day=50,
            notifications_per_day=60,
            weekend_screen_time=5.0,
        )
        m_healthy = compute_metrics(p_healthy)
        assert generate_clinical_interventions(p_healthy, m_healthy) == []

    def test_generate_recommendations_compatibility_wrapper(self) -> None:
        """Verify generate_recommendations produces identical output to generate_clinical_interventions."""
        profile = BehavioralProfileInput(
            daily_screen_time_hours=9.0,
            sleep_hours=6.0,
            social_media_hours=6.0,
            gaming_hours=1.0,
            app_opens_per_day=130,
            notifications_per_day=200,
            weekend_screen_time=12.0,
        )
        metrics = compute_metrics(profile)
        direct = generate_clinical_interventions(profile, metrics)
        wrapped = generate_recommendations(profile, metrics, probability=0.85)
        assert direct == wrapped
        assert len(wrapped) == 4


class TestClassificationHelpers:
    """Validation test suite for risk tier, severity, and status label generation."""

    @pytest.mark.parametrize(
        ("prob", "expected_severity", "expected_tier", "expected_spectrum"),
        [
            (0.95, "High", "HIGH", "Elevated Risk Tier"),
            (0.70, "High", "HIGH", "Elevated Risk Tier"),
            (0.69, "Moderate", "MODERATE", "Compensatory Usage Pattern"),
            (0.40, "Moderate", "MODERATE", "Compensatory Usage Pattern"),
            (0.39, "Healthy", "LOW", "Balanced Habit Profile"),
            (0.10, "Healthy", "LOW", "Balanced Habit Profile"),
        ],
    )
    def test_severity_tier_and_spectrum(
        self,
        prob: float,
        expected_severity: str,
        expected_tier: str,
        expected_spectrum: str,
    ) -> None:
        assert classify_severity(prob) == expected_severity
        assert classify_risk_tier(prob) == expected_tier
        assert classify_clinical_spectrum(prob) == expected_spectrum

    def test_get_authoritative_status_label(self) -> None:
        """Verify authoritative status label consolidation per ADR-0006."""
        # Addicted states
        assert get_authoritative_status_label(True, 0.85) == "ADDICTION DETECTED • HIGH RISK"
        assert get_authoritative_status_label(True, 0.55) == "ADDICTION DETECTED • MODERATE RISK"
        assert get_authoritative_status_label(True, 0.35) == "ADDICTION DETECTED • LOW RISK"

        # Non-addicted / healthy states
        assert get_authoritative_status_label(False, 0.25) == "HEALTHY PATTERN • LOW RISK"
        assert get_authoritative_status_label(False, 0.45) == "HEALTHY PATTERN • MODERATE RISK"
        assert get_authoritative_status_label(False, 0.72) == "HEALTHY PATTERN • HIGH RISK"
