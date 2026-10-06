"""Pydantic v2 domain schemas and data contracts for Smartphone Addiction Prediction platform."""

from typing import Any, Literal
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)
from typing_extensions import Self


class BehavioralProfileInput(BaseModel):
    """Core input schema capturing demographic attributes, daily digital time budgets, and interaction volumes."""

    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(
        default=25,
        ge=18,
        le=35,
        description="Participant age in years (18-35)",
    )
    gender: Literal["Female", "Male", "Other"] = Field(
        default="Male",
        description="Self-reported participant gender",
    )
    stress_level: Literal["Low", "Medium", "High"] = Field(
        default="Medium",
        description="Self-reported chronic stress level",
    )
    academic_work_impact: Literal["No", "Yes"] = Field(
        default="Yes",
        description="Observed negative impact on academic or professional performance",
    )
    daily_screen_time_hours: float = Field(
        default=7.5,
        ge=0.0,
        le=24.0,
        validation_alias=AliasChoices("daily_screen_time_hours", "daily_screen_time"),
        description="Average weekday daily screen time in hours (0.0-24.0)",
    )
    social_media_hours: float = Field(
        default=3.5,
        ge=0.0,
        le=24.0,
        description="Daily social media screen time in hours (0.0-24.0)",
    )
    gaming_hours: float = Field(
        default=1.2,
        ge=0.0,
        le=24.0,
        description="Daily gaming screen time in hours (0.0-24.0)",
    )
    work_study_hours: float = Field(
        default=2.5,
        ge=0.0,
        le=24.0,
        description="Daily productive work or study screen time in hours (0.0-24.0)",
    )
    weekend_screen_time: float = Field(
        default=9.5,
        ge=0.0,
        le=24.0,
        validation_alias=AliasChoices("weekend_screen_time", "weekend_screen_time_hours"),
        description="Average weekend daily screen time in hours (0.0-24.0)",
    )
    sleep_hours: float = Field(
        default=6.8,
        ge=1.0,
        le=18.0,
        validation_alias=AliasChoices("sleep_hours", "sleep_duration_hours"),
        description="Average daily sleep duration in hours (1.0-18.0)",
    )
    notifications_per_day: int = Field(
        default=140,
        ge=0,
        le=500,
        description="Total notifications received per day (0-500)",
    )
    app_opens_per_day: int = Field(
        default=100,
        ge=0,
        le=500,
        description="Total app open and device unlock events per day (0-500)",
    )
    threshold: float = Field(
        default=0.50,
        ge=0.05,
        le=0.95,
        validation_alias=AliasChoices("threshold", "decision_threshold"),
        description="Classification decision threshold tau (0.05-0.95)",
    )
    exceeds_daily_budget: bool = Field(
        default=False,
        description="Flag indicating if total accounted hours exceed 24.0h physiological limit",
    )

    @model_validator(mode="after")
    def _validate_physiological_budget(self) -> Self:
        """Verify daily physiological budget (screen + work/study + sleep <= 24.0h).

        Sets non-fatal warning flag exceeds_daily_budget without blocking inference.
        """
        total = self.daily_screen_time_hours + self.work_study_hours + self.sleep_hours
        if total > 24.0 or (self.daily_screen_time_hours + self.sleep_hours > 24.0):
            self.exceeds_daily_budget = True
        return self

    @property
    def decision_threshold(self) -> float:
        """Alias property for classification threshold tau."""
        return self.threshold

    @property
    def total_accounted_hours(self) -> float:
        """Sum of screen, work/study, and sleep hours."""
        return round(self.daily_screen_time_hours + self.work_study_hours + self.sleep_hours, 4)


class BehavioralMetrics(BaseModel):
    """Derived behavioral ratios and habit fragmentation indicators."""

    model_config = ConfigDict(populate_by_name=True)

    screen_to_sleep_ratio: float = Field(
        ...,
        validation_alias=AliasChoices("screen_to_sleep_ratio", "screen_to_sleep"),
        description="Ratio of daily screen time to sleep duration (screen / sleep)",
    )
    recreational_share: float = Field(
        ...,
        validation_alias=AliasChoices("recreational_share", "recreational_to_screen"),
        description="Proportion of screen time spent on recreational activities (social + gaming)",
    )
    avg_unlock_minutes: float = Field(
        ...,
        description="Estimated average continuous minutes between unlock events",
    )
    weekend_surge_hours: float = Field(
        ...,
        validation_alias=AliasChoices("weekend_surge_hours", "weekend_diff"),
        description="Difference between weekend and weekday daily screen time in hours",
    )
    total_accounted_hours: float = Field(
        ...,
        description="Sum of screen, work/study, and sleep hours",
    )
    exceeds_daily_budget: bool = Field(
        ...,
        description="Flag indicating if total accounted hours exceed 24.0h physiological limit",
    )

    @computed_field
    @property
    def screen_to_sleep(self) -> float:
        """Legacy / SPEC AC-1.4 alias for screen_to_sleep_ratio."""
        return self.screen_to_sleep_ratio

    @computed_field
    @property
    def recreational_to_screen(self) -> float:
        """Legacy / SPEC AC-1.4 alias for recreational_share."""
        return self.recreational_share

    @computed_field
    @property
    def weekend_diff(self) -> float:
        """Legacy / SPEC AC-1.4 alias for weekend_surge_hours."""
        return self.weekend_surge_hours


class PredictionResponse(BaseModel):
    """Diagnostic risk assessment returned by inference controllers."""

    model_config = ConfigDict(populate_by_name=True)

    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("probability", "addiction_probability"),
        description="Predicted addiction probability in range [0.0, 1.0]",
    )
    prediction: int = Field(
        ...,
        ge=0,
        le=1,
        description="Binary classification prediction (1 if probability >= threshold, else 0)",
    )
    classification: str = Field(
        ...,
        description="Diagnostic classification ('ADDICTION DETECTED' or 'HEALTHY')",
    )
    severity: str = Field(
        ...,
        validation_alias=AliasChoices("severity", "risk_tier"),
        description="Diagnostic severity tier ('High', 'Moderate', 'Healthy' or uppercase)",
    )
    status_label: str = Field(
        ...,
        description="Consolidated clinical status label (e.g., ADR-0006 authoritative status pill)",
    )
    ratios: BehavioralMetrics = Field(
        ...,
        validation_alias=AliasChoices("ratios", "metrics"),
        description="Granular derived behavioral metrics and ratios",
    )
    interventions: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("interventions", "recommendations"),
        description="Prioritized clinical recommendation action items",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Total inference computation duration in milliseconds",
    )
    decision_threshold: float = Field(
        default=0.50,
        ge=0.05,
        le=0.95,
        validation_alias=AliasChoices("decision_threshold", "threshold"),
        description="Evaluated classification threshold tau",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_prediction_and_classification(cls, data: Any) -> Any:
        """Coerce missing fields when initialized with alternate parameter conventions."""
        if isinstance(data, dict):
            # Coerce prediction if missing but is_addicted or probabilities are supplied
            if "prediction" not in data:
                if "is_addicted" in data:
                    data["prediction"] = 1 if data["is_addicted"] else 0
                elif "probability" in data or "addiction_probability" in data:
                    prob = data.get("probability", data.get("addiction_probability"))
                    tau = data.get("decision_threshold", data.get("threshold", 0.50))
                    if prob is not None:
                        data["prediction"] = 1 if prob >= tau else 0

            # Coerce classification if missing
            if "classification" not in data and "prediction" in data:
                data["classification"] = "ADDICTION DETECTED" if data["prediction"] == 1 else "HEALTHY"

            # Coerce severity if missing but risk_tier is supplied
            if "severity" not in data and "risk_tier" in data:
                data["severity"] = data["risk_tier"]

        return data

    @computed_field
    @property
    def addiction_probability(self) -> float:
        """Alias for probability."""
        return self.probability

    @computed_field
    @property
    def is_addicted(self) -> bool:
        """Boolean equivalent of prediction == 1."""
        return bool(self.prediction == 1)

    @computed_field
    @property
    def risk_tier(self) -> str:
        """Uppercase risk tier ('HIGH', 'MODERATE', 'LOW')."""
        s = self.severity.upper()
        if "HIGH" in s:
            return "HIGH"
        elif "MOD" in s:
            return "MODERATE"
        else:
            return "LOW"

    @computed_field
    @property
    def metrics(self) -> BehavioralMetrics:
        """Alias for ratios."""
        return self.ratios

    @computed_field
    @property
    def recommendations(self) -> list[str]:
        """Alias for interventions."""
        return self.interventions


class HealthResponse(BaseModel):
    """System diagnostics, telemetry, and model readiness status response."""

    model_config = ConfigDict(populate_by_name=True)

    status: str = Field(
        default="healthy",
        description="Service health status ('healthy' or 'degraded')",
    )
    model_loaded: bool = Field(
        default=True,
        description="True if LightGBM model is initialized in memory",
    )
    version: str = Field(
        default="1.0.0",
        description="API service version string",
    )
    model_family: str = Field(
        default="LightGBM",
        description="Champion model architecture family",
    )
    fold: int = Field(
        default=1,
        description="Model fold checkpoint identifier",
    )
    priors_loaded: bool = Field(
        default=True,
        description="True if population priors cache is loaded",
    )
    cohort_priors_cached: bool = Field(
        default=True,
        description="True if cohort priors are cached in memory",
    )
    cohort_cache_loaded: bool = Field(
        default=True,
        description="True if cohort analytics summary is loaded",
    )
    mock_mode: bool = Field(
        default=False,
        description="True if running under mock inference adapter",
    )
    uptime_seconds: float = Field(
        default=0.0,
        description="Server process uptime duration in seconds",
    )
