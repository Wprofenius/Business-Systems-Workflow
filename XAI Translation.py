#!/usr/bin/env python3
"""
Enterprise XAI Translation Engine
=================================

Purpose
-------
This script demonstrates a production-style Explainable AI translation layer for
CRM users. It takes raw model outputs, such as SHAP-like feature contributions,
and translates them into clear business explanations that can be used in a CRM
dashboard, customer-success workflow, or account-review process.

The engine produces:

1. A readable business narrative for non-technical users.
2. A structured dashboard/API payload.
3. Auditable metadata about the model, version, score, and explained drivers.
4. Validation safeguards so malformed model outputs are not explained as if they
   were reliable.

Important Assumption
--------------------
The feature_weights dictionary is treated as additive contribution data:

- A positive contribution increases the final score.
- A negative contribution decreases the final score.

Whether that increase is good or bad depends on the target metric. For example,
a higher Churn Risk score is unfavorable, while a higher Conversion Probability
score is favorable.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import json
import logging
import math
from collections.abc import Mapping
from typing import Any, Final


# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("xai_translation_engine")


# =============================================================================
# Enumerations and Constants
# =============================================================================


class MetricType(enum.Enum):
    """Supported business-facing prediction metrics."""

    CHURN_RISK = "Churn Risk"
    CONVERSION_PROBABILITY = "Conversion Probability"
    EXPANSION_POTENTIAL = "Expansion Potential"
    RENEWAL_CONFIDENCE = "Renewal Confidence"
    UNKNOWN = "Unknown Metric"


class DriverEffect(enum.Enum):
    """Direction of a feature's mathematical contribution to the score."""

    INCREASES_SCORE = "increases_score"
    DECREASES_SCORE = "decreases_score"
    NEUTRAL = "neutral"


class BusinessPolarity(enum.Enum):
    """
    Business meaning of a high final score.

    Examples
    --------
    Churn Risk:
        High score is unfavorable.

    Conversion Probability:
        High score is favorable.
    """

    FAVORABLE_WHEN_HIGH = "favorable_when_high"
    UNFAVORABLE_WHEN_HIGH = "unfavorable_when_high"
    NEUTRAL = "neutral"


class SeverityLevel(enum.Enum):
    """CRM-facing severity label."""

    CRITICAL = "Critical"
    ELEVATED = "Elevated"
    MODERATE = "Moderate"
    LOW = "Low"


class ImpactStrength(enum.Enum):
    """Readable magnitude bucket for a feature contribution."""

    VERY_STRONG = "very strong"
    STRONG = "strong"
    MODERATE = "moderate"
    LIGHT = "light"


class RecommendationPriority(enum.Enum):
    """CRM-facing action priority."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


SCORE_MIN: Final[float] = 0.0
SCORE_MAX: Final[float] = 100.0


# =============================================================================
# Data Models
# =============================================================================


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureMetadata:
    """
    Business metadata for a model feature.

    Attributes
    ----------
    label:
        Human-readable feature name.
    business_description:
        Plain-English explanation of what the feature measures.
    favorable_when_high:
        Whether a high raw feature value is generally favorable. This is used
        as background metadata, while contribution direction is interpreted
        against the target metric.
    recommended_action:
        Suggested action when this feature appears as an unfavorable driver.
    """

    label: str
    business_description: str
    favorable_when_high: bool | None
    recommended_action: str


@dataclasses.dataclass(frozen=True, slots=True)
class ModelPrediction:
    """
    Raw model output to be translated.

    Attributes
    ----------
    account_id:
        CRM account identifier.
    target_metric:
        Business metric being predicted, such as Churn Risk.
    final_score:
        Score on a 0-100 scale.
    feature_weights:
        Contribution map where each feature increased or decreased the score.
    baseline_score:
        Optional model baseline before feature contributions.
    model_name:
        Model family or registered model name.
    model_version:
        Version used to generate this prediction.
    prediction_timestamp:
        Optional timestamp from the model run.
    """

    account_id: str
    target_metric: str
    final_score: float
    feature_weights: dict[str, float]
    baseline_score: float | None = None
    model_name: str = "unspecified_model"
    model_version: str = "unspecified_version"
    prediction_timestamp: dt.datetime | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class DriverExplanation:
    """Structured explanation for one model driver."""

    feature_key: str
    feature_label: str
    contribution: float
    effect_on_score: DriverEffect
    impact_strength: ImpactStrength
    business_interpretation: str
    is_favorable_business_signal: bool | None
    recommended_action: str | None

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe driver payload."""
        return {
            "feature_key": self.feature_key,
            "feature_label": self.feature_label,
            "contribution": round(self.contribution, 4),
            "effect_on_score": self.effect_on_score.value,
            "impact_strength": self.impact_strength.value,
            "business_interpretation": self.business_interpretation,
            "is_favorable_business_signal": self.is_favorable_business_signal,
            "recommended_action": self.recommended_action,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ExplanationResult:
    """Complete explanation package for one model prediction."""

    account_id: str
    target_metric: str
    final_score: float
    severity: SeverityLevel
    priority: RecommendationPriority
    narrative: str
    top_drivers: list[DriverExplanation]
    unknown_feature_count: int
    audit_metadata: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe dashboard/API payload."""
        return {
            "account_id": self.account_id,
            "target_metric": self.target_metric,
            "final_score": round(self.final_score, 2),
            "severity": self.severity.value,
            "priority": self.priority.value,
            "narrative": self.narrative,
            "top_drivers": [driver.to_payload() for driver in self.top_drivers],
            "unknown_feature_count": self.unknown_feature_count,
            "audit_metadata": self.audit_metadata,
        }


# =============================================================================
# Exceptions
# =============================================================================


class PredictionValidationError(ValueError):
    """Raised when the incoming model prediction is unsafe or malformed."""


# =============================================================================
# Feature and Metric Configuration
# =============================================================================


FEATURE_LEXICON: Final[dict[str, FeatureMetadata]] = {
    "support_tickets_90d": FeatureMetadata(
        label="recent support ticket volume",
        business_description="the number of support tickets opened in the last 90 days",
        favorable_when_high=False,
        recommended_action=(
            "Review unresolved support issues and confirm whether the customer has "
            "a clear escalation path."
        ),
    ),
    "days_since_last_login": FeatureMetadata(
        label="account inactivity",
        business_description="the number of days since the account last logged in",
        favorable_when_high=False,
        recommended_action=(
            "Ask the account owner to confirm whether the customer is still actively "
            "using the product."
        ),
    ),
    "contract_value_usd": FeatureMetadata(
        label="contract value",
        business_description="the total value of the customer's active contract",
        favorable_when_high=None,
        recommended_action=(
            "Use contract value to prioritize outreach, but do not treat it alone "
            "as a health signal."
        ),
    ),
    "executive_sponsor_active": FeatureMetadata(
        label="executive sponsorship",
        business_description="whether an executive sponsor is actively engaged with the account",
        favorable_when_high=True,
        recommended_action=(
            "Identify or re-engage an executive sponsor before the next renewal conversation."
        ),
    ),
    "feature_adoption_rate": FeatureMetadata(
        label="core feature adoption",
        business_description="the percentage of relevant product features actively used by the account",
        favorable_when_high=True,
        recommended_action=(
            "Share adoption trends with the customer and reinforce the workflows "
            "that are already creating value."
        ),
    ),
    "training_completion_rate": FeatureMetadata(
        label="training completion",
        business_description="the share of assigned users who completed onboarding or enablement training",
        favorable_when_high=True,
        recommended_action="Offer targeted enablement for users who have not completed training.",
    ),
    "renewal_days_remaining": FeatureMetadata(
        label="time remaining before renewal",
        business_description="the number of days remaining before the contract renewal date",
        favorable_when_high=True,
        recommended_action=(
            "Begin renewal preparation and document open risks before the commercial conversation."
        ),
    ),
}


METRIC_POLARITY: Final[dict[str, BusinessPolarity]] = {
    MetricType.CHURN_RISK.value: BusinessPolarity.UNFAVORABLE_WHEN_HIGH,
    MetricType.CONVERSION_PROBABILITY.value: BusinessPolarity.FAVORABLE_WHEN_HIGH,
    MetricType.EXPANSION_POTENTIAL.value: BusinessPolarity.FAVORABLE_WHEN_HIGH,
    MetricType.RENEWAL_CONFIDENCE.value: BusinessPolarity.FAVORABLE_WHEN_HIGH,
}


# =============================================================================
# Utility Functions
# =============================================================================


def utc_now() -> dt.datetime:
    """Return a timezone-aware UTC timestamp."""
    return dt.datetime.now(dt.UTC)


def clean_metric_name(value: str) -> str:
    """Normalize a metric name while preserving readable business wording."""
    cleaned = " ".join(str(value).strip().split())
    return cleaned or MetricType.UNKNOWN.value


def validate_score(score: float, field_name: str = "score") -> None:
    """Validate that a score is numeric, finite, and within the expected range."""
    if not isinstance(score, (int, float)):
        raise PredictionValidationError(f"{field_name} must be numeric; received {score!r}.")

    numeric_score = float(score)

    if not math.isfinite(numeric_score):
        raise PredictionValidationError(f"{field_name} must be finite; received {score!r}.")

    if not SCORE_MIN <= numeric_score <= SCORE_MAX:
        raise PredictionValidationError(
            f"{field_name} must be between {SCORE_MIN:.0f} and {SCORE_MAX:.0f}; "
            f"received {score!r}."
        )


def validate_feature_weights(feature_weights: Mapping[str, float]) -> None:
    """Validate that feature weights are safe to explain."""
    if not isinstance(feature_weights, Mapping):
        raise PredictionValidationError(
            "feature_weights must be a mapping of feature names to numeric weights."
        )

    if not feature_weights:
        raise PredictionValidationError("feature_weights cannot be empty.")

    for feature_name, weight in feature_weights.items():
        if not isinstance(feature_name, str) or not feature_name.strip():
            raise PredictionValidationError(
                f"feature name must be a non-empty string; received {feature_name!r}."
            )

        if not isinstance(weight, (int, float)) or not math.isfinite(float(weight)):
            raise PredictionValidationError(
                f"feature weight for {feature_name!r} must be finite and numeric; "
                f"received {weight!r}."
            )


def score_to_severity(score: float, polarity: BusinessPolarity) -> SeverityLevel:
    """
    Convert a model score into a business severity label.

    For unfavorable metrics, such as Churn Risk, high scores mean higher severity.
    For favorable metrics, such as Conversion Probability, low scores mean higher
    severity because the account is underperforming.
    """

    if polarity == BusinessPolarity.FAVORABLE_WHEN_HIGH:
        if score < 25:
            return SeverityLevel.CRITICAL
        if score < 50:
            return SeverityLevel.ELEVATED
        if score < 75:
            return SeverityLevel.MODERATE
        return SeverityLevel.LOW

    if polarity == BusinessPolarity.UNFAVORABLE_WHEN_HIGH:
        if score >= 75:
            return SeverityLevel.CRITICAL
        if score >= 50:
            return SeverityLevel.ELEVATED
        if score >= 25:
            return SeverityLevel.MODERATE
        return SeverityLevel.LOW

    if score >= 75:
        return SeverityLevel.ELEVATED
    if score >= 50:
        return SeverityLevel.MODERATE
    return SeverityLevel.LOW


def severity_to_priority(severity: SeverityLevel) -> RecommendationPriority:
    """Map severity into a CRM action priority."""
    if severity == SeverityLevel.CRITICAL:
        return RecommendationPriority.HIGH

    if severity in {SeverityLevel.ELEVATED, SeverityLevel.MODERATE}:
        return RecommendationPriority.MEDIUM

    return RecommendationPriority.LOW


def contribution_to_strength(weight: float) -> ImpactStrength:
    """Bucket a numeric contribution into a readable impact label."""
    magnitude = abs(weight)

    if magnitude >= 20:
        return ImpactStrength.VERY_STRONG
    if magnitude >= 10:
        return ImpactStrength.STRONG
    if magnitude >= 5:
        return ImpactStrength.MODERATE
    return ImpactStrength.LIGHT


def contribution_effect(weight: float) -> DriverEffect:
    """Translate a numeric contribution into a score movement direction."""
    if weight > 0:
        return DriverEffect.INCREASES_SCORE
    if weight < 0:
        return DriverEffect.DECREASES_SCORE
    return DriverEffect.NEUTRAL


def infer_business_signal(
    contribution: float,
    metric_polarity: BusinessPolarity,
) -> bool | None:
    """
    Determine whether a feature contribution is favorable from a business perspective.

    Positive contribution means the feature raised the final score. Negative
    contribution means the feature lowered the final score. Whether that is good
    depends on the target metric.
    """

    if contribution == 0 or metric_polarity == BusinessPolarity.NEUTRAL:
        return None

    if metric_polarity == BusinessPolarity.FAVORABLE_WHEN_HIGH:
        return contribution > 0

    if metric_polarity == BusinessPolarity.UNFAVORABLE_WHEN_HIGH:
        return contribution < 0

    return None


def join_phrases_naturally(phrases: list[str]) -> str:
    """Join short phrases into natural prose."""
    if not phrases:
        return ""

    if len(phrases) == 1:
        return phrases[0]

    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"

    return f"{', '.join(phrases[:-1])}, and {phrases[-1]}"


# =============================================================================
# Narrative Translation Engine
# =============================================================================


class NarrativeTranslator:
    """
    Translates mathematical model outputs into business-facing explanations.

    The class separates validation, driver selection, driver interpretation,
    narrative generation, and structured payload generation.
    """

    def __init__(
        self,
        prediction: ModelPrediction,
        feature_lexicon: Mapping[str, FeatureMetadata] | None = None,
        max_drivers: int = 3,
    ) -> None:
        self.prediction = prediction
        self.feature_lexicon = feature_lexicon or FEATURE_LEXICON
        self.max_drivers = max(1, max_drivers)

    def _validate_prediction(self) -> None:
        """Validate the raw prediction before attempting explanation."""
        if not isinstance(self.prediction.account_id, str) or not self.prediction.account_id.strip():
            raise PredictionValidationError("account_id is required.")

        if not isinstance(self.prediction.target_metric, str) or not self.prediction.target_metric.strip():
            raise PredictionValidationError("target_metric is required.")

        validate_score(self.prediction.final_score, "final_score")
        validate_feature_weights(self.prediction.feature_weights)

        if self.prediction.baseline_score is not None:
            validate_score(self.prediction.baseline_score, "baseline_score")

    def _metric_polarity(self) -> BusinessPolarity:
        """Return the configured business polarity for the target metric."""
        metric = clean_metric_name(self.prediction.target_metric)
        return METRIC_POLARITY.get(metric, BusinessPolarity.NEUTRAL)

    def _sorted_driver_items(self) -> list[tuple[str, float]]:
        """Sort features by absolute contribution magnitude."""
        return sorted(
            (
                (feature_key.strip(), float(weight))
                for feature_key, weight in self.prediction.feature_weights.items()
            ),
            key=lambda item: abs(item[1]),
            reverse=True,
        )

    def _metadata_for_feature(self, feature_key: str) -> FeatureMetadata:
        """Return configured feature metadata or a safe fallback."""
        if feature_key in self.feature_lexicon:
            return self.feature_lexicon[feature_key]

        readable_name = feature_key.replace("_", " ").strip() or "unknown feature"

        return FeatureMetadata(
            label=readable_name,
            business_description=(
                "an unmapped model feature that does not yet have approved business metadata"
            ),
            favorable_when_high=None,
            recommended_action=(
                "Review this unmapped feature with the data science team before using it "
                "for customer-facing decisions."
            ),
        )

    def _explain_driver(
        self,
        feature_key: str,
        contribution: float,
        metric_polarity: BusinessPolarity,
    ) -> DriverExplanation:
        """Create a structured explanation for a single feature."""
        metadata = self._metadata_for_feature(feature_key)
        effect = contribution_effect(contribution)
        strength = contribution_to_strength(contribution)
        is_favorable_signal = infer_business_signal(contribution, metric_polarity)

        if effect == DriverEffect.INCREASES_SCORE:
            score_language = "increased the score"
        elif effect == DriverEffect.DECREASES_SCORE:
            score_language = "reduced the score"
        else:
            score_language = "had no measurable effect on the score"

        if is_favorable_signal is True:
            business_language = "This is a favorable business signal."
            recommended_action = None
        elif is_favorable_signal is False:
            business_language = "This is an unfavorable business signal."
            recommended_action = metadata.recommended_action
        else:
            business_language = (
                "This should be treated as contextual rather than clearly favorable "
                "or unfavorable."
            )
            recommended_action = metadata.recommended_action

        interpretation = (
            f"{metadata.label.capitalize()} {score_language} by "
            f"{abs(contribution):.1f} points. {business_language}"
        )

        return DriverExplanation(
            feature_key=feature_key,
            feature_label=metadata.label,
            contribution=contribution,
            effect_on_score=effect,
            impact_strength=strength,
            business_interpretation=interpretation,
            is_favorable_business_signal=is_favorable_signal,
            recommended_action=recommended_action,
        )

    def _generate_opening_sentence(
        self,
        severity: SeverityLevel,
        priority: RecommendationPriority,
    ) -> str:
        """Create the opening CRM-facing sentence."""
        metric = clean_metric_name(self.prediction.target_metric)
        score = float(self.prediction.final_score)

        if severity == SeverityLevel.CRITICAL:
            action_language = "requires immediate review"
        elif severity == SeverityLevel.ELEVATED:
            action_language = "should be reviewed soon"
        elif severity == SeverityLevel.MODERATE:
            action_language = "shows factors worth monitoring"
        else:
            action_language = "appears stable based on the current model output"

        return (
            f"Account {self.prediction.account_id} has a {metric} score of {score:.1f}. "
            f"The current severity is {severity.value.lower()}, so the recommended CRM "
            f"priority is {priority.value.lower()}; this account {action_language}."
        )

    def _generate_driver_sentence(self, drivers: list[DriverExplanation]) -> str:
        """Create a readable sentence summarizing the strongest drivers."""
        if not drivers:
            return "The model did not isolate any meaningful feature drivers for this prediction."

        driver_phrases: list[str] = []

        for driver in drivers:
            if driver.effect_on_score == DriverEffect.INCREASES_SCORE:
                movement = "raised the score"
            elif driver.effect_on_score == DriverEffect.DECREASES_SCORE:
                movement = "lowered the score"
            else:
                movement = "had limited effect on the score"

            driver_phrases.append(
                f"{driver.feature_label} {movement} by {abs(driver.contribution):.1f} points"
            )

        return f"The strongest drivers were {join_phrases_naturally(driver_phrases)}."

    def _generate_action_sentence(self, drivers: list[DriverExplanation]) -> str:
        """Create a concise recommended-action sentence."""
        recommended_actions = [
            driver.recommended_action
            for driver in drivers
            if driver.recommended_action and driver.is_favorable_business_signal is not True
        ]

        if not recommended_actions:
            return "No immediate intervention is suggested by the top model drivers."

        return f"Recommended next step: {recommended_actions[0]}"

    def explain(self) -> ExplanationResult:
        """Validate, translate, and return a complete explanation result."""
        self._validate_prediction()

        metric = clean_metric_name(self.prediction.target_metric)
        metric_polarity = self._metric_polarity()
        severity = score_to_severity(float(self.prediction.final_score), metric_polarity)
        priority = severity_to_priority(severity)

        sorted_drivers = self._sorted_driver_items()
        top_driver_items = sorted_drivers[: self.max_drivers]

        driver_explanations = [
            self._explain_driver(feature_key, contribution, metric_polarity)
            for feature_key, contribution in top_driver_items
        ]

        unknown_feature_count = sum(
            1
            for feature_key in self.prediction.feature_weights
            if feature_key not in self.feature_lexicon
        )

        narrative = " ".join(
            [
                self._generate_opening_sentence(severity, priority),
                self._generate_driver_sentence(driver_explanations),
                self._generate_action_sentence(driver_explanations),
            ]
        )

        prediction_timestamp = self.prediction.prediction_timestamp or utc_now()

        audit_metadata = {
            "model_name": self.prediction.model_name,
            "model_version": self.prediction.model_version,
            "prediction_timestamp": prediction_timestamp.isoformat(),
            "baseline_score": self.prediction.baseline_score,
            "explained_feature_count": len(driver_explanations),
            "total_feature_count": len(self.prediction.feature_weights),
            "unknown_feature_count": unknown_feature_count,
            "metric_polarity": metric_polarity.value,
        }

        logger.info(
            "Generated XAI explanation for account %s using %s/%s.",
            self.prediction.account_id,
            self.prediction.model_name,
            self.prediction.model_version,
        )

        return ExplanationResult(
            account_id=self.prediction.account_id,
            target_metric=metric,
            final_score=float(self.prediction.final_score),
            severity=severity,
            priority=priority,
            narrative=narrative,
            top_drivers=driver_explanations,
            unknown_feature_count=unknown_feature_count,
            audit_metadata=audit_metadata,
        )


# =============================================================================
# Demonstration Data
# =============================================================================


def build_demo_predictions() -> list[ModelPrediction]:
    """Create sample model outputs for demonstration."""
    return [
        ModelPrediction(
            account_id="ACC-8842-V",
            target_metric="Churn Risk",
            final_score=84.2,
            baseline_score=50.0,
            model_name="xgboost_churn_classifier",
            model_version="2026.06.1",
            prediction_timestamp=utc_now(),
            feature_weights={
                "days_since_last_login": 22.5,
                "support_tickets_90d": 18.1,
                "feature_adoption_rate": -12.0,
                "executive_sponsor_active": 8.4,
                "contract_value_usd": 1.2,
            },
        ),
        ModelPrediction(
            account_id="ACC-5530",
            target_metric="Conversion Probability",
            final_score=72.8,
            baseline_score=45.0,
            model_name="gradient_boosted_conversion_model",
            model_version="2026.06.1",
            prediction_timestamp=utc_now(),
            feature_weights={
                "feature_adoption_rate": 19.4,
                "training_completion_rate": 11.6,
                "support_tickets_90d": -4.1,
                "unknown_partner_signal": 7.7,
            },
        ),
    ]


# =============================================================================
# Main Execution
# =============================================================================


def main() -> None:
    """Run the XAI translation demonstration."""
    logger.info("Initializing Enterprise XAI Translation Engine.")

    predictions = build_demo_predictions()
    explanation_payloads: list[dict[str, Any]] = []

    for prediction in predictions:
        translator = NarrativeTranslator(prediction, max_drivers=3)

        try:
            explanation = translator.explain()
            explanation_payloads.append(explanation.to_payload())

            print("\n" + "=" * 80)
            print("RAW MODEL OUTPUT")
            print("=" * 80)
            print(
                json.dumps(
                    {
                        "account_id": prediction.account_id,
                        "target_metric": prediction.target_metric,
                        "final_score": prediction.final_score,
                        "baseline_score": prediction.baseline_score,
                        "feature_weights": prediction.feature_weights,
                        "model_name": prediction.model_name,
                        "model_version": prediction.model_version,
                    },
                    indent=2,
                )
            )

            print("\n" + "=" * 80)
            print("XAI TRANSLATED NARRATIVE")
            print("=" * 80)
            print(explanation.narrative)

            print("\n" + "=" * 80)
            print("STRUCTURED DASHBOARD PAYLOAD")
            print("=" * 80)
            print(json.dumps(explanation.to_payload(), indent=2))

        except PredictionValidationError as exc:
            logger.error(
                "Prediction for account %s could not be explained: %s",
                prediction.account_id,
                exc,
            )

    print("\n" + "=" * 80)
    print("BATCH EXPLANATION PAYLOAD")
    print("=" * 80)
    print(json.dumps(explanation_payloads, indent=2))


if __name__ == "__main__":
    main()
