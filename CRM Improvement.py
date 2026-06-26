#!/usr/bin/env python3
"""
Enterprise CRM Data Integration & Integrity Validation Pipeline
===============================================================

Purpose
-------
This script demonstrates a production-style CRM data pipeline that:

1. Ingests records from inconsistent legacy/source systems.
2. Normalizes account and opportunity data into predictable internal schemas.
3. Validates structural, relational, and business-rule integrity.
4. Quarantines records that should not be sent to a CRM API.
5. Produces JSON-safe payloads for downstream batch upsert/insert operations.

The design intentionally separates normalization, validation, quarantine, and
payload compilation so that each part is easier to test, explain, and maintain.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import json
import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any, Final


# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("crm_pipeline")


# =============================================================================
# Domain Models
# =============================================================================


class AccountTier(enum.Enum):
    """CRM-facing customer segmentation tiers."""

    ENTERPRISE = "Enterprise"
    MID_MARKET = "Mid-Market"
    SMB = "SMB"
    UNKNOWN = "Unknown"


@dataclasses.dataclass(frozen=True, slots=True)
class IngestedRecord:
    """
    Raw record wrapper used before normalization.

    Attributes
    ----------
    source_id:
        Unique identifier from the originating system.
    source_system:
        Name of the upstream system that produced the record.
    payload:
        Untrusted raw source data.
    timestamp:
        Time the record was received or exported.
    """

    source_id: str
    source_system: str
    payload: dict[str, Any]
    timestamp: dt.datetime


@dataclasses.dataclass(frozen=True, slots=True)
class NormalizedAccount:
    """Validated account shape used before CRM account upsert."""

    account_id: str
    legal_name: str
    domain: str
    annual_revenue: float
    tier: AccountTier
    billing_country: str
    is_active: bool

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe account payload."""
        return {
            "account_id": self.account_id,
            "legal_name": self.legal_name,
            "domain": self.domain,
            "annual_revenue": self.annual_revenue,
            "tier": self.tier.value,
            "billing_country": self.billing_country,
            "is_active": self.is_active,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class NormalizedOpportunity:
    """Validated opportunity shape used before CRM opportunity insert."""

    opportunity_id: str
    associated_account_id: str
    title: str
    contract_value: float
    currency_code: str
    stage: str
    close_date: dt.date

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe opportunity payload."""
        return {
            "opportunity_id": self.opportunity_id,
            "associated_account_id": self.associated_account_id,
            "title": self.title,
            "contract_value": self.contract_value,
            "currency_code": self.currency_code,
            "stage": self.stage,
            "close_date": self.close_date.isoformat(),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class QuarantineEntry:
    """Structured record of a rejected source payload."""

    source_id: str
    originating_system: str
    quarantine_timestamp: str
    failure_reasons: list[str]
    raw_data_dump: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(slots=True)
class PipelineMetrics:
    """Mutable counters for observability and execution reporting."""

    account_records_seen: int = 0
    opportunity_records_seen: int = 0
    accounts_verified: int = 0
    opportunities_verified: int = 0
    records_quarantined: int = 0

    @property
    def total_records_seen(self) -> int:
        return self.account_records_seen + self.opportunity_records_seen

    def to_payload(self) -> dict[str, int]:
        return {
            "account_records_seen": self.account_records_seen,
            "opportunity_records_seen": self.opportunity_records_seen,
            "total_records_seen": self.total_records_seen,
            "accounts_verified": self.accounts_verified,
            "opportunities_verified": self.opportunities_verified,
            "records_quarantined": self.records_quarantined,
        }


# =============================================================================
# Exceptions
# =============================================================================


class NormalizationError(ValueError):
    """Raised when raw source data cannot be safely converted."""


# =============================================================================
# Normalization Helpers
# =============================================================================


class Normalizer:
    """
    Converts inconsistent source values into stable Python types.

    These functions intentionally fail closed. In other words, questionable
    values are quarantined instead of being silently guessed into the payload.
    """

    TRUE_VALUES: Final[set[str]] = {"true", "t", "yes", "y", "1", "active", "enabled"}
    FALSE_VALUES: Final[set[str]] = {"false", "f", "no", "n", "0", "inactive", "disabled"}
    DATE_FORMATS: Final[tuple[str, ...]] = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y")

    @staticmethod
    def first_present(payload: Mapping[str, Any], keys: Iterable[str], default: Any = None) -> Any:
        """
        Return the first meaningful value from a source payload.

        This avoids the common `a or b or c` weakness where valid values like
        0 or False can be accidentally skipped.
        """
        for key in keys:
            if key in payload and payload[key] is not None and payload[key] != "":
                return payload[key]
        return default

    @staticmethod
    def clean_string(value: Any) -> str:
        """Convert a value into a stripped string without returning 'None'."""
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def parse_float(value: Any, field_name: str) -> float:
        """Parse a numeric field and raise a clear normalization error on failure."""
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise NormalizationError(f"{field_name} must be numeric; received {value!r}.") from exc

    @classmethod
    def parse_bool(cls, value: Any, field_name: str = "boolean field") -> bool:
        """Parse common boolean representations from legacy systems."""
        if isinstance(value, bool):
            return value

        if value is None:
            return True

        if isinstance(value, (int, float)):
            return value != 0

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in cls.TRUE_VALUES:
                return True
            if normalized in cls.FALSE_VALUES:
                return False

        raise NormalizationError(f"{field_name} must be boolean-like; received {value!r}.")

    @classmethod
    def parse_date(cls, value: Any, field_name: str) -> dt.date:
        """Parse supported date formats into a date object."""
        if isinstance(value, dt.datetime):
            return value.date()

        if isinstance(value, dt.date):
            return value

        if not isinstance(value, str) or not value.strip():
            raise NormalizationError(f"{field_name} is required and must be a valid date.")

        raw_value = value.strip()
        for date_format in cls.DATE_FORMATS:
            try:
                return dt.datetime.strptime(raw_value, date_format).date()
            except ValueError:
                continue

        raise NormalizationError(
            f"{field_name} has unsupported date format {value!r}. "
            f"Supported formats are: {', '.join(cls.DATE_FORMATS)}."
        )

    @staticmethod
    def normalize_domain(value: Any) -> str:
        """
        Convert raw URL/domain values into a bare domain.

        Examples
        --------
        https://www.example.com/path -> example.com
        EXAMPLE.com/landing          -> example.com
        """
        domain = Normalizer.clean_string(value).lower()
        domain = re.sub(r"^https?://", "", domain)
        domain = domain.removeprefix("www.")
        domain = domain.split("/")[0]
        domain = domain.split(":")[0]
        return domain


# =============================================================================
# Validation Engine
# =============================================================================


class ValidationEngine:
    """Applies structural, relational, and business-rule validation."""

    ACCOUNT_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{2,10}-\d{3,}$")
    OPPORTUNITY_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{2,10}-\d{3,}$")
    COUNTRY_ALPHA2_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{2}$")
    CURRENCY_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{3}$")
    DOMAIN_PATTERN: Final[re.Pattern[str]] = re.compile(
        r"^(?!-)(?:[a-z0-9-]{1,63}\.)+[a-z]{2,63}$",
        re.IGNORECASE,
    )
    VALID_STAGES: Final[set[str]] = {
        "Discovery",
        "Qualification",
        "Proposal",
        "Negotiation",
        "Closed Won",
        "Closed Lost",
    }

    @classmethod
    def validate_account(cls, account: NormalizedAccount) -> tuple[bool, list[str]]:
        """Validate one normalized account record."""
        errors: list[str] = []

        if not cls.ACCOUNT_ID_PATTERN.fullmatch(account.account_id):
            errors.append(
                f"Account ID must match a stable CRM pattern such as ACC-9921; "
                f"received {account.account_id!r}."
            )

        if not account.legal_name:
            errors.append("Legal name is required.")

        if not cls.DOMAIN_PATTERN.fullmatch(account.domain):
            errors.append(f"Domain is malformed: {account.domain!r}.")

        if account.annual_revenue < 0:
            errors.append(f"Annual revenue cannot be negative: {account.annual_revenue}.")

        if not cls.COUNTRY_ALPHA2_PATTERN.fullmatch(account.billing_country):
            errors.append(
                f"Billing country must use ISO alpha-2 format, such as US or CA; "
                f"received {account.billing_country!r}."
            )

        expected_tier = determine_account_tier(account.annual_revenue)
        if account.tier != expected_tier:
            errors.append(
                f"Tier mismatch: revenue {account.annual_revenue} maps to "
                f"{expected_tier.value}, not {account.tier.value}."
            )

        return not errors, errors

    @classmethod
    def validate_opportunity(
        cls,
        opportunity: NormalizedOpportunity,
        valid_account_ids: set[str],
        today: dt.date | None = None,
    ) -> tuple[bool, list[str]]:
        """Validate one normalized opportunity record."""
        errors: list[str] = []
        current_date = today or dt.date.today()

        if not cls.OPPORTUNITY_ID_PATTERN.fullmatch(opportunity.opportunity_id):
            errors.append(
                f"Opportunity ID must match a stable CRM pattern such as OPP-771; "
                f"received {opportunity.opportunity_id!r}."
            )

        if opportunity.associated_account_id not in valid_account_ids:
            errors.append(
                f"Orphaned opportunity: associated account ID "
                f"{opportunity.associated_account_id!r} was not verified."
            )

        if not opportunity.title:
            errors.append("Opportunity title is required.")

        if opportunity.contract_value < 0:
            errors.append(f"Contract value cannot be negative: {opportunity.contract_value}.")

        if not cls.CURRENCY_PATTERN.fullmatch(opportunity.currency_code):
            errors.append(
                f"Currency code must be a 3-letter ISO-style code; "
                f"received {opportunity.currency_code!r}."
            )

        if opportunity.stage not in cls.VALID_STAGES:
            errors.append(
                f"Unsupported opportunity stage {opportunity.stage!r}. "
                f"Allowed stages are: {sorted(cls.VALID_STAGES)}."
            )

        oldest_allowed_close_date = current_date - dt.timedelta(days=365)
        if opportunity.close_date < oldest_allowed_close_date:
            errors.append(
                f"Close date {opportunity.close_date.isoformat()} is older than the "
                f"allowed one-year lookback of {oldest_allowed_close_date.isoformat()}."
            )

        return not errors, errors


# =============================================================================
# Business Rules
# =============================================================================


def determine_account_tier(revenue: float) -> AccountTier:
    """Map annual revenue to a CRM account tier."""
    if revenue >= 10_000_000:
        return AccountTier.ENTERPRISE
    if revenue >= 1_000_000:
        return AccountTier.MID_MARKET
    if revenue >= 0:
        return AccountTier.SMB
    return AccountTier.UNKNOWN


def utc_now() -> dt.datetime:
    """Return a timezone-aware UTC timestamp."""
    return dt.datetime.now(dt.UTC)


# =============================================================================
# Pipeline Orchestration
# =============================================================================


class IngestionPipeline:
    """
    Coordinates CRM record processing.

    Accounts are processed before opportunities because opportunity validation
    depends on verified account IDs. Invalid records are quarantined instead of
    being passed downstream.
    """

    def __init__(self) -> None:
        self.metrics = PipelineMetrics()
        self.quarantine_registry: list[QuarantineEntry] = []
        self.verified_accounts: dict[str, NormalizedAccount] = {}
        self.verified_opportunities: dict[str, NormalizedOpportunity] = {}

    def _quarantine(self, record: IngestedRecord, reasons: list[str]) -> None:
        """Store an invalid record with clear failure reasons."""
        entry = QuarantineEntry(
            source_id=record.source_id,
            originating_system=record.source_system,
            quarantine_timestamp=utc_now().isoformat(),
            failure_reasons=reasons,
            raw_data_dump=record.payload,
        )
        self.quarantine_registry.append(entry)
        self.metrics.records_quarantined += 1
        logger.warning(
            "Record quarantined from %s (%s): %s",
            record.source_system,
            record.source_id,
            reasons,
        )

    def _normalize_account(self, record: IngestedRecord) -> NormalizedAccount:
        """Normalize a raw account record into the internal account schema."""
        payload = record.payload

        revenue = Normalizer.parse_float(
            Normalizer.first_present(payload, ("Annual_Rev", "estimated_revenue", "rev_amt"), 0.0),
            "annual_revenue",
        )

        account_id = Normalizer.clean_string(
            Normalizer.first_present(payload, ("legacy_id", "account_key"), "")
        ).upper()

        legal_name = Normalizer.clean_string(
            Normalizer.first_present(payload, ("company_name", "org_title"), "")
        )

        domain = Normalizer.normalize_domain(
            Normalizer.first_present(payload, ("web_domain", "url"), "")
        )

        billing_country = Normalizer.clean_string(
            Normalizer.first_present(payload, ("country", "billing_loc"), "")
        ).upper()

        is_active = Normalizer.parse_bool(
            Normalizer.first_present(payload, ("active_status", "is_active"), True),
            "active_status",
        )

        return NormalizedAccount(
            account_id=account_id,
            legal_name=legal_name,
            domain=domain,
            annual_revenue=revenue,
            tier=determine_account_tier(revenue),
            billing_country=billing_country,
            is_active=is_active,
        )

    def _normalize_opportunity(self, record: IngestedRecord) -> NormalizedOpportunity:
        """Normalize a raw opportunity record into the internal opportunity schema."""
        payload = record.payload

        close_date_raw = Normalizer.first_present(payload, ("close_dt", "expected_close"))

        return NormalizedOpportunity(
            opportunity_id=Normalizer.clean_string(
                Normalizer.first_present(payload, ("opp_uid", "deal_id"), "")
            ).upper(),
            associated_account_id=Normalizer.clean_string(
                Normalizer.first_present(payload, ("parent_account_key", "acc_id"), "")
            ).upper(),
            title=Normalizer.clean_string(
                Normalizer.first_present(payload, ("deal_name", "summary"), "")
            ),
            contract_value=Normalizer.parse_float(
                Normalizer.first_present(payload, ("value", "amount"), 0.0),
                "contract_value",
            ),
            currency_code=Normalizer.clean_string(
                Normalizer.first_present(payload, ("currency", "monetary_unit"), "USD")
            ).upper(),
            stage=Normalizer.clean_string(
                Normalizer.first_present(payload, ("pipeline_stage", "status"), "Discovery")
            ),
            close_date=Normalizer.parse_date(close_date_raw, "close_date"),
        )

    def process_account_stream(self, records: Iterable[IngestedRecord]) -> None:
        """Normalize and validate a stream of account records."""
        for record in records:
            self.metrics.account_records_seen += 1

            try:
                account = self._normalize_account(record)
                is_valid, errors = ValidationEngine.validate_account(account)

                if account.account_id in self.verified_accounts:
                    errors.append(f"Duplicate account ID detected: {account.account_id!r}.")
                    is_valid = False

                if is_valid:
                    self.verified_accounts[account.account_id] = account
                    self.metrics.accounts_verified += 1
                    logger.info("Verified account %s from %s.", account.account_id, record.source_system)
                else:
                    self._quarantine(record, errors)

            except NormalizationError as exc:
                self._quarantine(record, [str(exc)])

    def process_opportunity_stream(self, records: Iterable[IngestedRecord]) -> None:
        """Normalize and validate a stream of opportunity records."""
        valid_account_ids = set(self.verified_accounts)

        for record in records:
            self.metrics.opportunity_records_seen += 1

            try:
                opportunity = self._normalize_opportunity(record)
                is_valid, errors = ValidationEngine.validate_opportunity(
                    opportunity,
                    valid_account_ids,
                )

                if opportunity.opportunity_id in self.verified_opportunities:
                    errors.append(f"Duplicate opportunity ID detected: {opportunity.opportunity_id!r}.")
                    is_valid = False

                if is_valid:
                    self.verified_opportunities[opportunity.opportunity_id] = opportunity
                    self.metrics.opportunities_verified += 1
                    logger.info(
                        "Verified opportunity %s for account %s.",
                        opportunity.opportunity_id,
                        opportunity.associated_account_id,
                    )
                else:
                    self._quarantine(record, errors)

            except NormalizationError as exc:
                self._quarantine(record, [str(exc)])

    def compile_crm_payload(self) -> dict[str, Any]:
        """Assemble the final JSON-safe CRM operation payload."""
        return {
            "sync_metadata": {
                "execution_timestamp": utc_now().isoformat(),
                **self.metrics.to_payload(),
            },
            "payload_operations": {
                "upsert_accounts": [
                    account.to_payload()
                    for account in self.verified_accounts.values()
                ],
                "insert_opportunities": [
                    opportunity.to_payload()
                    for opportunity in self.verified_opportunities.values()
                ],
            },
        }

    def compile_quarantine_payload(self) -> list[dict[str, Any]]:
        """Return JSON-safe quarantine records for engineering review."""
        return [entry.to_payload() for entry in self.quarantine_registry]

    def print_execution_summary(self) -> None:
        """Print a concise human-readable execution summary."""
        print("\n" + "=" * 72)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 72)
        print(f"Account records seen:       {self.metrics.account_records_seen}")
        print(f"Opportunity records seen:   {self.metrics.opportunity_records_seen}")
        print(f"Accounts verified:          {self.metrics.accounts_verified}")
        print(f"Opportunities verified:     {self.metrics.opportunities_verified}")
        print(f"Records quarantined:        {self.metrics.records_quarantined}")


# =============================================================================
# Demonstration Data
# =============================================================================


def build_demo_account_records() -> list[IngestedRecord]:
    """Create sample account records from a hypothetical legacy billing export."""
    return [
        IngestedRecord(
            source_id="BILL-001",
            source_system="Legacy_Billing_SQL",
            timestamp=utc_now(),
            payload={
                "account_key": "ACC-9921",
                "company_name": "Apex Global Logistics",
                "url": "https://www.apex-logistics.com/profile",
                "Annual_Rev": 12_500_000.0,
                "country": "US",
                "active_status": "true",
            },
        ),
        IngestedRecord(
            source_id="BILL-002",
            source_system="Legacy_Billing_SQL",
            timestamp=utc_now(),
            payload={
                "account_key": "ACC-4012",
                "company_name": "Frontier Biotech",
                "url": "bad-url-format",
                "Annual_Rev": 450_000.0,
                "country": "USA",
            },
        ),
        IngestedRecord(
            source_id="BILL-003",
            source_system="Legacy_Billing_SQL",
            timestamp=utc_now(),
            payload={
                "legacy_id": "ACC-5530",
                "org_title": "Northstar Clinical Analytics",
                "web_domain": "northstarclinical.io",
                "estimated_revenue": "2750000",
                "billing_loc": "CA",
                "active_status": "active",
            },
        ),
        IngestedRecord(
            source_id="BILL-004",
            source_system="Legacy_Billing_SQL",
            timestamp=utc_now(),
            payload={
                "account_key": "ACC-9921",
                "company_name": "Duplicate Apex Record",
                "url": "duplicate-apex.com",
                "Annual_Rev": 1_000_000.0,
                "country": "US",
            },
        ),
    ]


def build_demo_opportunity_records() -> list[IngestedRecord]:
    """Create sample opportunity records from a hypothetical marketing CRM export."""
    return [
        IngestedRecord(
            source_id="MKT-808",
            source_system="HubSpot_Export",
            timestamp=utc_now(),
            payload={
                "deal_id": "OPP-771",
                "acc_id": "ACC-9921",
                "deal_name": "Apex Enterprise Expansion Phase 1",
                "amount": 150_000.0,
                "currency": "USD",
                "pipeline_stage": "Proposal",
                "close_dt": "2026-08-15",
            },
        ),
        IngestedRecord(
            source_id="MKT-809",
            source_system="HubSpot_Export",
            timestamp=utc_now(),
            payload={
                "deal_id": "OPP-772",
                "acc_id": "ACC-4012",
                "deal_name": "Frontier Baseline Package",
                "amount": -5_000.0,
                "currency": "USD",
                "pipeline_stage": "Discovery",
                "close_dt": "2026-09-01",
            },
        ),
        IngestedRecord(
            source_id="MKT-810",
            source_system="HubSpot_Export",
            timestamp=utc_now(),
            payload={
                "opp_uid": "OPP-913",
                "parent_account_key": "ACC-5530",
                "summary": "Northstar Analytics Implementation",
                "value": "85000",
                "monetary_unit": "CAD",
                "status": "Negotiation",
                "expected_close": "09/30/2026",
            },
        ),
        IngestedRecord(
            source_id="MKT-811",
            source_system="HubSpot_Export",
            timestamp=utc_now(),
            payload={
                "deal_id": "OPP-914",
                "acc_id": "ACC-5530",
                "deal_name": "Missing Date Opportunity",
                "amount": 25_000.0,
                "currency": "USD",
                "pipeline_stage": "Qualification",
            },
        ),
    ]


# =============================================================================
# Main Execution
# =============================================================================


def main() -> None:
    """Run the demonstration pipeline."""
    logger.info("Initializing CRM integration pipeline.")

    account_records = build_demo_account_records()
    opportunity_records = build_demo_opportunity_records()

    pipeline = IngestionPipeline()

    logger.info("Processing account records first to establish valid account relationships.")
    pipeline.process_account_stream(account_records)

    logger.info("Processing opportunity records after account verification.")
    pipeline.process_opportunity_stream(opportunity_records)

    pipeline.print_execution_summary()

    print("\n" + "=" * 72)
    print("CRM SAFE PAYLOAD")
    print("=" * 72)
    print(json.dumps(pipeline.compile_crm_payload(), indent=2))

    print("\n" + "=" * 72)
    print("QUARANTINE REGISTRY")
    print("=" * 72)
    print(json.dumps(pipeline.compile_quarantine_payload(), indent=2))


if __name__ == "__main__":
    main()
