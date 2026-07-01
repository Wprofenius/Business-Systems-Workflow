"""
Rex CRM :: External Lead Ingestion & Sync-Loop Prevention Pipeline
==================================================================

Object-oriented sanitization of incoming data from external listing services,
plus a state-management hashing mechanism that prevents legacy systems from
triggering infinite data synchronization loops.

Features strict type hinting, custom exception handling, and quantitative
validation checks to ensure zero malformed data reaches the primary database.

Target runtime: Python 3.9+ (verified on CPython 3.12). Recommended production
interpreter: 3.11 or 3.12 for mature typing support and long support windows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math  # HARDENED: needed to reject non-finite cooldown values (inf / nan).
import re
import threading
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from numbers import Real
from typing import Any, ClassVar, Dict, List, Optional, Union


# Configure robust logging for observability.
# The handler check prevents duplicate log lines when this module is imported elsewhere.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Custom exception for payload data failures."""
    pass


class DataSanitizer:
    """Handles the cleaning and normalization of external lead data."""

    MAX_EMAIL_LENGTH: ClassVar[int] = 254
    MAX_NAME_LENGTH: ClassVar[int] = 100
    MAX_SOURCE_LENGTH: ClassVar[int] = 80

    # HARDENED: Hard upper bound on the size of any single raw field BEFORE it is
    # run through per-character loops and regex. External listing services can (and
    # do) send megabyte-scale junk strings; without this cap those payloads become a
    # CPU/DoS vector even though they would ultimately be truncated or rejected.
    MAX_RAW_FIELD_LENGTH: ClassVar[int] = 10_000

    EMAIL_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,63}$",
        re.IGNORECASE
    )

    @staticmethod
    def _bounded_str(value: Any) -> str:
        """
        Coerce an arbitrary external value to a string and cap its length.

        HARDENED: Centralizes the ``str(...)`` coercion so a single malformed or
        maliciously oversized field cannot stall the worker. Truncation here is
        purely a safety valve; the field-specific validators still enforce their
        own (smaller) business limits afterward.
        """
        try:
            coerced = str(value)
        except Exception:
            # An object whose __str__ raises should never take down ingestion.
            logger.warning("Value could not be coerced to string. Treating as empty.")
            return ""

        if len(coerced) > DataSanitizer.MAX_RAW_FIELD_LENGTH:
            logger.warning(
                "Raw field exceeded %s chars (got %s). Truncating before processing.",
                DataSanitizer.MAX_RAW_FIELD_LENGTH,
                len(coerced)
            )
            coerced = coerced[:DataSanitizer.MAX_RAW_FIELD_LENGTH]

        return coerced

    @staticmethod
    def clean_phone(phone_raw: Any) -> Optional[str]:
        """Normalize a US phone number to ten digits, dropping invalid values."""
        if phone_raw is None or isinstance(phone_raw, bool):
            return None

        # Convert to string in case the external system sent an integer.
        # HARDENED: bounded coercion guards against oversized inputs.
        phone_str = DataSanitizer._bounded_str(phone_raw).strip()
        if not phone_str:
            return None

        # Remove common extension suffixes before checking the base number.
        # Example: "(512) 555-9932 ext. 101" -> "(512) 555-9932"
        phone_str = re.sub(
            r"(?:ext\.?|extension|x)\s*\d+\s*$",
            "",
            phone_str,
            flags=re.IGNORECASE
        )

        # Remove all non-numeric characters using regex.
        numeric_only = re.sub(r"\D+", "", phone_str)

        # Quantitative check: verify exact length of standard US numbers.
        if len(numeric_only) == 11 and numeric_only.startswith("1"):
            numeric_only = numeric_only[1:]  # Strip country code for standardization.

        if len(numeric_only) == 10:
            return numeric_only

        logger.warning(
            "Phone validation failed for digit length %s. Dropping field.",
            len(numeric_only)
        )
        return None

    @classmethod
    def clean_email(cls, email_raw: Any) -> Optional[str]:
        """Normalize and validate a practical CRM-safe email shape."""
        if email_raw is None or isinstance(email_raw, bool):
            return None

        # HARDENED: bounded coercion guards against oversized inputs.
        email_str = cls._bounded_str(email_raw).strip().lower()
        if not email_str:
            return None

        if len(email_str) > cls.MAX_EMAIL_LENGTH:
            logger.warning("Email validation failed due to excessive length. Dropping field.")
            return None

        if not cls.EMAIL_PATTERN.fullmatch(email_str):
            logger.warning("Email validation failed basic structure check. Dropping field.")
            return None

        local_part, domain = email_str.rsplit("@", 1)
        domain_labels = domain.split(".")

        if len(local_part) > 64:
            logger.warning("Email validation failed because local part is too long. Dropping field.")
            return None

        if local_part.startswith(".") or local_part.endswith(".") or ".." in local_part:
            logger.warning("Email validation failed local-part dot placement check. Dropping field.")
            return None

        if any(
            not label
            or len(label) > 63
            or label.startswith("-")
            or label.endswith("-")
            for label in domain_labels
        ):
            logger.warning("Email validation failed domain label check. Dropping field.")
            return None

        return email_str

    @classmethod
    def clean_name(cls, name_raw: Any) -> str:
        """Normalize names while preserving common human-name characters."""
        if name_raw is None or isinstance(name_raw, bool):
            return "Unknown"

        # HARDENED: bounded coercion guards against oversized inputs.
        name_str = cls._bounded_str(name_raw).strip()
        if not name_str:
            return "Unknown"

        # Normalize common apostrophe variants before character filtering.
        name_str = (
            name_str
            .replace("\u2019", "'")  # right single quotation mark
            .replace("\u2018", "'")  # left single quotation mark
            .replace("`", "'")
        )

        # Keep letters from any alphabet, spaces, hyphens, and apostrophes.
        # HARDENED: exclude surrogate code points explicitly. .isalpha() already
        # returns False for lone surrogates, but making it explicit documents intent
        # and protects the downstream JSON/hash step from unencodable characters.
        clean_str = "".join(
            char for char in name_str
            if (char.isalpha() and not ("\ud800" <= char <= "\udfff"))
            or char in {" ", "-", "'"}
        )

        # Collapse repeated whitespace and trim punctuation-only edges.
        clean_str = re.sub(r"\s+", " ", clean_str).strip(" -'")
        if not clean_str:
            return "Unknown"

        if len(clean_str) > cls.MAX_NAME_LENGTH:
            logger.warning("Name validation exceeded max length. Truncating field safely.")
            clean_str = clean_str[:cls.MAX_NAME_LENGTH].rstrip(" -'")

        return clean_str.title() or "Unknown"

    @classmethod
    def clean_source(cls, source_raw: Any) -> str:
        """Normalize the listing source while avoiding empty or control-character values."""
        if source_raw is None or isinstance(source_raw, bool):
            return "UNKNOWN"

        # HARDENED: bounded coercion guards against oversized inputs.
        source_str = cls._bounded_str(source_raw).strip()
        if not source_str:
            return "UNKNOWN"

        source_str = re.sub(r"[\r\n\t]+", " ", source_str)
        source_str = re.sub(r"\s+", " ", source_str).strip()

        # HARDENED: strip lone surrogate code points. Unlike names, this field is not
        # alphabet-filtered, so a surrogate from bad upstream data could otherwise
        # survive here and later crash json/utf-8 encoding in the hashing step.
        if any("\ud800" <= ch <= "\udfff" for ch in source_str):
            logger.warning("Source contained unencodable surrogate characters. Stripping them.")
            source_str = "".join(ch for ch in source_str if not ("\ud800" <= ch <= "\udfff"))
            source_str = source_str.strip()
            if not source_str:
                return "UNKNOWN"

        if len(source_str) > cls.MAX_SOURCE_LENGTH:
            logger.warning("Source exceeded max length. Truncating field safely.")
            source_str = source_str[:cls.MAX_SOURCE_LENGTH].rstrip()

        return source_str.upper() or "UNKNOWN"

    @classmethod
    def normalize_payload(cls, raw_payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Iterates through the raw payload and applies all normalization protocols."""
        if not isinstance(raw_payload, Mapping):
            raise DataValidationError(
                f"Expected payload to be a mapping/dictionary, got {type(raw_payload).__name__}."
            )

        normalized: Dict[str, Any] = {
            "first_name": cls.clean_name(raw_payload.get("first_name")),
            "last_name": cls.clean_name(raw_payload.get("last_name")),
            "email": cls.clean_email(raw_payload.get("email")),
            "phone": cls.clean_phone(raw_payload.get("phone")),
            "source": cls.clean_source(raw_payload.get("source")),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Failsafe: reject payload entirely if both contact methods are invalid.
        if not normalized["email"] and not normalized["phone"]:
            raise DataValidationError("Payload lacks both valid email and valid phone.")

        return normalized


class SyncLoopPreventer:
    """Manages state to prevent legacy PMS from bouncing updates back to Funnel."""

    HASH_EXCLUDED_FIELDS: ClassVar[frozenset[str]] = frozenset({"timestamp"})

    # HARDENED: A finite ceiling on cooldown length. timedelta cannot represent
    # arbitrarily large spans and will raise OverflowError past its internal C-int
    # limits; capping here keeps construction from crashing on absurd config values.
    MAX_COOLDOWN_MINUTES: ClassVar[int] = 60 * 24 * 365  # one year

    # HARDENED: Hard cap on the in-memory hash cache. In production this is Redis and
    # bounded by its own eviction policy, but the self-contained dict below must not
    # grow without limit if a burst of unique payloads arrives inside one cooldown
    # window. When exceeded, the oldest entries are evicted (see _enforce_capacity).
    DEFAULT_MAX_TRACKED_HASHES: ClassVar[int] = 100_000

    def __init__(
        self,
        cooldown_minutes: Union[int, float] = 5,
        max_tracked_hashes: int = DEFAULT_MAX_TRACKED_HASHES,
    ):
        # HARDENED: reject bool, non-numeric, non-positive, AND non-finite values.
        # inf/nan previously passed validation and then crashed timedelta() below.
        if (
            isinstance(cooldown_minutes, bool)
            or not isinstance(cooldown_minutes, Real)
            or not math.isfinite(float(cooldown_minutes))
            or cooldown_minutes <= 0
        ):
            raise ValueError("cooldown_minutes must be a positive, finite number.")

        # HARDENED: reject spans that timedelta cannot represent, with a clear error.
        if cooldown_minutes > self.MAX_COOLDOWN_MINUTES:
            raise ValueError(
                f"cooldown_minutes must not exceed {self.MAX_COOLDOWN_MINUTES} "
                f"(got {cooldown_minutes})."
            )

        # HARDENED: validate the capacity cap too, so a bad config fails fast and loud.
        if isinstance(max_tracked_hashes, bool) or not isinstance(max_tracked_hashes, int):
            raise ValueError("max_tracked_hashes must be an integer.")
        if max_tracked_hashes < 1:
            raise ValueError("max_tracked_hashes must be a positive integer.")
        self.max_tracked_hashes: int = max_tracked_hashes

        # In a production environment, this cache would be managed by Redis.
        # A standard dictionary is used here for script self-containment.
        self.processed_hashes: Dict[str, datetime] = {}

        # HARDENED: belt-and-suspenders guard around timedelta construction. Even with
        # the validation above, this converts any residual overflow into a clean error.
        try:
            self.cooldown = timedelta(minutes=float(cooldown_minutes))
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"cooldown_minutes produced an invalid duration: {exc}") from exc

        self._lock = threading.Lock()

    def generate_payload_hash(self, payload: Mapping[str, Any]) -> str:
        """Creates a stable cryptographic hash of the contact data."""
        if not isinstance(payload, Mapping):
            raise TypeError(f"Expected payload mapping, got {type(payload).__name__}.")

        # Remove timestamp before hashing to ensure data-only comparison.
        hashable_data = {
            key: value
            for key, value in payload.items()
            if key not in self.HASH_EXCLUDED_FIELDS
        }

        # Sort keys and compact separators to ensure consistent hashing regardless of dictionary order.
        # HARDENED: wrap serialization so an unexpectedly unserializable value raises a
        # descriptive TypeError instead of an opaque one mid-pipeline. default=str keeps
        # exotic-but-stringifiable values from breaking the hash.
        try:
            serialized = json.dumps(
                hashable_data,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Payload could not be serialized for hashing: {exc}") from exc

        # HARDENED: encode with errors="surrogatepass" so any lone surrogate code point
        # that slipped through upstream sanitization is hashed deterministically rather
        # than raising UnicodeEncodeError (which the caller's broad except would swallow,
        # silently dropping a VALID lead and mislabeling it a system failure).
        return hashlib.sha256(serialized.encode("utf-8", "surrogatepass")).hexdigest()

    def is_duplicate(self, payload: Mapping[str, Any]) -> bool:
        """Checks if identical data was recently processed."""
        payload_hash = self.generate_payload_hash(payload)
        current_time = datetime.now(timezone.utc)

        with self._lock:
            # Clean up expired hashes to prevent memory leaks.
            self._purge_expired_hashes(current_time)

            if payload_hash in self.processed_hashes:
                # Refresh timestamp to make cooldown a sliding window during active bounce loops.
                self.processed_hashes[payload_hash] = current_time
                logger.info("Duplicate payload detected. Sync loop prevented. Dropping event.")
                return True

            # Log the new hash with a timestamp.
            self.processed_hashes[payload_hash] = current_time

            # HARDENED: enforce the capacity ceiling after inserting, so a flood of
            # unique payloads inside one cooldown window cannot exhaust memory.
            self._enforce_capacity()
            return False

    def _purge_expired_hashes(self, current_time: datetime) -> None:
        """Removes hashes older than the cooldown window. Call while holding self._lock."""
        expired_keys = [
            key
            for key, processed_at in self.processed_hashes.items()
            if current_time - processed_at >= self.cooldown
        ]

        for key in expired_keys:
            del self.processed_hashes[key]

    def _enforce_capacity(self) -> None:
        """
        Evict the oldest tracked hashes if the cache exceeds its cap.

        HARDENED: Safety valve for the self-contained dict cache. Call while holding
        self._lock. Eviction is by processed_at (oldest first), which mirrors how a
        real Redis TTL/LRU policy would behave. This never runs on normal traffic;
        it only engages under pathological unique-payload bursts.

        Eviction batches down to a low-water mark (~90% of the cap) rather than
        trimming a single entry per insert. That keeps the O(n log n) sort from
        running on every event while the cache sits at capacity, and avoids emitting
        one log line per insert under sustained overflow.
        """
        if len(self.processed_hashes) <= self.max_tracked_hashes:
            return

        # Trim down to the low-water mark so this only fires periodically.
        low_water = max(1, int(self.max_tracked_hashes * 0.9))
        remove_count = len(self.processed_hashes) - low_water
        if remove_count <= 0:
            return

        # Oldest-first ordering; delete only the excess.
        oldest_keys: List[str] = sorted(
            self.processed_hashes,
            key=lambda key: self.processed_hashes[key]
        )[:remove_count]

        for key in oldest_keys:
            del self.processed_hashes[key]

        logger.warning(
            "Hash cache exceeded capacity (%s). Evicted %s oldest entries down to %s.",
            self.max_tracked_hashes,
            remove_count,
            low_water
        )


def redact_payload_for_logs(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Redacts PII before writing sanitized payload details to logs."""
    # HARDENED: guard against a non-mapping being passed in so log redaction itself
    # can never raise and expose raw PII via an error path.
    if not isinstance(payload, Mapping):
        logger.warning("redact_payload_for_logs received a non-mapping. Returning empty redaction.")
        return {}

    redacted: Dict[str, Any] = dict(payload)

    for name_key in ("first_name", "last_name"):
        if redacted.get(name_key) and redacted[name_key] != "Unknown":
            redacted[name_key] = "***"

    email = redacted.get("email")
    if isinstance(email, str) and "@" in email:
        local_part, domain = email.split("@", 1)
        redacted["email"] = f"{local_part[:2]}***@{domain}"
    elif email:
        redacted["email"] = "***"

    phone = redacted.get("phone")
    if isinstance(phone, str) and len(phone) >= 4:
        redacted["phone"] = f"***-***-{phone[-4:]}"
    elif phone:
        redacted["phone"] = "***"

    return redacted


def process_incoming_webhook(
    raw_data: Mapping[str, Any],
    state_manager: SyncLoopPreventer
) -> bool:
    """Main execution thread simulating an AWS Lambda or serverless function."""
    logger.info("Incoming webhook received. Initiating processing sequence.")

    if not isinstance(state_manager, SyncLoopPreventer):
        logger.critical("Invalid state manager supplied. Terminating webhook processing.")
        return False

    try:
        # Step 1: Normalize external data.
        clean_data = DataSanitizer.normalize_payload(raw_data)
        logger.info("Data normalization successful.")

        # Step 2: Check for bi-directional loop.
        if state_manager.is_duplicate(clean_data):
            return False  # Event is dropped safely.

        # Step 3: Proceed with API push to the CRM.
        # HARDENED: default=str + ensure_ascii (default True) keep this log line from
        # ever raising on exotic values; redaction guarantees no raw PII is emitted.
        logger.info(
            "Pushing sanitized data to CRM API: %s",
            json.dumps(redact_payload_for_logs(clean_data), sort_keys=True, default=str)
        )
        return True

    except DataValidationError as dve:
        logger.error("Data Validation Failure: %s", str(dve))
        return False
    except (TypeError, ValueError) as exc:
        # HARDENED: surface configuration/serialization errors distinctly from truly
        # unexpected failures, while still never propagating out of the handler.
        logger.error("Processing error: %s", str(exc))
        return False
    except Exception:
        logger.exception("Unexpected system failure while processing webhook.")
        return False


# --- Execution Simulation ---
if __name__ == "__main__":
    # HARDENED: wrap the whole simulation so the demo can never exit with a traceback.
    try:
        # Initialize the state manager with a 5-minute cooldown.
        sync_manager = SyncLoopPreventer(cooldown_minutes=5)

        # Simulating a messy payload from an external Internet Listing Service.
        messy_external_payload = {
            "first_name": "  jOhn_ ",
            "last_name": "D'Angelo!!",
            "email": " JOHN.dangelo@example.com ",
            "phone": "+1 (512) 555-9932",
            "source": "Zillow"
        }

        print("--- FIRST PROCESSING ATTEMPT ---")
        first_result = process_incoming_webhook(messy_external_payload, sync_manager)
        print(f"Processed: {first_result}")

        print("\n--- SECOND PROCESSING ATTEMPT (SIMULATING PMS BOUNCE-BACK) ---")
        # Simulating the legacy system pushing the exact same data back immediately.
        second_result = process_incoming_webhook(messy_external_payload, sync_manager)
        print(f"Processed: {second_result}")

        # HARDENED: demonstrate that the previously crash-prone edge cases now degrade
        # gracefully instead of taking down the worker.
        print("\n--- EDGE CASE: MISSING BOTH CONTACT METHODS (rejected cleanly) ---")
        third_result = process_incoming_webhook(
            {"first_name": "Jane", "last_name": "Doe", "source": "Realtor"},
            sync_manager
        )
        print(f"Processed: {third_result}")

        print("\n--- EDGE CASE: SURROGATE IN SOURCE (no longer crashes the hash) ---")
        fourth_result = process_incoming_webhook(
            {"phone": "512-555-0000", "source": "Zillow\ud800"},
            sync_manager
        )
        print(f"Processed: {fourth_result}")

        print("\n--- EDGE CASE: INVALID COOLDOWN CONFIG (rejected at construction) ---")
        for bad_cfg in (float("inf"), float("nan"), 0, -3):
            try:
                SyncLoopPreventer(cooldown_minutes=bad_cfg)
                print(f"  cooldown={bad_cfg!r}: unexpectedly accepted")
            except ValueError as exc:
                print(f"  cooldown={bad_cfg!r}: rejected -> {exc}")

    except Exception:
        logger.exception("Fatal error in execution simulation.")
