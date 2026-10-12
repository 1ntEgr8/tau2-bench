# Copyright Sierra
"""Published per-entity-type fold rules for the intake domain.

Reward in this domain is final-DB hash equality against a golden replay, so
two surface forms of the *same captured value* must enter the DB as the same
byte string. The seam is **canonicalization at write time**: ``submit_fields``
folds every field into its canonical stored form before it lands in the DB.
An orthography coin-flip ("maria lopez",
lowercase VINs, "555 123 2002" vs "555-123-2002") must not fail the reward; a
real capture error (digit swap, wrong record, accepted decoy) must.

The canonical form is **purely syntactic**: computable from the writer's input
alone, deterministic, and collision-free over the entity banks (truth never
folds together with any of its decoys — asserted exhaustively in
``tests/test_domains/test_intake/test_folds_intake.py``). The tool cannot know
the pinned truth at runtime, so no rule may depend on it; in particular there
is no "restore bank casing" step — where casing cannot be reconstructed
syntactically (names), the stored form IS the folded form.

Fold rules per entity type:

- ``PHONE`` — the bare digit sequence, with the two purely syntactic dialing
  prefixes normalized away: a ``+`` marker and the international ``00``
  prefix are dropped (``+44…`` == ``0044…`` == ``44…``), and an 11-digit
  sequence starting with ``1`` drops the NANP country code
  (``+1 (206) 555-0111`` == ``(206) 555-0111``). Grouping, separators, and
  extension markers ("ext", "x") carry no identity; every other digit —
  including any remaining leading zero — does.
  ``555 123 2002`` == ``555-123-2002`` -> ``5551232002``;
  ``+44 20 7946 0958`` -> ``442079460958``.
- ``CODE`` — uppercase alphanumeric sequence: separators and spaces dropped,
  case folded up. Covers VINs, plates, member/loyalty IDs, and resolved
  record ids (``prv-203`` == ``PRV 203`` -> ``PRV203``).
- ``NAME`` — person / place / coined names and addresses: diacritics folded,
  case folded down, apostrophes dropped, ``&`` folded to the word "and" and
  ``%`` to "percent" (both are what the symbol says out loud),
  hyphens / commas / slashes / parentheses folded to spaces, periods
  dropped only when neither neighbor is a digit, a space inserted at every
  digit<->letter boundary (``K9-Plus`` == ``K9 Plus`` == ``K9Plus`` ->
  ``k 9 plus``), whitespace collapsed.
  Unit words retain their written form: "milligram", "milligrams", and
  ``mg`` remain distinct. Numeric periods are preserved, so ``0.5 mg``
  cannot match ``05 mg`` or ``5 mg``.
  Title-casing is deliberately NOT restored ("van der / de la" makes it
  unsafe), so the stored form is the folded lowercase form:
  ``Ana de la Vega-Marchetti`` -> ``ana de la vega marchetti``.
- ``EMAIL`` — case folded down, trimmed; **separators preserved**. Dots vs
  underscores in a local part are identity-bearing (the hard-email decoys
  differ only by them), so nothing beyond case may fold.
- ``DATE`` — no format conversion. Only surrounding whitespace is trimmed;
  the agent must submit the field's required ``YYYY-MM-DD`` representation.
  Alternative formats remain as submitted and fail the golden-record match.
- ``NAME_LIST`` — list-valued fields: split on ``;`` / ``,`` / newlines, each
  item NAME-folded, joined with the canonical ``"; "``. Item **order is
  preserved** — for ``current_medications`` the order is caller-pinned
  (document order), not caller-free.
- ``TIME`` — no format conversion. Only surrounding whitespace is trimmed;
  the agent must submit ``H:MM AM`` / ``H:MM PM`` (no leading hour zero,
  two-digit minutes, uppercase meridiem). Alternate spellings, 24-hour times,
  and words such as "noon" remain as submitted and fail the golden-record match.
- ``AMOUNT`` — monetary amount as its minimal decimal digit string: currency
  markers ("$", a "USD"/"dollars" word), thousands separators, and trailing
  decimal zeros carry no identity (``$1,250.00`` == ``1250``); every
  remaining digit does (``125.5`` != ``125.05``).
- Explicit none — the pinned sentinel is JSON null (``None``). On fields that
  accept an explicit none, the bare tokens ``none`` / ``n/a`` / ``null``
  (case-insensitive) fold to the sentinel.

Values that cannot fold (a phone without digits, an empty code) are returned
trimmed-as-given: canonicalization never invents a value, and a nonsense
value failing the DB match is the correct outcome.
"""

import re
import unicodedata
from enum import Enum
from typing import Optional

# 1.2.0: require date/time formats and preserve numeric periods in names.
# 1.1.0: preserve written units; milligram/milligrams no longer fold to mg.
SCORING_VERSION = "1.2.0"

NONE_TOKENS = frozenset({"none", "n/a", "null"})

LIST_JOINER = "; "


class FoldKind(str, Enum):
    """Closed set of per-entity-type fold rules."""

    NATIVE_NAME = "native_name"  # NFC/case only: marks and punctuation carry identity
    NATIVE_NAME_LIST = "native_name_list"
    NAME = "name"  # person / place / coined names, addresses
    CODE = "code"  # VINs, plates, member ids, resolved record ids
    PHONE = "phone"  # phone numbers
    EMAIL = "email"  # email addresses
    DATE = "date"  # calendar dates
    NAME_LIST = "name_list"  # list-valued name fields ("; "-joined)
    TIME = "time"  # times of day ("h:mm AM/PM", meridiem identity-bearing)
    AMOUNT = "amount"  # monetary amounts (minimal decimal digit string)


def fold_phone(value: str) -> str:
    """Bare digit sequence, dialing prefixes normalized. See module docstring."""
    trimmed = value.strip()
    digits = "".join(ch for ch in trimmed if ch.isdigit())
    if digits.startswith("00"):
        # International dialing prefix: '0044…' says the same thing as
        # '+44…'. No subscriber number begins with two zeros.
        digits = digits[2:]
    if len(digits) == 11 and digits.startswith("1"):
        # NANP country code: '+1 (206) 555-0111' == '(206) 555-0111'.
        digits = digits[1:]
    if not digits:
        return trimmed
    return digits


def fold_code(value: str) -> str:
    """Uppercase alphanumeric sequence. See module docstring."""
    trimmed = value.strip()
    key = "".join(ch for ch in trimmed if ch.isalnum()).upper()
    return key or trimmed


# Latin letters that carry a diacritic-like distinction but do NOT decompose
# under NFD, so mark-stripping alone cannot fold them. Casefolding "I" gives
# "i" while "ı" stays "ı" — without this map the uppercase form of a Turkish
# name would fold differently from the name itself.
_NON_DECOMPOSING = str.maketrans(
    {
        "ı": "i",
        "ø": "o",
        "ł": "l",
        "đ": "d",
        "ð": "d",
        "þ": "th",
        "æ": "ae",
        "œ": "oe",
    }
)


def fold_name(value: str) -> str:
    """Case-, diacritic-, and separator-folded name form. See module docstring."""
    decomposed = unicodedata.normalize("NFD", value.strip())
    folded = "".join(
        ch for ch in decomposed if not unicodedata.combining(ch)
    ).casefold()
    folded = folded.translate(_NON_DECOMPOSING)
    folded = folded.replace("'", "").replace("’", "")
    folded = folded.replace("&", " and ").replace("%", " percent ")
    folded = re.sub(r"[-,/()]", " ", folded)
    # Keep numeric periods: deleting one can change a medication's strength.
    # Ordinary initial/abbreviation periods, such as "Dr.", still fold away.
    folded = re.sub(r"(?<!\d)\.(?!\d)", "", folded)
    # 'K9Plus', 'K9 Plus', and 'K9-Plus' are the same spoken name; a digit
    # next to a letter is a boundary the writer may or may not mark.
    folded = re.sub(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def fold_native_name(value: str) -> str:
    """Canonical Unicode composition and casing, retaining script-bearing marks.

    No transliteration, accent deletion, English symbol expansion or fuzzy
    matching: authored variants that differ in written identity remain distinct.
    """
    return unicodedata.normalize("NFC", " ".join(value.strip().casefold().split()))


def fold_native_name_list(value: str) -> str:
    """Preserve list order and all identity-bearing marks in each native name."""
    return LIST_JOINER.join(
        fold_native_name(item) for item in re.split(r"[;,\n]", value) if item.strip()
    )


def fold_email(value: str) -> str:
    """Case-folded, trimmed; separators preserved. See module docstring."""
    return value.strip().casefold()


def fold_name_list(value: str) -> str:
    """Order-preserving canonical ``"; "`` join of NAME-folded items."""
    items = [fold_name(item) for item in re.split(r"[;,\n]", value)]
    return LIST_JOINER.join(item for item in items if item)


_AMOUNT_PATTERN = re.compile(
    r"\$?\s*(?P<digits>\d{1,3}(?:,\d{3})*|\d+)(?:\.(?P<cents>\d+))?"
    r"(?:\s*(?:usd|dollars?))?",
    re.IGNORECASE,
)


def fold_amount(value: str) -> str:
    """Minimal decimal digit string of a money amount. See module docstring."""
    trimmed = value.strip()
    match = _AMOUNT_PATTERN.fullmatch(trimmed)
    if not match:
        return trimmed
    whole = match.group("digits").replace(",", "").lstrip("0") or "0"
    cents = (match.group("cents") or "").rstrip("0")
    if cents:
        return f"{whole}.{cents}"
    return whole


_FOLDERS = {
    FoldKind.NAME: fold_name,
    FoldKind.NATIVE_NAME: fold_native_name,
    FoldKind.NATIVE_NAME_LIST: fold_native_name_list,
    FoldKind.CODE: fold_code,
    FoldKind.PHONE: fold_phone,
    FoldKind.EMAIL: fold_email,
    FoldKind.DATE: str.strip,
    FoldKind.NAME_LIST: fold_name_list,
    FoldKind.TIME: str.strip,
    FoldKind.AMOUNT: fold_amount,
}


def fold_value(kind: FoldKind, value: str) -> str:
    """Fold ``value`` with the rule for ``kind``."""
    return _FOLDERS[kind](value)


def fold_field(
    kind: FoldKind, value: Optional[str], allows_none: bool = False
) -> Optional[str]:
    """Canonical stored form of one submitted field.

    ``None`` stays the explicit-none sentinel. On fields that accept an
    explicit none, the bare NONE_TOKENS fold to the sentinel too; everywhere
    else they stay literal strings (and correctly fail the match).
    """
    if value is None:
        return None
    if allows_none and value.strip().casefold() in NONE_TOKENS:
        return None
    return fold_value(kind, value)
