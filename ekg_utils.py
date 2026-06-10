from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional
import re

import numpy as np

from ekg_models import Fragment


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\xa0", " ").replace("ё", "е").lower()
    text = re.sub(r"[^\w\s%.,:/()'’\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = (
        str(value).replace("\xa0", "").replace(" ", "")
        .replace("'", "").replace("’", "").replace(",", ".").strip()
    )
    if text.lower() in {"", "-", "—", "n/d", "n\\d", "na", "n/a"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def unit_family(unit: str) -> str:
    text = clean_text(unit)
    if "руб" in text or "rub" in text:
        return "money"
    if "%" in text:
        return "percent"
    if "человек" in text or "сотрудник" in text:
        return "people"
    if "коэффициент" in text or text == "-":
        return "ratio"
    return "other"


def normalize_value(value: Optional[float], unit: str) -> Optional[float]:
    if value is None:
        return None
    text = clean_text(unit)
    if unit_family(unit) != "money":
        return value
    if "трлн" in text:
        return value * 1_000_000_000_000
    if "млрд" in text:
        return value * 1_000_000_000
    if "млн" in text or "mln" in text:
        return value * 1_000_000
    if "тыс" in text:
        return value * 1_000
    return value


def auto_role(filename: str) -> str:
    name = filename.lower()
    if name.endswith((".xls", ".xlsx")) and ("esg" in name or "gri" in name):
        return "esg_table"
    if name.endswith((".xls", ".xlsx")) and any(t in name for t in ["additional", "fin", "бух", "balance"]):
        return "financial_table"
    if name.endswith(".pdf") and any(t in name for t in ["fs", "финанс", "statement"]):
        return "financial_statements_pdf"
    if name.endswith(".pdf") and any(t in name for t in ["ar", "годов", "report"]):
        return "annual_report_pdf"
    if name.endswith(".pdf") and any(t in name for t in ["polit", "kodeks", "кодекс", "policy"]):
        return "policy_pdf"
    return "other"


def fragment_to_dict(fragment: Fragment) -> dict[str, Any]:
    return asdict(fragment)
