from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import re

import pandas as pd
import pymupdf

from ekg_models import Fragment, normalize_source_group
from ekg_utils import (
    auto_role,
    clean_text,
    fragment_to_dict,
    normalize_value,
    parse_number,
    unit_family,
)


def _append_fragment(
    fragments: list[Fragment],
    filename: str,
    role: str,
    location: str,
    text: str,
    metric_label: str = "",
    year: Optional[int] = None,
    raw_value: Optional[float] = None,
    unit: str = "",
    structured: bool = False,
    extraction_route: str = "PARSED",
) -> None:
    fragments.append(
        Fragment(
            document_id=f"DOC-{len(fragments)+1}",
            filename=filename,
            role=role,
            location=location,
            text=clean_text(text),
            metric_label=str(metric_label or "").strip(),
            year=year,
            value=normalize_value(raw_value, unit),
            unit=unit,
            unit_family=unit_family(unit),
            structured=structured,
            extraction_route=extraction_route,
        )
    )


def _append_text_rows(fragments: list[Fragment], path: Path, role: str, sheet_name: str, raw: pd.DataFrame) -> None:
    for row_idx, row in raw.iterrows():
        parts = [str(v).strip() for v in row.tolist() if pd.notna(v) and str(v).strip()]
        joined = " | ".join(parts)
        if len(joined) >= 30:
            _append_fragment(
                fragments, path.name, role, f"{sheet_name}!text-row{row_idx+1}",
                joined, parts[0] if parts else "", None, None, "", False, "TABLE_TEXT_ROW"
            )


def _parse_financial_table(path: Path, role: str, sheets: dict[str, pd.DataFrame]) -> list[Fragment]:

    fragments: list[Fragment] = []
    for sheet_name, raw in sheets.items():
        if raw.empty:
            continue
        period_row = None
        for idx in range(min(15, len(raw))):
            exact_years = []
            for value in raw.iloc[idx].tolist():
                text_value = str(value).replace('.0', '').strip() if pd.notna(value) else ''
                if re.fullmatch(r"20\d{2}", text_value):
                    exact_years.append(int(text_value))
            if len(exact_years) >= 2:
                period_row = idx
                break
        if period_row is None:
            _append_text_rows(fragments, path, role, sheet_name, raw)
            continue
        declared_unit = str(raw.iloc[period_row, 0]).strip() if pd.notna(raw.iloc[period_row, 0]) else "RUB mln"
        annual_columns: dict[int, int] = {}
        for col in range(1, raw.shape[1]):
            value = raw.iloc[period_row, col]
            header = str(value).replace('.0', '').strip() if pd.notna(value) else ''
            if re.fullmatch(r"20\d{2}", header):
                annual_columns[col] = int(header)
        for row_idx in range(period_row + 1, len(raw)):
            metric = raw.iloc[row_idx, 0]
            if pd.isna(metric) or not str(metric).strip():
                continue
            metric_text = str(metric).strip()
            unit = "%" if "%" in metric_text else declared_unit
            for col, year in annual_columns.items():
                value = parse_number(raw.iloc[row_idx, col])
                if value is None:
                    continue
                _append_fragment(
                    fragments, path.name, role, f"{sheet_name}!row{row_idx+1}",
                    f"{metric_text}; период {year}", metric_text, year, value,
                    unit, True, "FINANCIAL_TABLE_ANNUAL_WITH_DECLARED_UNIT"
                )
        _append_text_rows(fragments, path, role, sheet_name, raw)
    return fragments


def parse_table_file(path: Path, role: str) -> list[Fragment]:

    engine = "xlrd" if path.suffix.lower() == ".xls" else None
    sheets = pd.read_excel(path, sheet_name=None, header=None, engine=engine)
    parser_role = auto_role(path.name)
    if parser_role == "financial_table":
        return _parse_financial_table(path, role, sheets)

    fragments: list[Fragment] = []
    for sheet_name, raw in sheets.items():
        if raw.empty:
            continue

        
        long_header = None
        for idx in range(min(10, len(raw))):
            row_tokens = [clean_text(v) for v in raw.iloc[idx].tolist()]
            if any(token in {"год", "year"} for token in row_tokens) and any(
                token in {"значение", "value"} for token in row_tokens
            ):
                long_header = idx
                break
        if long_header is not None:
            frame = raw.iloc[long_header + 1:].copy()
            frame.columns = [clean_text(v) for v in raw.iloc[long_header].tolist()]
            year_col = next((c for c in frame.columns if c in {"год", "year"}), None)
            value_col = next((c for c in frame.columns if c in {"значение", "value"}), None)
            unit_col = next((c for c in frame.columns if "единиц" in c or c == "unit"), None)
            text_cols = [c for c in frame.columns if c not in {year_col, value_col, unit_col}]
            if year_col and value_col:
                for row_idx, row in frame.iterrows():
                    year = parse_number(row.get(year_col))
                    value = parse_number(row.get(value_col))
                    if year is None or value is None:
                        continue
                    unit = str(row.get(unit_col, "") if unit_col else "")
                    label = " | ".join(
                        str(row.get(c)).strip() for c in text_cols
                        if pd.notna(row.get(c)) and str(row.get(c)).strip()
                    )
                    _append_fragment(
                        fragments, path.name, role, f"{sheet_name}!row{row_idx+1}",
                        label, label, int(year), value, unit, True, "TABLE_LONG_FORMAT"
                    )

        
        year_header = None
        year_columns: dict[int, int] = {}
        for idx in range(min(20, len(raw))):
            detected: dict[int, int] = {}
            for col in range(raw.shape[1]):
                value = parse_number(raw.iloc[idx, col])
                if value is not None and 2010 <= int(value) <= 2035:
                    detected[col] = int(value)
            if len(detected) >= 2:
                year_header = idx
                year_columns = detected
                break
        if year_header is not None:
            first_year_col = min(year_columns)
            pre_columns = list(range(first_year_col))
            metric_col = max(
                pre_columns,
                key=lambda col: sum(
                    1 for value in raw.iloc[year_header+1:, col].tolist()
                    if isinstance(value, str) and len(value.strip()) >= 8
                ),
                default=0,
            )
            unit_col = metric_col + 1 if metric_col + 1 < first_year_col else None
            for row_idx in range(year_header + 1, len(raw)):
                metric = raw.iloc[row_idx, metric_col]
                if pd.isna(metric) or not str(metric).strip():
                    continue
                metric_text = str(metric).strip()
                unit = (
                    str(raw.iloc[row_idx, unit_col]).strip()
                    if unit_col is not None and pd.notna(raw.iloc[row_idx, unit_col])
                    else ""
                )
                for col, year in year_columns.items():
                    value = parse_number(raw.iloc[row_idx, col])
                    if value is None:
                        continue
                    _append_fragment(
                        fragments, path.name, role, f"{sheet_name}!row{row_idx+1}",
                        f"{sheet_name} | {metric_text}", metric_text, year, value,
                        unit, True, "TABLE_WIDE_FORMAT"
                    )
        _append_text_rows(fragments, path, role, sheet_name, raw)
    return fragments


def parse_pdf_file(
    path: Path,
    role: str,
    enable_ocr: bool,
    ocr_language: str,
    ocr_min_chars: int,
) -> tuple[list[Fragment], list[dict[str, Any]]]:
    fragments: list[Fragment] = []
    audit: list[dict[str, Any]] = []
    document = pymupdf.open(str(path))

    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        embedded = page.get_text("text", sort=True).strip()
        text = embedded
        extraction_route = "PDF_TEXT"
        ocr_status = "not_needed"

        if enable_ocr and len(clean_text(embedded)) < ocr_min_chars:
            try:
                text_page = page.get_textpage_ocr(language=ocr_language, dpi=250, full=True)
                ocr_text = page.get_text("text", textpage=text_page, sort=True).strip()
                if len(clean_text(ocr_text)) > len(clean_text(embedded)):
                    text = ocr_text
                    extraction_route = "PDF_OCR"
                    ocr_status = "used"
                else:
                    ocr_status = "no_improvement"
            except Exception as exc:
                ocr_status = f"error: {str(exc)[:100]}"

        audit.append(
            {
                "filename": path.name,
                "page": page_index + 1,
                "embedded_chars": len(embedded),
                "text_chars_used": len(text),
                "route": extraction_route,
                "ocr_status": ocr_status,
            }
        )
        if not text:
            continue
        for start in range(0, len(text), 900):
            chunk = text[start : start + 900].strip()
            if len(chunk) >= 30:
                _append_fragment(
                    fragments, path.name, role, f"page {page_index+1}",
                    chunk, "", None, None, "", False, extraction_route
                )
    return fragments, audit


def inject_known_document_facts(path: Path, role: str) -> list[Fragment]:

    return []


def parse_documents(
    file_rows: pd.DataFrame,
    temp_dir: Path,
    enable_ocr: bool = False,
    ocr_language: str = "rus+eng",
    ocr_min_chars: int = 70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fragments: list[Fragment] = []
    extraction_audit: list[dict[str, Any]] = []
    for _, row in file_rows.iterrows():
        path = temp_dir / row["filename"]
        role = normalize_source_group(row.get("source_group", "other_sources"))
        suffix = path.suffix.lower()
        if suffix in {".xls", ".xlsx"}:
            fragments.extend(parse_table_file(path, role))
        elif suffix == ".pdf":
            pdf_fragments, page_audit = parse_pdf_file(
                path, role, False, ocr_language, ocr_min_chars
            )
            fragments.extend(pdf_fragments)
            extraction_audit.extend(page_audit)
        fragments.extend(inject_known_document_facts(path, role))
    return (
        pd.DataFrame([fragment_to_dict(f) for f in fragments]),
        pd.DataFrame(extraction_audit),
    )
