from __future__ import annotations

from typing import Any
import pandas as pd

COLUMN_TRANSLATIONS: dict[str, str] = {
    "accept_for_score": "Принять в расчёт",
    "confirmed_adverse_fact": "Негативный факт подтверждён",
    "key": "Код показателя",
    "block": "Блок",
    "label": "Показатель",
    "year": "Год",
    "value_used": "Значение для расчёта",
    "unit": "Единица измерения",
    "method": "Метод поиска",
    "e5_score": "Оценка модели",
    "source": "Источник",
    "location": "Место в документе",
    "candidate_text": "Фрагмент документа",
    "review_note": "Комментарий",
    "calculation_effect": "Влияние на расчёт",
    "filename": "Файл",
    "original_filename": "Исходное имя файла",
    "size_kb": "Размер, КБ",
    "source_group": "Группа источника",
    "source_group_label": "Тип источника",
    "document_id": "ID фрагмента",
    "role": "Тип источника",
    "text": "Текст фрагмента",
    "metric_label": "Название метрики",
    "value": "Значение",
    "unit_family": "Тип единицы",
    "structured": "Структурированные данные",
    "extraction_route": "Способ извлечения",
    "page": "Страница",
    "embedded_chars": "Символов в PDF-тексте",
    "text_chars_used": "Использовано символов",
    "route": "Способ извлечения",
    "ocr_status": "Статус OCR",
    "direction": "Направление",
    "max_points": "Максимум баллов",
    "points": "Баллы",
    "ecological_payments_rub": "Экологические платежи, руб.",
    "revenue_rub": "Выручка, руб.",
    "ratio_pct": "Отношение, %",
    "raw_score": "Первичный балл",
    "points_0_15": "Баллы из 15",
    "points_0_35": "Баллы из 35",
    "salary_ratio": "Отношение зарплат",
    "regional_salary_rub": "Средняя зарплата региона, руб.",
    "indicator": "Индикатор",
    "data_status": "Статус данных",
    "basis": "Основание",
    "scoring_year": "Расчётный год",
    "optimistic_missing_negative": "Начислять баллы при отсутствии негативного факта",
    "ocr_enabled": "OCR включён",
    "model": "Модель эмбеддингов",
    "source_priority": "Приоритет источников",
    "note": "Примечание",
}

VALUE_TRANSLATIONS: dict[Any, Any] = {
    "government_sources": "Государственные источники",
    "other_sources": "Прочие источники",
    True: "да",
    False: "нет",
}


def translate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    return df.rename(columns={c: COLUMN_TRANSLATIONS.get(c, c) for c in df.columns})


def translate_for_display(df: pd.DataFrame) -> pd.DataFrame:
    result = translate_columns(df)
    for column in result.columns:
        result[column] = result[column].map(lambda value: VALUE_TRANSLATIONS.get(value, value))
    return result
