from __future__ import annotations

import pandas as pd


def _accepted_value(inputs: pd.DataFrame, key: str) -> float:
    rows = inputs[inputs["key"] == key]
    if rows.empty:
        return 0.0
    row = rows.iloc[0]
    return float(row["value_used"]) if bool(row["accept_for_score"]) else 0.0


def _has_accepted(inputs: pd.DataFrame, key: str) -> bool:
    rows = inputs[inputs["key"] == key]
    return False if rows.empty else bool(rows.iloc[0]["accept_for_score"])


def _eco_raw(payments: float, revenue: float) -> float:
    if revenue <= 0:
        return 0.0
    ratio = payments / revenue * 100
    if payments <= 1000: return 100.0
    if ratio < 0.00001: return 90.0
    if ratio <= 0.0001: return 80.0
    if ratio <= 0.001: return 70.0
    if ratio <= 0.01: return 60.0
    if ratio <= 0.1: return 50.0
    if ratio <= 1.0: return 40.0
    if ratio <= 10.0: return 30.0
    return 0.0


def _cadres_raw(ratio: float) -> float:
    thresholds = [
        (2.00, 100), (1.90, 95), (1.80, 90), (1.70, 85), (1.60, 80),
        (1.50, 75), (1.40, 70), (1.30, 65), (1.20, 60), (1.10, 55),
        (1.00, 50), (0.95, 45), (0.90, 40), (0.85, 35), (0.80, 30),
        (0.75, 25), (0.70, 20), (0.65, 15), (0.60, 10), (0.55, 5),
    ]
    return float(next((score for threshold, score in thresholds if ratio >= threshold), 0))


def calculate_score(
    inputs: pd.DataFrame,
    adverse: pd.DataFrame,
    scoring_year: int,
    optimistic_missing_negative: bool,
) -> dict[str, pd.DataFrame]:
    eco_rows: list[dict[str, Any]] = []
    for year in [scoring_year - 2, scoring_year - 1, scoring_year]:
        payments = (
            _accepted_value(inputs, f"nvos_{year}")
            + _accepted_value(inputs, f"eco_fines_{year}")
            + _accepted_value(inputs, f"eco_damage_{year}")
        )
        revenue = _accepted_value(inputs, f"revenue_{year}")
        raw = _eco_raw(payments, revenue)
        ratio = payments / revenue * 100 if revenue else 0.0
        eco_rows.append(
            {
                "year": year, "ecological_payments_rub": payments,
                "revenue_rub": revenue, "ratio_pct": ratio,
                "raw_score": raw, "points_0_15": raw / 100 * 15,
            }
        )
    ecology = pd.DataFrame(eco_rows)
    
    
    ecology_points = float(ecology["points_0_15"].min())

    ratio = _accepted_value(inputs, "salary_ratio")
    cadres_raw = _cadres_raw(ratio)
    cadres = pd.DataFrame(
        [{
            "salary_ratio": ratio,
            "regional_salary_rub": _accepted_value(inputs, "regional_salary"),
            "raw_score": cadres_raw,
            "points_0_35": cadres_raw / 100 * 35,
        }]
    )
    cadres_points = float(cadres.iloc[0]["points_0_35"])

    ca = _accepted_value(inputs, "current_assets")
    cl = _accepted_value(inputs, "short_liabilities")
    di = _accepted_value(inputs, "deferred_income")
    assets = _accepted_value(inputs, "assets_total")
    ltl = _accepted_value(inputs, "long_liabilities")
    equity = _accepted_value(inputs, "equity")
    profit = _accepted_value(inputs, "profit")
    nca_current = _accepted_value(inputs, "nca_current")
    nca_previous = _accepted_value(inputs, "nca_previous")

    has_liquidity = _has_accepted(inputs, "current_assets") and _has_accepted(inputs, "short_liabilities")
    has_solvency = _has_accepted(inputs, "assets_total") and _has_accepted(inputs, "long_liabilities") and _has_accepted(inputs, "short_liabilities")
    has_autonomy = _has_accepted(inputs, "equity") and _has_accepted(inputs, "assets_total")
    has_roa = _has_accepted(inputs, "profit") and _has_accepted(inputs, "assets_total")
    has_growth = _has_accepted(inputs, "nca_current") and _has_accepted(inputs, "nca_previous")

    liquidity = ca / (cl - di) if has_liquidity and (cl - di) else 0.0
    solvency = assets / (ltl + cl) if has_solvency and (ltl + cl) else 0.0
    autonomy = equity / assets if has_autonomy and assets else 0.0
    roa = profit / assets * 100 if has_roa and assets else 0.0
    nca_growth = (nca_current - nca_previous) / nca_previous * 100 if has_growth and nca_previous else 0.0

    def liq(v: float) -> float: return 100 if v >= 1.1 else 50 if v >= 0.95 else 25 if v >= 0.8 else 0
    def sol(v: float) -> float: return 100 if v >= 1.3 else 50 if v >= 1.1 else 25 if v >= 1.0 else 0
    def aut(v: float) -> float: return 100 if v >= 0.5 else 50 if v >= 0.3 else 25 if v >= 0.25 else 0
    def roa_score(v: float) -> float: return 100 if v >= 10 else 50 if v >= 5 else 25 if v >= 1 else 0
    def growth(v: float) -> float: return 100 if v >= 5 else 50 if v >= 0 else 25 if v >= -5 else 0

    state_financial = pd.DataFrame(
        [
            {"indicator": "Текущая ликвидность", "value": liquidity, "raw_score": liq(liquidity) if has_liquidity else 0, "data_status": "найдено" if has_liquidity else "не найдено"},
            {"indicator": "Общая платежеспособность", "value": solvency, "raw_score": sol(solvency) if has_solvency else 0, "data_status": "найдено" if has_solvency else "не найдено"},
            {"indicator": "Автономия", "value": autonomy, "raw_score": aut(autonomy) if has_autonomy else 0, "data_status": "найдено" if has_autonomy else "не найдено"},
            {"indicator": "Рентабельность активов, %", "value": roa, "raw_score": roa_score(roa) if has_roa else 0, "data_status": "найдено" if has_roa else "не найдено"},
            {"indicator": "Рост внеоборотных активов, %", "value": nca_growth, "raw_score": growth(nca_growth) if has_growth else 0, "data_status": "найдено" if has_growth else "не найдено"},
        ]
    )
    state_fin_points = float(state_financial["raw_score"].sum() / 500 * 15)

    adverse_map = {
        row["key"]: bool(row["confirmed_adverse_fact"])
        for _, row in adverse.iterrows()
    }
    tax_load = _accepted_value(inputs, "company_tax_load")
    industry_load = _accepted_value(inputs, "industry_tax_load")
    tax_load_score = 100.0 if industry_load and tax_load / industry_load >= 1 else 0.0

    def no_adverse_points(key: str) -> float:
        return 0.0

    age = _accepted_value(inputs, "company_age")
    state_trust = pd.DataFrame(
        [
            {"indicator": "Отклонение налоговой нагрузки", "raw_score": tax_load_score, "basis": "числовые данные"},
            {"indicator": "Нет существенной недоимки", "raw_score": no_adverse_points("arrears"), "basis": "нет подтверждающей информации в загруженных документах"},
            {"indicator": "Нет офшорных учредителей", "raw_score": no_adverse_points("offshore"), "basis": "нет подтверждающей информации в загруженных документах"},
            {"indicator": "Нет исполнительного производства", "raw_score": no_adverse_points("enforcement"), "basis": "нет подтверждающей информации в загруженных документах"},
            {"indicator": "Возраст компании более пяти лет", "raw_score": 100.0 if age >= 5 else 0.0, "basis": "найденное/введённое значение"},
        ]
    )
    state_trust_points = float(state_trust["raw_score"].sum() / 500 * 35)
    state_points = state_fin_points + state_trust_points
    total = ecology_points + cadres_points + state_points

    summary = pd.DataFrame(
        [
            {"direction": "Экология", "max_points": 15.0, "points": ecology_points},
            {"direction": "Кадры", "max_points": 35.0, "points": cadres_points},
            {"direction": "Государство / Финансовая устойчивость", "max_points": 15.0, "points": state_fin_points},
            {"direction": "Государство / Налоговая история и благонадёжность", "max_points": 35.0, "points": state_trust_points},
            {"direction": "Итог", "max_points": 100.0, "points": total},
        ]
    )
    missing_info = inputs[inputs["accept_for_score"] != True].copy()
    if not missing_info.empty:
        missing_info["calculation_effect"] = "В расчёте использовано 0: значение не найдено или не подтверждено."
        missing_info = missing_info[[
            "key", "block", "label", "year", "method", "source",
            "location", "candidate_text", "review_note", "calculation_effect"
        ]]
    else:
        missing_info = pd.DataFrame(columns=[
            "key", "block", "label", "year", "method", "source",
            "location", "candidate_text", "review_note", "calculation_effect"
        ])

    return {
        "summary": summary,
        "ecology": ecology,
        "cadres": cadres,
        "state_financial": state_financial,
        "state_trust": state_trust,
        "missing_info": missing_info,
    }

