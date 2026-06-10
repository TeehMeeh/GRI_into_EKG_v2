from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import torch

from ekg_pipeline import (
    SOURCE_GROUP_LABELS,
    RetrievalEngine,
    calculate_score,
    make_excel_report,
    parse_documents,
)
from ekg_translations import COLUMN_TRANSLATIONS, translate_for_display

MODEL_ID = "ai-forever/ru-en-RoSBERTa"
DEFAULT_SCORING_YEAR = 2025


def _base_column_config() -> dict[str, object]:
    return {key: st.column_config.Column(label) for key, label in COLUMN_TRANSLATIONS.items()}


st.set_page_config(page_title="Перевод GRI в ЭКГ", layout="wide")
st.title("Перевод GRI-отчета в ЭКГ-рейтинг")

scoring_year = st.number_input(
    "Расчётный год",
    min_value=2018,
    max_value=2035,
    value=DEFAULT_SCORING_YEAR,
    step=1,
)

st.subheader("1. Загрузка источников")
col_gov, col_other = st.columns(2)

with col_gov:
    government_uploads = st.file_uploader(
        "Государственные источники",
        type=["xls", "xlsx", "pdf"],
        accept_multiple_files=True,
        key="government_uploads",
    )

with col_other:
    other_uploads = st.file_uploader(
        "Прочие источники",
        type=["xls", "xlsx", "pdf"],
        accept_multiple_files=True,
        key="other_uploads",
    )

uploads_with_groups = []
for item in government_uploads or []:
    uploads_with_groups.append((item, "government_sources"))
for item in other_uploads or []:
    uploads_with_groups.append((item, "other_sources"))

if uploads_with_groups:
    st.info(f"Загружено файлов: {len(uploads_with_groups)}.")

    if st.button("Рассчитать входные данные", type="primary"):
        workdir = Path(tempfile.mkdtemp(prefix="ekg_upload_"))
        file_rows = []
        used_names: dict[str, int] = {}

        for item, source_group in uploads_with_groups:
            original_name = item.name
            used_names[original_name] = used_names.get(original_name, 0) + 1
            stored_name = original_name
            if used_names[original_name] > 1:
                stem = Path(original_name).stem
                suffix = Path(original_name).suffix
                stored_name = f"{stem}_{used_names[original_name]}{suffix}"
            file_bytes = item.getvalue()
            (workdir / stored_name).write_bytes(file_bytes)
            file_rows.append(
                {
                    "filename": stored_name,
                    "original_filename": original_name,
                    "size_kb": round(len(file_bytes) / 1024, 1),
                    "source_group": source_group,
                    "source_group_label": SOURCE_GROUP_LABELS[source_group],
                }
            )

        documents_df = pd.DataFrame(file_rows)

        with st.spinner("Извлекаю данные из документов..."):
            fragments, extraction_audit = parse_documents(
                documents_df,
                workdir,
                enable_ocr=False,
                ocr_language="rus+eng",
                ocr_min_chars=10**9,
            )

        if fragments.empty:
            st.error("Не удалось получить фрагменты из загруженных файлов.")
            st.stop()

        with st.spinner("Загружаю RoSBERTa и выполняю поиск кандидатов..."):
            @st.cache_resource(show_spinner=False)
            def load_model() -> SentenceTransformer:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                cache_dir = str(Path.cwd() / ".rosberta_model_cache")
                return SentenceTransformer(MODEL_ID, cache_folder=cache_dir, device=device)

            model = load_model()
            engine = RetrievalEngine(fragments, model)
            input_candidates = engine.build_input_candidates(int(scoring_year))
            negative_candidates = engine.build_negative_candidates()

        st.session_state["ekg_data"] = {
            "fragments": fragments,
            "extraction_audit": extraction_audit,
            "documents": documents_df,
            "inputs": input_candidates,
            "negative": negative_candidates,
            "scoring_year": int(scoring_year),
            "workdir": str(workdir),
        }

if "ekg_data" in st.session_state:
    data = st.session_state["ekg_data"]

    tab_inputs, tab_negative, tab_index = st.tabs(
        ["Полученные данные", "Негативные факты", "Индекс"]
    )

    with tab_inputs:
        input_column_config = _base_column_config()
        input_column_config.update(
            {
                "accept_for_score": st.column_config.CheckboxColumn("Принять в расчёт"),
                "value_used": st.column_config.NumberColumn("Значение для расчёта", format="%.6f"),
                "candidate_text": st.column_config.TextColumn("Фрагмент документа", width="large"),
                "review_note": st.column_config.TextColumn("Комментарий", width="large"),
            }
        )
        edited_inputs = st.data_editor(
            data["inputs"],
            hide_index=True,
            disabled=[
                "key", "block", "label", "year", "unit", "method", "e5_score",
                "source", "location", "candidate_text", "review_note",
            ],
            column_config=input_column_config,
            key="inputs_editor",
            use_container_width=True,
        )

    with tab_negative:
        negative_column_config = _base_column_config()
        negative_column_config.update(
            {
                "confirmed_adverse_fact": st.column_config.CheckboxColumn("Негативный факт подтверждён"),
                "candidate_text": st.column_config.TextColumn("Фрагмент документа", width="large"),
            }
        )
        edited_negative = st.data_editor(
            data["negative"],
            hide_index=True,
            disabled=["key", "label", "method", "source", "location", "candidate_text"],
            column_config=negative_column_config,
            key="negative_editor",
            use_container_width=True,
        )

    with tab_index:
        st.write("Документы")
        st.dataframe(translate_for_display(data["documents"]), use_container_width=True, hide_index=True)
        st.write("Журнал извлечения PDF")
        st.dataframe(translate_for_display(data["extraction_audit"]), use_container_width=True, hide_index=True)
        st.write("Первые 200 фрагментов индекса")
        st.dataframe(translate_for_display(data["fragments"].head(200)), use_container_width=True, hide_index=True)

    if st.button("Рассчитать score и подготовить Excel", type="primary"):
        scoring = calculate_score(
            edited_inputs,
            edited_negative,
            int(data["scoring_year"]),
            optimistic_missing_negative=False,
        )
        total = float(scoring["summary"].iloc[-1]["points"])
        st.subheader("Результат")
        col1, col2 = st.columns([1, 2])
        col1.metric("Score, из 100", f"{total:.2f}")
        col2.dataframe(translate_for_display(scoring["summary"]), hide_index=True, use_container_width=True)

        st.subheader("Не найдено / не принято в расчёт")
        if scoring["missing_info"].empty:
            st.success("Все требуемые входы были найдены и приняты в расчёт.")
        else:
            st.dataframe(translate_for_display(scoring["missing_info"]), hide_index=True, use_container_width=True)

        settings = {
            "scoring_year": int(data["scoring_year"]),
            "optimistic_missing_negative": False,
            "ocr_enabled": False,
            "model": MODEL_ID,
            "source_priority": "Государственные источники > Прочие источники",
            "note": "В расчёт входят только найденные и принятые значения.",
        }
        excel = make_excel_report(
            scoring,
            edited_inputs,
            edited_negative,
            data["fragments"],
            data["documents"],
            data["extraction_audit"],
            settings,
        )
        st.download_button(
            "Скачать Excel-отчёт",
            data=excel,
            file_name="ЭКГ_результат_только_по_найденным_данным.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
else:
    st.info("Загрузите документы и нажмите «Рассчитать входные данные».")
