from __future__ import annotations

from html import escape
from pathlib import Path
import math
import sys

import pandas as pd
import streamlit as st


CURRENT_FILE = Path(__file__).resolve()
APP_DIR = CURRENT_FILE.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from multidataset_core import (
    build_dataset_catalog,
    delete_uploaded_dataset,
    first_available,
    get_display_row,
    load_retrieval_data,
    persist_uploaded_dataset,
    read_csv_fallback,
    reset_search_state,
    run_search_cached,
    score_for_display,
    canonicalize_mode,
    MODE_BM25,
    MODE_SBERT,
    MODE_HYBRID,
    source_display_name,
)


PAGE_SIZE = 20

st.set_page_config(
    page_title="Hybrid Retrieval Metadata Buku",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Hybrid Retrieval Metadata Buku Perpustakaan")
st.caption("BM25 • Fine-Tuned Sentence-BERT • Hybrid Retrieval")

# CSS mengikuti tampilan kartu pada aplikasi referensi pengguna.
st.markdown(
    """
    <style>
    .result-card-link {
        text-decoration: none !important;
        color: inherit !important;
        display: block;
        margin-bottom: 14px;
    }

    .result-card-link:hover {
        text-decoration: none !important;
        color: inherit !important;
    }

    .result-card {
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 16px 18px;
        background: #FFFFFF;
        transition: 0.2s ease-in-out;
    }

    .result-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
        transform: translateY(-1px);
    }

    .result-row {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 16px;
    }

    .result-left {
        display: flex;
        gap: 14px;
        flex: 1;
        min-width: 0;
    }

    .result-rank {
        min-width: 36px;
        height: 36px;
        border-radius: 999px;
        background: #F1F5F9;
        color: #0F172A;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 15px;
        margin-top: 4px;
    }

    .result-body {
        min-width: 0;
        flex: 1;
    }

    .result-title {
        font-size: 20px;
        font-weight: 700;
        color: #111827;
        margin: 0 0 6px 0;
        line-height: 1.35;
    }

    .result-subject {
        font-size: 14px;
        color: #334155;
        margin: 0 0 8px 0;
        line-height: 1.5;
    }

    .result-meta {
        font-size: 12px;
        color: #64748B;
        margin: 0;
        line-height: 1.5;
    }

    .result-score-box {
        min-width: 170px;
        text-align: right;
        flex-shrink: 0;
    }

    .result-score-label {
        font-size: 11px;
        color: #64748B;
        margin-bottom: 4px;
    }

    .result-score-value {
        font-size: 16px;
        font-weight: 700;
        color: #0F172A;
    }

    .info-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 12px 14px;
        border-radius: 12px;
        margin-bottom: 12px;
        color: #334155;
        font-size: 14px;
    }

    @media (max-width: 760px) {
        .result-row {
            flex-direction: column;
        }

        .result-score-box {
            min-width: 0;
            width: 100%;
            text-align: left;
            padding-left: 50px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# State
# ============================================================

defaults = {
    "active_dataset_key": "opac-usu",
    "last_results": None,
    "last_query": "",
    "last_mode": MODE_HYBRID,
    "last_elapsed": None,
    "current_page": 1,
    "query_input": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "pending_dataset_selection" in st.session_state:
    st.session_state.active_dataset_key = (
        st.session_state.pop("pending_dataset_selection")
    )


# ============================================================
# Alert satu kali setelah proses dan rerun
# ============================================================

FLASH_ALERT_KEY = "_flash_alert"


def queue_alert(
    message: str,
    alert_type: str = "success",
) -> None:
    """Menyimpan alert agar tetap muncul setelah st.rerun()."""
    st.session_state[FLASH_ALERT_KEY] = {
        "message": message,
        "type": alert_type,
    }


def render_queued_alert() -> None:
    """
    Menampilkan notifikasi selama 5 detik, lalu menghilang otomatis.

    Alert disimpan sementara di session_state agar tetap dapat muncul
    setelah st.rerun(), kemudian langsung dihapus setelah dibaca.
    """
    alert = st.session_state.pop(FLASH_ALERT_KEY, None)
    if not isinstance(alert, dict):
        return

    message = str(alert.get("message", "")).strip()
    alert_type = str(alert.get("type", "success")).strip().lower()

    if not message:
        return

    icons = {
        "success": "✅",
        "info": "ℹ️",
        "warning": "⚠️",
        "error": "❌",
    }

    try:
        # Streamlit versi baru mendukung durasi dalam satuan detik.
        st.toast(
            message,
            icon=icons.get(alert_type, "✅"),
            duration=5,
        )
    except TypeError:
        # Fallback untuk Streamlit lama. Default toast tampil ±4 detik.
        st.toast(
            message,
            icon=icons.get(alert_type, "✅"),
        )


render_queued_alert()


# ============================================================
# Routing detail menggunakan st.button + st.switch_page agar session_state tetap utuh.

# ============================================================
# Dataset selector dan penambahan permanen
# ============================================================

# Bangun katalog sebelum digunakan oleh selector dan validasi state.
catalog = build_dataset_catalog()

if st.session_state.active_dataset_key not in catalog:
    st.session_state.active_dataset_key = "opac-usu"

dataset_keys = list(catalog.keys()) + ["__add_new__"]


def dataset_label(dataset_key: str) -> str:
    if dataset_key == "__add_new__":
        return "➕ Tambah Dataset Baru"
    return catalog[dataset_key]["label"]


selected_key = st.sidebar.selectbox(
    "Sumber dataset",
    options=dataset_keys,
    index=dataset_keys.index(
        st.session_state.active_dataset_key
    ),
    format_func=dataset_label,
    key="dataset_selector_widget",
)

if selected_key == "__add_new__":
    st.sidebar.markdown("### Tambah Dataset Permanen")

    uploaded_file = st.sidebar.file_uploader(
        "Unggah CSV",
        type=["csv"],
        help=(
            "Saat disimpan, aplikasi langsung menjalankan preprocessing, "
            "membentuk indeks BM25, dan membuat embedding SBERT float16."
        ),
    )

    if uploaded_file is None:
        st.info(
            "Pilih CSV melalui sidebar. Dataset yang berhasil diproses "
            "akan tersimpan permanen dan langsung siap digunakan pada "
            "BM25, Fine-Tuned Sentence-BERT, dan Hybrid Retrieval."
        )
        st.stop()

    if uploaded_file.size > 50 * 1024 * 1024:
        st.error("Ukuran file maksimal 50 MB.")
        st.stop()

    uploaded_bytes = uploaded_file.getvalue()
    raw_df = read_csv_fallback(uploaded_bytes)

    if len(raw_df) > 100_000:
        st.error("Dataset maksimal 100.000 baris.")
        st.stop()

    if len(raw_df.columns) < 2:
        st.error("CSV minimal memiliki dua kolom.")
        st.stop()

    columns = list(raw_df.columns)
    lowered = {
        str(column).strip().lower(): index
        for index, column in enumerate(columns)
    }

    title_index = next(
        (
            lowered[name]
            for name in [
                "title", "judul", "judul_buku",
                "book_title", "name",
            ]
            if name in lowered
        ),
        0,
    )
    subject_index = next(
        (
            lowered[name]
            for name in [
                "subject", "subjek", "category",
                "categories", "genre", "genres",
            ]
            if name in lowered
        ),
        min(1, len(columns) - 1),
    )

    dataset_name = st.sidebar.text_input(
        "Nama dataset",
        value=Path(uploaded_file.name).stem,
    ).strip()

    title_column = st.sidebar.selectbox(
        "Kolom judul",
        columns,
        index=title_index,
    )
    subject_column = st.sidebar.selectbox(
        "Kolom subjek/kategori",
        columns,
        index=subject_index,
    )

    st.sidebar.caption(
        "Proses Simpan Dataset mencakup preprocessing, penyimpanan CSV, "
        "indeks BM25, dan embedding SBERT float16."
    )

    if st.sidebar.button(
        "Simpan dan Siapkan Hybrid",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Memproses dataset dan membuat embedding SBERT..."
            ):
                entry = persist_uploaded_dataset(
                    uploaded_bytes=uploaded_bytes,
                    uploaded_filename=uploaded_file.name,
                    raw_df=raw_df,
                    title_column=title_column,
                    subject_column=subject_column,
                    dataset_name=dataset_name,
                )

            st.session_state.pending_dataset_selection = entry["key"]
            reset_search_state()
            st.session_state.query_input = ""
            queue_alert(
                f"Dataset '{entry['name']}' berhasil ditambahkan, "
                "dipreproses, dan siap digunakan untuk BM25, "
                "Fine-Tuned Sentence-BERT, serta Hybrid Retrieval.",
                "success",
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Dataset gagal ditambahkan: {exc}")

    st.stop()


if selected_key != st.session_state.active_dataset_key:
    selected_dataset_label = catalog[selected_key]["label"]
    st.session_state.active_dataset_key = selected_key
    reset_search_state()
    st.session_state.query_input = ""
    queue_alert(
        f"Dataset aktif berhasil diubah menjadi "
        f"'{selected_dataset_label}'.",
        "info",
    )
    st.rerun()

active_key = st.session_state.active_dataset_key
active_config = catalog[active_key]
retrieval_df = load_retrieval_data(active_key)

st.sidebar.caption(
    f"Dataset aktif: **{active_config['label']}**  \n"
    f"Jumlah metadata: **{len(retrieval_df):,}**"
)

if active_config.get("is_user_dataset"):
    with st.sidebar.expander("Kelola dataset ini"):
        st.caption(
            "CSV hasil preprocessing dan embedding tersimpan di folder "
            "proyek sehingga dataset tetap tersedia setelah restart."
        )
        if st.button(
            "Hapus Dataset",
            type="secondary",
            use_container_width=True,
        ):
            deleted_dataset_name = active_config["label"]
            try:
                delete_uploaded_dataset(active_key)
                st.session_state.pending_dataset_selection = "opac-usu"
                reset_search_state()
                st.session_state.query_input = ""
                queue_alert(
                    f"Dataset '{deleted_dataset_name}' berhasil dihapus. "
                    "Dataset aktif dikembalikan ke OPAC USU.",
                    "success",
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Dataset gagal dihapus: {exc}")


# ============================================================
# Pengaturan dan pencarian
# ============================================================

st.sidebar.markdown("### Pengaturan Pencarian")

st.session_state.last_mode = canonicalize_mode(
    st.session_state.last_mode
)
modes = [MODE_BM25, MODE_SBERT, MODE_HYBRID]
mode = st.sidebar.selectbox(
    "Metode",
    modes,
    index=(
        modes.index(st.session_state.last_mode)
        if st.session_state.last_mode in modes
        else 2
    ),
)

# Saat metode diganti, query terakhir dijalankan ulang seperti aplikasi referensi.
if (
    mode != st.session_state.last_mode
    and st.session_state.last_query
):
    try:
        results, elapsed = run_search_cached(
            active_key,
            st.session_state.last_query,
            mode,
        )
        st.session_state.last_results = results
        st.session_state.last_mode = mode
        st.session_state.last_elapsed = elapsed
        st.session_state.current_page = 1
        queue_alert(
            f"Metode pencarian berhasil diubah menjadi '{mode}'. "
            f"Ditemukan {len(results):,} metadata dalam "
            f"{elapsed:.2f} detik.",
            "success",
        )
        st.rerun()
    except Exception as exc:
        st.error(f"Terjadi error saat mengganti metode: {exc}")
        st.stop()


def clear_search_form() -> None:
    """Mengosongkan query dan hasil pencarian sebelum widget dirender ulang."""
    reset_search_state()
    st.session_state["query_input"] = ""
    queue_alert(
        "Query dan hasil pencarian metadata berhasil dibersihkan.",
        "success",
    )


with st.form("search_form"):
    col_query, col_search, col_clear = st.columns([8, 1.2, 1.2])

    with col_query:
        st.text_input(
            "Masukkan query pencarian",
            key="query_input",
            placeholder=(
                "Contoh: basis data, algoritma, pemrograman python, "
                "akuntansi"
            ),
        )

    with col_search:
        st.markdown(
            "<div style='height:28px'></div>",
            unsafe_allow_html=True,
        )
        submitted = st.form_submit_button(
            "Cari",
            use_container_width=True,
        )

    with col_clear:
        st.markdown(
            "<div style='height:28px'></div>",
            unsafe_allow_html=True,
        )
        cleared = st.form_submit_button(
            "Clear",
            use_container_width=True,
            on_click=clear_search_form,
        )

if submitted:
    cleaned_query = st.session_state.query_input.strip()

    if not cleaned_query:
        st.warning("Query tidak boleh kosong.")
    else:
        try:
            results, elapsed = run_search_cached(
                active_key,
                cleaned_query,
                mode,
            )
            st.session_state.last_results = results
            st.session_state.last_query = cleaned_query
            st.session_state.last_mode = mode
            st.session_state.last_elapsed = elapsed
            st.session_state.current_page = 1

            if results.empty:
                queue_alert(
                    f"Pencarian untuk query '{cleaned_query}' selesai, "
                    "tetapi tidak ditemukan metadata yang sesuai.",
                    "info",
                )
            else:
                queue_alert(
                    f"Pencarian metadata berhasil. Ditemukan "
                    f"{len(results):,} hasil untuk query "
                    f"'{cleaned_query}' menggunakan metode '{mode}' "
                    f"dalam {elapsed:.2f} detik.",
                    "success",
                )
            st.rerun()
        except Exception as exc:
            st.error(f"Terjadi error saat pencarian: {exc}")


# ============================================================
# Hasil pencarian
# ============================================================

st.markdown("---")
results = st.session_state.last_results

if results is None:
    st.info("Silakan masukkan query untuk memulai pencarian.")
    st.stop()

st.subheader("Hasil Pencarian")
st.write(f"**Query:** `{st.session_state.last_query}`")
st.write(f"**Mode:** {st.session_state.last_mode}")
st.write(f"**Sumber Dataset:** {active_config['label']}")

if st.session_state.last_elapsed is not None:
    st.write(
        f"**Waktu proses:** "
        f"{st.session_state.last_elapsed:.2f} detik"
    )

if results.empty:
    st.info("Tidak ada hasil yang ditemukan.")
    st.stop()

results = results.reset_index(drop=True)
total_results = len(results)
total_pages = max(1, math.ceil(total_results / PAGE_SIZE))
st.session_state.current_page = min(
    st.session_state.current_page,
    total_pages,
)

st.markdown(
    f"""
    <div class="info-box">
        Menampilkan {total_results} hasil. Setiap halaman berisi
        maksimal {PAGE_SIZE} metadata buku.
    </div>
    """,
    unsafe_allow_html=True,
)


def render_pagination(location: str) -> None:
    col_previous, col_page, col_next = st.columns([1, 2, 1])

    with col_previous:
        if st.button(
            "← Sebelumnya",
            disabled=st.session_state.current_page <= 1,
            key=f"previous-{location}",
        ):
            st.session_state.current_page -= 1
            st.rerun()

    with col_page:
        st.markdown(
            f"<div style='text-align:center;padding-top:8px'>"
            f"Halaman <b>{st.session_state.current_page}</b> "
            f"dari <b>{total_pages}</b></div>",
            unsafe_allow_html=True,
        )

    with col_next:
        if st.button(
            "Berikutnya →",
            disabled=st.session_state.current_page >= total_pages,
            key=f"next-{location}",
        ):
            st.session_state.current_page += 1
            st.rerun()


render_pagination("top")

start = (st.session_state.current_page - 1) * PAGE_SIZE
page_df = results.iloc[start:start + PAGE_SIZE].reset_index(drop=True)

for index, result_row in page_df.iterrows():
    doc_id = str(result_row.get("doc_id", ""))
    result_source = first_available(
        result_row,
        ["source_dataset"],
    )

    metadata_row = get_display_row(
        active_key,
        doc_id,
        None if result_source == "-" else result_source,
    )
    row = metadata_row if metadata_row is not None else result_row

    title = first_available(
        row,
        ["judul_buku", "title", "judul"],
    )
    subject = first_available(
        row,
        [
            "subjek", "subject", "categories",
            "category", "genre",
        ],
    )
    author = first_available(
        row,
        ["pengarang", "authors", "author", "penulis"],
    )
    year = first_available(
        row,
        ["tahun", "published_year", "year", "tahun_terbit"],
    )

    raw_source = first_available(row, ["source_dataset"])
    if raw_source == "-":
        raw_source = active_config["label"]
    display_source = source_display_name(raw_source)

    # Jenis literatur sengaja tidak ditampilkan pada tile.
    metadata_parts = [
        value for value in [author, year]
        if value != "-"
    ]
    metadata_parts.append(f"Sumber: {display_source}")
    metadata_text = " • ".join(metadata_parts)

    rank_number = start + index + 1
    score = score_for_display(
        result_row,
        st.session_state.last_mode,
    )

    # Card tetap memakai CSS referensi, tetapi navigasi dilakukan dengan
    # tombol native Streamlit agar query, metode, hasil, dan dataset_key
    # tidak hilang ketika pindah ke halaman detail.
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-row">
                <div class="result-left">
                    <div class="result-rank">{rank_number}</div>
                    <div class="result-body">
                        <div class="result-title">
                            {escape(title)}
                        </div>
                        <div class="result-subject">
                            {escape(subject)}
                        </div>
                        <div class="result-meta">
                            {escape(metadata_text)}
                        </div>
                    </div>
                </div>
                <div class="result-score-box">
                    <div class="result-score-label">Skor Raw</div>
                    <div class="result-score-value">
                        {score:.4f}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "Buka Detail Metadata",
        key=f"open-detail-{active_key}-{doc_id}-{start + index}",
        use_container_width=True,
    ):
        # Konteks detail disimpan eksplisit sebelum berpindah halaman.
        # Tidak ada fallback ke OPAC pada halaman detail.
        st.session_state.detail_context = {
            "dataset_key": active_key,
            "doc_id": doc_id,
            "source_dataset": raw_source,
            "result_row": result_row.to_dict(),
            "query": st.session_state.get("last_query", ""),
            "mode": st.session_state.get(
                "last_mode", MODE_HYBRID
            ),
            "return_page": st.session_state.get("current_page", 1),
        }
        st.switch_page("pages/1_Detail_Metadata.py")


st.markdown("---")
render_pagination("bottom")

st.markdown("---")
st.markdown(
    """
Aplikasi mendukung:
- **BM25** untuk *lexical retrieval*
- **Fine-Tuned Sentence-BERT** untuk *semantic retrieval*
- **Hybrid Retrieval** untuk gabungan BM25 dan SBERT *fine-tuned*
"""
)
