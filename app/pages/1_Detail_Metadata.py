from __future__ import annotations

from html import escape
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


CURRENT_FILE = Path(__file__).resolve()
APP_DIR = CURRENT_FILE.parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from multidataset_core import (
    ALPHA,
    BM25_B,
    BM25_K1,
    build_detail_table,
    clean_text,
    first_available,
    get_bm25_engine,
    get_dataset_config,
    get_dense_engine,
    get_display_row,
    load_retrieval_data,
    normalize_source,
    run_search_cached,
    safe_text,
    source_display_name,
    canonicalize_mode,
    MODE_BM25,
    MODE_SBERT,
    MODE_HYBRID,
    RRF_K,
)


st.set_page_config(
    page_title="Detail Metadata Buku",
    page_icon="📖",
    layout="wide",
)

st.markdown(
    """
    <style>
    .detail-header {
        padding: 6px 0 14px 0;
    }

    .detail-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 6px;
        color: #111827;
        line-height: 1.3;
    }

    .detail-subject {
        font-size: 16px;
        color: #475569;
        margin-bottom: 8px;
    }

    .detail-meta {
        font-size: 13px;
        color: #64748B;
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
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Konteks detail: tidak boleh fallback ke OPAC
# ============================================================

context = st.session_state.get("detail_context")

if not isinstance(context, dict):
    st.warning(
        "Metadata belum dipilih dari halaman hasil pencarian."
    )
    if st.button("← Kembali ke Pencarian"):
        st.switch_page("streamlit_app.py")
    st.stop()

dataset_key = str(context.get("dataset_key", "")).strip()
doc_id = str(context.get("doc_id", "")).strip()
source = str(context.get("source_dataset", "")).strip()
query = str(
    context.get("query")
    or st.session_state.get("last_query")
    or st.session_state.get("query_input")
    or ""
).strip()
mode = str(
    context.get("mode")
    or st.session_state.get("last_mode")
    or MODE_HYBRID
).strip()
mode = canonicalize_mode(mode)

# Pulihkan state utama agar tombol kembali menampilkan hasil pencarian
# dengan query dan metode yang sama.
if query:
    st.session_state.last_query = query
    st.session_state.query_input = query
if mode:
    st.session_state.last_mode = mode
result_row_dict = context.get("result_row")

if not dataset_key or not doc_id:
    st.error("Identitas dataset atau dokumen tidak lengkap.")
    if st.button("← Kembali ke Pencarian"):
        st.switch_page("streamlit_app.py")
    st.stop()

try:
    dataset_config = get_dataset_config(dataset_key)
except Exception:
    st.error(
        "Dataset asal tidak ditemukan. Detail tidak dialihkan ke OPAC "
        "karena dapat menampilkan metadata yang salah."
    )
    if st.button("← Kembali ke Pencarian"):
        st.switch_page("streamlit_app.py")
    st.stop()

# Dataset pengguna selalu memiliki doc_id berawalan dataset_key.
if (
    dataset_config.get("is_user_dataset")
    and not doc_id.startswith(f"{dataset_key}::")
):
    st.error(
        "Identitas dokumen tidak cocok dengan dataset asal. "
        "Halaman detail dibatalkan agar tidak mengambil metadata OPAC."
    )
    if st.button("← Kembali ke Pencarian"):
        st.switch_page("streamlit_app.py")
    st.stop()

row = get_display_row(
    dataset_key=dataset_key,
    doc_id=doc_id,
    source_dataset=source or None,
)

if row is None:
    st.error(
        "Metadata tidak ditemukan pada dataset asal. Sistem tidak "
        "melakukan fallback ke dataset lain."
    )
    if st.button("← Kembali ke Hasil Pencarian"):
        st.switch_page("streamlit_app.py")
    st.stop()

# Validasi tambahan: doc_id yang dibuka harus identik.
if str(row.get("doc_id", "")) != doc_id:
    st.error("Metadata yang ditemukan tidak memiliki ID yang sesuai.")
    st.stop()

row_source = first_available(row, ["source_dataset"])
if (
    source
    and row_source != "-"
    and normalize_source(source) != normalize_source(row_source)
):
    st.error(
        "Sumber metadata tidak cocok dengan hasil pencarian. "
        "Detail tidak ditampilkan untuk mencegah pertukaran dataset."
    )
    st.stop()

if st.button("← Kembali ke Hasil Pencarian"):
    st.switch_page("streamlit_app.py")


# ============================================================
# Informasi metadata
# ============================================================

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

source_name = first_available(row, ["source_dataset"])
if source_name == "-":
    source_name = dataset_config["label"]
source_name = source_display_name(source_name)

meta_parts = [
    value for value in [author, year]
    if value != "-"
]
meta_parts.append(f"Sumber: {source_name}")

st.markdown(
    f"""
    <div class="detail-header">
        <div class="detail-title">{escape(title)}</div>
        <div class="detail-subject">{escape(subject)}</div>
        <div class="detail-meta">
            {escape(" • ".join(meta_parts))}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

thumbnail = first_available(
    row,
    ["thumbnail", "image_url", "cover_url"],
)
description = first_available(
    row,
    ["deskripsi", "description", "summary", "abstract"],
)

if (
    thumbnail != "-"
    and thumbnail.lower().startswith(("http://", "https://"))
):
    st.image(
        thumbnail,
        width=190,
        caption=title,
    )

st.markdown("## Informasi Detail Metadata")
detail_table = build_detail_table(row)

if detail_table.empty:
    st.info("Tidak ada metadata tambahan yang tersedia.")
else:
    st.dataframe(
        detail_table,
        use_container_width=True,
        hide_index=True,
    )

if description != "-":
    st.markdown("## Deskripsi")
    st.write(description)


# ============================================================
# Ambil row skor yang benar dari dataset yang sama
# ============================================================

def resolve_result_row() -> pd.Series | None:
    if isinstance(result_row_dict, dict):
        candidate = pd.Series(result_row_dict)
        if str(candidate.get("doc_id", "")) == doc_id:
            return candidate

    if not query or not mode:
        return None

    try:
        results, _ = run_search_cached(
            dataset_key=dataset_key,
            query=query,
            mode=mode,
        )
    except Exception as exc:
        st.warning(
            f"Skor pencarian tidak dapat dimuat ulang: {exc}"
        )
        return None

    if results is None or results.empty:
        return None

    matched = results[
        results["doc_id"].astype(str) == doc_id
    ]

    if source and "source_dataset" in matched.columns:
        source_matched = matched[
            matched["source_dataset"].map(normalize_source)
            == normalize_source(source)
        ]
        if not source_matched.empty:
            matched = source_matched

    return None if matched.empty else matched.iloc[0]


result_row = resolve_result_row()


# ============================================================
# Fungsi analisis dari aplikasi referensi
# ============================================================

def simple_tokenize_for_explain(text: str) -> list[str]:
    text = "" if text is None else str(text).lower()
    text = re.sub(
        r"[^a-zA-Z0-9\u00C0-\u024F\u1E00-\u1EFF\s]",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return [token for token in text.split() if token]


def document_text_for_explain() -> str:
    values = [
        value for value in [title, subject]
        if value != "-"
    ]
    return " ".join(values)


def highlight_matching_tokens(
    text_value: str,
    query_value: str,
) -> str:
    text_value = safe_text(text_value)
    query_tokens = simple_tokenize_for_explain(query_value)

    if not query_tokens or text_value == "-":
        return escape(text_value)

    highlighted = escape(text_value)
    for token in sorted(set(query_tokens), key=len, reverse=True):
        pattern = re.compile(
            rf"(?i)\b({re.escape(token)})\b"
        )
        highlighted = pattern.sub(
            r"<span style='color:#DC2626;font-weight:700;'>\1</span>",
            highlighted,
        )
    return highlighted


def highlight_query_tokens(
    query_value: str,
    document_text: str,
) -> str:
    query_tokens = simple_tokenize_for_explain(query_value)
    document_tokens = set(
        simple_tokenize_for_explain(document_text)
    )

    output: list[str] = []
    for token in query_tokens:
        escaped_token = escape(token)
        if token in document_tokens:
            output.append(
                "<span style='color:#DC2626;font-weight:700;'>"
                f"{escaped_token}</span>"
            )
        else:
            output.append(escaped_token)

    return " ".join(output) if output else "-"


def render_bm25_analysis() -> None:
    st.markdown("### Analisis Relevansi BM25")
    st.write(
        "BM25 menilai relevansi berdasarkan kecocokan token query "
        "dengan metadata buku. Analisis berikut menunjukkan TF, DF, "
        "IDF, dan kontribusi setiap token terhadap skor."
    )

    engine = get_bm25_engine(dataset_key)
    tokenizer = getattr(engine, "tokenize", None)

    if callable(tokenizer):
        query_tokens = tokenizer(query)
    else:
        query_tokens = clean_text(query).split()

    if not query_tokens:
        st.info("Query tidak menghasilkan token yang dapat dianalisis.")
        return

    engine_df = getattr(engine, "df", None)
    tokenized_corpus = getattr(engine, "tokenized_corpus", None)
    bm25_model = getattr(engine, "bm25", None)

    if (
        engine_df is None
        or tokenized_corpus is None
        or bm25_model is None
        or "doc_id" not in engine_df.columns
    ):
        st.warning(
            "Struktur indeks BM25 tidak lengkap untuk analisis."
        )
        return

    positions = np.flatnonzero(
        engine_df["doc_id"].astype(str).to_numpy() == doc_id
    )
    if len(positions) == 0:
        st.warning(
            "Dokumen tidak ditemukan pada indeks BM25 dataset asal."
        )
        return

    selected_position = int(positions[0])
    selected_tokens = tokenized_corpus[selected_position]

    corpus_size = len(tokenized_corpus)
    document_length = len(selected_tokens)
    average_length = float(
        getattr(bm25_model, "avgdl", 0.0) or 0.0
    )
    k1_value = float(
        getattr(bm25_model, "k1", BM25_K1)
    )
    b_value = float(
        getattr(bm25_model, "b", BM25_B)
    )

    st.markdown("**Query:**")
    st.markdown(
        highlight_query_tokens(
            query,
            document_text_for_explain(),
        ),
        unsafe_allow_html=True,
    )

    st.markdown("**Judul:**")
    st.markdown(
        highlight_matching_tokens(title, query),
        unsafe_allow_html=True,
    )

    st.markdown("**Subjek/Kategori:**")
    st.markdown(
        highlight_matching_tokens(subject, query),
        unsafe_allow_html=True,
    )

    col_n, col_dl, col_avgdl, col_param = st.columns(4)
    col_n.metric(
        "Jumlah metadata (N)",
        f"{corpus_size:,}".replace(",", "."),
    )
    col_dl.metric(
        "Panjang dokumen",
        f"{document_length} token",
    )
    col_avgdl.metric(
        "Rata-rata panjang",
        f"{average_length:.2f}".replace(".", ","),
    )
    col_param.metric(
        "Parameter",
        f"k1={k1_value:.2f}; b={b_value:.2f}".replace(".", ","),
    )

    rows: list[dict[str, object]] = []
    total_contribution = 0.0
    total_match_frequency = 0

    for token in dict.fromkeys(query_tokens):
        query_frequency = query_tokens.count(token)
        term_frequency = selected_tokens.count(token)
        total_match_frequency += term_frequency

        document_frequency = sum(
            1
            for document_tokens in tokenized_corpus
            if token in document_tokens
        )
        document_percentage = (
            document_frequency / corpus_size * 100.0
            if corpus_size
            else 0.0
        )

        idf_value = float(
            getattr(bm25_model, "idf", {}).get(token, 0.0)
        )

        length_normalization = (
            1.0
            - b_value
            + b_value * (
                document_length / average_length
            )
            if average_length > 0
            else 1.0
        )

        denominator = (
            term_frequency
            + k1_value * length_normalization
        )
        contribution_once = (
            idf_value
            * (
                term_frequency * (k1_value + 1.0)
                / denominator
            )
            if term_frequency > 0 and denominator > 0
            else 0.0
        )
        token_contribution = (
            contribution_once * query_frequency
        )
        total_contribution += token_contribution

        rows.append(
            {
                "Token Query": token,
                "TF Query": query_frequency,
                "TF Dokumen": term_frequency,
                "DF Korpus": (
                    f"{document_frequency} dari {corpus_size}"
                ),
                "% Metadata": (
                    f"{document_percentage:.2f}%"
                    .replace(".", ",")
                ),
                "IDF": f"{idf_value:.4f}".replace(".", ","),
                "Kontribusi Skor": (
                    f"{token_contribution:.4f}"
                    .replace(".", ",")
                ),
            }
        )

    st.markdown("**Rincian kontribusi setiap token query:**")
    st.table(pd.DataFrame(rows))
    st.caption(
        "DF menunjukkan jumlah metadata yang memuat token. "
        "Token yang lebih jarang umumnya memiliki IDF lebih besar."
    )

    raw_score = 0.0
    if result_row is not None:
        value = result_row.get("bm25_score", 0.0)
        if value is not None and not pd.isna(value):
            raw_score = float(value)

    st.markdown("**Perhitungan ringkas:**")
    st.write(
        (
            f"Jumlah kontribusi token = **{total_contribution:.4f}**, "
            f"sedangkan BM25 raw score pada hasil retrieval = "
            f"**{raw_score:.4f}**."
        ).replace(".", ",")
    )

    if abs(total_contribution - raw_score) > 1e-4:
        st.warning(
            "Jumlah kontribusi belum sama dengan raw score. Hal ini "
            "dapat terjadi bila parameter, preprocessing, atau indeks "
            "analisis berbeda dari proses pencarian."
        )

    if total_match_frequency == 0:
        st.info(
            "Tidak terdapat token query yang cocok secara langsung "
            "pada metadata ini."
        )


def render_sbert_analysis() -> None:
    st.markdown(
        "### Analisis Relevansi Sentence-BERT Fine-Tuned"
    )

    engine = get_dense_engine(dataset_key)
    engine_df = getattr(engine, "df", None)
    embeddings = getattr(engine, "doc_embeddings", None)
    model = getattr(engine, "model", None)

    if (
        engine_df is None
        or embeddings is None
        or model is None
        or "doc_id" not in engine_df.columns
    ):
        st.info(
            "Data atau embedding dokumen belum tersedia."
        )
        return

    positions = np.flatnonzero(
        engine_df["doc_id"].astype(str).to_numpy() == doc_id
    )
    if len(positions) == 0:
        st.info(
            "Embedding dokumen tidak ditemukan pada dataset asal."
        )
        return

    selected_position = int(positions[0])

    cleaned_query = (
        engine.clean_text(query)
        if hasattr(engine, "clean_text")
        else clean_text(query)
    )

    if hasattr(engine, "encode_query"):
        query_embedding = engine.encode_query(cleaned_query)
    elif hasattr(model, "encode_query"):
        query_embedding = model.encode_query(
            cleaned_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    else:
        query_embedding = model.encode(
            [cleaned_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

    document_embedding = np.asarray(
        embeddings[selected_position],
        dtype=np.float32,
    )
    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).reshape(-1)

    query_norm = float(np.linalg.norm(query_embedding))
    document_norm = float(np.linalg.norm(document_embedding))

    if query_norm == 0.0 or document_norm == 0.0:
        st.warning(
            "Norm embedding bernilai nol sehingga cosine similarity "
            "tidak dapat dihitung."
        )
        return

    query_unit = query_embedding / query_norm
    document_unit = document_embedding / document_norm

    similarity = float(
        np.clip(
            np.dot(query_unit, document_unit),
            -1.0,
            1.0,
        )
    )
    cosine_distance = 1.0 - similarity
    embedding_dimension = int(query_unit.shape[0])

    # Grafik mengikuti file referensi: query pada titik 0 dan dokumen
    # pada cosine distance asli, tanpa reduksi dimensi.
    figure, axis = plt.subplots(figsize=(9, 3))
    axis.plot(
        [0.0, cosine_distance],
        [0.0, 0.0],
        linestyle="--",
        linewidth=2,
    )
    axis.scatter(
        [0.0],
        [0.0],
        s=190,
        marker="*",
        label="Query",
        zorder=3,
    )
    axis.scatter(
        [cosine_distance],
        [0.0],
        s=150,
        marker="o",
        label="Dokumen yang dibuka",
        zorder=3,
    )
    axis.annotate(
        "Query\n0,0000",
        xy=(0.0, 0.0),
        xytext=(0.0, 0.18),
        ha="center",
        va="bottom",
    )
    axis.annotate(
        (
            f"Dokumen\n{cosine_distance:.4f}"
            .replace(".", ",")
        ),
        xy=(cosine_distance, 0.0),
        xytext=(cosine_distance, 0.18),
        ha="center",
        va="bottom",
    )
    axis.set_xlim(-0.05, 2.05)
    axis.set_ylim(-0.35, 0.45)
    axis.set_yticks([])
    axis.set_xlabel(
        f"Cosine distance pada embedding asli "
        f"{embedding_dimension} dimensi "
        "(0 = identik, 2 = berlawanan)"
    )
    axis.set_title(
        "Jarak Asli Query dan Dokumen pada Ruang Embedding"
    )
    axis.legend(loc="lower right")
    axis.grid(axis="x", alpha=0.25)

    st.pyplot(figure, clear_figure=True)
    plt.close(figure)

    col_similarity, col_distance, col_dimension = st.columns(3)
    col_similarity.metric(
        "Cosine similarity",
        f"{similarity:.4f}".replace(".", ","),
    )
    col_distance.metric(
        "Cosine distance",
        f"{cosine_distance:.4f}".replace(".", ","),
    )
    col_dimension.metric(
        "Dimensi embedding",
        str(embedding_dimension),
    )

    st.write(
        (
            f"Grafik tidak menggunakan reduksi dimensi. Jarak dihitung "
            f"langsung dari seluruh {embedding_dimension} dimensi. "
            f"Dokumen berada pada cosine distance "
            f"**{cosine_distance:.4f}**; semakin mendekati 0, "
            "semakin mirip maknanya."
        ).replace(".", ",")
    )

    st.markdown("**Perhitungan ringkas:**")
    st.write(
        (
            f"Cosine similarity raw = **{similarity:.4f}**, "
            f"sehingga cosine distance = 1 − {similarity:.4f} "
            f"= **{cosine_distance:.4f}**."
        ).replace(".", ",")
    )


def render_hybrid_calculation() -> None:
    if result_row is None:
        st.info(
            "Skor hybrid tidak tersedia pada hasil pencarian."
        )
        return

    bm25_norm = float(
        result_row.get("bm25_norm", 0.0) or 0.0
    )
    dense_norm = float(
        result_row.get("dense_norm", 0.0) or 0.0
    )
    hybrid_score = float(
        result_row.get(
            "hybrid_score",
            (1.0 - ALPHA) * bm25_norm
            + ALPHA * dense_norm,
        )
        or 0.0
    )

    st.markdown("### Perhitungan Skor Hybrid")
    calculation = pd.DataFrame(
        [
            {
                "Komponen": "BM25",
                "Skor Normalisasi": bm25_norm,
                "Bobot": 1.0 - ALPHA,
                "Kontribusi": (
                    (1.0 - ALPHA) * bm25_norm
                ),
            },
            {
                "Komponen": "SBERT Fine-Tuned",
                "Skor Normalisasi": dense_norm,
                "Bobot": ALPHA,
                "Kontribusi": ALPHA * dense_norm,
            },
        ]
    )

    formatted = calculation.copy()
    for column in [
        "Skor Normalisasi", "Bobot", "Kontribusi"
    ]:
        formatted[column] = formatted[column].map(
            lambda value: (
                f"{value:.4f}".replace(".", ",")
            )
        )

    st.table(formatted)
    bm25_weight = 1.0 - ALPHA
    formula_text = (
        "Hybrid raw score = "
        f"({bm25_weight:.2f} × BM25_norm) + "
        f"({ALPHA:.2f} × SBERT_norm) "
        f"= **{hybrid_score:.4f}**"
    ).replace(".", ",")
    st.write(formula_text)

    rrf_score = float(result_row.get("rrf_score", 0.0) or 0.0)
    st.caption(
        "RRF digunakan sebagai pengurutan tambahan (tie-breaker), "
        f"bukan sebagai skor utama. rrf_k={RRF_K}; "
        f"RRF score dokumen={rrf_score:.6f}."
    )


# ============================================================
# Tampilkan analisis sesuai metode
# ============================================================

st.markdown("---")
st.markdown("## Analisis Relevansi Query dan Dokumen")

if not query:
    st.info(
        "Analisis relevansi belum tersedia karena halaman detail tidak "
        "dibuka melalui tombol Buka Detail Metadata pada hasil pencarian."
    )
    st.stop()

st.write(f"**Query:** `{query}`")
st.write(f"**Metode:** {mode}")

if mode == MODE_BM25:
    render_bm25_analysis()

elif mode == MODE_SBERT:
    render_sbert_analysis()

elif mode == MODE_HYBRID:
    st.markdown("### Analisis Relevansi Hybrid Retrieval")
    st.write(
        "Hybrid Retrieval menggabungkan BM25 untuk kecocokan kata "
        "dan Sentence-BERT untuk kemiripan makna."
    )
    render_bm25_analysis()
    st.markdown("---")
    render_sbert_analysis()
    render_hybrid_calculation()

else:
    st.error(f"Metode pencarian tidak dikenali: {mode}")
