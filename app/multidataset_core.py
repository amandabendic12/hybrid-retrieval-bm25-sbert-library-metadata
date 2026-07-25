from __future__ import annotations

from pathlib import Path
from typing import Any
import gc
import hashlib
import io
import json
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Lokasi proyek dan aset
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent
while PROJECT_ROOT != PROJECT_ROOT.parent:
    if (
        (PROJECT_ROOT / "app").exists()
        and (PROJECT_ROOT / "src").exists()
        and (PROJECT_ROOT / "data").exists()
    ):
        break
    PROJECT_ROOT = PROJECT_ROOT.parent

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# Aset final hasil pelatihan konfigurasi C3.
# C3_corpus_doc_ids.csv digunakan sebagai korpus retrieval karena urutan
# barisnya dijamin sama dengan C3_corpus_embeddings_float16.npy.
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "C3_corpus_doc_ids.csv"
)
DISPLAY_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "books_raw_enriched.csv"
)
DATASET2_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "books_dataset2_preprocessed.csv"
)

MODEL_PATH = (
    PROJECT_ROOT / "data" / "processed" / "final_model_C3"
)
OPAC_EMBEDDING_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "C3_corpus_embeddings_float16.npy"
)

APP_LIGHT_EMBEDDING_DIR = (
    PROJECT_ROOT / "data" / "processed" / "app_light" / "embeddings"
)
# Nama file menyertakan C3 agar embedding lama tidak digunakan kembali.
DATASET2_FP16_PATH = (
    APP_LIGHT_EMBEDDING_DIR / "dataset2_C3_finetuned_fp16.npy"
)

USER_DATASET_DIR = (
    PROJECT_ROOT / "data" / "processed" / "user_datasets"
)
USER_DATASET_REGISTRY = USER_DATASET_DIR / "registry.json"

OFFICIAL_DATASET2_LABEL = "7k Books (Kaggle)"

# Konfigurasi terbaik hasil pengujian akhir.
BM25_K1 = 1.2
BM25_B = 0.50
ALPHA = 0.70
RRF_K = 60
POOL_K = 50
TOP_K = 50

MODE_BM25 = "BM25"
MODE_SBERT = "Fine-Tuned Sentence-BERT"
MODE_HYBRID = "Hybrid Retrieval"


def canonicalize_mode(mode: str) -> str:
    """Menyamakan label metode lama dan baru pada session_state."""
    aliases = {
        "BM25": MODE_BM25,
        "Dense Fine-Tuned": MODE_SBERT,
        "Sentence-BERT Fine-Tuned": MODE_SBERT,
        "SBERT Fine-Tuned": MODE_SBERT,
        "Fine-Tuned Sentence-BERT": MODE_SBERT,
        "Hybrid Retrieval": MODE_HYBRID,
    }
    return aliases.get(str(mode).strip(), str(mode).strip())


# ============================================================
# Utilitas teks dan metadata
# ============================================================

def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.lower()
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"[^\w\s+#&'\-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    text = str(value).strip()
    return text if text else "-"


def first_available(row: pd.Series, candidates: list[str]) -> str:
    for column in candidates:
        if column not in row.index:
            continue
        value = safe_text(row[column])
        if value != "-":
            return value
    return "-"


def normalize_source(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def read_csv_fallback(source: str | Path | bytes) -> pd.DataFrame:
    errors: list[str] = []

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            if isinstance(source, bytes):
                return pd.read_csv(
                    io.BytesIO(source),
                    encoding=encoding,
                    low_memory=False,
                )
            return pd.read_csv(
                Path(source),
                encoding=encoding,
                low_memory=False,
            )
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")

    raise ValueError("CSV tidak dapat dibaca. " + " | ".join(errors))


def source_display_name(value: Any) -> str:
    normalized = normalize_source(value)
    aliases = {
        "opacusu": "OPAC USU",
        "usuopac": "OPAC USU",
        "kaggle7kbooks": OFFICIAL_DATASET2_LABEL,
        "kaggle7kbook": OFFICIAL_DATASET2_LABEL,
        "7kbooks": OFFICIAL_DATASET2_LABEL,
        "kaggle7kbooksmetadata": OFFICIAL_DATASET2_LABEL,
    }
    return aliases.get(normalized, safe_text(value))


# ============================================================
# Registry dataset permanen milik pengguna
# ============================================================

def _empty_registry() -> dict[str, dict[str, Any]]:
    return {}


def load_user_registry() -> dict[str, dict[str, Any]]:
    USER_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if not USER_DATASET_REGISTRY.exists():
        return _empty_registry()

    try:
        data = json.loads(
            USER_DATASET_REGISTRY.read_text(encoding="utf-8")
        )
        return data if isinstance(data, dict) else _empty_registry()
    except Exception:
        return _empty_registry()


def save_user_registry(registry: dict[str, dict[str, Any]]) -> None:
    USER_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    temporary = USER_DATASET_REGISTRY.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(USER_DATASET_REGISTRY)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "dataset"


def prepare_uploaded_dataset(
    raw_df: pd.DataFrame,
    title_column: str,
    subject_column: str,
    dataset_name: str,
    dataset_key: str,
) -> pd.DataFrame:
    if title_column == subject_column:
        raise ValueError("Kolom judul dan subjek/kategori harus berbeda.")

    if title_column not in raw_df.columns:
        raise ValueError("Kolom judul tidak ditemukan.")
    if subject_column not in raw_df.columns:
        raise ValueError("Kolom subjek/kategori tidak ditemukan.")

    result = raw_df.copy().fillna("")

    result["judul_buku"] = result[title_column].map(clean_text)
    result["subjek"] = result[subject_column].map(clean_text)
    result["document_text"] = (
        result["judul_buku"]
        .str.cat(result["subjek"], sep=" ")
        .map(clean_text)
    )

    result = result[
        (result["judul_buku"].str.len() > 0)
        & (result["document_text"].str.len() > 0)
    ].copy()

    result = result.drop_duplicates(
        subset=["document_text"],
        keep="first",
    ).reset_index(drop=True)

    result["original_doc_id"] = result.index.astype(str)
    result["doc_id"] = (
        dataset_key + "::" + result["original_doc_id"]
    )
    result["source_dataset"] = dataset_name

    ordered = [
        "doc_id",
        "judul_buku",
        "subjek",
        "document_text",
        "source_dataset",
        "original_doc_id",
    ]
    remaining = [c for c in result.columns if c not in ordered]
    return result[ordered + remaining]


def persist_uploaded_dataset(
    uploaded_bytes: bytes,
    uploaded_filename: str,
    raw_df: pd.DataFrame,
    title_column: str,
    subject_column: str,
    dataset_name: str,
) -> dict[str, Any]:
    """
    Menyimpan dataset pengguna sebagai dataset permanen yang siap dipakai
    BM25, SBERT, dan Hybrid Retrieval.

    Tahap yang dijalankan:
    1. Validasi dan preprocessing.
    2. Penyimpanan CSV hasil preprocessing.
    3. Registrasi dataset.
    4. Pembentukan indeks BM25.
    5. Pembentukan embedding SBERT float16.
    6. Penyimpanan laporan preprocessing dan embedding.

    Jika embedding gagal, seluruh file dataset baru di-rollback agar tidak ada
    dataset setengah jadi pada dropdown.
    """
    dataset_name = dataset_name.strip()
    if not dataset_name:
        raise ValueError("Nama dataset tidak boleh kosong.")

    if len(raw_df) > 100_000:
        raise ValueError("Dataset maksimal 100.000 baris.")

    initial_rows = int(len(raw_df))
    digest = hashlib.sha256(uploaded_bytes).hexdigest()[:12]
    dataset_key = f"user-{slugify(dataset_name)}-{digest}"

    processed = prepare_uploaded_dataset(
        raw_df=raw_df,
        title_column=title_column,
        subject_column=subject_column,
        dataset_name=dataset_name,
        dataset_key=dataset_key,
    )

    if processed.empty:
        raise ValueError("Tidak ada data valid setelah preprocessing.")

    USER_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    APP_LIGHT_EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    output_path = USER_DATASET_DIR / f"{dataset_key}.csv"
    report_path = USER_DATASET_DIR / f"{dataset_key}_processing.json"

    processed.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    entry = {
        "key": dataset_key,
        "name": dataset_name,
        "original_filename": uploaded_filename,
        "data_path": str(output_path),
        "title_column": title_column,
        "subject_column": subject_column,
        "initial_row_count": initial_rows,
        "row_count": int(len(processed)),
        "removed_row_count": int(initial_rows - len(processed)),
        "file_hash": digest,
        "preprocessing_completed": True,
        "embedding_completed": False,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    registry = load_user_registry()
    registry[dataset_key] = entry
    save_user_registry(registry)

    try:
        readiness = prepare_dataset_for_hybrid(
            dataset_key=dataset_key,
            force_rebuild=True,
        )

        entry.update(
            {
                "embedding_completed": True,
                "embedding_path": readiness["embedding_path"],
                "embedding_shape": readiness["embedding_shape"],
                "embedding_dtype": readiness["embedding_dtype"],
                "embedding_size_mb": readiness["embedding_size_mb"],
                "embedding_seconds": readiness["embedding_seconds"],
                "bm25_ready": True,
                "ready_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        registry = load_user_registry()
        registry[dataset_key] = entry
        save_user_registry(registry)

        report = {
            "dataset_key": dataset_key,
            "dataset_name": dataset_name,
            "source_file": uploaded_filename,
            "title_column": title_column,
            "subject_column": subject_column,
            "initial_rows": initial_rows,
            "final_rows": int(len(processed)),
            "removed_rows": int(initial_rows - len(processed)),
            "preprocessing": [
                "penanganan nilai kosong",
                "case folding",
                "normalisasi Unicode",
                "pembersihan karakter",
                "normalisasi spasi",
                "pembentukan document_text",
                "penghapusan metadata kosong",
                "penghapusan duplikat document_text",
                "pembentukan doc_id unik",
            ],
            "bm25_ready": True,
            "embedding": readiness,
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return entry

    except Exception as exc:
        # Rollback agar dataset yang belum siap Hybrid tidak muncul.
        delete_uploaded_dataset(dataset_key)
        if report_path.exists():
            report_path.unlink()
        raise RuntimeError(
            "Dataset berhasil dipreproses, tetapi embedding SBERT gagal "
            f"dibentuk. Penambahan dataset dibatalkan. Detail: {exc}"
        ) from exc


def _close_numpy_memmap(value: Any) -> None:
    """Menutup file .npy yang dibuka melalui np.load(..., mmap_mode='r')."""
    if value is None:
        return

    mmap_object = getattr(value, "_mmap", None)
    if mmap_object is not None:
        try:
            mmap_object.close()
        except Exception:
            pass


def _unlink_with_retry(
    path: Path,
    attempts: int = 8,
    delay_seconds: float = 0.25,
) -> None:
    """
    Menghapus file dengan beberapa percobaan.

    Pada Windows, file .npy tidak dapat dihapus selama masih digunakan
    oleh NumPy memmap atau proses lain.
    """
    if not path.exists():
        return

    last_error: PermissionError | None = None

    for _ in range(attempts):
        try:
            path.unlink()
            return
        except PermissionError as exc:
            last_error = exc
            gc.collect()
            time.sleep(delay_seconds)

    if last_error is not None:
        raise PermissionError(
            f"File masih digunakan oleh proses lain dan belum dapat dihapus: {path}. "
            "Tutup tab aplikasi lain yang memakai dataset tersebut, lalu coba kembali."
        ) from last_error


def delete_uploaded_dataset(dataset_key: str) -> None:
    """
    Menghapus dataset pengguna beserta embedding dan laporan pemrosesannya.

    Cache dan NumPy memmap ditutup lebih dahulu agar file embedding dapat
    dihapus pada Windows tanpa WinError 32.
    """
    registry = load_user_registry()
    entry = registry.get(dataset_key)

    if entry is None:
        return

    data_path = Path(entry.get("data_path", ""))
    source_name = str(entry.get("name", dataset_key))

    embedding_path = (
        APP_LIGHT_EMBEDDING_DIR
        / f"runtime_{dataset_key}_C3_fp16.npy"
    )
    processing_report = (
        USER_DATASET_DIR
        / f"{dataset_key}_processing.json"
    )

    # Tutup memmap dari engine dataset yang sedang tersimpan di cache.
    try:
        if data_path.exists():
            dense_engine = get_runtime_dense_engine(
                dataset_key,
                str(data_path),
                source_name,
            )
            _close_numpy_memmap(
                getattr(dense_engine, "doc_embeddings", None)
            )
            dense_engine.doc_embeddings = None
            del dense_engine
    except Exception:
        # Penghapusan tetap dilanjutkan; cache dibersihkan di bawah.
        pass

    # Hapus semua cache yang mungkin masih memegang referensi dataset.
    try:
        run_search_cached.clear()
    except Exception:
        pass

    try:
        get_runtime_bm25_engine.clear()
        get_runtime_dense_engine.clear()
        load_dataset_file.clear()
    except Exception:
        pass

    gc.collect()
    time.sleep(0.15)

    # Hapus embedding lebih dahulu karena file ini yang paling mungkin terkunci.
    _unlink_with_retry(embedding_path)
    _unlink_with_retry(processing_report)
    _unlink_with_retry(data_path)

    registry.pop(dataset_key, None)
    save_user_registry(registry)


def build_dataset_catalog() -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {
        "opac-usu": {
            "key": "opac-usu",
            "label": "OPAC USU",
            "data_path": str(DATA_PATH),
            "display_path": str(DISPLAY_DATA_PATH),
            "engine_type": "default",
            "is_user_dataset": False,
        }
    }

    if DATASET2_PATH.exists():
        catalog["7k-books-kaggle"] = {
            "key": "7k-books-kaggle",
            "label": OFFICIAL_DATASET2_LABEL,
            "data_path": str(DATASET2_PATH),
            "display_path": str(DATASET2_PATH),
            "engine_type": "runtime",
            "is_user_dataset": False,
        }

    for key, entry in load_user_registry().items():
        data_path = Path(entry.get("data_path", ""))
        if not data_path.exists():
            continue

        catalog[key] = {
            "key": key,
            "label": entry.get("name", key),
            "data_path": str(data_path),
            "display_path": str(data_path),
            "engine_type": "runtime",
            "is_user_dataset": True,
            "row_count": entry.get("row_count"),
        }

    return catalog


def get_dataset_config(dataset_key: str) -> dict[str, Any]:
    catalog = build_dataset_catalog()
    if dataset_key not in catalog:
        raise KeyError(f"Dataset tidak ditemukan: {dataset_key}")
    return catalog[dataset_key]


# ============================================================
# Normalisasi struktur dataset
# ============================================================

def ensure_internal_dataset(
    df: pd.DataFrame,
    source_name: str,
    drop_duplicates: bool = False,
) -> pd.DataFrame:
    result = df.copy().fillna("")

    if "judul_buku" not in result.columns:
        title_candidates = [
            "title", "judul", "book_title", "name"
        ]
        matched = next(
            (c for c in title_candidates if c in result.columns),
            None,
        )
        if matched is None:
            raise ValueError("Kolom judul tidak ditemukan.")
        result["judul_buku"] = result[matched]

    if "subjek" not in result.columns:
        subject_candidates = [
            "subject", "category", "categories",
            "genre", "genres",
        ]
        matched = next(
            (c for c in subject_candidates if c in result.columns),
            None,
        )
        result["subjek"] = (
            result[matched] if matched is not None else ""
        )

    result["judul_buku"] = result["judul_buku"].map(clean_text)
    result["subjek"] = result["subjek"].map(clean_text)

    if "document_text" not in result.columns:
        result["document_text"] = (
            result["judul_buku"]
            .str.cat(result["subjek"], sep=" ")
            .map(clean_text)
        )
    else:
        result["document_text"] = result["document_text"].map(clean_text)

    result = result[
        (result["judul_buku"].str.len() > 0)
        & (result["document_text"].str.len() > 0)
    ].copy()

    if drop_duplicates:
        result = result.drop_duplicates(
            subset=["document_text"],
            keep="first",
        )

    result = result.reset_index(drop=True)

    if "doc_id" not in result.columns:
        result["doc_id"] = result.index.astype(str)
    else:
        result["doc_id"] = result["doc_id"].astype(str)

    if "source_dataset" not in result.columns:
        result["source_dataset"] = source_name
    else:
        empty = (
            result["source_dataset"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        )
        result.loc[empty, "source_dataset"] = source_name

    return result


@st.cache_data(show_spinner=False)
def load_dataset_file(
    path_string: str,
    modified_time_ns: int,
    source_name: str,
) -> pd.DataFrame:
    del modified_time_ns
    path = Path(path_string)
    df = read_csv_fallback(path)
    return ensure_internal_dataset(
        df,
        source_name=source_name,
        drop_duplicates=False,
    )


def load_retrieval_data(dataset_key: str) -> pd.DataFrame:
    config = get_dataset_config(dataset_key)
    path = Path(config["data_path"])
    return load_dataset_file(
        str(path),
        path.stat().st_mtime_ns,
        config["label"],
    )


def load_display_data(dataset_key: str) -> pd.DataFrame:
    config = get_dataset_config(dataset_key)
    path = Path(config["display_path"])

    # Data display OPAC dapat memiliki struktur lebih kaya.
    df = read_csv_fallback(path)
    if "doc_id" in df.columns:
        df["doc_id"] = df["doc_id"].astype(str)
    return df.fillna("")


# ============================================================
# Engine retrieval
# ============================================================

class RuntimeBM25Engine:
    def __init__(self, df: pd.DataFrame):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "rank-bm25 belum terpasang."
            ) from exc

        self.df = df.copy().reset_index(drop=True)
        self.df["doc_id"] = self.df["doc_id"].astype(str)
        self.tokenized_corpus = [
            self.tokenize(value)
            for value in self.df["document_text"].astype(str)
        ]
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=BM25_K1,
            b=BM25_B,
        )

    @staticmethod
    def tokenize(value: Any) -> list[str]:
        return [
            token for token in clean_text(value).split()
            if token
        ]

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        nonzero_only: bool = False,
    ) -> pd.DataFrame:
        tokens = self.tokenize(query)
        scores = np.asarray(
            self.bm25.get_scores(tokens),
            dtype=np.float32,
        )
        positions = np.argsort(-scores)

        if nonzero_only:
            positions = positions[scores[positions] > 0]

        positions = positions[:top_k]
        columns = [
            c for c in [
                "doc_id", "judul_buku", "subjek",
                "jenis_koleksi", "source_dataset",
            ]
            if c in self.df.columns
        ]
        result = self.df.iloc[positions][columns].copy()
        result["bm25_score"] = scores[positions]
        return result.reset_index(drop=True)

    def search_with_filter_nonzero(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> pd.DataFrame:
        return self.search(
            query=query,
            top_k=top_k,
            nonzero_only=True,
        )


class RuntimeDenseEngine:
    def __init__(
        self,
        df: pd.DataFrame,
        model: Any,
        embeddings: np.ndarray,
    ):
        self.df = df.copy().reset_index(drop=True)
        self.df["doc_id"] = self.df["doc_id"].astype(str)
        self.model = model
        self.doc_embeddings = embeddings

    def encode_query(self, query: str) -> np.ndarray:
        cleaned = clean_text(query)

        if hasattr(self.model, "encode_query"):
            vector = self.model.encode_query(
                cleaned,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        else:
            vector = self.model.encode(
                [cleaned],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0]

        return np.asarray(vector, dtype=np.float32).reshape(-1)

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> pd.DataFrame:
        query_vector = self.encode_query(query)
        similarities = self.doc_embeddings @ query_vector
        positions = np.argsort(-similarities)[:top_k]

        columns = [
            c for c in [
                "doc_id", "judul_buku", "subjek",
                "jenis_koleksi", "source_dataset",
            ]
            if c in self.df.columns
        ]
        result = self.df.iloc[positions][columns].copy()
        result["dense_finetuned_score"] = similarities[positions]
        return result.reset_index(drop=True)


@st.cache_resource(show_spinner="Memuat Fine-Tuned Sentence-BERT C3...")
def load_sentence_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model C3 tidak ditemukan. Letakkan folder final_model_C3 di: "
            f"{MODEL_PATH}"
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Library sentence-transformers belum terpasang."
        ) from exc

    model = SentenceTransformer(str(MODEL_PATH))
    model.max_seq_length = 128
    return model


@st.cache_resource(show_spinner="Menyiapkan BM25 OPAC...")
def get_default_bm25_engine() -> RuntimeBM25Engine:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Korpus C3 tidak ditemukan. Letakkan C3_corpus_doc_ids.csv di: "
            f"{DATA_PATH}"
        )

    df = load_dataset_file(
        str(DATA_PATH),
        DATA_PATH.stat().st_mtime_ns,
        "OPAC USU",
    )
    return RuntimeBM25Engine(df)


@st.cache_resource(show_spinner="Memuat model dan embedding OPAC C3...")
def get_default_dense_engine() -> RuntimeDenseEngine:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Korpus C3 tidak ditemukan. Letakkan C3_corpus_doc_ids.csv di: "
            f"{DATA_PATH}"
        )
    if not OPAC_EMBEDDING_PATH.exists():
        raise FileNotFoundError(
            "Embedding C3 tidak ditemukan. Letakkan "
            "C3_corpus_embeddings_float16.npy di: "
            f"{OPAC_EMBEDDING_PATH}"
        )

    df = load_dataset_file(
        str(DATA_PATH),
        DATA_PATH.stat().st_mtime_ns,
        "OPAC USU",
    )
    embeddings = np.load(OPAC_EMBEDDING_PATH, mmap_mode="r")

    if len(embeddings) != len(df):
        raise RuntimeError(
            "Jumlah embedding C3 tidak sama dengan jumlah baris korpus. "
            f"Embedding={len(embeddings):,}; korpus={len(df):,}. "
            "Gunakan C3_corpus_embeddings_float16.npy bersama "
            "C3_corpus_doc_ids.csv dari proses training yang sama."
        )
    if embeddings.ndim != 2 or embeddings.shape[1] != 384:
        raise RuntimeError(
            "Bentuk embedding C3 tidak sesuai. Diharapkan (N, 384), "
            f"ditemukan {embeddings.shape}."
        )

    return RuntimeDenseEngine(
        df=df,
        model=load_sentence_model(),
        embeddings=embeddings,
    )


def encode_documents(
    model: Any,
    documents: list[str],
) -> np.ndarray:
    if hasattr(model, "encode_document"):
        vectors = model.encode_document(
            documents,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    else:
        vectors = model.encode(
            documents,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    return np.asarray(vectors, dtype=np.float16)


@st.cache_resource(show_spinner="Menyiapkan BM25 dataset aktif...")
def get_runtime_bm25_engine(
    dataset_key: str,
    data_path: str,
    source_name: str,
) -> RuntimeBM25Engine:
    del dataset_key
    path = Path(data_path)
    df = load_dataset_file(
        str(path),
        path.stat().st_mtime_ns,
        source_name,
    )
    return RuntimeBM25Engine(df)


@st.cache_resource(
    show_spinner="Menyiapkan embedding dataset aktif..."
)
def get_runtime_dense_engine(
    dataset_key: str,
    data_path: str,
    source_name: str,
) -> RuntimeDenseEngine:
    path = Path(data_path)
    df = load_dataset_file(
        str(path),
        path.stat().st_mtime_ns,
        source_name,
    )

    model = load_sentence_model()
    APP_LIGHT_EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    if dataset_key == "opac-usu":
        embedding_path = OPAC_EMBEDDING_PATH
    elif dataset_key == "7k-books-kaggle":
        embedding_path = DATASET2_FP16_PATH
    else:
        embedding_path = (
            APP_LIGHT_EMBEDDING_DIR
            / f"runtime_{dataset_key}_C3_fp16.npy"
        )

    embeddings = None
    if embedding_path.exists():
        candidate = np.load(
            embedding_path,
            mmap_mode="r",
        )
        if len(candidate) == len(df):
            embeddings = candidate
        elif dataset_key == "opac-usu":
            raise RuntimeError(
                "Embedding OPAC C3 tidak sejajar dengan korpus C3. "
                "Gunakan pasangan file hasil training yang sama."
            )

    if embeddings is None:
        if dataset_key == "opac-usu":
            raise FileNotFoundError(
                f"Embedding OPAC C3 tidak ditemukan: {embedding_path}"
            )
        vectors = encode_documents(
            model,
            df["document_text"].astype(str).tolist(),
        )
        np.save(embedding_path, vectors)
        embeddings = np.load(
            embedding_path,
            mmap_mode="r",
        )

    return RuntimeDenseEngine(
        df=df,
        model=model,
        embeddings=embeddings,
    )


def get_bm25_engine(dataset_key: str) -> Any:
    config = get_dataset_config(dataset_key)

    if config["engine_type"] == "default":
        return get_default_bm25_engine()

    return get_runtime_bm25_engine(
        dataset_key,
        config["data_path"],
        config["label"],
    )


def get_dense_engine(dataset_key: str) -> Any:
    config = get_dataset_config(dataset_key)

    if config["engine_type"] == "default":
        return get_default_dense_engine()

    return get_runtime_dense_engine(
        dataset_key,
        config["data_path"],
        config["label"],
    )


def prepare_dataset_for_hybrid(
    dataset_key: str,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """
    Menyiapkan dataset runtime agar langsung siap digunakan untuk Hybrid:
    membangun indeks BM25 dan embedding SBERT float16.

    Fungsi ini dipanggil saat tombol Simpan Dataset ditekan, sehingga dataset
    pengguna tidak hanya tersimpan sebagai CSV.
    """
    config = get_dataset_config(dataset_key)
    if config["engine_type"] == "default":
        raise ValueError("Dataset OPAC tidak diproses melalui fungsi runtime.")

    data_path = Path(config["data_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path}")

    embedding_path = (
        DATASET2_FP16_PATH
        if dataset_key == "7k-books-kaggle"
        else APP_LIGHT_EMBEDDING_DIR
        / f"runtime_{dataset_key}_C3_fp16.npy"
    )

    APP_LIGHT_EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    if force_rebuild and embedding_path.exists():
        embedding_path.unlink()

    # Bersihkan cache agar file yang baru ditulis selalu dibaca ulang.
    get_runtime_bm25_engine.clear()
    get_runtime_dense_engine.clear()
    load_dataset_file.clear()

    started = time.perf_counter()
    bm25_engine = get_runtime_bm25_engine(
        dataset_key,
        config["data_path"],
        config["label"],
    )
    dense_engine = get_runtime_dense_engine(
        dataset_key,
        config["data_path"],
        config["label"],
    )
    elapsed = time.perf_counter() - started

    embeddings = getattr(dense_engine, "doc_embeddings", None)
    if embeddings is None:
        raise RuntimeError("Embedding dokumen tidak berhasil dibentuk.")

    if len(embeddings) != len(getattr(dense_engine, "df", [])):
        raise RuntimeError(
            "Jumlah embedding tidak sama dengan jumlah metadata."
        )

    actual_embedding_path = (
        DATASET2_FP16_PATH
        if dataset_key == "7k-books-kaggle"
        else embedding_path
    )
    if not actual_embedding_path.exists():
        raise RuntimeError(
            f"File embedding tidak ditemukan: {actual_embedding_path}"
        )

    return {
        "dataset_key": dataset_key,
        "bm25_ready": getattr(bm25_engine, "bm25", None) is not None,
        "embedding_path": str(actual_embedding_path),
        "embedding_shape": [int(v) for v in embeddings.shape],
        "embedding_dtype": str(embeddings.dtype),
        "embedding_size_mb": round(
            actual_embedding_path.stat().st_size / (1024 ** 2),
            4,
        ),
        "embedding_seconds": round(elapsed, 4),
        "normalize_embeddings": True,
    }


# ============================================================
# Pencarian
# ============================================================

def minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)
    minimum = values.min()
    maximum = values.max()

    if minimum == maximum:
        return pd.Series(
            np.zeros(len(values)),
            index=values.index,
        )

    return (values - minimum) / (maximum - minimum)


def rrf(rank: Any) -> float:
    if rank is None or pd.isna(rank):
        return 0.0
    return 1.0 / (RRF_K + float(rank))


def hybrid_search(
    dataset_key: str,
    query: str,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    bm25_engine = get_bm25_engine(dataset_key)
    dense_engine = get_dense_engine(dataset_key)

    if hasattr(bm25_engine, "search_with_filter_nonzero"):
        bm25_df = bm25_engine.search_with_filter_nonzero(
            query=query,
            top_k=POOL_K,
        ).copy()
    else:
        bm25_df = bm25_engine.search(
            query=query,
            top_k=POOL_K,
        ).copy()

    dense_df = dense_engine.search(
        query=query,
        top_k=POOL_K,
    ).copy()

    bm25_df["doc_id"] = bm25_df["doc_id"].astype(str)
    dense_df["doc_id"] = dense_df["doc_id"].astype(str)
    bm25_df["bm25_rank"] = np.arange(1, len(bm25_df) + 1)
    dense_df["dense_rank"] = np.arange(1, len(dense_df) + 1)

    if (
        "dense_finetuned_score" not in dense_df.columns
        and "dense_score" in dense_df.columns
    ):
        dense_df = dense_df.rename(
            columns={"dense_score": "dense_finetuned_score"}
        )

    merged = pd.merge(
        bm25_df,
        dense_df,
        on="doc_id",
        how="outer",
        suffixes=("_bm25", "_dense"),
    )

    for base_column in [
        "judul_buku",
        "subjek",
        "jenis_koleksi",
        "source_dataset",
    ]:
        left = f"{base_column}_bm25"
        right = f"{base_column}_dense"

        if left in merged.columns and right in merged.columns:
            merged[base_column] = merged[left].combine_first(
                merged[right]
            )
        elif left in merged.columns:
            merged[base_column] = merged[left]
        elif right in merged.columns:
            merged[base_column] = merged[right]

    if "bm25_score" not in merged.columns:
        merged["bm25_score"] = 0.0
    if "dense_finetuned_score" not in merged.columns:
        merged["dense_finetuned_score"] = 0.0

    merged["bm25_score"] = pd.to_numeric(
        merged["bm25_score"],
        errors="coerce",
    ).fillna(0.0)
    merged["dense_finetuned_score"] = pd.to_numeric(
        merged["dense_finetuned_score"],
        errors="coerce",
    ).fillna(0.0)

    merged["bm25_norm"] = minmax(merged["bm25_score"])
    merged["dense_norm"] = minmax(
        merged["dense_finetuned_score"]
    )
    merged["hybrid_score"] = (
        (1.0 - ALPHA) * merged["bm25_norm"]
        + ALPHA * merged["dense_norm"]
    )
    merged["rrf_score"] = (
        merged["bm25_rank"].map(rrf)
        + merged["dense_rank"].map(rrf)
    )

    merged = merged.sort_values(
        by=[
            "hybrid_score",
            "rrf_score",
            "bm25_score",
            "dense_finetuned_score",
        ],
        ascending=False,
    )

    output_columns = [
        c for c in [
            "doc_id", "judul_buku", "subjek",
            "jenis_koleksi", "source_dataset",
            "bm25_score", "dense_finetuned_score",
            "bm25_norm", "dense_norm",
            "hybrid_score", "rrf_score",
        ]
        if c in merged.columns
    ]

    return merged.head(top_k)[output_columns].reset_index(drop=True)


def enrich_result_metadata(
    dataset_key: str,
    results: pd.DataFrame,
) -> pd.DataFrame:
    if results is None or results.empty:
        return results

    retrieval = load_retrieval_data(dataset_key)
    columns = [
        c for c in [
            "doc_id", "jenis_koleksi", "source_dataset"
        ]
        if c in retrieval.columns
    ]

    if "doc_id" not in columns:
        return results

    metadata = retrieval[columns].copy()
    metadata["doc_id"] = metadata["doc_id"].astype(str)
    metadata = metadata.drop_duplicates("doc_id")

    enriched = results.copy()
    enriched["doc_id"] = enriched["doc_id"].astype(str)
    enriched = enriched.merge(
        metadata,
        on="doc_id",
        how="left",
        suffixes=("", "_source"),
    )

    for column in ["jenis_koleksi", "source_dataset"]:
        source_column = f"{column}_source"
        if source_column not in enriched.columns:
            continue

        if column not in enriched.columns:
            enriched[column] = enriched[source_column]
        else:
            blank = (
                enriched[column]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq("")
            )
            enriched.loc[blank, column] = enriched.loc[
                blank, source_column
            ]
        enriched = enriched.drop(columns=[source_column])

    return enriched


@st.cache_data(show_spinner=False)
def run_search_cached(
    dataset_key: str,
    query: str,
    mode: str,
) -> tuple[pd.DataFrame, float]:
    mode = canonicalize_mode(mode)
    started = time.perf_counter()

    if mode == MODE_BM25:
        engine = get_bm25_engine(dataset_key)
        if hasattr(engine, "search_with_filter_nonzero"):
            result = engine.search_with_filter_nonzero(
                query=query,
                top_k=TOP_K,
            )
        else:
            result = engine.search(
                query=query,
                top_k=TOP_K,
            )

    elif mode == MODE_SBERT:
        result = get_dense_engine(dataset_key).search(
            query=query,
            top_k=TOP_K,
        )

    elif mode == MODE_HYBRID:
        result = hybrid_search(
            dataset_key=dataset_key,
            query=query,
            top_k=TOP_K,
        )

    else:
        raise ValueError("Metode pencarian tidak dikenali.")

    elapsed = time.perf_counter() - started
    result = enrich_result_metadata(dataset_key, result)
    return result.reset_index(drop=True), elapsed


def score_for_display(row: pd.Series, mode: str) -> float:
    mode = canonicalize_mode(mode)
    candidates: list[str]

    if mode == MODE_BM25:
        candidates = ["bm25_score"]
    elif mode == MODE_SBERT:
        candidates = [
            "dense_finetuned_score",
            "dense_score",
        ]
    else:
        candidates = [
            "hybrid_score",
            "rrf_score",
        ]

    for column in candidates:
        if column in row.index:
            try:
                return float(row[column])
            except Exception:
                continue

    return 0.0


# ============================================================
# Akses detail
# ============================================================

def get_display_row(
    dataset_key: str,
    doc_id: str,
    source_dataset: str | None = None,
) -> pd.Series | None:
    expected_source = normalize_source(source_dataset)

    for df in (
        load_display_data(dataset_key),
        load_retrieval_data(dataset_key),
    ):
        if "doc_id" not in df.columns:
            continue

        matched = df[
            df["doc_id"].astype(str) == str(doc_id)
        ]

        if (
            expected_source
            and "source_dataset" in matched.columns
            and not matched.empty
        ):
            source_match = matched[
                matched["source_dataset"].map(normalize_source)
                == expected_source
            ]
            if not source_match.empty:
                matched = source_match

        if not matched.empty:
            return matched.iloc[0]

    return None


def build_detail_table(row: pd.Series) -> pd.DataFrame:
    preferred = [
        ("source_dataset", "Sumber Dataset"),
        ("doc_id", "ID Dokumen"),
        ("judul_buku", "Judul Buku"),
        ("title", "Judul Buku"),
        ("subjek", "Subjek/Kategori"),
        ("categories", "Subjek/Kategori"),
        ("category", "Subjek/Kategori"),
        ("jenis_koleksi", "Jenis Koleksi"),
        ("pengarang", "Pengarang"),
        ("authors", "Pengarang"),
        ("author", "Pengarang"),
        ("tahun", "Tahun Terbit"),
        ("published_year", "Tahun Terbit"),
        ("year", "Tahun Terbit"),
        ("penerbit", "Penerbit"),
        ("publisher", "Penerbit"),
        ("isbn", "ISBN"),
        ("isbn13", "ISBN-13"),
        ("isbn10", "ISBN-10"),
        ("bahasa", "Bahasa"),
        ("language", "Bahasa"),
        ("no_panggil", "Nomor Panggil"),
        ("lokasi", "Lokasi"),
        ("location", "Lokasi"),
        ("edisi", "Edisi"),
        ("jumlah_halaman", "Jumlah Halaman"),
        ("page_count", "Jumlah Halaman"),
        ("deskripsi_fisik", "Deskripsi Fisik"),
    ]

    hidden = {
        "document_text", "original_doc_id",
        "bm25_score", "dense_score",
        "dense_finetuned_score", "bm25_norm",
        "dense_norm", "hybrid_score", "rrf_score",
        "thumbnail", "image_url", "cover_url",
        "deskripsi", "description", "summary", "abstract",
    }

    rows: list[dict[str, str]] = []
    used_columns: set[str] = set()
    used_labels: set[str] = set()

    for column, label in preferred:
        if column not in row.index:
            continue
        value = safe_text(row[column])
        if value == "-" or label in used_labels:
            continue

        if column == "source_dataset":
            value = source_display_name(value)

        rows.append({"Atribut": label, "Nilai": value})
        used_columns.add(column)
        used_labels.add(label)

    for column in row.index:
        if column in used_columns or column in hidden:
            continue
        if str(column).startswith("fallback_"):
            continue

        value = safe_text(row[column])
        if value == "-":
            continue

        rows.append(
            {
                "Atribut": str(column).replace("_", " ").title(),
                "Nilai": value,
            }
        )

    return pd.DataFrame(rows)


def reset_search_state() -> None:
    for key, value in {
        "last_results": None,
        "last_query": "",
        "last_mode": MODE_HYBRID,
        "last_elapsed": None,
        "current_page": 1,
    }.items():
        st.session_state[key] = value
