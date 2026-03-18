import json
import math
import os
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
from fastapi import HTTPException

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RAG_DIR = DATA_DIR / "rag"
DOCS_JSON = RAG_DIR / "docs.json"
CHUNKS_JSONL = RAG_DIR / "chunks.jsonl"


def ensure_rag_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    if not DOCS_JSON.exists():
        DOCS_JSON.write_text("[]", encoding="utf-8")
    if not CHUNKS_JSONL.exists():
        CHUNKS_JSONL.write_text("", encoding="utf-8")

"""处理 TXT/MD 的编码问题，优先 utf-8，失败就 gbk，再兜底忽略错误"""
def read_text_bytes(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return raw.decode("gbk")
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="ignore")

"""统一换行符，压缩多余空行，便于 chunking"""
def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

"""根据文件扩展名分流"""
def extract_text(filename: str, raw: bytes) -> str:
    name = (filename or "").lower().strip()

    if name.endswith(".txt") or name.endswith(".md"):
        return clean_text(read_text_bytes(raw))

    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(raw))
        parts: List[str] = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
        return clean_text("\n\n".join(parts))

    if name.endswith(".docx"):
        import docx
        d = docx.Document(BytesIO(raw))
        parts: List[str] = []
        for p in d.paragraphs:
            t = (p.text or "").strip()
            if t:
                parts.append(t)
        return clean_text("\n".join(parts))

    if name.endswith(".doc"):
        import textract
        out = textract.process(BytesIO(raw))
        if isinstance(out, bytes):
            return clean_text(out.decode("utf-8", errors="ignore"))
        return clean_text(str(out))

    if name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")):
        from PIL import Image
        import pytesseract
        img = Image.open(BytesIO(raw))
        return clean_text(pytesseract.image_to_string(img))

    raise HTTPException(status_code=400, detail=f"不支持的文件类型: {filename}")

"""固定窗口切分 + overlap 滑窗；返回 (start, end, chunk) 列表， 保持上下文连续"""
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[Tuple[int, int, str]]:
    text = text.strip()
    if not text:
        return []
    chunks: List[Tuple[int, int, str]] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append((start, end, piece))
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def load_docs() -> List[Dict[str, Any]]:
    ensure_rag_dirs()
    return json.loads(DOCS_JSON.read_text(encoding="utf-8"))


def save_docs(docs: List[Dict[str, Any]]) -> None:
    DOCS_JSON.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")


def append_chunk(row: Dict[str, Any]) -> None:
    ensure_rag_dirs()
    with CHUNKS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_chunks() -> List[Dict[str, Any]]:
    ensure_rag_dirs()
    rows: List[Dict[str, Any]] = []
    for line in CHUNKS_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def l2_norm(vec: List[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def normalize(vec: List[float]) -> List[float]:
    n = l2_norm(vec)
    if n == 0:
        return vec
    return [x / n for x in vec]


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


async def embed_texts(texts: List[str]) -> List[List[float]]:
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    if not api_key:
        raise HTTPException(status_code=500, detail="缺少 OPENAI_API_KEY")

    payload = {"model": model, "input": texts}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{base_url}/embeddings", headers=headers, json=payload)
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    return [item["embedding"] for item in data["data"]]


def build_rag_messages_with_history(query: str, hits: List[Dict[str, Any]], history: list[dict] = []) -> List[Dict[str, str]]:
    ctx = "\n\n".join(
        [f"[{i}] ({h['filename']} | {h['chunk_id']}) {h['text']}" for i, h in enumerate(hits, start=1)]
    )
    system = "你是一个严谨的助手。只能依据给定的 Context 回答。历史仅用于理解指代，不得引入 Context 外信息；若 Context 不含答案，回答“不知道“。"
    msgs = [{"role": "system", "content": system}]
    for m in history[-6:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": f"问题：{query}\n\nContext:\n{ctx}"})
    return msgs


def make_doc_item(doc_id: str, filename: str, size: int) -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "filename": filename,
        "size": size,
        "uploaded_at": datetime.utcnow().isoformat(),
    }

def build_search_query(query: str, history: list[dict]) -> str:
    q = (query or "").strip()
    if len(q) >= 8:
        return q

    recent_user = [m["content"] for m in history if m.get("role") == "user" and m.get("content")]
    recent_user = recent_user[-2:]
    if not recent_user:
        return q

    return " ".join(recent_user + [q])

