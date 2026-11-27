"""最小 RAG：读文档 → 按标题切分 → 向量化 → 余弦检索。

三个设计取舍（面试可以直接讲）：
1. 切分按结构（二级标题）而不是固定字数：语义边界由文档作者给出，比固定窗口准。
2. 不引向量库：当前十几条片段，numpy 直接算余弦足够。向量库解决的是百万级规模、
   持久化、元数据过滤，不是"检索是否正确"。
3. 索引落盘：embedding 是 IO + 算力成本，不该每次启动重算。
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DOCS_DIR = Path("data/docs")
INDEX_DIR = Path("data/index")
VEC_FILE = INDEX_DIR / "vectors.npy"
META_FILE = INDEX_DIR / "chunks.json"
FP_FILE = INDEX_DIR / "corpus.fingerprint"

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


@lru_cache
def _embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=EMBED_MODEL)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def load_chunks() -> list[dict]:
    chunks: list[dict] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for block in text.split("\n## ")[1:]:
            lines = [line for line in block.strip().splitlines() if line.strip()]
            if not lines:
                continue
            heading = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            if not body:
                continue
            chunks.append({"doc": path.stem, "title": heading, "text": f"{heading}\n{body}"})
    return chunks


def _corpus_fingerprint() -> str:
    """语料指纹：文件名 + 大小 + 修改时间。任何一个变了就说明索引过期。"""
    hasher = hashlib.sha256()
    for path in sorted(DOCS_DIR.glob("*.md")):
        stat = path.stat()
        hasher.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return hasher.hexdigest()[:16]


def build_index(rebuild: bool = False) -> tuple[np.ndarray, list[dict]]:
    fingerprint = _corpus_fingerprint()

    # 缓存有效的条件：三个文件都在，且指纹与当前语料一致。
    # 只看"文件存在"是不够的 —— 改了文档却没重建索引时不会报错，
    # 检索会静默地继续用旧片段（c04 就是这么挂的）。
    if not rebuild and VEC_FILE.exists() and META_FILE.exists() and FP_FILE.exists():
        if FP_FILE.read_text(encoding="utf-8").strip() == fingerprint:
            return np.load(VEC_FILE), json.loads(META_FILE.read_text(encoding="utf-8"))

    chunks = load_chunks()
    if not chunks:
        raise RuntimeError(f"{DOCS_DIR} 下没有可用的 .md 文档")

    vectors = np.array(list(_embedder().embed([c["text"] for c in chunks])), dtype="float32")
    vectors = _normalize(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VEC_FILE, vectors)
    META_FILE.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    FP_FILE.write_text(fingerprint, encoding="utf-8")
    return vectors, chunks


def search(query: str, top_k: int = 2) -> list[dict]:
    vectors, chunks = build_index()
    query_vector = np.array(list(_embedder().embed([query]))[0], dtype="float32")
    query_vector = _normalize(query_vector.reshape(1, -1))[0]

    scores = vectors @ query_vector
    order = np.argsort(-scores)[:top_k]
    return [{**chunks[int(i)], "score": float(scores[int(i)])} for i in order]