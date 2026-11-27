"""rag.py 的测试。

分两类，跑起来的代价差别很大：

- **纯逻辑**（归一化 / 切分 / 指纹 / 索引缓存）：不联网、不用模型，毫秒级。
- **检索质量**（需要真实 embedding 模型）：首次运行要加载 ~95MB 模型，几秒钟。

这里刻意**不 mock embedding**：检索质量是这条链的核心行为，把模型换掉就等于把要测的
东西换掉了 —— 那只能测「代码不报错」，测不出「检索对不对」。要控制成本就在用例数量上
控制，而不是在保真度上妥协。
"""

import numpy as np
import pytest

from app import rag

# ── 归一化：纯数学，最快也最值得锁死 ────────────────────────────


def test_normalize_makes_rows_unit_length():
    matrix = np.array([[3.0, 4.0, 0.0], [1.0, 0.0, 0.0]], dtype="float32")

    out = rag._normalize(matrix)

    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    # 方向不变：(3,4) → (0.6,0.8)
    assert np.allclose(out[0], [0.6, 0.8, 0.0])


def test_normalize_survives_zero_vector():
    """零向量不能产生 nan。

    这是 `norms[norms == 0] = 1.0` 那行的意义：一个空片段混进索引，
    如果除出 nan，整张分数表都会被污染（nan 会传染），而且不会报错。
    """
    matrix = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype="float32")

    out = rag._normalize(matrix)

    assert not np.isnan(out).any()
    assert np.allclose(out[0], [0.0, 0.0, 0.0])


# ── 切分：只读真实语料，不写任何东西 ────────────────────────────


def test_load_chunks_returns_enough_chunks():
    assert len(rag.load_chunks()) >= 10


def test_load_chunks_fields_are_clean():
    for chunk in rag.load_chunks():
        assert chunk["doc"], "doc 不能为空"
        assert chunk["title"], "title 不能为空"
        assert chunk["text"], "text 不能为空"
        assert not chunk["title"].startswith("#"), "标题里不该带着 markdown 的井号"
        assert chunk["title"] in chunk["text"], "embedding 用的是 text，标题必须拼进去"


# ── 语料指纹与索引缓存：这是「索引自动失效」的依据 ──────────────


def test_fingerprint_is_stable_then_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "DOCS_DIR", tmp_path)
    doc = tmp_path / "a.md"
    doc.write_text("# A\n\n## 标题一\n内容一\n", encoding="utf-8")

    first = rag._corpus_fingerprint()
    assert rag._corpus_fingerprint() == first, "语料没动，指纹必须一样"

    doc.write_text("# A\n\n## 标题一\n内容改过了，更长一些\n", encoding="utf-8")
    assert rag._corpus_fingerprint() != first, "语料变了，指纹必须变"


class _FakeEmbedder:
    """固定维度的假模型：只用来验证缓存行为，不参与检索质量测试。"""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        for _ in texts:
            yield np.array([1.0] + [0.0] * 7, dtype="float32")


def test_build_index_reuses_cache_and_rebuilds_on_change(tmp_path, monkeypatch):
    """指纹一致就复用索引；语料一变就必须重建 —— 否则新文档永远进不了检索范围。

    这正是 c04 那次失败的根因：语料加了「报销提交时限」，索引还是旧的 24 条。
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\n## 标题一\n内容一\n", encoding="utf-8")

    index_dir = tmp_path / "index"
    monkeypatch.setattr(rag, "DOCS_DIR", docs)
    monkeypatch.setattr(rag, "INDEX_DIR", index_dir)
    monkeypatch.setattr(rag, "VEC_FILE", index_dir / "vectors.npy")
    monkeypatch.setattr(rag, "META_FILE", index_dir / "chunks.json")
    monkeypatch.setattr(rag, "FP_FILE", index_dir / "corpus.fingerprint")

    fake = _FakeEmbedder()
    monkeypatch.setattr(rag, "_embedder", lambda: fake)

    rag.build_index()
    assert fake.calls == 1, "第一次必须真正算向量"

    rag.build_index()
    assert fake.calls == 1, "指纹没变，不该重算"

    (docs / "a.md").write_text("# A\n\n## 标题一\n内容二，变长了\n", encoding="utf-8")
    rag.build_index()
    assert fake.calls == 2, "语料变了，必须重建"


# ── 检索质量：需要真实模型 ──────────────────────────────────────


@pytest.fixture(scope="module")
def rag_index():
    pytest.importorskip("fastembed")
    rag.build_index()
    return True


def test_search_finds_annual_leave(rag_index):
    from app.tools import MIN_SCORE

    hits = rag.search("年假有几天", top_k=3)

    assert hits
    assert "年假" in hits[0]["title"]
    assert hits[0]["score"] >= MIN_SCORE


def test_search_scores_unrelated_query_low(rag_index):
    from app.tools import MIN_SCORE

    hits = rag.search("帮我写一首关于秋天的诗", top_k=1)

    assert hits[0]["score"] < MIN_SCORE


def test_search_kb_answers_when_found_and_refuses_when_not(rag_index):
    """工具层的拒绝是业务规则，不是 rag 的内部细节，所以放在这一层验证。"""
    from app.tools import search_kb

    assert "年假" in search_kb("年假有几天")
    assert "没有找到" in search_kb("帮我写一首关于秋天的诗")
