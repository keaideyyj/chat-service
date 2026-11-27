"""探针：打印若干查询的召回分数，用来确定 MIN_SCORE。

原则：正例的最低分 与 负例的最高分 之间取阈值。
运行 .\.venv\Scripts\python.exe -m scripts.probe_scores
"""

from app.rag import build_index, search

build_index(rebuild=True)

positive = ["年假有几天", "报销多久内提交", "入职要带什么材料"]
negative = ["公司股票代码是多少", "今天天气怎么样", "帮我写一首诗"]

for group, queries in (("正例", positive), ("负例", negative)):
    print(f"\n== {group} ==")
    for query in queries:
        hits = search(query, top_k=3)
        detail = ", ".join(f"{h['score']:.3f} {h['title']}" for h in hits)
        print(f"  {query:20s} -> {detail}")