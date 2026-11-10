# -*- coding: utf-8 -*-
"""
RAG 评测脚本 —— 这个文件不用改，直接跑。

用法：
    python eval.py                # 用现有 index.npz 评测，输出 Hit@1/3/5/10 与失败案例
    python eval.py --sweep        # 扫 chunk size，做对照实验（简历上的数字来源）
    python eval.py --refuse       # 只测拒答（语料里没有的问题，模型该不该编）

关于指标口径（面试可能追问，务必看懂这段）：
    本脚本算的是 Hit@k（命中率）= 「top-k 里至少有一块来自正确文档」的问题占比。
    严格意义的 Recall@k 是「召回的 gold 块数 / 全部 gold 块数」，需要标注到块级。
    文档级 Hit@k 是工程上最常用的简化口径，因为它和你真正关心的事一致：
    「答案所在的文档，有没有被翻出来」。说的时候要说 Hit@k，别说成 Recall@k。
"""
import argparse
import json

import numpy as np

import rag

KS = (1, 3, 5, 10)


def measure(questions, vecs, chunks, metas, ks=KS):
    """对每条可答问题检索 top-10，记录 gold 文档首次出现的位置。

    固定按 top-10 检索再按 ks 截断统计，保证不同 ks 配置之间 MRR 口径一致。
    """
    ranks = []
    detail = []
    for q in questions:
        if q["kind"] == "absent":
            continue
        hits = rag.retrieve(q["q"], vecs, chunks, top_k=10)
        got = [metas[h[1]]["doc_id"] for h in hits]
        rank = got.index(q["gold"]) + 1 if q["gold"] in got else None
        ranks.append(rank)
        detail.append({"qid": q["qid"], "q": q["q"], "gold": q["gold"],
                       "kind": q["kind"], "got": got, "rank": rank})
    out = {}
    for k in ks:
        out[k] = sum(1 for r in ranks if r is not None and r <= k) / len(ranks)
    mrr = sum(1 / r for r in ranks if r is not None) / len(ranks)
    out["mrr"] = mrr
    out["detail"] = detail
    out["n"] = len(ranks)
    return out


def show(res, title):
    print(f"\n{title}  （{res['n']} 条可答问题）")
    line = "  ".join(f"Hit@{k}={res[k]:.3f}" for k in KS)
    print(f"  {line}   MRR={res['mrr']:.3f}")

    by_kind = {}
    for d in res["detail"]:
        by_kind.setdefault(d["kind"], []).append(d["rank"])
    print("  分类：", end="")
    for kind, rs in by_kind.items():
        h3 = sum(1 for r in rs if r is not None and r <= 3) / len(rs)
        print(f"  {kind} Hit@3={h3:.2f}(n={len(rs)})", end="")
    print()

    miss = [d for d in res["detail"] if d["rank"] is None or d["rank"] > 3]
    if miss:
        print(f"\n  失败案例（{len(miss)} 条，写进 README 的'失败分析'一节）：")
        for d in miss:
            print(f"    {d['qid']} [{d['kind']}] {d['q'][:26]}")
            print(f"        期望 {d['gold']}  实际 {d['got'][:3]}")


def build_in_memory(size, overlap=50):
    """不落盘地重建一套索引，用于 sweep 对照实验。"""
    docs = rag.load_docs()
    chunks, metas = [], []
    for d in docs:
        for c in rag.chunk_text(d["text"], size=size, overlap=overlap):
            chunks.append(c)
            metas.append({"doc_id": d["doc_id"], "title": d["title"]})
    vecs = rag.embed(chunks)
    return vecs, chunks, metas


def sweep():
    qs = rag.load_questions()
    print("对照实验：不同 chunk size 对检索质量的影响")
    print("（overlap 固定 50；块数越多不代表越好，看 Hit@3 和单块信息完整度的平衡）\n")
    print(f"{'size':>6} {'块数':>6} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'MRR':>7}")
    print("-" * 46)
    rows = []
    for size in (150, 200, 300, 400, 600):
        vecs, chunks, metas = build_in_memory(size)
        r = measure(qs, vecs, chunks, metas, ks=(1, 3, 5))
        rows.append((size, len(chunks), r[1], r[3], r[5], r["mrr"]))
        print(f"{size:>6} {len(chunks):>6} {r[1]:>7.3f} {r[3]:>7.3f} {r[5]:>7.3f} {r['mrr']:>7.3f}")
    best = max(rows, key=lambda x: x[3])
    print(f"\nHit@3 最优：size={best[0]}（{best[3]:.3f}）。把这个结论写进 README，"
          f"并解释为什么不是越大越好。")


def refuse_test():
    """拒答测试：语料里明确没有答案，模型应该说不知道而不是编。"""
    vecs, chunks, metas = rag.load_index()
    qs = [q for q in rag.load_questions() if q["kind"] == "absent"]
    ok = 0
    for q in qs:
        hits = rag.retrieve(q["q"], vecs, chunks, top_k=3)
        ans = rag.generate(q["q"], hits, metas)
        refused = any(w in ans for w in ["未提及", "没有", "未包含", "无法确定", "资料中未", "无相关"])
        # 同时检查它是否"编造"了具体数字（最常见的幻觉形态）
        fabricated = any(ch.isdigit() for ch in ans) and not refused
        ok += refused
        print(f"  {'[拒答]' if refused else '[幻觉]'} {q['q']}")
        if not refused:
            print(f"        → {ans[:120]}")
    print(f"\n拒答正确率 {ok}/{len(qs)} = {ok/len(qs):.0%}")
    print("低于 100% 就去改 PROMPT：把'资料中没有答案时直接回答未提及'放到第一句。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--refuse", action="store_true")
    a = ap.parse_args()

    if a.sweep:
        sweep()
        return
    if a.refuse:
        refuse_test()
        return

    vecs, chunks, metas = rag.load_index()
    qs = rag.load_questions()
    res = measure(qs, vecs, chunks, metas)
    show(res, f"基线（已存索引，{len(chunks)} 块）")
    print("\n下一步：python eval.py --sweep   拿到对照实验数字")
    print("        python eval.py --refuse  测拒答")


if __name__ == "__main__":
    main()
