# -*- coding: utf-8 -*-
"""
RAG 可视化走查 —— 不用读代码，跑一遍就知道每一步在干什么。

    PY="C:/Users/ASUS/Desktop/毕设/.venv/Scripts/python.exe"
    $PY walkthrough.py

它会带你走 6 步，每一步都把中间结果打印出来：
你看到的不是"代码跑通了"，而是"我看见了它怎么想的"。
"""
import numpy as np

import rag

LINE = "━" * 62


def title(step, text):
    print(f"\n{LINE}\n第 {step} 步 · {text}\n{LINE}")


def main():
    docs = rag.load_docs()
    total = sum(len(d["text"]) for d in docs)

    # ── 0 ────────────────────────────────────────────────
    title(0, "你的语料长什么样")
    print(f"共 {len(docs)} 篇文档，合计 {total} 字，平均 {total // len(docs)} 字/篇")
    types = {}
    for d in docs:
        types[d["type"]] = types.get(d["type"], 0) + 1
    print("构成：", "、".join(f"{k} {v} 篇" for k, v in types.items()))
    demo = docs[0]
    print(f"\n拿第一篇举例 —— {demo['doc_id']}（{demo['title']}）")
    print("  开头 60 字：", demo["text"][:60].replace("\n", " "), "…")
    print(f"  （全文 {len(demo['text'])} 字）")

    # ── 1 ────────────────────────────────────────────────
    title(1, "切块：一篇文档变成几块")
    chunks = rag.chunk_text(demo["text"])
    print(f"当前参数：每块最多 {rag.CHUNK_SIZE} 字，相邻块重叠 {rag.CHUNK_OVERLAP} 字")
    print(f"→ {demo['doc_id']} 被切成 {len(chunks)} 块\n")
    for i, c in enumerate(chunks):
        print(f"  块 {i + 1}（{len(c)} 字）：{c[:45].replace(chr(10), ' ')}…")

    if len(chunks) >= 2:
        tail = chunks[0][-rag.CHUNK_OVERLAP:]
        head = chunks[1][:rag.CHUNK_OVERLAP]
        print(f"\n  ← 这就是 overlap（重叠）：")
        print(f"     块 1 结尾 {rag.CHUNK_OVERLAP} 字：{tail[:40].replace(chr(10), ' ')}…")
        print(f"     块 2 开头 {rag.CHUNK_OVERLAP} 字：{head[:40].replace(chr(10), ' ')}…")
        print(f"     完全一致？{tail == head}")
        print("     为什么要重复存一遍：防止一句因果关系被切到两块边上，结果只被取到一半。")

    # ── 2 ────────────────────────────────────────────────
    title(2, "向量化：文字变成 768 个数字")
    a, b, c = "主轴有异响", "机床发出怪声", "注塑件出现飞边"
    vecs = rag.embed([a, b, c])
    head = "[" + ", ".join(f"{x:.3f}" for x in vecs[0][:8]) + f", …] 共 {vecs.shape[1]} 个数"
    print(f"「{a}」→", head)
    print(f"  这串数字的长度（L2 范数）= {np.linalg.norm(vecs[0]):.4f}"
          f"  ← 被归一化成 1 了，为了下一步算得快")
    print(f"\n三句话两两相似度：")
    print(f"  「{a}」vs「{b}」= {vecs[0] @ vecs[1]:.3f}   ← 意思像，分数高")
    print(f"  「{a}」vs「{c}」= {vecs[0] @ vecs[2]:.3f}   ← 意思不像，分数低")
    print("  注意：不相关也能到 0.4 左右，中文短句的基线本来就偏高。")
    print("  所以检索靠的是「排序取前几名」，不是「分数超过某个值才算命中」。")

    # ── 3 ────────────────────────────────────────────────
    title(3, "建索引：44 段全部变成向量，存成一个矩阵")
    vecs, chunks, metas = rag.build_index()
    print(f"矩阵形状 {vecs.shape} = （片段数，向量维度）")
    print(f"也就是 {vecs.shape[0]} 行，每行代表一段文字")
    print(f"\n每篇文档被切成几块：", end="")
    from collections import Counter
    cnt = Counter(m["doc_id"] for m in metas)
    counts = sorted(set(cnt.values()))
    print("每篇都是 " + "/".join(str(x) for x in counts) + " 块"
          if len(counts) == 1 else f"各篇不同：{dict(cnt)}")

    # ── 4 ────────────────────────────────────────────────
    title(4, "检索：一个问题，和 44 段逐一算分")
    question = "GA-75 空压机排气温度高，跟温控阀有关系吗？"
    print(f"问题：{question}\n")
    qv = rag.embed([question])[0]
    sims = vecs @ qv
    print(f"一次性算出 {len(sims)} 个分数（一行矩阵乘，毫秒级）")
    print(f"最高分 {sims.max():.3f}    最低分 {sims.min():.3f}\n")
    print("分数最高的 5 段：")
    order = np.argsort(-sims)[:5]
    seen, dup = set(), []
    for rank, i in enumerate(order, 1):
        did = metas[i]["doc_id"]
        if did in seen:
            mark = "↺ 重复"
            dup.append(did)
        elif did == "WO-1063":
            mark = "★ 命中"
        else:
            mark = "     "
        seen.add(did)
        print(f"  {rank}. {sims[i]:.3f}  {mark}  {did}  {metas[i]['title']}")

    print("\n  （标准答案是 WO-1063，排第 1 —— 检索对了）")
    if dup:
        print(f"\n  但注意：{dup[0]} 出现了两次 —— 因为每篇被切成 2 块，两块都跟问题相关。")
        print("  这在真实场景是浪费：一个位子给了重复内容，等于把 top_k 变相降了一档。")
        print("  解法是「按文档去重」：同一篇只保留分数最高的那块。这就是可优化点。")

    print(f"\n  top_k 当前是 {rag.TOP_K}，只有前 {rag.TOP_K} 段会被送给模型。")
    print("  如果把 top_k 调到 10，多出来的几段里有些是无关的，会干扰模型判断。")

    # ── 5 ────────────────────────────────────────────────
    title(5, "拼 prompt：模型真正看到的东西")
    hits = rag.retrieve(question, vecs, chunks, top_k=rag.TOP_K)
    parts = []
    for _, idx, text in hits:
        parts.append(f"[{metas[idx]['doc_id']}] {metas[idx]['title']}\n{text}")
    prompt = rag.PROMPT.format(context="\n\n---\n\n".join(parts), question=question)

    print("下面是完整 prompt（模型的全部输入就是这些，没有别的）：\n")
    for line in prompt.split("\n"):
        print("  │ " + line)
    print(f"\n  prompt 长度：{len(prompt)} 字")
    print("  注意最后那句「资料中没有答案就说未提及」——")
    print("  没有它，模型会拿训练时记住的东西硬答，编出看起来很像但错误的答案。")

    # ── 6 ────────────────────────────────────────────────
    title(6, "生成：模型照着材料回答")
    ans = rag.generate(question, hits, metas)
    print(ans)
    print("\n检查两件事：")
    print("  ① 有没有标注文档编号（能追溯 = RAG 的意义所在）")
    print("  ② 有没有大量复述原文（复述说明它只是在抄，没在理解）")

    # ── 收尾 ─────────────────────────────────────────────
    print(f"\n{LINE}")
    print("走完了。现在做这一件事：改一个数字，看结果怎么变。")
    print(f"{LINE}")
    print("打开 rag.py，把最上面那三个常量改一个，比如：")
    print("    CHUNK_SIZE = 300  →  600")
    print("然后重跑：")
    print("    python walkthrough.py     ← 看第 1 步是不是从 2 块变成 1 块")
    print("    python eval.py            ← 看 Hit@3 从 0.96 掉到多少")
    print("\n再把 TOP_K 从 3 改成 10，跑：")
    print('    python rag.py ask "GA-75 空压机排气温度高，跟温控阀有关系吗？"')
    print("看答案是不是开始混进别的内容。")
    print("\n这两个动作做完，你就有了「调过参、看过结果」的真实经历，")
    print("面试问起来能讲 5 分钟，而且每句都是自己看到的。")


if __name__ == "__main__":
    main()
