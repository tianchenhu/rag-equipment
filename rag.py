# -*- coding: utf-8 -*-
"""
最小可运行 RAG —— 完整版，每一行都有白话注释。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
先看这张图，整份代码就是它：

  离线（建库，只跑一次）
    你的文档  ──切块──▶  44 个小片段  ──变成向量──▶  44 × 768 的数字表  ──存盘──▶  index.npz

  在线（每次提问都跑）
    你的问题  ──变成向量──▶  1 × 768
                              │
                              ├─ 和那 44 行逐一比相似度（一次矩阵乘）
                              ├─ 取分数最高的 3 行
                              └─ 把这 3 段原文塞进 prompt ──▶  大模型  ──▶  答案

一句话：RAG 不改模型，改的是"提问时给模型看什么材料"。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

怎么跑（用毕设那个环境，依赖全是现成的）：
    PY="C:/Users/ASUS/Desktop/毕设/.venv/Scripts/python.exe"

    $PY walkthrough.py         # ← 先跑这个。它会把每一步的中间结果打印给你看
    $PY rag.py                 # 全流程跑一遍
    $PY rag.py ask "空压机排气温度高怎么办"
    $PY eval.py                # 评测：Hit@3 / MRR
    $PY eval.py --sweep        # 参数对照实验

只想改一件事试试？改上面那三个常量（CHUNK_SIZE / CHUNK_OVERLAP / TOP_K），
然后重跑 eval.py，看数字怎么变。这就是"调参实验"。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
INDEX = os.path.join(HERE, "index.npz")

EMB_MODEL = "shibing624/text2vec-base-chinese"   # 你毕设里用过的那个句向量模型
LLM_MODEL = "qwen2.5:7b"                          # 本地 ollama 里的模型
OLLAMA_URL = "http://localhost:11434/api/chat"

# 这三个数字是全部可调参数。改它们就是"做实验"。
CHUNK_SIZE = 300      # 每块最多多少字
CHUNK_OVERLAP = 50    # 相邻两块重叠多少字
TOP_K = 3             # 每次取回几段


# ══════════════════════════════════════════════════════════════
# 第 1 站：切块
# ══════════════════════════════════════════════════════════════
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    把一篇长文档切成若干小块。

    为什么要切？因为向量模型只擅长处理几百字的短文本。400 字还能勉强塞进去，
    换成 50 页的设备手册就必须先切。

    为什么不按段落切？因为段落长短太不均匀——本语料最短 200 字、最长 500 字。
    长段落会被模型截断，超出的部分悄无声息地丢掉。

    为什么要 overlap（重叠）？因为答案经常横跨两块的边界。
    比如一块的末尾讲"温控阀卡滞"，下一块开头讲"导致排气温度高"。
    没有重叠时，问题只命中其中一块，你就只拿到半个因果。
    重叠是用"多存一点"换"不漏掉"。
    """
    if not text:
        return []

    step = size - overlap             # 每次往前挪 250 个字（300 - 50）
    out = []                          # 装所有切出来的块
    start = 0                         # 当前从第几个字开始切
    n = len(text)                     # 全文多少字

    while start < n:                  # 只要还没切到末尾，就继续
        out.append(text[start:start + size])   # 切出 [start, start+300) 这一段
        if start + size >= n:         # 如果这一块已经碰到文件末尾了
            break                     # 就收工，不用再切一块几乎重复的
        start += step                 # 否则往前挪 250 字，切下一块

    return out


# ══════════════════════════════════════════════════════════════
# 第 2 站：把文字变成向量
# ══════════════════════════════════════════════════════════════
_MODEL = None      # 模型加载一次约 3 秒、占 400 MB，所以缓存起来反复用


def get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        # local_files_only=True：只用本地已下载的权重，不联网
        _MODEL = SentenceTransformer(EMB_MODEL, local_files_only=True)
    return _MODEL


def embed(texts):
    """
    文字 → 一串 768 个数字。

    为什么要变成数字？因为计算机没法直接比"两句话像不像"，
    但可以比"两组数字离得近不近"。这个模型是专门训练过的：
    意思相近的句子，会被映射到向量空间里相近的位置。

    normalize_embeddings=True 是关键的一步：它顺便把每个向量缩放成"长度 = 1"。
    这样下一步比较相似度时，"余弦相似度"就简化成了一次点积（省掉除法）。

    为什么查询和文档必须用同一个模型？
    两个不同模型各自训练的坐标系毫无关系。拿 A 的坐标去和 B 的坐标比距离，
    结果等于随机数。这是 RAG 最硬的一条铁律。
    """
    if isinstance(texts, str):      # 传进来一个字符串，就包成列表
        texts = [texts]
    return get_model().encode(
        texts,
        normalize_embeddings=True,   # 归一化，长度变成 1
        convert_to_numpy=True,       # 返回 numpy 数组，方便后面做矩阵运算
    )


# ══════════════════════════════════════════════════════════════
# 第 3 站：把向量存起来
# ══════════════════════════════════════════════════════════════
def load_docs():
    """读 22 篇文档。jsonl = 每行一个 JSON 对象。"""
    with open(os.path.join(DATA, "corpus.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_questions():
    """读 30 条测试问题（25 条可答 + 5 条应该拒答）。"""
    with open(os.path.join(DATA, "questions.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_index():
    """
    离线阶段：读文档 → 切块 → 变向量 → 存盘。只跑一次。

    存三样东西，缺一不可：
      vecs   : 44 行 × 768 列的数字表，用来算相似度
      chunks : 44 段原文，检索命中后要拿它去拼 prompt
      metas  : 每段来自哪篇文档（doc_id），用来溯源和评测
    """
    chunks = []       # 所有块的原文本
    metas = []        # 每块对应的来源信息，和 chunks 一一对应

    for doc in load_docs():
        for piece in chunk_text(doc["text"]):
            chunks.append(piece)
            metas.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "type": doc["type"],
            })

    vecs = embed(chunks)          # 44 段文字 → 44 × 768

    np.savez(
        INDEX,
        vecs=vecs,
        chunks=np.array(chunks, dtype=object),   # dtype=object：因为每段长度不同
        metas=np.array(metas, dtype=object),     # 存字典也得用 object
    )
    print(f"索引完成：{len(chunks)} 块 × {vecs.shape[1]} 维 → index.npz")
    return vecs, chunks, metas


def load_index():
    """读盘。allow_pickle=True 是因为存了字符串和字典。"""
    z = np.load(INDEX, allow_pickle=True)
    return z["vecs"], list(z["chunks"]), list(z["metas"])


# ══════════════════════════════════════════════════════════════
# 第 4 站：检索
# ══════════════════════════════════════════════════════════════
def retrieve(query, vecs, chunks, top_k=TOP_K):
    """
    给一个问题，找出最相关的 top_k 段。

    返回 [(分数, 在 chunks 里的下标, 原文), ...]，分数从高到低。

    为什么一行矩阵乘就够？
    因为第 2 站已经把所有向量都缩放成长度 1 了。
    长度都是 1 的时候，"余弦相似度"就等于"点积"，
    所以 vecs @ qv 一次就算出了问题跟全部 44 段的相似度。
    """
    qv = embed([query])[0]        # 问题 → 1 个 768 维向量

    sims = vecs @ qv              # (44, 768) @ (768,) → (44,) 44 个分数

    order = np.argsort(-sims)     # 从大到小排（加负号是因为默认从小到大排）
    top = order[:top_k]           # 取前 top_k 个下标

    return [(float(sims[i]), int(i), chunks[i]) for i in top]


# ══════════════════════════════════════════════════════════════
# 第 5 站：拼 prompt，让模型照着材料回答
# ══════════════════════════════════════════════════════════════
PROMPT = """你是一名设备维修助理。请只依据下面提供的资料回答问题。

规则：
1. 如果资料中没有答案，直接回答"资料中未提及"，不要猜测，不要用常识补充。
2. 回答时标注依据的文档编号，例如（WO-1063）。
3. 用简洁的中文，不要复述资料原文，不要加开场白。

资料：
{context}

问题：{question}
"""


def generate(question, hits, metas):
    """
    把检索到的几段资料拼成一段 prompt，交给本地大模型。

    prompt 里那条"没有答案就说未提及"是整份代码里最重要的一句话。
    不写它，模型会用训练时记住的东西去补全，编出一个看起来很像、
    但完全错误的答案（幻觉）。RAG 存在的意义就是"有据可查"，
    一旦允许它自由发挥，这层意义就没了。

    要求标注文档编号，是为了让"可追溯"这件事看得见——
    用户能拿着编号回去核对原文，这是 RAG 比直接问模型唯一硬的地方。
    """
    import requests

    # 把 3 段资料拼在一起，每段前面标来源，段与段之间用 --- 隔开
    parts = []
    for _, idx, text in hits:
        parts.append(f"[{metas[idx]['doc_id']}] {metas[idx]['title']}\n{text}")
    context = "\n\n---\n\n".join(parts)

    prompt = PROMPT.format(context=context, question=question)

    resp = requests.post(OLLAMA_URL, json={
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1},   # 压低随机性，同一个问题别答出两种
    }, timeout=180)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


# ══════════════════════════════════════════════════════════════
# 以下是自测与串联，不需要看懂
# ══════════════════════════════════════════════════════════════
def _ok(cond, msg):
    print(("  [OK]   " if cond else "  [FAIL] ") + msg)
    return cond


def test1():
    print("第 1 站 · 切块")
    text = "".join(f"{i:03d}" for i in range(1, 61))   # 180 字的测试文本
    cs = chunk_text(text, size=50, overlap=10)
    ok = _ok(len(cs) >= 3, f"切出 {len(cs)} 块（应 >= 3）")
    ok &= _ok(all(len(c) <= 50 for c in cs), f"每块长度 <= 50（最长 {max(len(c) for c in cs)}）")
    ok &= _ok(all(c for c in cs), "没有空块")
    ok &= _ok(cs[1][:10] == cs[0][-10:], "相邻块重叠 10 字")
    cover = sum(len(c) for c in cs) - 10 * (len(cs) - 1)
    ok &= _ok(cover >= len(text), f"覆盖全文（有效覆盖 {cover} 字 / 原文 {len(text)} 字）")
    doc = load_docs()[0]
    real = chunk_text(doc["text"])
    print(f"  例：{doc['doc_id']}（{len(doc['text'])} 字）→ {len(real)} 块，长度 {[len(c) for c in real]}")
    return ok


def test2():
    print("第 2 站 · 向量化")
    v = embed(["主轴有异响", "机床发出怪声", "注塑件出现飞边"])
    ok = _ok(v.shape == (3, 768), f"shape = {v.shape}（应 (3, 768)）")
    norms = np.linalg.norm(v, axis=1)
    ok &= _ok(np.allclose(norms, 1.0, atol=1e-4), f"已归一化（长度 {norms.round(4).tolist()}）")
    sim = v @ v.T
    # 中文短句的余弦基线偏高，两句无关的话也可能到 0.4，所以比"相对差距"而不是卡绝对值
    ok &= _ok(sim[0, 1] > 0.6, f"近义句相似度 {sim[0,1]:.3f} > 0.6")
    ok &= _ok(sim[0, 1] - sim[0, 2] > 0.15,
              f"近义({sim[0,1]:.3f}) 与 无关({sim[0,2]:.3f}) 差距 {sim[0,1]-sim[0,2]:.3f} > 0.15")
    return ok


def test3():
    print("第 3 站 · 建索引")
    vecs, chunks, metas = build_index()
    ok = _ok(len(chunks) == vecs.shape[0] == len(metas),
             f"三者条数一致：{len(chunks)} / {vecs.shape[0]} / {len(metas)}")
    ok &= _ok(30 <= len(chunks) <= 80, f"切片总数 {len(chunks)}（合理区间 30~80）")
    v2, c2, _ = load_index()
    ok &= _ok(v2.shape == vecs.shape and c2[0] == chunks[0], "存盘后读回一致")
    return ok


def test4():
    print("第 4 站 · 检索")
    vecs, chunks, metas = load_index()
    cases = [
        ("GA-75 空压机排气温度高的原因", "WO-1063"),
        ("机器人手腕转起来咔咔响", "WO-1115"),
        ("循环冷却水多久检测一次硬度", "SOP-07"),
    ]
    ok = True
    for q, gold in cases:
        hits = retrieve(q, vecs, chunks, top_k=3)
        ok &= _ok(len(hits) == 3, "返回 3 条")
        ok &= _ok(all(hits[i][0] >= hits[i + 1][0] for i in range(len(hits) - 1)), "分数降序")
        got = [metas[h[1]]["doc_id"] for h in hits]
        ok &= _ok(gold in got, f"「{q}」→ {got}（期望含 {gold}）")
    return ok


def test5():
    print("第 5 站 · 生成")
    vecs, chunks, metas = load_index()
    q = "GA-75 空压机排气温度高，跟温控阀有关系吗？"
    ans = generate(q, retrieve(q, vecs, chunks, top_k=3), metas)
    print("  回答：", ans.replace("\n", " ")[:300])
    ok = _ok("WO-1063" in ans or "温控阀" in ans, "答案切题")
    q2 = "KR-16 机器人的最大运动速度是多少？"
    ans2 = generate(q2, retrieve(q2, vecs, chunks, top_k=3), metas)
    print("  拒答测试：", ans2.replace("\n", " ")[:200])
    ok &= _ok(any(w in ans2 for w in ["未提及", "没有", "无法", "资料中未"]), "语料无答案时拒答")
    return ok


def run_all():
    import time
    t0 = time.time()
    for i, fn in enumerate([test1, test2, test3, test4, test5], 1):
        if not fn():
            print(f"\n第 {i} 站有问题，先看看上面的 FAIL。")
            return
        print()
    print(f"五站全通，耗时 {time.time()-t0:.1f} 秒。接下来跑：python eval.py")


def ask(question, top_k=TOP_K):
    vecs, chunks, metas = load_index()
    hits = retrieve(question, vecs, chunks, top_k=top_k)
    print("检索到的片段：")
    for s, i, _ in hits:
        print(f"  {s:.3f}  {metas[i]['doc_id']}  {metas[i]['title']}")
    print("\n回答：", generate(question, hits, metas))


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg in ("1", "2", "3", "4", "5"):
        {"1": test1, "2": test2, "3": test3, "4": test4, "5": test5}[arg]()
    elif arg == "ask":
        ask(" ".join(sys.argv[2:]))
    else:
        run_all()
