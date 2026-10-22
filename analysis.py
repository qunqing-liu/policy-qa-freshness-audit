# -*- coding: utf-8 -*-
"""
政策问答失效审计分析（可复现脚本）
数据：policy_queries.csv（950 行，任务发放数据，未做任何修改）
运行：python analysis.py  （依赖 pandas / numpy，见 requirements.txt）
所有 README 中引用的数字均由本脚本对发放数据计算得出。
"""
import pandas as pd
import numpy as np

HOUR_INVALID = 75   # 口径（任务给定）：每次失效政策答错损失 75 工时
HOUR_VERIFY  = 12   # 口径（任务给定）：每提示版本待核验一次花费 12 工时

df = pd.read_csv("policy_queries.csv")
df["query_time"]   = pd.to_datetime(df["query_time"], utc=True)
df["doc_synced_at"] = pd.to_datetime(df["doc_synced_at"], utc=True)

# 同步时滞：问答发生时间 - 命中文档最后同步时间
df["lag_min"] = (df["query_time"] - df["doc_synced_at"]).dt.total_seconds() / 60.0
# 版本代理：同类别 + 同一同步分钟视为同一文档版本（数据无文档 ID，见 README 假设）
df["version"] = df["policy_category"] + "|" + df["doc_synced_at"].dt.strftime("%Y%m%d%H%M")

def sec(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)

# ---------------------------------------------------------------- 1. 全貌
sec("1. 数据全貌")
n = len(df)
n_inv, n_val, n_pend = (df.is_valid == 0).sum(), (df.is_valid == 1).sum(), df.is_valid.isna().sum()
print(f"总记录 {n} 条；时间范围 {df.query_time.min():%Y-%m-%d} ~ {df.query_time.max():%Y-%m-%d}（7 天）")
print(f"业务复核：有效 {n_val}，失效 {n_inv}，待复核 {n_pend}")
print(f"已复核无效率 = {n_inv} / {n_val + n_inv} = {n_inv / (n_val + n_inv) * 100:.1f}%")
print(f"失效答错造成的返工损失 = {n_inv} × {HOUR_INVALID} = {n_inv * HOUR_INVALID} 工时/周")

# ---------------------------------------------------------------- 2. freshness_score 检查
sec("2. freshness_score（知识库自评健康度）是否可信")
print(f"freshness_score 范围 {df.freshness_score.min()}~{df.freshness_score.max()}（名义 0-100），均值 {df.freshness_score.mean():.1f}")
print(f"与 is_valid 的 Pearson 相关 = {df['freshness_score'].corr(df['is_valid']):.3f}")
print(f"与 sync_lag_min 的 Pearson 相关 = {df['freshness_score'].corr(df['lag_min']):.3f}")
print("分桶无效率：")
b = pd.cut(df["freshness_score"], bins=[-1, 5, 10, 15, 20, 25, 30],
           labels=["0-5", "6-10", "11-15", "16-20", "21-25", "26-30"])
g = df[df.is_valid.notna()].groupby(b, observed=True).agg(
    n=("query_id", "count"), invalid=("is_valid", lambda s: int((s == 0).sum())))
g["rate"] = (g["invalid"] / g["n"] * 100).round(1)
print(g.to_string())

# ---------------------------------------------------------------- 3. 同步时滞
sec("3. 同步时滞（query_time - doc_synced_at）与无效率")
print("失效 / 有效 / 待复核三组的时滞（分钟）：")
print(df.groupby(df.is_valid.map({1.0: "valid", 0.0: "invalid", np.nan: "pending"}))
        .lag_min.describe().round(1).to_string())
b2 = pd.cut(df["lag_min"], bins=[-1, 15, 20, 25, 30, 40, 50, 70],
            labels=["<15", "15-20", "20-25", "25-30", "30-40", "40-50", "50-70"])
g2 = df[df.is_valid.notna()].groupby(b2, observed=True).agg(
    n=("query_id", "count"), invalid=("is_valid", lambda s: int((s == 0).sum())))
g2["rate"] = (g2["invalid"] / g2["n"] * 100).round(1)
print("按时滞分桶的无效率：")
print(g2.to_string())

# ---------------------------------------------------------------- 4. 拦截规则 + 成本寻优
sec("4. 拦截规则成本寻优（口径：核验 12 工时/版本，答错 75 工时/次）")
iv = df[df.is_valid == 0]
baseline = HOUR_INVALID * len(iv)
print(f"基线（不拦截）：{baseline} 工时/周")
print("规则：命中文档满足 [时滞≥L 分钟 或 freshness≤F] 则先核验再作答")
print(f"{'L':>4} {'F':>3} {'核验版本':>7} {'拦下失效':>7} {'漏过失效':>7} {'总成本':>7} {'节省':>7}")
rows = []
for L in [10, 15, 20, 25, 30, 40, 50]:
    for F in [0, 5, 10]:
        flag = (df["lag_min"] >= L) | (df["freshness_score"] <= F)
        flagV = set(df[flag]["version"])
        saved = iv[iv["version"].isin(flagV)]
        missed = iv[~iv["version"].isin(flagV)]
        cost = HOUR_VERIFY * len(flagV) + HOUR_INVALID * len(missed)
        rows.append((cost, L, F, len(flagV), len(saved), len(missed), baseline - cost))
        print(f"{L:>4} {F:>3} {len(flagV):>7} {len(saved):>7} {len(missed):>7} {cost:>7} {baseline - cost:>7}")
rows.sort()
best = rows[0]
print(f"\n最优规则：时滞≥{best[1]} 分钟 或 freshness≤{best[2]}")
print(f"  核验 {best[3]} 个版本（{best[3] * HOUR_VERIFY} 工时），拦下失效 {best[4]}/{len(iv)}（召回 {best[4] / len(iv) * 100:.0f}%），"
      f"漏过 {best[5]} 次（{best[5] * HOUR_INVALID} 工时）")
print(f"  总成本 {best[0]} 工时/周，较基线节省 {best[6]} 工时/周（{best[6] / baseline * 100:.0f}%）")

# ---------------------------------------------------------------- 5. 最优规则细节
sec("5. 最优规则细节与优先处置清单")
L, F = best[1], best[2]
flag = (df["lag_min"] >= L) | (df["freshness_score"] <= F)
flagV = set(df[flag]["version"])
fl = df[df["version"].isin(flagV)]
print(f"被拦截的查询 {flag.sum()} 条（{len(flagV)} 个版本）；其中已复核部分：失效 {int((fl.is_valid == 0).sum())}，有效 {int((fl.is_valid == 1).sum())}")
print(f"拦截精度 = 失效 / 已复核 = {(fl.is_valid == 0).sum() / (fl.is_valid.notna()).sum() * 100:.1f}%")
pend_flag = df[df.is_valid.isna() & df["version"].isin(flagV)]
print(f"39 条待复核中有 {len(pend_flag)} 条命中规则 → 建议优先核验（预期可避免 {len(pend_flag) * 0.75:.0f} 次失效，约 {len(pend_flag) * 0.75 * HOUR_INVALID:.0f} 工时）")
print("\n失效返工工时按类别（Top5，全量）：")
top = df[df.is_valid == 0].groupby("policy_category").agg(
    invalid=("query_id", "count"), hours=("query_id", lambda s: len(s) * HOUR_INVALID))
print(top.sort_values("hours", ascending=False).head(5).to_string())
print("\n失效返工工时按部门：")
dep = df[df.is_valid == 0].groupby("dept").agg(
    invalid=("query_id", "count"), hours=("query_id", lambda s: len(s) * HOUR_INVALID))
print(dep.sort_values("hours", ascending=False).to_string())
