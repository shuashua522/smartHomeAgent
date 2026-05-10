import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings

warnings.filterwarnings('ignore')


def welch_anova(groups):
    """手动实现Welch ANOVA，兼容低版本scipy"""
    k = len(groups)
    if k < 2:
        return None, None
    means = [g.mean() for g in groups]
    vars = [g.var() for g in groups]
    ns = [len(g) for g in groups]
    total_mean = pd.concat(groups).mean()

    # 计算分子和分母
    numerator = sum(n * (m - total_mean) ** 2 for n, m in zip(ns, means)) / (k - 1)
    denominator = sum((1 - n / sum(ns)) * v / n for n, v in zip(ns, vars)) / (sum(ns) - k)

    if denominator == 0:
        return float('inf'), 0.0  # 组内方差为0时返回inf和p=0
    f_stat = numerator / denominator

    # Welch-Satterthwaite自由度
    temp = sum((v / n) ** 2 / (n - 1) for v, n in zip(vars, ns))
    df2 = (sum(v / n for v, n in zip(vars, ns)) ** 2) / temp if temp != 0 else sum(ns) - k
    df1 = k - 1
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)
    return f_stat, p_value


def one_way_anova_by_scenario(df, factor="系统", scenarios=["场景1", "场景2", "场景3", "场景4"],
                              metrics=["输入负担", "可用性", "接受度"]):
    all_results = []
    for scene in scenarios:
        print(f"\n==================== {scene} 分析结果 ====================")
        df_scene = df[df["场景"] == scene].copy()
        if len(df_scene) == 0:
            print(f"{scene} 无数据，跳过")
            continue

        sys_count = df_scene[factor].nunique()
        if sys_count < 2:
            print(f"{scene} 只有{sys_count}个系统数据，无法分析，跳过")
            continue

        for metric in metrics:
            print(f"\n--- 指标：{metric} ---")
            try:
                # 1. 数据过滤：每组至少2个样本，去除空值
                groups = []
                sys_names = []
                for sys in df_scene[factor].unique():
                    sys_data = df_scene[df_scene[factor] == sys][metric].dropna()
                    if len(sys_data) >= 2:
                        groups.append(sys_data)
                        sys_names.append(sys)
                        # 检查组内是否有波动
                        if sys_data.std() == 0:
                            print(f"   ⚠️ {sys}的{metric}得分完全相同（无波动）")

                if len(groups) < 2:
                    print(f"   ❌ 有效系统数不足（需≥2组），跳过")
                    all_results.append({"场景": scene, "指标": metric, "状态": "跳过", "原因": "有效系统数不足"})
                    continue

                # 2. 正态性检验
                normality_pass = True
                print("1. 正态性检验（Shapiro-Wilk）：")
                for sys, data in zip(sys_names, groups):
                    stat, p = stats.shapiro(data)
                    p = max(p, 1e-10)  # 避免p=0的极端情况
                    print(f"   {sys}: 统计量={stat:.3f}, p值={p:.3f} → {'符合' if p > 0.05 else '不符合'}")
                    if p <= 0.05:
                        normality_pass = False

                # 3. 方差齐性检验（优化nan处理）
                try:
                    stat_levene, p_levene = stats.levene(*groups)
                    if pd.isna(stat_levene) or pd.isna(p_levene) or any(g.std() == 0 for g in groups):
                        homogeneity_pass = False
                        print(f"2. 方差齐性检验：数据无波动 → 判定为不符合方差齐性")
                    else:
                        homogeneity_pass = p_levene > 0.05
                        print(
                            f"2. 方差齐性检验：统计量={stat_levene:.3f}, p值={p_levene:.3f} → {'符合' if homogeneity_pass else '不符合'}")
                except:
                    homogeneity_pass = False
                    print(f"2. 方差齐性检验：计算失败 → 判定为不符合方差齐性")

                # 4. 方差分析
                f_stat, p_value = welch_anova(groups) if not homogeneity_pass else stats.f_oneway(*groups)
                anova_type = "Welch ANOVA" if not homogeneity_pass else "标准ANOVA"

                if f_stat == float('inf'):
                    print(f"3. {anova_type} 结果：F=inf, p值={p_value:.3f} → 组内无波动，组间差异显著")
                else:
                    print(f"3. {anova_type} 结果：F={f_stat:.3f}, p值={p_value:.3f}")

                result_flag = p_value < 0.05 if not pd.isna(p_value) else False
                if result_flag:
                    print(f"   → 结论：不同系统的{metric}得分存在显著差异")

                    # 5. 事后检验（Tukey HSD）
                    print("4. 事后检验（Tukey HSD）：")
                    tukey_data = pd.DataFrame({
                        factor: [sys for sys, data in zip(sys_names, groups) for _ in data],
                        metric: [val for data in groups for val in data]
                    })
                    tukey = pairwise_tukeyhsd(endog=tukey_data[metric], groups=tukey_data[factor], alpha=0.05)
                    print(tukey)

                    # 6. 均值排名（修复核心bug：用zip配对，脱离索引i）
                    mean_scores = {sys: data.mean() for sys, data in zip(sys_names, groups)}
                    # 输入负担：升序（低→优）；其他指标：降序（高→优）
                    sort_asc = metric == "输入负担"
                    sorted_scores = sorted(mean_scores.items(), key=lambda x: x[1], reverse=not sort_asc)

                    print(f"5. {metric} 得分排名（均值）：")
                    for idx, (sys, mean) in enumerate(sorted_scores, 1):
                        rank_desc = "最优" if idx == 1 else "最差" if idx == len(sorted_scores) else "中等"
                        print(f"   {idx}. {sys}：{mean:.2f} → {rank_desc}")
                else:
                    print(f"   → 结论：不同系统的{metric}得分无显著差异")

                # 保存结果
                all_results.append({
                    "场景": scene, "指标": metric, "正态性符合": normality_pass,
                    "方差齐性符合": homogeneity_pass, "F值": f_stat, "p值": p_value,
                    "是否显著差异": result_flag, "状态": "成功"
                })
            except Exception as e:
                error_info = str(e)[:50]
                print(f"   ❌ 分析失败：{error_info}")
                all_results.append({
                    "场景": scene, "指标": metric, "状态": "失败", "原因": error_info
                })
    return pd.DataFrame(all_results)

def get_original_data():
    pass
# ===================== 主程序 =====================
if __name__ == "__main__":
    # 替换为你的真实数据！
    data = {
        "系统": ["系统A"] * 40 + ["系统B"] * 40 + ["系统C"] * 40,
        "场景": (["场景1"] * 10 + ["场景2"] * 10 + ["场景3"] * 10 + ["场景4"] * 10) * 3,
        "输入负担": [2.1] * 10 + [1.8] * 10 + [2.8] * 10 + [1.5] * 10 +
                    [3.2] * 10 + [2.5] * 10 + [3.5] * 10 + [2.2] * 10 +
                    [1.9] * 10 + [2.2] * 10 + [2.5] * 10 + [2.8] * 10,
        "可用性": [4.8] * 10 + [4.9] * 10 + [4.2] * 10 + [5.0] * 10 +
                  [4.1] * 10 + [4.0] * 10 + [3.8] * 10 + [4.2] * 10 +
                  [5.0] * 10 + [4.5] * 10 + [4.5] * 10 + [4.8] * 10,
        "接受度": [4.7] * 10 + [4.8] * 10 + [4.1] * 10 + [4.9] * 10 +
                  [4.0] * 10 + [3.9] * 10 + [3.7] * 10 + [4.1] * 10 +
                  [4.9] * 10 + [4.4] * 10 + [4.4] * 10 + [4.7] * 10
    }
    df = pd.DataFrame(data)

    # 执行分析
    results_df = one_way_anova_by_scenario(
        df=df,
        scenarios=["场景1", "场景2", "场景3", "场景4"],
        metrics=["输入负担", "可用性", "接受度"]
    )

    # 保存结果
    results_df.to_excel("各场景系统分析结果_修复版.xlsx", index=False)
    print("\n✅ 修复版分析完成！结果已保存")