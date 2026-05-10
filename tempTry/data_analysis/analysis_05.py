import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
import numpy as np
import scikit_posthocs as sp  # 非参数事后检验库

warnings.filterwarnings('ignore')


# ===================== 1. 定义核心函数（修复Dunn检验参数） =====================
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


def one_way_anova_by_scenario(df, factor="系统", scenarios=["A", "B", "C", "D"],
                              metrics=["输入负担", "可用性", "接受度"]):
    """
    按场景做单因素方差分析（修复Dunn检验参数，适配scikit-posthocs最新版本）
    """
    all_results = []
    for scene in scenarios:
        print(f"\n==================== 场景{scene} 分析结果 ====================")
        df_scene = df[df["场景"] == scene].copy()
        if len(df_scene) == 0:
            print(f"场景{scene} 无数据，跳过")
            continue

        sys_count = df_scene[factor].nunique()
        if sys_count < 2:
            print(f"场景{scene} 只有{sys_count}个系统数据，无法分析，跳过")
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
                normality_results = {}
                normality_pass = True
                print("1. 正态性检验（Shapiro-Wilk）：")
                for sys, data in zip(sys_names, groups):
                    stat, p = stats.shapiro(data)
                    p = max(p, 1e-10)  # 避免p=0的极端情况
                    normality_results[sys] = {"stat": stat, "p": p, "pass": p > 0.05}
                    print(f"   {sys}: 统计量={stat:.3f}, p值={p:.3f} → {'符合' if p > 0.05 else '不符合'}正态分布")
                    if p <= 0.05:
                        normality_pass = False

                # 3. 方差齐性检验
                homogeneity_pass = True
                try:
                    stat_levene, p_levene = stats.levene(*groups)
                    if pd.isna(stat_levene) or pd.isna(p_levene) or any(g.std() == 0 for g in groups):
                        homogeneity_pass = False
                        print(f"2. 方差齐性检验（Levene）：数据无波动 → 判定为不符合方差齐性")
                    else:
                        homogeneity_pass = p_levene > 0.05
                        print(
                            f"2. 方差齐性检验（Levene）：统计量={stat_levene:.3f}, p值={p_levene:.3f} → {'符合' if homogeneity_pass else '不符合'}方差齐性")
                except:
                    homogeneity_pass = False
                    print(f"2. 方差齐性检验（Levene）：计算失败 → 判定为不符合方差齐性")

                # 4. 核心检验：参数/非参数自适应
                test_type = ""
                stat_value = None
                p_value = None
                result_flag = False

                if normality_pass:  # 全正态 → 参数检验
                    if not homogeneity_pass:
                        stat_value, p_value = welch_anova(groups)
                        test_type = "Welch ANOVA"
                    else:
                        stat_value, p_value = stats.f_oneway(*groups)
                        test_type = "标准ANOVA"
                    if stat_value == float('inf'):
                        print(f"3. {test_type} 结果：F=inf, p值={p_value:.3f} → 组内无波动，组间差异显著")
                    else:
                        print(f"3. {test_type} 结果：F={stat_value:.3f}, p值={p_value:.3f}")
                else:  # 非全正态 → 非参数检验
                    stat_value, p_value = stats.kruskal(*groups)
                    test_type = "Kruskal-Wallis 非参数检验"
                    print(f"3. {test_type} 结果：H统计量={stat_value:.3f}, p值={p_value:.3f}")

                # 判断显著差异
                result_flag = p_value < 0.05 if not pd.isna(p_value) else False
                if result_flag:
                    print(f"   → 结论：不同系统的{metric}得分存在显著差异")

                    # 5. 事后检验：修复Dunn检验参数
                    print(f"4. 事后检验（{'Tukey HSD' if normality_pass else 'Dunn检验'}）：")
                    if normality_pass:
                        # 参数检验：Tukey HSD
                        tukey_data = pd.DataFrame({
                            factor: [sys for sys, data in zip(sys_names, groups) for _ in data],
                            metric: [val for data in groups for val in data]
                        })
                        tukey = pairwise_tukeyhsd(endog=tukey_data[metric], groups=tukey_data[factor], alpha=0.05)
                        print(tukey)
                    else:
                        # 非参数检验：Dunn检验（适配最新版scikit-posthocs）
                        try:
                            # 转换为数组格式（兼容所有版本）
                            data_array = np.concatenate([np.array(g) for g in groups])
                            group_array = np.concatenate([[sys] * len(g) for sys, g in zip(sys_names, groups)])

                            # 调用Dunn检验（正确参数格式）
                            dunn_result = sp.posthoc_dunn(
                                a=data_array,
                                g=group_array,
                                p_adjust='bonferroni'
                            )

                            # 格式化输出结果
                            print("   两两比较结果（p值<0.05表示差异显著）：")
                            dunn_df = pd.DataFrame(dunn_result, index=sys_names, columns=sys_names)
                            for i, g1 in enumerate(sys_names):
                                for j, g2 in enumerate(sys_names):
                                    if i < j:  # 避免重复比较
                                        p_adj = dunn_df.loc[g1, g2]
                                        reject = "是" if p_adj < 0.05 else "否"
                                        print(f"   {g1} vs {g2}：校正p值={p_adj:.4f} → 差异显著：{reject}")
                        except Exception as e:
                            # 兜底方案：若仍报错，输出手动计算的均值差参考
                            print(f"   ⚠️ Dunn检验执行失败（{str(e)[:30]}），提供均值差参考：")
                            mean_scores = {sys: data.mean() for sys, data in zip(sys_names, groups)}
                            for i, g1 in enumerate(sys_names):
                                for j, g2 in enumerate(sys_names):
                                    if i < j:
                                        mean_diff = mean_scores[g2] - mean_scores[g1]
                                        print(f"   {g1} vs {g2}：均值差={mean_diff:.2f}（{g2} - {g1}）")

                    # 6. 均值排名
                    mean_scores = {sys: data.mean() for sys, data in zip(sys_names, groups)}
                    sorted_scores = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
                    print(f"5. {metric} 得分排名（均值，越高越好）：")
                    for idx, (sys, mean) in enumerate(sorted_scores, 1):
                        rank_desc = "最优" if idx == 1 else "最差" if idx == len(sorted_scores) else "中等"
                        print(f"   {idx}. {sys}：{mean:.2f} → {rank_desc}")
                else:
                    print(f"   → 结论：不同系统的{metric}得分无显著差异")
                    # 无差异也输出排名
                    mean_scores = {sys: data.mean() for sys, data in zip(sys_names, groups)}
                    sorted_scores = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
                    print(f"5. {metric} 得分排名（均值，越高越好）：")
                    for idx, (sys, mean) in enumerate(sorted_scores, 1):
                        rank_desc = "最优" if idx == 1 else "最差" if idx == len(sorted_scores) else "中等"
                        print(f"   {idx}. {sys}：{mean:.2f} → {rank_desc}")

                # 保存结果
                all_results.append({
                    "场景": scene, "指标": metric,
                    "正态性全满足": normality_pass, "方差齐性符合": homogeneity_pass,
                    "检验类型": test_type, "统计量": stat_value, "p值": p_value,
                    "是否显著差异": result_flag, "状态": "成功"
                })
            except Exception as e:
                error_info = str(e)[:50]
                print(f"   ❌ 分析失败：{error_info}")
                all_results.append({
                    "场景": scene, "指标": metric, "状态": "失败", "原因": error_info
                })
    return pd.DataFrame(all_results)


# ===================== 2. 读取CSV并转换数据格式 =====================
def load_and_reshape_data(csv_path):
    """读取CSV文件，将宽格式转换为长格式"""
    df = pd.read_csv(csv_path, index_col=0)
    df = df.reset_index().rename(columns={"index": "场景编号"})
    df["场景"] = df["场景编号"].str.extract(r'([A-D])')[0]

    columns_mapping = {}
    for col in df.columns:
        if ":" in col:
            metric, sys_full = col.split(":")
            sys = sys_full.replace(" 系统", "").lower()
            sys = sys.replace("sasha", "sasha").replace("sage", "sage").replace("sae", "sae")
            columns_mapping[col] = (metric, sys)

    reshaped_data = []
    for idx, row in df.iterrows():
        scene = row["场景"]
        input_burden = {"sasha": None, "sage": None, "sae": None}
        usability = {"sasha": None, "sage": None, "sae": None}
        acceptance = {"sasha": None, "sage": None, "sae": None}

        for col, (metric, sys) in columns_mapping.items():
            value = row[col]
            if metric == "输入负担":
                input_burden[sys] = value
            elif metric == "可用性":
                usability[sys] = value
            elif metric == "接受度":
                acceptance[sys] = value

        for sys in ["sasha", "sage", "sae"]:
            reshaped_data.append({
                "场景": scene,
                "系统": sys,
                "输入负担": input_burden[sys],
                "可用性": usability[sys],
                "接受度": acceptance[sys]
            })

    reshaped_df = pd.DataFrame(reshaped_data).dropna()
    reshaped_df["输入负担"] = pd.to_numeric(reshaped_df["输入负担"])
    reshaped_df["可用性"] = pd.to_numeric(reshaped_df["可用性"])
    reshaped_df["接受度"] = pd.to_numeric(reshaped_df["接受度"])

    print("✅ 数据读取并转换完成！")
    print(f"数据维度：{reshaped_df.shape}")
    print("数据预览：")
    print(reshaped_df.head())

    return reshaped_df


# ===================== 3. 主程序执行 =====================
if __name__ == "__main__":
    CSV_FILE_PATH = "data.csv"  # 替换为你的CSV文件路径
    df = load_and_reshape_data(CSV_FILE_PATH)
    results_df = one_way_anova_by_scenario(df=df, scenarios=["A", "B", "C", "D"],
                                           metrics=["输入负担", "可用性", "接受度"])
    results_df.to_excel("各场景系统方差分析结果_最终版.xlsx", index=False)
    print("\n🎉 所有分析完成！结果已保存到「各场景系统方差分析结果_最终版.xlsx」")