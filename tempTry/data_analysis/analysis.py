import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings

warnings.filterwarnings('ignore')


# ===================== 1. 定义核心函数（修复版ANOVA） =====================
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
    按场景做单因素方差分析（适配你的数据格式）
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
                normality_pass = True
                print("1. 正态性检验（Shapiro-Wilk）：")
                for sys, data in zip(sys_names, groups):
                    stat, p = stats.shapiro(data)
                    p = max(p, 1e-10)  # 避免p=0的极端情况
                    print(f"   {sys}: 统计量={stat:.3f}, p值={p:.3f} → {'符合' if p > 0.05 else '不符合'}正态分布")
                    if p <= 0.05:
                        normality_pass = False

                # 3. 方差齐性检验（优化nan处理）
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

                # 4. 方差分析
                if not homogeneity_pass:
                    f_stat, p_value = welch_anova(groups)
                    anova_type = "Welch ANOVA"
                else:
                    f_stat, p_value = stats.f_oneway(*groups)
                    anova_type = "标准ANOVA"

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


# ===================== 2. 读取CSV并转换数据格式 =====================
def load_and_reshape_data(csv_path):
    """
    读取CSV文件，将宽格式转换为长格式
    :param csv_path: CSV文件路径（如"问卷数据.csv"）
    :return: 长格式DataFrame，列：场景、系统、输入负担、可用性、接受度
    """
    # 1. 读取CSV（跳过空行，处理索引）
    df = pd.read_csv(csv_path, index_col=0)

    # 2. 重置索引，提取场景（A/B/C/D，去掉数字）
    df = df.reset_index().rename(columns={"index": "场景编号"})
    df["场景"] = df["场景编号"].str.extract(r'([A-D])')[0]  # 提取A/B/C/D

    # 3. 拆分列名，转换为长格式
    # 定义列名映射
    columns_mapping = {}
    for col in df.columns:
        if ":" in col:
            metric, sys_full = col.split(":")
            # 提取系统名称并统一小写
            sys = sys_full.replace(" 系统", "").lower()
            # 统一系统名称（sasha/sage/sae）
            sys = sys.replace("sasha", "sasha").replace("sage", "sage").replace("sae", "sae")
            columns_mapping[col] = (metric, sys)

    # 4. 重塑数据
    reshaped_data = []
    for idx, row in df.iterrows():
        scene = row["场景"]
        # 遍历每个指标-系统组合
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

        # 为每个系统生成一行数据
        for sys in ["sasha", "sage", "sae"]:
            reshaped_data.append({
                "场景": scene,
                "系统": sys,
                "输入负担": input_burden[sys],
                "可用性": usability[sys],
                "接受度": acceptance[sys]
            })

    # 5. 转换为DataFrame并清理空值
    reshaped_df = pd.DataFrame(reshaped_data)
    reshaped_df = reshaped_df.dropna()  # 去除空值
    # 确保得分是数值类型
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
    # ---------------- 请修改这里的CSV文件路径 ----------------
    CSV_FILE_PATH = "data.csv"  # 替换为你的CSV文件实际路径（如：F:/data/问卷数据.csv）
    # ---------------------------------------------------------

    # 步骤1：读取并转换数据
    df = load_and_reshape_data(CSV_FILE_PATH)

    # 步骤2：执行单因素方差分析
    results_df = one_way_anova_by_scenario(
        df=df,
        scenarios=["A", "B", "C", "D"],  # 你的场景是A/B/C/D
        metrics=["输入负担", "可用性", "接受度"]
    )

    # 步骤3：保存结果到Excel
    results_df.to_excel("各场景系统方差分析结果.xlsx", index=False)
    print("\n🎉 所有分析完成！结果已保存到「各场景系统方差分析结果.xlsx」")