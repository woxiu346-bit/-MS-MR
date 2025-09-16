
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSMR (部署版 · 支持 mid_max + policy)
------------------------------------
- 新增：在 MEDIUM 段支持 mid_max（分别对 POS/NEG），驱动函数：
    drive(t) = { (t/s_mid)*weak,                        t < s_mid
               { lerp(mid_min, mid_max),               s_mid ≤ t < s_strong
               { strong,                                t ≥ s_strong
- 新增：保留研究版分流策略，在结果中增加 policy 字段：
    仅当 POS×STRONG → policy = "HUMAN_REVIEW"，其余 "AUTO_SOFT"（不改分，仅打标）。
- 保留：原部署版的“仅 NEG×STRONG 时使用 C_alt = 100 - (MS - ms_shift)”（可通过 --apply 调整为 always）。
- 其它：输出包含 MS、MS-30、MR(一位小数)、P(六位小数)、冲突标签与 policy。

用法
----
python MSMR_deploy_midmax_policy.py \
  --input data.csv \
  --out-csv out.csv --out-xlsx out.xlsx \
  --ms-shift 25 --apply neg_strong --show-head 10

输入文件需包含列：MS, MR（百分刻度 0–100）。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd


# ------------------------ 参数（含 mid_max 支持） ------------------------
@dataclass
class MSMRParams:
    # 冲突阈值
    eps_dir: float = 0.02
    sconf_mid: float = 0.20
    sconf_strong: float = 0.40

    # 方向偏置与非线性
    dir_pow: float = 1.25
    k_dir_pos: float = 0.35
    k_dir_neg: float = -0.85

    # POS 增益（弱→中段下限→中段上限→强）
    pos_gain_weak: float = 0.20
    pos_gain_mid_min: float = 0.40
    pos_gain_mid_max: float = 0.65
    pos_gain_strong: float = 1.20

    # NEG 增益
    neg_gain_weak: float = 0.20
    neg_gain_mid_min: float = 0.60
    neg_gain_mid_max: float = 0.60
    neg_gain_strong: float = 1.20

    # α 的稳定包络
    alpha_raw: float = 0.90
    alpha_min_base: float = 0.07
    alpha_min_scale_u: float = 0.20

    # 稳态增益 G（一致轻抬、冲突轻抑）
    g_pos: float = 0.15
    g_conf: float = 0.15
    g_min: float = 0.75
    g_max: float = 1.25


# ------------------------ 工具函数 ------------------------
def clip01(x: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, np.maximum(0.0, x))


def lerp(a: float, b: float, w: np.ndarray) -> np.ndarray:
    return a + (b - a) * w


def drive_piecewise_with_midmax(
    t: np.ndarray, weak: float, mid_min: float, mid_max: float, strong: float,
    s_mid: float, s_strong: float
) -> np.ndarray:
    """
    分段驱动：弱段线性到 weak；中段在 [mid_min, mid_max] 线性插值；强段常数 strong。
    """
    # 预分配
    d = np.empty_like(t, dtype=float)

    # 弱段：0 → s_mid
    w1 = np.clip(t / s_mid, 0.0, 1.0)
    d_weak = weak * w1

    # 中段：s_mid → s_strong （按 mid_min→mid_max 线性）
    span = max(s_strong - s_mid, 1e-9)
    w2 = np.clip((t - s_mid) / span, 0.0, 1.0)
    d_mid = lerp(mid_min, mid_max, w2)

    # 组装
    d = np.where(t < s_mid, d_weak, np.where(t < s_strong, d_mid, strong))
    return d


def make_conflict_labels(delta: np.ndarray, t: np.ndarray,
                         eps_dir: float, s_mid: float, s_strong: float
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 (dir_code, level_code, labels)
    dir_code: -1 NEG, 0 CONSISTENT, 1 POS
    level_code: 0 WEAK, 1 MEDIUM, 2 STRONG
    """
    dir_code = np.zeros_like(delta, dtype=np.int8)
    dir_code[delta >= eps_dir] = 1
    dir_code[delta <= -eps_dir] = -1

    level_code = np.zeros_like(t, dtype=np.int8)
    level_code[(t >= s_mid) & (t < s_strong)] = 1
    level_code[t >= s_strong] = 2

    dir_map = { -1: "NEG", 0: "CONSISTENT", 1: "POS" }
    lvl_map = { 0: "WEAK", 1: "MEDIUM", 2: "STRONG" }
    labels = np.array([f"{dir_map[d]}×{lvl_map[l]}" for d, l in zip(dir_code, level_code)], dtype=object)
    return dir_code, level_code, labels


# ------------------------ 核心计算 ------------------------
def compute_msmr(
    df: pd.DataFrame,
    params: MSMRParams,
    ms_shift: float = 25.0,
    apply_mode: str = "neg_strong"
) -> pd.DataFrame:
    """
    输入：DataFrame，含 MS, MR（百分刻度）。
    返回：加入 MS-30, 冲突标签, policy 与 P。
    """
    MS = df["MS"].astype(float).to_numpy()
    MR = df["MR"].astype(float).to_numpy()

    # 可视化/报表需要的列：MS-30、MR 保留一位
    MS_m30 = MS - 30.0
    MR_1d = np.round(MR, 1)

    # 人/机侧证据刻度 [0,1]
    C = 100.0 - MS
    e_ms = C / 100.0
    e_mr = MR / 100.0

    # 基础合成量
    u = 0.5 * (e_mr + e_ms)            # 均值作锚点（用于 G 缩放）
    delta = e_mr - e_ms
    t = np.abs(delta)

    # 标签
    dir_code, level_code, labels = make_conflict_labels(
        delta, t, params.eps_dir, params.sconf_mid, params.sconf_strong
    )

    # 研究版分流策略（仅 POS×STRONG → 人审）
    policy = np.where((dir_code == 1) & (level_code == 2), "HUMAN_REVIEW", "AUTO_SOFT")

    # 方向驱动（含 mid_max）
    h = np.power(t, params.dir_pow)     # 冲突非线性
    drive_pos = drive_piecewise_with_midmax(
        t, params.pos_gain_weak, params.pos_gain_mid_min, params.pos_gain_mid_max,
        params.pos_gain_strong, params.sconf_mid, params.sconf_strong
    )
    drive_neg = drive_piecewise_with_midmax(
        t, params.neg_gain_weak, params.neg_gain_mid_min, params.neg_gain_mid_max,
        params.neg_gain_strong, params.sconf_mid, params.sconf_strong
    )

    alpha_base = np.full_like(t, 0.5, dtype=float)
    mask_pos = (dir_code == 1)
    mask_neg = (dir_code == -1)
    alpha_base[mask_pos] = 0.5 + params.k_dir_pos * drive_pos[mask_pos] * h[mask_pos]
    alpha_base[mask_neg] = 0.5 + params.k_dir_neg * drive_neg[mask_neg] * h[mask_neg]
    alpha_base = clip01(alpha_base)

    # α 稳定包络
    a_min = params.alpha_min_base * (1.0 + params.alpha_min_scale_u * u)
    alpha = a_min + (params.alpha_raw - a_min) * alpha_base
    alpha = clip01(alpha)

    # 人侧替代项（部署策略保留）
    if apply_mode not in ("neg_strong", "always"):
        raise ValueError("--apply 仅支持 'neg_strong' 或 'always'")
    use_alt = (dir_code == -1) & (level_code == 2) if apply_mode == "neg_strong" else np.ones_like(dir_code, dtype=bool)
    C_alt = 100.0 - (MS - ms_shift)
    e_ms_used = np.where(use_alt, C_alt / 100.0, e_ms)

    # 融合（基础）
    base = alpha * e_mr + (1.0 - alpha) * e_ms_used

    # 稳态增益：一致轻抬、冲突轻抑（以 u 为锚点缩放）
    g = np.where(dir_code == 0, 1.0 + params.g_pos, 1.0 - params.g_conf)
    g = np.clip(g, params.g_min, params.g_max)
    P = u + g * (base - u)
    P = clip01(P)

    # 组装结果
    out = pd.DataFrame({
        "MS": MS,
        "MS-30": MS_m30,
        "MR": MR_1d,
        "P": np.round(P, 6),
        "delta": np.round(delta, 6),
        "direction": np.where(dir_code == 1, "POS", np.where(dir_code == -1, "NEG", "CONSISTENT")),
        "level": np.where(level_code == 2, "STRONG", np.where(level_code == 1, "MEDIUM", "WEAK")),
        "label": labels,
        "policy": policy,
    })

    return out


# ------------------------ CLI ------------------------
def read_table(path: Optional[str]) -> pd.DataFrame:
    if path is None:
        # 内置样例
        demo = pd.DataFrame({"MS": [19,34,27,33,26,24,34,29], "MR":[10.4,3.2,6.6,3.7,7.1,8.0,3.2,5.6]})
        return demo
    # 简单按扩展名判断
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description="MSMR 部署版（支持 mid_max 与 policy）。")
    parser.add_argument("--input", type=str, default=None, help="输入 CSV/XLSX，需含列 MS, MR。")
    parser.add_argument("--out-csv", type=str, default="msmr_out.csv", help="输出 CSV 路径。")
    parser.add_argument("--out-xlsx", type=str, default="msmr_out.xlsx", help="输出 XLSX 路径。")
    parser.add_argument("--ms-shift", type=float, default=25.0, help="C_alt = 100 - (MS - ms_shift) 的 shift。")
    parser.add_argument("--apply", type=str, default="neg_strong", choices=["neg_strong","always"],
                        help="应用 C_alt 的场景：仅 NEG×STRONG 或 全局。")
    parser.add_argument("--show-head", type=int, default=10, help="打印前 N 行结果。")

    args = parser.parse_args()

    params = MSMRParams()  # 使用默认参数（含 mid_max）
    df_in = read_table(args.input)

    out = compute_msmr(df_in, params, ms_shift=args.ms_shift, apply_mode=args.apply)

    # 保存
    with pd.ExcelWriter(args.out_xlsx, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="MSMR")
    out.to_csv(args.out_csv, index=False, float_format="%.6f")

    # 预览
    if args.show_head and args.show_head > 0:
        print(out.head(args.show_head).to_string(index=False))


if __name__ == "__main__":
    main()
