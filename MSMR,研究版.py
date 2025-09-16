# -*- coding: utf-8 -*-
"""
MSMR v5.4 (s_conf-only) — 最新实现
- 输入刻度：MS, MR ∈ [0,100]
- 中间量/概率刻度：[0,1]
- 仅“正向×强冲突”分流；其余自动软融合（打标不改分）
- 去除超椭圆门/软偏置η/负向抬升uplift；统一用 s_conf 分级与方向偏置

CLI:
  # 逐点计算（JSON 列表）
  python msmr_v54.py --pairs '[[96,35],[60,45],[20,15]]'
  # 生成 200x200 栅格并输出 CI 摘要
  python msmr_v54.py --grid 200 200 --ci
  # 覆盖默认参数
  python msmr_v54.py --params '{"sconf_mid":0.25,"dir_pow":1.0}'
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import argparse, json, math
import numpy as np

# -----------------------------
# 通用工具
# -----------------------------
def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

# -----------------------------
# 参数（研究调优版：s_mid=0.25、dir_pow=1.25 等）
# -----------------------------

# --- Research tuning note ---
# Defaults changed per user request:
# k_dir_neg=-0.85, neg_gain_strong=1.20, neg_gain_mid_max=0.60,
# dir_pow=1.25, alpha_min_base=0.07
# --------------------------------

@dataclass
class MSMRParams:
    # 冲突阈值
    sconf_mid: float = 0.20
    sconf_strong: float = 0.40
    eps_dir: float = 0.02        # 方向阈（Δ的死区）

    # 方向与增益（POS朝MR、NEG朝C）
    dir_pow: float = 1.25
    k_dir_pos: float = 0.35
    k_dir_neg: float = -0.85

    # POS 增益：弱/中(线性ramp)/强
    pos_gain_weak: float = 0.20
    pos_gain_mid_min: float = 0.40
    pos_gain_mid_max: float = 0.65   # ← 新
    pos_gain_strong: float = 1.20

    # NEG 增益：弱/中(线性ramp)/强
    neg_gain_weak: float = 0.20
    neg_gain_mid_min: float = 0.60
    neg_gain_mid_max: float = 0.60   # ← 新
    neg_gain_strong: float = 1.20

    # α 的稳定包络（no hard floor）
    alpha_raw: float = 0.90
    alpha_min_base: float = 0.07
    alpha_min_scale_u: float = 0.20

    # 稳态增益 G（一致轻抬、冲突轻抑）
    g_pos: float = 0.15
    g_conf: float = 0.15
    g_min: float = 0.75
    g_max: float = 1.25

# -----------------------------
# MS 四项计算（AS / CV_norm / CBS / CS）
# -----------------------------
def calculate_as(correct: int, total: int) -> float:
    if total <= 0 or correct < 0 or correct > total:
        raise ValueError("Invalid counts for AS.")
    return correct / float(total)

def _cv_of_sliding_accuracy(ans: List[int], window: int) -> float:
    m = len(ans) - window + 1
    r = [sum(ans[i:i+window]) / float(window) for i in range(m)]
    mu = float(np.mean(r))
    if mu == 0.0:
        return float("inf")
    sigma = float(np.std(r, ddof=1))   # 样本标准差
    return sigma / mu

def calculate_cv_norm(
    answers: List[int], window_size: int = 5,
    cv_max: Optional[float] = None,
    prior_answers: Optional[List[List[int]]] = None
) -> float:
    if not answers or window_size <= 0 or window_size > len(answers):
        raise ValueError("Empty answers or invalid window_size")
    cv = _cv_of_sliding_accuracy(answers, window_size)
    if cv_max is None and prior_answers:
        cvs = []
        for a in prior_answers:
            if isinstance(a, list) and len(a) >= window_size:
                cvs.append(_cv_of_sliding_accuracy(a, window_size))
        if cvs:
            cv_max = float(np.percentile(cvs, 95))
    if cv_max is None:
        cv_max = 0.8
    cvn = 1.0 - min(max(cv, 0.0), cv_max) / cv_max
    return float(np.clip(cvn, 0.0, 1.0))

def calculate_cbs(confidence: List[float], correctness: List[int]) -> float:
    if len(confidence) != len(correctness) or not confidence:
        raise ValueError("Length mismatch or empty inputs")
    if any((c < 1 or c > 5) for c in confidence):
        raise ValueError("Confidence must be in 1..5")
    x = [(c - 1.0) / 4.0 for c in confidence]        # 映射到[0,1]
    bias = [abs(xi - yi) for xi, yi in zip(x, correctness)]
    cbs = 1.0 - float(np.mean(bias))
    return float(np.clip(cbs, 0.0, 1.0))

def calculate_cs(confidence: List[float]) -> float:
    if not confidence:
        raise ValueError("Empty confidence")
    if any((c < 1 or c > 5) for c in confidence):
        raise ValueError("Confidence must be in 1..5")
    x = [(c - 1.0) / 4.0 for c in confidence]        # [0,1]
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=0))                   # 总体标准差
    cs = 0.5 * mean + 0.5 * (1.0 - std)
    return float(np.clip(cs, 0.0, 1.0))

def mastery_components(
    answers: List[int], confidence: List[float],
    window_size: int = 5, prior_answers: Optional[List[List[int]]] = None
) -> Dict[str, float]:
    total = len(answers)
    correct = sum(answers)
    AS  = calculate_as(correct, total)
    CVn = calculate_cv_norm(answers, window_size, cv_max=None, prior_answers=prior_answers)
    CBS = calculate_cbs(confidence, answers)
    CS  = calculate_cs(confidence)
    ms  = 0.4 * AS + 0.25 * CVn + 0.225 * CBS + 0.125 * CS
    return {
        "AS": round(AS, 4),
        "CV_norm": round(CVn, 4),
        "CBS": round(CBS, 4),
        "CS": round(CS, 4),
        "MS(0-100)": int(round(ms * 100))
    }

# -----------------------------
# MSMR 核心：冲突分级、α、G、p、路由
# -----------------------------
def piecewise_ramp(s_conf: float, p: MSMRParams) -> Tuple[float, str]:
    """返回 (r, level)；WEAK/MEDIUM 线性上升，STRONG=1"""
    if s_conf < p.sconf_mid:
        r = s_conf / p.sconf_mid if p.sconf_mid > 1e-9 else 0.0
        return r, "WEAK"
    elif s_conf < p.sconf_strong:
        denom = (p.sconf_strong - p.sconf_mid) if p.sconf_strong > p.sconf_mid else 1.0
        return (s_conf - p.sconf_mid) / denom, "MEDIUM"
    else:
        return 1.0, "STRONG"

def _dir_and_level(delta: float, s_conf: float, p: MSMRParams) -> Tuple[str, float, str]:
    if delta >= p.eps_dir:
        direction = "POS"
    elif delta <= -p.eps_dir:
        direction = "NEG"
    else:
        direction = "CONSISTENT"
    r, level = piecewise_ramp(s_conf, p)
    return direction, r, level

def predict_msmr(MS: float, MR: float, params: Optional[MSMRParams] = None) -> Dict[str, object]:
    """主推断：返回 p(0..1) 与中间量；仅 POS×STRONG 分流到 HUMAN_REVIEW。"""
    p = params or MSMRParams()

    # 归一化
    C = 100.0 - MS
    e_ms = C / 100.0
    e_mr = MR / 100.0
    u = 0.5 * (e_mr + e_ms)
    delta = e_mr - e_ms
    s_conf = abs(delta)

    # 方向与分级
    direction, r, level = _dir_and_level(delta, s_conf, p)

    # α_base（方向偏置）
    alpha_base = 0.5
    if direction in ("POS", "NEG"):
        # 方向系数与分级增益
        if direction == "POS":
            k = p.k_dir_pos
            if level == "WEAK":
                g = p.pos_gain_weak
            elif level == "MEDIUM":
                g = lerp(p.pos_gain_mid_min, p.pos_gain_mid_max, r)
            else:
                g = p.pos_gain_strong
        else:  # NEG
            k = p.k_dir_neg
            if level == "WEAK":
                g = p.neg_gain_weak
            elif level == "MEDIUM":
                g = lerp(p.neg_gain_mid_min, p.neg_gain_mid_max, r)
            else:
                g = p.neg_gain_strong

        h = s_conf ** p.dir_pow
        alpha_base = 0.5 + k * g * r * h

    alpha_base = clip01(alpha_base)

    # α 的稳定包络
    a_min = p.alpha_min_base * (1.0 + p.alpha_min_scale_u * u)
    alpha = a_min + (p.alpha_raw - a_min) * alpha_base
    alpha = clip01(alpha)

    # 线性基线与稳态增益
    base = alpha * MR + (1.0 - alpha) * C   # 百分刻度
    G = 1.0 + p.g_pos * (1.0 - s_conf) - p.g_conf * s_conf
    G = max(p.g_min, min(p.g_max, G))

    # 最终概率
    prob = clip01(G * base / 100.0)

    # 路由（仅 POS×STRONG 分流）
    if (direction == "POS") and (s_conf >= p.sconf_strong):
        policy = "HUMAN_REVIEW"
    else:
        policy = "AUTO_SOFT"

    return {
        "MS": round(MS, 2), "MR": round(MR, 2), "C": round(C, 2),
        "alpha": round(alpha, 4), "alpha_base": round(alpha_base, 4),
        "G": round(G, 4), "s_conf": round(s_conf, 4),
        "direction": direction, "level": level, "policy": policy,
        "p": round(prob, 6)
    }

# -----------------------------
# 栅格评估与简易 CI（幅度阈值为主）
# -----------------------------
def grid_eval(params: MSMRParams, ms_cells=200, mr_cells=200) -> Dict[str, object]:
    MS_vals = np.linspace(1, 99, ms_cells)
    MR_vals = np.linspace(1, 99, mr_cells)
    rows = []
    for MS in MS_vals:
        for MR in MR_vals:
            rows.append(predict_msmr(float(MS), float(MR), params))
    import pandas as pd
    df = pd.DataFrame(rows)

    # 单调性（幅度诊断）：行向不降、列向不升
    EPS = 1e-9
    row_viol, row_max_drop = 0, 0.0
    for ms, g in df.groupby("MS"):
        g = g.sort_values("MR")
        p = g["p"].to_numpy()
        drops = np.maximum(p[:-1] - p[1:], 0.0)
        if np.any(drops > EPS):
            row_viol += 1
            row_max_drop = max(row_max_drop, float(np.max(drops)))

    col_viol, col_max_rise = 0, 0.0
    for mr, g in df.groupby("MR"):
        g = g.sort_values("MS")
        p = g["p"].to_numpy()
        rises = np.maximum(p[1:] - p[:-1], 0.0)
        if np.any(rises > EPS):
            col_viol += 1
            col_max_rise = max(col_max_rise, float(np.max(rises)))

    summary = {
        "n_points": int(df.shape[0]),
        "mean_p": float(df["p"].mean()),
        "median_p": float(df["p"].median()),
        "row_nonmono_rate": row_viol / len(df["MS"].unique()),
        "row_max_drop": row_max_drop,
        "col_nonmono_rate": col_viol / len(df["MR"].unique()),
        "col_max_rise": col_max_rise,
        "policy_counts": df["policy"].value_counts().to_dict(),
        "direction_counts": df["direction"].value_counts().to_dict(),
        "level_counts": df["level"].value_counts().to_dict(),
    }
    return {"summary": summary, "frame": df}

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="MSMR v5.4 (s_conf-only) predictor")
    ap.add_argument("--pairs", type=str, default="",
                    help="JSON 列表 [[MS,MR],...]；留空则不跑逐点")
    ap.add_argument("--grid", nargs=2, type=int, metavar=("MS_CELLS","MR_CELLS"),
                    help="生成栅格并输出 CI 摘要")
    ap.add_argument("--ci", action="store_true", help="与 --grid 配合，输出 CI 摘要 JSON")
    ap.add_argument("--params", type=str, default="", help="JSON 覆盖参数")
    args = ap.parse_args()

    params = MSMRParams()
    if args.params:
        try:
            cfg = json.loads(args.params)
            for k, v in cfg.items():
                if hasattr(params, k):
                    setattr(params, k, v)
        except Exception as e:
            print(f"[WARN] 参数解析失败：{e}")

    out = {}

    if args.pairs:
        try:
            pairs = json.loads(args.pairs)
            out["pairs"] = [predict_msmr(float(ms), float(mr), params) for ms, mr in pairs]
        except Exception as e:
            out["pairs_error"] = f"pairs 解析失败：{e}"

    if args.grid:
        ms_cells, mr_cells = args.grid
        res = grid_eval(params, ms_cells, mr_cells)
        if args.ci:
            out["ci_summary"] = res["summary"]

    if not out:
        # demo
        demo = [(96,35), (96,45), (60,45), (20,15), (20,45)]
        out["demo"] = [predict_msmr(ms, mr, params) for ms, mr in demo]

    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
