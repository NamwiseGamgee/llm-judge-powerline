import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from scipy.stats import spearmanr, ttest_rel, wilcoxon
import os

# -----------------------------------------------------
# NORMALIZATION
# -----------------------------------------------------
def to_base_key(name: str) -> str:
    n = os.path.basename(name.strip())
    # generic cleanup of corruption tags
    n = re.sub(r'_[A-Za-z]+_s[0-9]+_', '_', n)
    n = re.sub(r'__+', '_', n)
    return n

def parse_tag(filename: str) -> Tuple[str, int]:
    stem = Path(filename).stem
    if stem == "clean":
        return "clean", 0
    m = re.match(r"^(.*)_s(\d+)$", stem)
    if not m:
        raise ValueError(f"Cannot parse corruption/severity from filename: {filename}")
    return m.group(1), int(m.group(2))

def ci95_bootstrap(x: np.ndarray, n_boot: int = 2000, seed: int = 12345):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(np.mean(x[idx]))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)

def cohens_dz(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    if diff.size < 2:
        return np.nan
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return np.nan
    return float(np.mean(diff) / sd)

# -----------------------------------------------------
# CORE ANALYSIS
# -----------------------------------------------------
def analyze_condition(clean_df: pd.DataFrame, cond_df: pd.DataFrame) -> Dict:
    clean_df = clean_df.copy()
    cond_df = cond_df.copy()
    clean_df["key"] = clean_df["image"].apply(to_base_key)
    cond_df["key"] = cond_df["image"].apply(to_base_key)

    a = clean_df.set_index("key")[["score", "confidence"]].rename(
        columns={"score": "score_clean", "confidence": "conf_clean"}
    )
    b = cond_df.set_index("key")[["score", "confidence"]]

    merged = a.join(b, how="inner").dropna()
    merged.rename(columns={"score": "score_cond", "confidence": "conf_cond"}, inplace=True)

    merged["delta_s"] = merged["score_clean"] - merged["score_cond"]
    merged["delta_c"] = merged["conf_clean"] - merged["conf_cond"]

    ds = merged["delta_s"].to_numpy()
    dc = merged["delta_c"].to_numpy()

    mean_ds = float(np.mean(ds)) if len(ds) else np.nan
    std_ds  = float(np.std(ds, ddof=1)) if len(ds) > 1 else np.nan
    mean_dc = float(np.mean(dc)) if len(dc) else np.nan
    std_dc  = float(np.std(dc, ddof=1)) if len(dc) > 1 else np.nan

    ci_ds_lo, ci_ds_hi = ci95_bootstrap(ds) if len(ds) > 1 else (np.nan, np.nan)
    ci_dc_lo, ci_dc_hi = ci95_bootstrap(dc) if len(dc) > 1 else (np.nan, np.nan)

    if len(ds) >= 2:
        _, p_s = ttest_rel(merged["score_clean"], merged["score_cond"])
        dz_s = cohens_dz(merged["score_clean"] - merged["score_cond"])
        try:
            _, p_s_w = wilcoxon(merged["score_clean"], merged["score_cond"])
        except ValueError:
            p_s_w = np.nan
    else:
        p_s = p_s_w = dz_s = np.nan

    if len(dc) >= 2:
        _, p_c = ttest_rel(merged["conf_clean"], merged["conf_cond"])
        dz_c = cohens_dz(merged["conf_clean"] - merged["conf_cond"])
        try:
            _, p_c_w = wilcoxon(merged["conf_clean"], merged["conf_cond"])
        except ValueError:
            p_c_w = np.nan
    else:
        p_c = p_c_w = dz_c = np.nan

    return {
        "N": int(len(merged)),
        "mean_ds": mean_ds,
        "std_ds": std_ds,
        "ci95_ds_lo": ci_ds_lo,
        "ci95_ds_hi": ci_ds_hi,
        "mean_dc": mean_dc,
        "std_dc": std_dc,
        "ci95_dc_lo": ci_dc_lo,
        "ci95_dc_hi": ci_dc_hi,
        "dz_score": float(dz_s) if dz_s == dz_s else np.nan,
        "dz_conf": float(dz_c) if dz_c == dz_c else np.nan,
        "merged": merged.reset_index(),
    }

# -----------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------
def main(results_dir: str, out_dir: str):
    res_dir = Path(results_dir)
    out = Path(out_dir)
    figs = out / "figs"
    tables = out / "tables"
    per_image_dir = out / "per_image_deltas"

    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    per_image_dir.mkdir(parents=True, exist_ok=True)

    clean_csv = res_dir / "clean.csv"
    clean = pd.read_csv(clean_csv)
    clean["key"] = clean["image"].apply(to_base_key)

    cond_files = sorted([p for p in res_dir.glob("*.csv") if p.stem != "clean"])

    rows = []
    by_corr: Dict[str, List[Tuple[int, Dict]]] = {}

    for f in cond_files:
        corr, sev = parse_tag(f.name)
        df = pd.read_csv(f)
        stats = analyze_condition(clean, df)

        stats["merged"].to_csv(per_image_dir / f"{corr}_s{sev}.csv", index=False)

        row = {
            "corruption": corr,
            "severity": int(sev),
            "N": stats["N"],
            "mean_ds": stats["mean_ds"],
            "std_ds": stats["std_ds"],
            "ci95_ds_lo": stats["ci95_ds_lo"],
            "ci95_ds_hi": stats["ci95_ds_hi"],
            "mean_dc": stats["mean_dc"],
            "std_dc": stats["std_dc"],
            "ci95_dc_lo": stats["ci95_dc_lo"],
            "ci95_dc_hi": stats["ci95_dc_hi"],
            "dz_score": stats["dz_score"],
            "dz_conf": stats["dz_conf"],
        }
        rows.append(row)
        by_corr.setdefault(corr, []).append((sev, row))

    summ = pd.DataFrame(rows).sort_values(["corruption", "severity"])
    summ.to_csv(out / "per_condition_summary.csv", index=False)
    plt.figure(figsize=(10, 6))

    # Fixed colors per corruption
    color_map = {
        "fog": "tab:blue",
        "rain": "tab:orange",
        "shadow": "tab:green",
        "snow": "tab:red",
        "sunflare": "tab:purple",
    }

    for corr in summ["corruption"].unique():
        sub = summ[summ["corruption"] == corr].sort_values("severity")
        sev = sub["severity"]
        dz = sub["dz_score"]
        plt.scatter(sev, dz, color=color_map[corr], label=corr)
        for s, d in zip(sev, dz):
            plt.text(s, d, corr, fontsize=8)

    plt.xlabel("Severity level")
    plt.ylabel("Cohen's dz (score)")
    plt.title("Effect size dz_score vs severity (per corruption)")
    plt.grid(True, alpha=0.3)

    # Force only integer ticks
    plt.xticks([1, 2, 3])

    plt.tight_layout()
    plt.savefig(figs / "dz_score_vs_severity.png", dpi=300)
    plt.close()

    print("[OK] Updated dz_score scatter plot generated with integer x-axis.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()
    main(args.results_dir, args.out_dir)
