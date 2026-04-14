import os, io, json, base64, time, argparse
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import re, json

PROMPT_SYSTEM = (
    "You are a safety judge for powerline segmentation overlays. "
    "You must ALWAYS return a valid JSON object, even if the image is unclear or corrupted. "
    "No prose, no markdown — only JSON. "
    "Format: {\"score\": int in [1,5], \"confidence\": float in [0,1], \"explanation\": string <= 140 chars}. "
    "Scoring rubric: "
    "5 = perfect continuity, no spurious detections; "
    "4 = minor defects; "
    "3 = noticeable gaps or a few spurious lines; "
    "2 = significant errors; "
    "1 = unusable or image too unclear to evaluate. "
    "If you cannot confidently judge due to blur, fog, snow, or corruption, "
    "still return JSON with score=1, confidence=0.1, and explanation='Image too unclear or corrupted to evaluate.'"
)

PROMPT_USER = (
    "Evaluate this RGB image with a semi-transparent red mask overlay for powerline segmentation quality. "
    "Return ONLY valid JSON with the exact fields: score, confidence, explanation. "
    "If you cannot see or interpret the mask properly, still respond with: "
    "{\"score\": 1, \"confidence\": 0.1, \"explanation\": \"Image too unclear or corrupted to evaluate.\"}"
)

JSON_SCHEMA_HINT = (
    'Return: {"score": <1-5 integer>, "confidence": <0-1 float>, "explanation": "<<=140 chars>"}'
)


def b64_image(img_path: Path) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def judge_with_openai(image_path: Path, model: str = "gpt-4o") -> dict:
    import openai
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    img_b64 = b64_image(image_path)
    messages = [
        {"role": "system", "content": PROMPT_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT_USER + "\n" + JSON_SCHEMA_HINT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_path.suffix.lstrip('.').lower()};base64,{img_b64}"
                    },
                },
            ],
        },
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=128
    )
    text = resp.choices[0].message.content.strip()
    return parse_llm_json(text)


def judge_with_gemini(image_path: Path, model: str = "gemini-1.5-pro") -> dict:
    """
    Requires: export GOOGLE_API_KEY
    Uses google-generativeai. Adjust if your SDK differs.
    """
    import google.generativeai as genai  # pip install google-generativeai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    from google.generativeai.types import HarmCategory, HarmBlockThreshold

    img = Image.open(image_path).convert("RGB")
    model_ = genai.GenerativeModel(model)
    prompt = PROMPT_SYSTEM + "\n" + PROMPT_USER + "\n" + JSON_SCHEMA_HINT
    resp = model_.generate_content(
        [prompt, img],
        generation_config={"temperature": 0},
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_VIOLENCE: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUAL: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    text = resp.text.strip()
    return parse_llm_json(text)

def judge_with_mock(image_path: Path, seed: int = 7) -> dict:
    """
    Deterministic pseudo-judge for pipeline testing.
    Derives score from filename to be repeatable.
    """
    rng = np.random.default_rng(abs(hash(image_path.stem)) % (2**32))
    # Mild deterministic variation by corruption keywords
    name = image_path.name.lower()
    base = 4.5
    if "break" in name:   base -= 1.6
    if "spu" in name:     base -= 1.2
    if "erode" in name:   base -= 0.8
    if "dilate" in name:  base -= 0.6
    if "fog" in name:     base -= 1.0
    if "rain" in name:    base -= 0.9
    if "snow" in name:    base -= 0.7
    if "sunflare" in name:base -= 1.1
    if "shadow" in name:  base -= 0.5
    if "motion_blur" in name: base -= 0.8
    if "defocus_blur" in name:base -= 0.7
    if "gauss_noise" in name: base -= 0.6
    if "jpeg" in name:    base -= 0.5
    if "brightness" in name: base -= 0.4
    # severity hint
    for s in [3,2,1]:
        if f"_s{s}" in name: base -= 0.2*(s-1)
    score = int(np.clip(round(base), 1, 5))
    conf  = float(np.clip(0.6 + 0.08*(score-3), 0.05, 0.99))
    expl  = f"Score {score}: {'good continuity' if score>=4 else 'noticeable defects'}."
    return {"score": score, "confidence": conf, "explanation": expl}

def parse_llm_json(text: str) -> dict:
    """Robust JSON parser that extracts the first {...} block from the response."""
    if not text or not text.strip():
        raise ValueError("Empty response from model")

    # Remove code fences if present
    text = text.strip()
    for fence in ["```json", "```"]:
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]

    # Extract the first {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON object found in: {text[:200]!r}")
    block = match.group(0)

    # Try parse
    obj = json.loads(block)
    s = int(obj.get("score", 0))
    c = float(obj.get("confidence", 0))
    e = str(obj.get("explanation", ""))[:140]

    if not (1 <= s <= 5):
        raise ValueError("score out of range")
    if not (0.0 <= c <= 1.0):
        raise ValueError("confidence out of range")
    return {"score": s, "confidence": c, "explanation": e}

# =====================
# Judging pipeline
# =====================

def judge_group(images, backend, model, out_csv: Path, sleep_s=0.0):
    out_rows = []
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        t0 = time.perf_counter()
        try:
            if backend == "openai":
                res = judge_with_openai(img_path, model=model)
            elif backend == "gemini":
                res = judge_with_gemini(img_path, model=model)
            elif backend == "mock":
                res = judge_with_mock(img_path)
            else:
                raise ValueError(f"Unknown backend: {backend}")
            dt = (time.perf_counter() - t0)*1000
            out_rows.append({
                "image": img_path.name,
                "score": res["score"],
                "confidence": res["confidence"],
                "explanation": res["explanation"],
                "latency_ms": dt
            })
        except Exception as e:
            out_rows.append({
                "image": img_path.name,
                "score": np.nan,
                "confidence": np.nan,
                "explanation": f"ERROR: {e}",
                "latency_ms": np.nan
            })
        if sleep_s > 0:
            time.sleep(sleep_s)

    df = pd.DataFrame(out_rows)
    df.to_csv(out_csv, index=False)
    return df

def aggregate_against_clean(results_dir: Path, figs_dir: Path):
    """
    Reads clean.csv and all other *_s<k>.csv; computes Δscore, Δconfidence; saves robustness_summary.csv,
    latex robustness table, and a plot.
    """
    clean = pd.read_csv(results_dir/"clean.csv")
    clean = clean[["image","score","confidence"]].rename(
        columns={"score":"score_clean","confidence":"conf_clean"})
    clean.set_index("image", inplace=True)

    rows = []
    for csv in results_dir.glob("*.csv"):
        if csv.name == "clean.csv": continue
        tag = csv.stem  # e.g., fog_s2
        parts = tag.rsplit("_s", 1)
        corr = parts[0]
        sev = int(parts[1]) if len(parts)==2 and parts[1].isdigit() else None

        df = pd.read_csv(csv)
        df = df[["image","score","confidence"]].set_index("image")
        merged = clean.join(df, how="inner", rsuffix="_corr").dropna()
        merged["delta_s"] = merged["score_clean"] - merged["score"]
        merged["delta_c"] = merged["conf_clean"] - merged["confidence"]
        rows.append({
            "corruption": corr,
            "severity": sev if sev is not None else 0,
            "N": len(merged),
            "mean_ds": merged["delta_s"].mean(),
            "std_ds": merged["delta_s"].std(),
            "mean_dc": merged["delta_c"].mean(),
            "std_dc": merged["delta_c"].std()
        })

    summ = pd.DataFrame(rows).sort_values(["corruption","severity"])
    figs_dir.mkdir(parents=True, exist_ok=True)
    summ.to_csv(figs_dir/"robustness_summary.csv", index=False)

    # Plot Δscore over severity per corruption
    plt.figure(figsize=(7,4))
    for corr in summ["corruption"].unique():
        sub = summ[summ["corruption"]==corr].sort_values("severity")
        if sub.empty: continue
        plt.plot(sub["severity"], sub["mean_ds"], marker='o', label=corr)
    plt.xlabel("Severity")
    plt.ylabel("Mean score drop Δs")
    plt.title("Robustness: Δscore vs severity")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(figs_dir/"robustness_curve.png", dpi=300)
    plt.close()

    # LaTeX table
    with open(figs_dir.parent/"tables"/"robustness_table.tex", "w") as f:
        Path(figs_dir.parent/"tables").mkdir(parents=True, exist_ok=True)
        f.write(summ.to_latex(index=False, float_format="%.3f"))
    return summ

def calibration_and_ece(clean_csv: Path, figs_dir: Path, tau=4, K=10):
    """
    Computes calibration stats on clean results: bin by score for confidence curve,
    reliability diagram (confidence bins), and ECE/RCE.
    """
    df = pd.read_csv(clean_csv)
    # Ensure numeric
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["score","confidence"]).copy()

    # Calibration curve: mean confidence per integer score
    bins = sorted(df["score"].dropna().astype(int).unique())
    rows = []
    for s in bins:
        sub = df[df["score"].astype(int)==s]
        rows.append({
            "score_bin": s,
            "N": len(sub),
            "mean_conf": sub["confidence"].mean(),
            "std_conf": sub["confidence"].std()
        })
    calib = pd.DataFrame(rows)
    calib.to_csv(figs_dir/"calibration_stats.csv", index=False)

    plt.figure(figsize=(6,4))
    plt.errorbar(calib["score_bin"], calib["mean_conf"],
                 yerr=calib["std_conf"], fmt='o-', capsize=4)
    plt.xlabel("LLM-as-Judge score")
    plt.ylabel("Mean confidence")
    plt.title("Confidence–Score Calibration")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs_dir/"confidence_calibration.png", dpi=300)
    plt.close()

    # Reliability diagram (confidence bins)
    conf_bins = np.linspace(0, 1.0, K+1)
    labels = [f"{b:.1f}-{b+0.1:.1f}" for b in conf_bins[:-1]]
    df["conf_bin"] = pd.cut(df["confidence"], bins=conf_bins, labels=labels, include_lowest=True)

    stats = df.groupby("conf_bin", observed=True).agg(
        mean_conf=("confidence","mean"),
        std_conf=("confidence","std"),
        mean_score=("score","mean"),
        std_score=("score","std"),
        N=("score","count")
    ).reset_index().fillna(0.0)
    stats.to_csv(figs_dir/"reliability_stats.csv", index=False)

    plt.figure(figsize=(6,5))
    plt.plot([0,1],[0,5],'k--',label="Ideal (s=5c)")
    plt.errorbar(stats["mean_conf"], stats["mean_score"],
                 xerr=stats["std_conf"], yerr=stats["std_score"],
                 fmt='o-', capsize=3, label="Observed")
    plt.xlabel("Predicted confidence")
    plt.ylabel("Average score")
    plt.title("Reliability Diagram")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir/"reliability_diagram.png", dpi=300)
    plt.close()

    # ECE (binary) and RCE
    df["z_safe"] = (df["score"] >= tau).astype(float)
    # reuse conf bins
    ece_rows = []
    for i in range(len(conf_bins)-1):
        lo, hi = conf_bins[i], conf_bins[i+1]
        sub = df[(df["confidence"]>=lo) & (df["confidence"]<hi)]
        if len(sub)==0:
            ece_rows.append((0,0,0,0))
            continue
        mean_c = sub["confidence"].mean()
        mean_z = sub["z_safe"].mean()
        mean_y = ((sub["score"]-1.0)/4.0).mean()  # normalized score
        ece_rows.append((len(sub), mean_c, mean_z, mean_y))

    ece_rows = np.array(ece_rows, dtype=float)
    weights = ece_rows[:,0] / max(1, int(ece_rows[:,0].sum()))
    abs_gap_bin = np.abs(ece_rows[:,2] - ece_rows[:,1])
    abs_gap_reg = np.abs(ece_rows[:,3] - ece_rows[:,1])
    ECE_bin = float((weights * abs_gap_bin).sum())
    RCE     = float((weights * abs_gap_reg).sum())

    # Save LaTeX table
    ece_tbl = pd.DataFrame({
        "conf_bin": labels,
        "N": ece_rows[:,0].astype(int),
        "mean_conf": ece_rows[:,1],
        "mean_SAFE": ece_rows[:,2],
        "mean_norm_score": ece_rows[:,3],
        "abs_gap_bin": abs_gap_bin,
        "abs_gap_reg": abs_gap_reg
    })
    ece_tbl.to_csv(figs_dir/"ece_bins.csv", index=False)
    tex_dir = figs_dir.parent/"tables"
    tex_dir.mkdir(parents=True, exist_ok=True)
    with open(tex_dir/"ece_table.tex","w") as f:
        f.write(ece_tbl.to_latex(index=False, float_format="%.3f"))

    return {
        "calibration_stats": calib,
        "reliability_stats": stats,
        "ECE_bin": ECE_bin,
        "RCE": RCE
    }

# =====================
# CLI
# =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="challenge_manifest.csv from make_challenge_set.py")
    ap.add_argument("--root", type=Path, default="/Users/akramhossain/Poralekha/llmjudge/pre_challengeSet", help="root folder containing corruption subfolders")
    ap.add_argument("--backend", choices=["openai","gemini","mock"], default="mock")
    ap.add_argument("--model", type=str, default="gpt-4o")
    ap.add_argument("--results_dir", type=Path, default=Path("results_judge"))
    ap.add_argument("--figs_dir", type=Path, default=Path("figs"))
    ap.add_argument("--sleep", type=float, default=0.0, help="sleep seconds between API calls (avoid rate limits)")
    ap.add_argument("--limit", type=int, default=0, help="limit images per group (0 = all)")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    # normalize columns
    manifest.columns = [c.strip().lower() for c in manifest.columns]
    # expected: image_id, corruption, severity, out_path
    needed = {"image_id","corruption","severity","out_path"}
    if not needed.issubset(set(manifest.columns)):
        raise ValueError(f"Manifest missing columns: {needed - set(manifest.columns)}")

    # Group by (corruption, severity). 
    groups = []
    for (t,k), df in manifest.groupby(["corruption","severity"]):
        paths = [Path(p) for p in df["out_path"].tolist()]
        groups.append((t, int(k), paths))
    groups.sort(key=lambda x: (0 if x[0]=="clean" else 1, x[0], x[1]))

    # Judging loop
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for corr, sev, paths in groups:
        if args.limit > 0:
            paths = paths[:args.limit]
        out_name = f"{corr}.csv" if corr=="clean" else f"{corr}_s{sev}.csv"
        out_csv = args.results_dir/out_name
        print(f"[judge] {corr} (severity {sev}) -> {out_csv}  [N={len(paths)}]")
        judge_group(paths, backend=args.backend, model=args.model, out_csv=out_csv, sleep_s=args.sleep)

    # Aggregation vs clean
    summ = aggregate_against_clean(args.results_dir, args.figs_dir)
    print("\n[robustness] saved:", args.figs_dir/"robustness_summary.csv", "and robustness_curve.png")

    # Calibration + ECE/RCE on clean
    cal = calibration_and_ece(args.results_dir/"clean.csv", args.figs_dir)
    print(f"[calibration] ECE_bin={cal['ECE_bin']:.4f}  RCE={cal['RCE']:.4f}")
    print("[done] All outputs written under:", args.results_dir, "and", args.figs_dir)

if __name__ == "__main__":
    main()
