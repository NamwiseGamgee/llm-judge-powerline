import os
import json
import base64
import time
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def parse_llm_json_strict(text: str) -> dict:
    t = text.strip()
    if t.startswith("```json"):
        t = t[7:]
    if t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    obj = json.loads(t)

    score = int(round(float(obj["score"])))
    score = max(1, min(5, score))  # clamp 1..5

    conf = float(obj["confidence"])
    conf = conf / 100.0 if conf > 1.0 else conf
    conf = max(0.0, min(1.0, conf))
    conf = round(conf, 3)  # deterministic precision

    expl = str(obj.get("explanation", "")).strip()
    if len(expl) > 140:
        expl = expl[:140]

    return {"score": score, "confidence": conf, "explanation": expl}


def b64_image(img_path: Path) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def judge_with_openai(image_path: Path, model: str = "gpt-4o") -> dict:
    import openai

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = openai.OpenAI(api_key=api_key)
    img_b64 = b64_image(image_path)

    system_msg = (
        "You are a deterministic visual safety judge for powerline segmentation overlays. "
        "Return STRICT JSON only, matching the provided schema."
    )
    user_text = (
        "Evaluate this RGB image with a semi-transparent red mask overlay for powerline "
        "segmentation quality. Score 1..5 (5=perfect continuity, 1=unusable). "
        "Confidence in [0,1]. Keep explanation ≤ 140 chars."
    )

    schema = {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "explanation": {"type": "string", "maxLength": 140},
        },
        "required": ["score", "confidence", "explanation"],
        "additionalProperties": False,
    }

    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_path.suffix.lstrip('.').lower()};base64,{img_b64}"
                    },
                },
            ],
        },
    ]

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        max_tokens=128,
        seed=42,
    )

    try:
        resp = client.chat.completions.create(
            **kwargs,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_response",
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        text = resp.choices[0].message.content
        return parse_llm_json_strict(text)
    except Exception:
        last_err = None
        for _ in range(3):
            try:
                resp = client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content
                return parse_llm_json_strict(text)
            except Exception as e:
                last_err = e
                time.sleep(0.4)
        raise last_err


def judge_with_gemini(image_path: Path, model: str = "gemini-1.5-pro") -> dict:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    genai.configure(api_key=api_key)

    img = Image.open(image_path).convert("RGB")
    sys_and_user = (
        "SYSTEM: You are a deterministic visual safety judge. "
        "Return STRICT JSON ONLY on a single line, with keys exactly: "
        '{"score": <1-5 int>, "confidence": <0-1 float>, "explanation": "<=140 chars>"}.\n'
        "USER: Evaluate this RGB image with a red semi-transparent powerline mask overlay. "
        "Score 5=perfect continuity, 1=unusable. Confidence in [0,1]."
    )

    model_ = genai.GenerativeModel(model)
    retries, last_err = 3, None
    for _ in range(retries):
        try:
            resp = model_.generate_content(
                [sys_and_user, img],
                generation_config={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_output_tokens": 128,
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_VIOLENCE: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUAL: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                },
            )
            text = (resp.text or "").strip()
            return parse_llm_json_strict(text)
        except Exception as e:
            last_err = e
            time.sleep(0.4)
    raise last_err


def judge_with_mock(image_path: Path) -> dict:
    stem = image_path.stem.lower()
    h = abs(hash(stem)) % 1000
    base = 4 + ((h % 3) - 1) * 0.3  # 3.7, 4.0, or 4.3
    score = int(np.clip(round(base), 1, 5))
    conf = float(np.clip(0.78 + 0.06 * (score - 3), 0.05, 0.99))
    expl = f"Score {score}: {'clean and continuous' if score >= 4 else 'minor issues'}."
    return {"score": score, "confidence": round(conf, 3), "explanation": expl}


def judge_one(image_path: Path, backend: str, model: str):
    if backend == "openai":
        return judge_with_openai(image_path, model=model)
    if backend == "gemini":
        return judge_with_gemini(image_path, model=model)
    if backend == "mock":
        return judge_with_mock(image_path)
    raise ValueError(f"Unknown backend {backend}")


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize_words(s: str) -> set:
    s = normalize_text(s)
    return set(re.findall(r"\b\w+\b", s))


def word_overlap_iou(texts) -> float:
    
    word_sets = [tokenize_words(t) for t in texts]

    if not any(len(ws) for ws in word_sets):
        return 1.0

    union = set().union(*word_sets)
    inter = set.intersection(*word_sets) if all(len(ws) for ws in word_sets) else set()

    if len(union) == 0:
        return 1.0
    return len(inter) / float(len(union))


def pairwise_agreement_discrete(M: np.ndarray) -> np.ndarray:
    R = M.shape[1]
    A = np.eye(R)
    for i in range(R):
        for j in range(i + 1, R):
            agree = np.mean(M[:, i] == M[:, j])
            A[i, j] = A[j, i] = agree
    return A


def pairwise_corr(M: np.ndarray) -> np.ndarray:
    R = M.shape[1]
    C = np.eye(R)
    for i in range(R):
        for j in range(i + 1, R):
            x = M[:, i]
            y = M[:, j]
            if np.std(x) < 1e-8 or np.std(y) < 1e-8:
                C[i, j] = C[j, i] = np.nan
            else:
                C[i, j] = C[j, i] = np.corrcoef(x, y)[0, 1]
    return C


def icc_1_1(X: np.ndarray) -> float:
    """
    ICC(1,1): one-way random effects (Shrout & Fleiss, 1979).
    X: shape (N subjects, k raters) -> here subjects = images, raters = runs.
    """
    N, k = X.shape
    x_bar_i = X.mean(axis=1)
    ms_between = k * np.var(x_bar_i, ddof=1)
    ms_within = np.mean(np.var(X - x_bar_i[:, None], axis=1, ddof=1))
    return float((ms_between - ms_within) / (ms_between + (k - 1) * ms_within + 1e-12))


def run_judge_once(images, backend, model, out_csv: Path, sleep_s=0.0, fused_k: int = 1):
    rows = []
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    for p in images:
        t0 = time.perf_counter()
        try:
            if fused_k <= 1:
                res = judge_one(p, backend, model)
            else:
                scores, confs, exps = [], [], []
                for _ in range(fused_k):
                    r = judge_one(p, backend, model)
                    scores.append(int(r["score"]))
                    confs.append(float(r["confidence"]))
                    exps.append(r["explanation"])

                score = int(sorted(scores, key=lambda x: (-scores.count(x), x))[0])
                conf = round(float(np.mean(confs)), 3)
                expl = min(exps, key=len)
                res = {"score": score, "confidence": conf, "explanation": expl}

            dt = (time.perf_counter() - t0) * 1000
            rows.append(
                {
                    "image": p.name,
                    "score": res["score"],
                    "confidence": res["confidence"],
                    "explanation": res["explanation"],
                    "latency_ms": dt,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "image": p.name,
                    "score": np.nan,
                    "confidence": np.nan,
                    "explanation": f"ERROR: {e}",
                    "latency_ms": np.nan,
                }
            )

        if sleep_s > 0:
            time.sleep(sleep_s)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def analyze_repeatability(run_csvs, out_dir: Path, eps_conf: float = 1e-6):
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = [pd.read_csv(f) for f in run_csvs]
    for i, df in enumerate(dfs):
        df.columns = [c.lower() for c in df.columns]
        dfs[i] = df[["image", "score", "confidence", "explanation"]].copy()

    df_merged = dfs[0][["image"]].copy()
    for r, df in enumerate(dfs, start=1):
        df_merged = df_merged.merge(
            df.rename(
                columns={
                    "score": f"score_r{r}",
                    "confidence": f"conf_r{r}",
                    "explanation": f"expl_r{r}",
                }
            ),
            on="image",
            how="inner",
        )

    # ----- Score agreement -----
    score_mat = df_merged[[f"score_r{r}" for r in range(1, 6)]].to_numpy(dtype=float)
    score_identical = np.all(score_mat == score_mat[:, [0]], axis=1)
    score_agree_rate = float(np.mean(score_identical))
    score_pair_agree = pairwise_agreement_discrete(score_mat)
    icc = icc_1_1(score_mat)
    score_corr = pairwise_corr(score_mat)

    # ----- Confidence stability -----
    conf_mat = df_merged[[f"conf_r{r}" for r in range(1, 6)]].to_numpy(dtype=float)
    conf_std = np.std(conf_mat, axis=1, ddof=1)
    conf_identical = (np.max(conf_mat, axis=1) - np.min(conf_mat, axis=1)) <= eps_conf
    conf_ident_rate = float(np.mean(conf_identical))
    conf_std_mean = float(conf_std.mean())
    conf_std_p95 = float(np.percentile(conf_std, 95))
    conf_corr = pairwise_corr(conf_mat)

    # ----- Text explanation word-overlap -----
    ex_mat = df_merged[[f"expl_r{r}" for r in range(1, 6)]].astype(str).to_numpy()
    expl_word_overlap = np.array([word_overlap_iou(row) for row in ex_mat], dtype=float)
    expl_word_overlap_mean = float(np.mean(expl_word_overlap))

    ex_norm = np.vectorize(normalize_text)(ex_mat)
    expl_identical = np.all(ex_norm == ex_norm[:, [0]], axis=1)

    # ----- Combined numeric stability A_{s,c} -----
    combined_ident = score_identical & conf_identical
    combined_agree_rate = float(np.mean(combined_ident))

    # ----- Save per-image stats -----
    per_image = df_merged[["image"]].copy()
    per_image["score_identical_5runs"] = score_identical
    per_image["conf_std"] = conf_std
    per_image["conf_identical_5runs"] = conf_identical
    per_image["expl_wordoverlap_iou"] = expl_word_overlap
    per_image["expl_identical_5runs_debug"] = expl_identical
    per_image["combined_identical_5runs"] = combined_ident
    per_image.to_csv(out_dir / "per_image_stats.csv", index=False)

    # ----- Save matrices -----
    pd.DataFrame(
        score_pair_agree,
        columns=[f"r{i}" for i in range(1, 6)],
        index=[f"r{i}" for i in range(1, 6)],
    ).to_csv(out_dir / "pairwise_agreement_scores.csv")

    pd.DataFrame(
        score_corr,
        columns=[f"r{i}" for i in range(1, 6)],
        index=[f"r{i}" for i in range(1, 6)],
    ).to_csv(out_dir / "pairwise_corr_scores.csv")

    pd.DataFrame(
        conf_corr,
        columns=[f"r{i}" for i in range(1, 6)],
        index=[f"r{i}" for i in range(1, 6)],
    ).to_csv(out_dir / "pairwise_corr_conf.csv")

    # ----- Summary + JSON -----
    summary = {
        "N_images": int(len(df_merged)),
        "score_agreement_all5": score_agree_rate,
        "icc_scores_1_1": icc,
        "conf_identical_all5": conf_ident_rate,
        "conf_std_mean": conf_std_mean,
        "conf_std_p95": conf_std_p95,
        "text_expl_wordoverlap_mean": expl_word_overlap_mean,
        "combined_numeric_stability": combined_agree_rate,
    }
    with open(out_dir / "repeatability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ----- LaTeX table -----
    tex = r"""\begin{table}[!t]
\caption{Repeatability metrics over 5 runs on original overlays (schema-constrained).}
\centering
\begin{tabular}{lc}
\hline
Metric & Value \\\hline
Images ($N$) & %d \\
Score agreement (all 5 runs) & %.2f \\
ICC(1,1) for scores & %.3f \\
Confidence identical (all 5 runs) & %.2f \\
Confidence std (mean) & %.4f \\
Confidence std (95th pct.) & %.4f \\
Text explanation word-overlap (IoU) & %.2f \\
Combined numeric stability ($A_{s,c}$) & %.2f \\\hline
\end{tabular}
\label{tab:repeatability}
\end{table}
""" % (
        summary["N_images"],
        summary["score_agreement_all5"] * 100.0,
        summary["icc_scores_1_1"],
        summary["conf_identical_all5"] * 100.0,
        summary["conf_std_mean"],
        summary["conf_std_p95"],
        summary["text_expl_wordoverlap_mean"] * 100.0,
        summary["combined_numeric_stability"] * 100.0,
    )

    tables_dir = out_dir.parent / "tables_repeat"
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "repeatability_table.tex", "w") as f:
        f.write(tex)

    # ----- Optional simple plots -----
    try:
        s1 = score_mat[:, 0]
        s5 = score_mat[:, -1]
        plt.figure(figsize=(4, 4))
        plt.scatter(s1, s5, s=15)
        plt.plot([1, 5], [1, 5], "r--", linewidth=1)
        plt.xlabel("Score (Run 1)")
        plt.ylabel("Score (Run 5)")
        plt.title("Repeatability: Scores Run1 vs Run5")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "scatter_scores_r1_r5.png", dpi=200)
        plt.close()

        c1 = conf_mat[:, 0]
        c5 = conf_mat[:, -1]
        plt.figure(figsize=(4, 4))
        plt.scatter(c1, c5, s=15)
        plt.plot([0, 1], [0, 1], "r--", linewidth=1)
        plt.xlabel("Confidence (Run 1)")
        plt.ylabel("Confidence (Run 5)")
        plt.title("Repeatability: Confidence Run1 vs Run5")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "scatter_conf_r1_r5.png", dpi=200)
        plt.close()
    except Exception:
        pass

    return summary


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--overlays_dir",
        type=Path,
        required=True,
        help="Folder of ORIGINAL overlay images (Section A).",
    )
    ap.add_argument("--backend", choices=["openai", "gemini", "mock"], default="mock")
    ap.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name for selected backend.",
    )
    ap.add_argument("--results_dir", type=Path, default=Path("results_repeat"))
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between calls (avoid rate limits).",
    )
    ap.add_argument("--limit", type=int, default=0, help="Limit # images (0 = all).")
    ap.add_argument(
        "--fused_k",
        type=int,
        default=1,
        help="If >1, run k judgments/image and fuse (majority score, mean conf).",
    )
    args = ap.parse_args()

    images = sorted(
        [p for p in args.overlays_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    )
    if args.limit > 0:
        images = images[: args.limit]
    if len(images) == 0:
        raise RuntimeError("No images found in overlays_dir.")

    csv_paths = []
    for r in range(1, 6):
        out_csv = args.results_dir / f"run{r}_clean.csv"
        print(f"[run {r}/5] judging {len(images)} images -> {out_csv}")
        _ = run_judge_once(
            images,
            backend=args.backend,
            model=args.model,
            out_csv=out_csv,
            sleep_s=args.sleep,
            fused_k=args.fused_k,
        )
        csv_paths.append(out_csv)

    print("[analyze] computing repeatability metrics...")
    summary = analyze_repeatability(csv_paths, out_dir=Path("figs_repeat"))

    print("\n=== Repeatability Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nOutputs:")
    print("  - 5 run CSVs:", *[str(p) for p in csv_paths], sep="\n    ")
    print("  - per-image stats: figs_repeat/per_image_stats.csv")
    print("  - matrices: figs_repeat/pairwise_*.csv")
    print("  - LaTeX table: tables_repeat/repeatability_table.tex")
    print("  - summary JSON: figs_repeat/repeatability_summary.json")
    print("  - plots: figs_repeat/scatter_*.png")


if __name__ == "__main__":
    main()
