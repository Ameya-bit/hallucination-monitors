"""Gate A analysis: confidence histograms, probe AUROC by layer, axis cosines.

Reads a pilot run's records.jsonl and saved activations, writes metrics.json and figures.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#26251f"
MUTED = "#6f6f69"
ACCENT = "#bd6593"
PAPER = "#eeece6"
N_FOLDS = 5
N_BOOT = 1000
SEED = 0


def load_run(run_dir: Path, act_dir: Path) -> dict:
    records = [json.loads(line) for line in (run_dir / "records.jsonl").open()]
    acts = {
        "last_prompt": np.load(act_dir / "last_prompt_token.npy").astype(np.float32),
        "first_answer": np.load(act_dir / "first_answer_token.npy").astype(np.float32),
    }
    return {
        "y": np.array([r["contains_match"] for r in records], dtype=int),
        "conf": np.array([r["first_tok_prob"] for r in records], dtype=float),
        "acts": acts,
    }


def safe_auroc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def oof_scores(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold decision scores for an L2 logistic probe and a difference-of-means projection."""
    probe, dom = np.zeros(len(y)), np.zeros(len(y))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in skf.split(x, y):
        scaler = StandardScaler().fit(x[tr])
        xtr, xte = scaler.transform(x[tr]), scaler.transform(x[te])
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(xtr, y[tr])
        probe[te] = clf.decision_function(xte)
        axis = xtr[y[tr] == 1].mean(0) - xtr[y[tr] == 0].mean(0)
        dom[te] = xte @ axis
    return probe, dom


def bootstrap_ci(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], score[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def axis_cosines(x: np.ndarray, y: np.ndarray, is_conf: np.ndarray) -> float:
    corr_axis = x[y == 1].mean(0) - x[y == 0].mean(0)
    conf_axis = x[is_conf].mean(0) - x[~is_conf].mean(0)
    return float(corr_axis @ conf_axis / (np.linalg.norm(corr_axis) * np.linalg.norm(conf_axis) + 1e-12))


def per_layer_metrics(acts: np.ndarray, y: np.ndarray, is_conf: np.ndarray) -> dict:
    n_layers = acts.shape[1]
    out = {k: [] for k in ("probe_all", "probe_conf", "dom_all", "dom_conf", "cos_corr_conf")}
    best = {"auroc": -1.0, "layer": -1, "scores": None}
    for layer in range(n_layers):
        x = acts[:, layer, :]
        probe, dom = oof_scores(x, y)
        out["probe_all"].append(safe_auroc(y, probe))
        out["probe_conf"].append(safe_auroc(y[is_conf], probe[is_conf]))
        out["dom_all"].append(safe_auroc(y, dom))
        out["dom_conf"].append(safe_auroc(y[is_conf], dom[is_conf]))
        out["cos_corr_conf"].append(axis_cosines(x, y, is_conf))
        if out["probe_conf"][-1] > best["auroc"]:
            best = {"auroc": out["probe_conf"][-1], "layer": layer, "scores": probe}
        print(f"  layer {layer:2d}  probe all {out['probe_all'][-1]:.3f}  conf {out['probe_conf'][-1]:.3f}"
              f"  dom all {out['dom_all'][-1]:.3f}  cos {out['cos_corr_conf'][-1]:+.3f}", flush=True)
    out["best_conf_layer"] = best["layer"]
    out["best_conf_ci95"] = bootstrap_ci(y[is_conf], best["scores"][is_conf])
    return out


def style_axes(ax) -> None:
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="y", color="#c9c6bd", linewidth=0.6)
    ax.set_axisbelow(True)


def fig_confidence_hist(conf: np.ndarray, y: np.ndarray, thr: float, label: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=PAPER)
    style_axes(ax)
    bins = np.linspace(0, 1, 21)
    ax.hist(conf[y == 1], bins=bins, color=INK, alpha=0.9, label=f"correct (n={int(y.sum())})")
    ax.hist(conf[y == 0], bins=bins, color=ACCENT, alpha=0.75, label=f"wrong (n={int((1 - y).sum())})")
    ax.axvline(thr, color=MUTED, linestyle="--", linewidth=1)
    ax.text(thr + 0.01, ax.get_ylim()[1] * 0.95, f"confident ≥ {thr}", color=MUTED, fontsize=10, va="top")
    ax.set_xlabel("first answer-token probability", color=INK)
    ax.set_ylabel("questions", color=INK)
    ax.set_title(f"First answer-token probability, correct vs wrong\n{label}", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_auroc_by_layer(metrics: dict, baseline_all: float, baseline_conf: float, position: str, label: str,
                       path: Path) -> None:
    layers = np.arange(len(metrics["probe_all"]))
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=PAPER)
    style_axes(ax)
    ax.plot(layers, metrics["probe_all"], color=INK, linewidth=2, label="probe, all answers")
    ax.plot(layers, metrics["probe_conf"], color=ACCENT, linewidth=2, label="probe, confident subset")
    ax.plot(layers, metrics["dom_all"], color=INK, linewidth=1.2, linestyle=":", label="diff-of-means, all")
    ax.axhline(baseline_all, color=INK, linewidth=1, linestyle="--", alpha=0.5)
    ax.axhline(baseline_conf, color=ACCENT, linewidth=1, linestyle="--", alpha=0.6)
    ax.axhline(0.5, color=MUTED, linewidth=0.8)
    ax.text(layers[-1], baseline_all + 0.005, f"token-prob baseline {baseline_all:.2f}", color=INK, fontsize=9, ha="right", alpha=0.7)
    ax.text(layers[-1], baseline_conf + 0.005, f"token-prob on confident subset {baseline_conf:.2f}", color=ACCENT, fontsize=9, ha="right")
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("layer", color=INK)
    ax.set_ylabel("AUROC (higher is better)", color=INK)
    ax.set_title(f"Probe AUROC by layer vs token-probability baseline\n{label}, {position.replace('_', ' ')} token",
                 color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_cosine_by_layer(metrics: dict, dim: int, position: str, label: str, path: Path) -> None:
    layers = np.arange(len(metrics["cos_corr_conf"]))
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=PAPER)
    style_axes(ax)
    rand_sd = 1 / np.sqrt(dim)
    ax.fill_between(layers, -2 * rand_sd, 2 * rand_sd, color="#c9c6bd", alpha=0.6, label="random-direction band (±2 SD)")
    ax.plot(layers, metrics["cos_corr_conf"], color=ACCENT, linewidth=2, marker="o", markersize=4,
            label="cos(correctness axis, confidence axis)")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("layer", color=INK)
    ax.set_ylabel("cosine", color=INK)
    ax.set_title(f"Cosine between correctness axis and confidence axis by layer\n{label}, {position.replace('_', ' ')} token",
                 color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/<run> directory")
    ap.add_argument("--acts", required=True, help="data/activations/<run> directory")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--label", default="", help="model/dataset label for figure subtitles")
    args = ap.parse_args()
    run_dir, act_dir = Path(args.run), Path(args.acts)
    figs = run_dir / "figs"
    figs.mkdir(exist_ok=True)

    data = load_run(run_dir, act_dir)
    y, conf = data["y"], data["conf"]
    is_conf = conf >= args.threshold
    baseline_all = safe_auroc(y, conf)
    baseline_conf = safe_auroc(y[is_conf], conf[is_conf])
    summary = {
        "n": int(len(y)), "n_correct": int(y.sum()), "n_confident": int(is_conf.sum()),
        "n_confident_wrong": int((is_conf & (y == 0)).sum()), "n_confident_correct": int((is_conf & (y == 1)).sum()),
        "threshold": args.threshold, "baseline_auroc_all": baseline_all, "baseline_auroc_conf": baseline_conf,
        "positions": {},
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "positions"}, indent=2), flush=True)
    fig_confidence_hist(conf, y, args.threshold, args.label, figs / "confidence_hist.png")

    for position, acts in data["acts"].items():
        print(f"position: {position}", flush=True)
        metrics = per_layer_metrics(acts, y, is_conf)
        summary["positions"][position] = metrics
        fig_auroc_by_layer(metrics, baseline_all, baseline_conf, position, args.label,
                           figs / f"auroc_by_layer_{position}.png")
        fig_cosine_by_layer(metrics, acts.shape[2], position, args.label, figs / f"cosine_by_layer_{position}.png")

    with (run_dir / "metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", run_dir / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
