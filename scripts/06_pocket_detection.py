#!/usr/bin/env python3
"""
Week 6 - Pocket Detection (final)
Fix: P2Rank outputs a visualizations/ subdir — skip directories when copying results.
"""

import matplotlib
matplotlib.use("Agg")

import os, subprocess, shutil, tarfile, requests, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

STRUCTURES_DIR = Path("results/structures")
POCKETS_DIR    = Path("results/pockets")
TMP_TOOL       = Path("/tmp/p2rank_tool")
TMP_RUN        = Path("/tmp/p2rank_run")
VERSION        = "2.4"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ── Setup ─────────────────────────────────────────────────────────────────────
def setup_p2rank():
    prank = TMP_TOOL / "prank"
    if prank.exists():
        log(f"P2Rank ready at {prank}")
        return True
    log(f"Installing P2Rank {VERSION} into /tmp/p2rank_tool/ ...")
    TMP_TOOL.mkdir(parents=True, exist_ok=True)
    url = (f"https://github.com/rdk/p2rank/releases/download/"
           f"{VERSION}/p2rank_{VERSION}.tar.gz")
    try:
        r = requests.get(url, timeout=180, stream=True)
        r.raise_for_status()
        tar = TMP_TOOL / "tmp.tar.gz"
        tar.write_bytes(r.content)
        with tarfile.open(tar, "r:gz") as t:
            t.extractall(TMP_TOOL)
        tar.unlink(missing_ok=True)
        for sub in list(TMP_TOOL.iterdir()):
            if sub.is_dir():
                for item in sub.iterdir():
                    dst = TMP_TOOL / item.name
                    if not dst.exists():
                        shutil.move(str(item), str(dst))
                try: sub.rmdir()
                except Exception: pass
        os.chmod(prank, 0o755)
        log("P2Rank installed")
        return True
    except Exception as e:
        log(f"Install failed: {e}")
        return False


def check_java():
    try:
        r = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=5)
        log(f"Java: {(r.stderr+r.stdout).splitlines()[0]}")
        return True
    except Exception:
        log("Java not found")
        return False


# ── Run P2Rank ────────────────────────────────────────────────────────────────
def run_p2rank_all():
    pdb_files = sorted(STRUCTURES_DIR.glob("*.pdb"))
    if not pdb_files:
        log("No PDB files found")
        return []

    log(f"Running P2Rank on {len(pdb_files)} structures (cwd={TMP_TOOL})...")
    if TMP_RUN.exists():
        shutil.rmtree(TMP_RUN)
    TMP_RUN.mkdir(parents=True)
    POCKETS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for pdb in pdb_files:
        gene    = pdb.stem
        tmp_pdb = TMP_RUN / pdb.name
        tmp_out = TMP_RUN / f"{gene}_out"
        tmp_out.mkdir(exist_ok=True)
        shutil.copy(str(pdb), str(tmp_pdb))

        log(f"  {gene}...")
        try:
            proc = subprocess.run(
                ["./prank", "predict",
                 "-f", str(tmp_pdb), "-o", str(tmp_out),
                 "-c", "alphafold", "-threads", "2"],
                cwd=str(TMP_TOOL),          # dirname $0 = "." — no spaces
                capture_output=True, text=True, timeout=180,
            )
            ok = proc.returncode == 0
            if not ok:
                log(f"    Error: {proc.stderr.strip()[:150]}")
            else:
                proj_out = POCKETS_DIR / gene
                proj_out.mkdir(parents=True, exist_ok=True)
                for f in tmp_out.iterdir():
                    if f.is_file():                  # ← skip subdirectories
                        shutil.copy(str(f), str(proj_out / f.name))
                log(f"    OK")
            results.append((gene, pdb, POCKETS_DIR / gene, ok))
        except subprocess.TimeoutExpired:
            log(f"    Timeout")
            results.append((gene, pdb, POCKETS_DIR / gene, False))
        except FileNotFoundError as e:
            log(f"    {e}")
            break

    log(f"P2Rank: {sum(ok for *_,ok in results)}/{len(pdb_files)} succeeded")
    return results


# ── Parse P2Rank CSV ──────────────────────────────────────────────────────────
def parse_p2rank(results):
    rows = []
    for gene, _, out_dir, ok in results:
        if not ok:
            continue
        pred = (list(out_dir.glob("*.pdb_predictions.csv")) +
                list(out_dir.glob("*predictions.csv")))
        if not pred:
            continue
        try:
            df = pd.read_csv(pred[0])
            df.columns = [c.strip() for c in df.columns]
            if df.empty:
                continue
            best      = df.iloc[0]
            sc        = next((c for c in ["score","Score"] if c in df.columns), df.columns[1])
            prob_col  = next((c for c in ["probability","prob"] if c in df.columns), None)
            rows.append({
                "gene":              gene,
                "n_pockets":         len(df),
                "best_pocket_score": float(best[sc]),
                "best_pocket_prob":  float(best[prob_col]) if prob_col else 0.5,
                "best_pocket_size":  int(best["sas_points"]) if "sas_points" in df.columns else 0,
                "center_x":          float(best.get("center_x", 0) or 0),
                "center_y":          float(best.get("center_y", 0) or 0),
                "center_z":          float(best.get("center_z", 0) or 0),
                "status":            "p2rank",
            })
            log(f"  {gene:<15} pockets={len(df)}  score={float(best[sc]):.2f}")
        except Exception as e:
            log(f"  Parse error {gene}: {e}")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Geometric center from CA atoms ───────────────────────────────────────────
def geometric_center(pdb_path):
    coords = []
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    try:
                        coords.append([float(line[30:38]),
                                       float(line[38:46]),
                                       float(line[46:54])])
                    except ValueError:
                        pass
    except Exception:
        pass
    if not coords:
        return 0.0, 0.0, 0.0
    a = np.array(coords)
    return round(float(a[:,0].mean()),2), round(float(a[:,1].mean()),2), round(float(a[:,2].mean()),2)


# ── Manual fallback ───────────────────────────────────────────────────────────
def manual_analysis():
    log("Manual pocket estimation with real geometric centers...")
    pdb_files = sorted(STRUCTURES_DIR.glob("*.pdb"))

    try:
        ml    = pd.read_csv("results/target_scores.csv")
        smap  = dict(zip(ml["gene"], ml["druggability_score"]))
    except Exception:
        smap = {}

    try:
        pr = pd.read_csv("results/structure_results_local.csv")
        pr["gc"] = pr["gene"].str.replace(r"^\d+_","",regex=True)
        pmap = dict(zip(pr["gc"], pr["mean_plddt"]))
    except Exception:
        pmap = {}

    rows = []
    for pdb in pdb_files:
        gene  = pdb.stem
        clean = gene.split("_",1)[-1] if "_" in gene else gene
        bfs   = []
        try:
            with open(pdb) as f:
                for line in f:
                    if line.startswith("ATOM"):
                        try: bfs.append(float(line[60:66]))
                        except ValueError: pass
        except Exception:
            continue
        if not bfs:
            continue

        plddt  = float(pmap.get(clean, float(np.mean(bfs))))
        ml_s   = float(smap.get(clean, 0.65))
        score  = float(ml_s*0.5 + (plddt/100)*0.35 + min(len(bfs)/6000, 0.15))
        cx, cy, cz = geometric_center(str(pdb))

        rows.append({
            "gene":              gene,
            "gene_clean":        clean,
            "n_pockets":         max(1, len(bfs)//300),
            "best_pocket_score": score,
            "best_pocket_prob":  float(ml_s * 0.85),
            "best_pocket_size":  len(bfs)//5,
            "mean_plddt":        plddt,
            "center_x":          cx,
            "center_y":          cy,
            "center_z":          cz,
            "status":            "estimated",
        })
        log(f"  {clean:<12} score={score:.3f}  center=({cx:.1f},{cy:.1f},{cz:.1f})")

    return pd.DataFrame(rows).sort_values("best_pocket_score", ascending=False)


# ── Merge ─────────────────────────────────────────────────────────────────────
def merge_scores(pocket_df):
    if "gene_clean" not in pocket_df.columns:
        pocket_df = pocket_df.copy()
        pocket_df["gene_clean"] = pocket_df["gene"].str.replace(r"^\d+_","",regex=True)

    try:
        ml = pd.read_csv("results/target_scores.csv")[["gene","druggability_score","priority"]]
    except Exception:
        ml = pd.DataFrame(columns=["gene","druggability_score","priority"])

    try:
        pr = pd.read_csv("results/structure_results_local.csv")
        pr["gene_clean"] = pr["gene"].str.replace(r"^\d+_","",regex=True)
        pr = pr[["gene_clean","mean_plddt","pct_confident"]]
    except Exception:
        pr = pd.DataFrame(columns=["gene_clean","mean_plddt","pct_confident"])

    m = pocket_df.merge(ml, left_on="gene_clean", right_on="gene", how="left", suffixes=("","_ml"))
    m = m.merge(pr, on="gene_clean", how="left")

    def sc(df, col, default):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
        return pd.Series([float(default)]*len(df))

    ml_v  = sc(m, "druggability_score", 0.65)
    pl_v  = sc(m, "mean_plddt", 88.0) / 100.0
    pk_v  = sc(m, "best_pocket_score", 0.5)
    mx    = pk_v.max()
    pk_n  = pk_v/mx if mx > 0 else pk_v*0+0.5

    m["composite_score"] = (ml_v*0.40 + pk_n*0.35 + pl_v*0.25).astype(float).round(4)
    m = m.sort_values("composite_score", ascending=False).reset_index(drop=True)
    m["final_rank"] = range(1, len(m)+1)
    return m


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(df):
    genes = df["gene_clean"].tolist()
    n = len(genes)
    fig, axes = plt.subplots(1, 3, figsize=(16, max(5, n*0.45)))

    cols = ["#E05C3A" if s>=0.6 else "#F5A623" if s>=0.4 else "#B0C4DE"
            for s in df["best_pocket_score"].astype(float)]
    for ax, vals, title, color in [
        (axes[0], df["best_pocket_score"].astype(float), "Best pocket score", cols),
        (axes[1], df["n_pockets"].astype(int),           "Estimated pockets",  "#4A90D9"),
        (axes[2], df["composite_score"].astype(float),   "Composite ranking",  "#5BAD8F"),
    ]:
        c = color if isinstance(color, list) else [color]*n
        ax.barh(range(n), vals, color=c, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(n)); ax.set_yticklabels(genes, fontsize=9)
        ax.invert_yaxis(); ax.set_title(title, fontsize=11)
    for i, v in enumerate(df["composite_score"].astype(float)):
        axes[2].text(v+0.002, i, f"{v:.3f}", va="center", fontsize=8)
    axes[2].set_xlabel("Score (0–1)")

    plt.suptitle("K. pneumoniae — Pocket Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path("visualizations").mkdir(exist_ok=True)
    plt.savefig("visualizations/pocket_scores.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("Saved: visualizations/pocket_scores.png")


# ── Save ──────────────────────────────────────────────────────────────────────
def save(df):
    df.to_csv("results/pocket_summary.csv", index=False)
    log("Saved: results/pocket_summary.csv")
    keep = ["final_rank","gene_clean","composite_score","druggability_score",
            "best_pocket_score","mean_plddt","n_pockets","center_x","center_y","center_z"]
    keep = [c for c in keep if c in df.columns]
    out = df.head(10)[keep].rename(columns={"gene_clean":"gene"})
    out.to_csv("results/final_targets_for_docking.csv", index=False)
    log("Saved: results/final_targets_for_docking.csv")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*55)
    print(f" Week 6 - Pocket Detection  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*55 + "\n")
    Path("results").mkdir(exist_ok=True)

    use_p2rank = False
    if check_java() and setup_p2rank():
        res = run_p2rank_all()
        if res:
            pocket_df = parse_p2rank(res)
            if not pocket_df.empty:
                use_p2rank = True

    if not use_p2rank:
        log("\nFalling back to manual + geometric centers")
        pocket_df = manual_analysis()

    if pocket_df.empty:
        log("ERROR: No results")
        return

    log("\nMerging...")
    merged = merge_scores(pocket_df)
    docking = save(merged)
    plot(merged)

    print("\n" + "─"*55)
    print(" FINAL RANKING")
    print("─"*55)
    show = [c for c in ["final_rank","gene_clean","composite_score",
                        "center_x","center_y","center_z"] if c in merged.columns]
    print(merged[show].head(10).to_string(index=False))

    print("\n" + "="*55)
    print(f" Week 6 complete! Mode: {'P2Rank' if use_p2rank else 'manual + geometric centers'}")
    print(f" Next: python scripts/07_docking.py")
    print("="*55)


if __name__ == "__main__":
    main()