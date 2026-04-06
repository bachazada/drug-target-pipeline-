#!/usr/bin/env python3
"""
Week 7 - Molecular Docking (AutoDock Vina) — clean rebuild
"""

import matplotlib
matplotlib.use("Agg")

import os, subprocess, shutil, requests, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ── Directories ───────────────────────────────────────────────────────────────
STRUCTURES_DIR = Path("results/structures")
DOCKING_DIR    = Path("results/docking")
TMP            = Path("/tmp/vina_dock")   # no spaces — WSL safe

# ── Ligand database ───────────────────────────────────────────────────────────
LIGANDS = {
    "murA":  ("fosfomycin",    446987,   -5.8),
    "murC":  ("UNAM-1",        5311272,  -6.2),
    "fabI":  ("triclosan",     5564,     -8.1),
    "fabl":  ("triclosan",     5564,     -8.1),
    "gyrA":  ("ciprofloxacin", 2764,     -7.4),
    "gyrB":  ("novobiocin",    54675779, -9.2),
    "metG":  ("REP3123",       9908089,  -8.5),
    "pheT":  ("mupirocin",     446596,   -7.8),
    "groEL": ("geldanamycin",  5288382,  -9.1),
    "ftsI":  ("ampicillin",    6249,     -6.5),
    "ftsl":  ("ampicillin",    6249,     -6.5),
    "era":   ("GDP",           135398513,-6.0),
    "rpsC":  ("tetracycline",  54675776, -5.9),
    "rpsD":  ("tetracycline",  54675776, -5.9),
    "secA":  ("azide",         33558,    -5.2),
}
DEFAULT_LIGAND = ("ciprofloxacin", 2764, -6.0)
BOX_SIZE = 20.0


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ── Tool detection — robust version ──────────────────────────────────────────
OBABEL_PATH = "/home/bacha/miniforge3/envs/drug_target_pipeline/bin/obabel"
VINA_NAMES  = ["vina", "autodock_vina"]
OBABEL_NAMES = ["obabel", "openbabel", OBABEL_PATH]


def find_tool(candidates, test_args, success_string):
    """Try each candidate command; return first one that works."""
    for cmd in candidates:
        try:
            r = subprocess.run(
                [cmd] + test_args,
                capture_output=True, text=True, timeout=8
            )
            combined = r.stdout + r.stderr
            if success_string.lower() in combined.lower():
                return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def detect_tools():
    vina   = find_tool(VINA_NAMES,   ["--version"], "autodock vina")
    obabel = find_tool(OBABEL_NAMES, ["-V"],         "open babel")

    if vina:
        log(f"Vina:     {vina}")
    else:
        log("Vina not found  → conda install -c conda-forge vina -y")

    if obabel:
        log(f"Open Babel: {obabel}")
    else:
        log("obabel not found → conda install -c conda-forge openbabel -y")

    return vina, obabel


# ── Ligand download ───────────────────────────────────────────────────────────
def get_ligand_sdf(name, cid, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 100:
        return True
    log(f"    Downloading {name} (PubChem CID {cid})...")
    for record in ["3d", ""]:
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
               f"{cid}/SDF{'?record_type=3d' if record == '3d' else ''}")
        try:
            r = requests.get(url, timeout=30)
            if r.ok and len(r.text) > 100:
                dest.write_text(r.text)
                return True
        except Exception:
            continue
    log(f"    Download failed for {name}")
    return False


# ── Format conversions ────────────────────────────────────────────────────────
def convert(obabel, in_fmt, in_file, out_fmt, out_file, extra_flags=None):
    cmd = [obabel, f"-i{in_fmt}", str(in_file),
           f"-o{out_fmt}", f"-O{str(out_file)}",
           "--partialcharge", "gasteiger", "-h"]
    if extra_flags:
        cmd += extra_flags
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return Path(out_file).exists() and Path(out_file).stat().st_size > 0
    except Exception as e:
        log(f"    convert error: {e}")
        return False


# ── AutoDock Vina ─────────────────────────────────────────────────────────────
def run_vina(vina, receptor, ligand, cx, cy, cz, out_file):
    cmd = [
        vina,
        "--receptor",       str(receptor),
        "--ligand",         str(ligand),
        "--out",            str(out_file),
        "--center_x",       f"{cx:.3f}",
        "--center_y",       f"{cy:.3f}",
        "--center_z",       f"{cz:.3f}",
        "--size_x",         str(BOX_SIZE),
        "--size_y",         str(BOX_SIZE),
        "--size_z",         str(BOX_SIZE),
        "--exhaustiveness", "8",
        "--num_modes",      "9",
        "--energy_range",   "3",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = r.stdout + r.stderr
        affinities = []
        for line in output.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    mode = int(parts[0])
                    aff  = float(parts[1])
                    if -20 < aff < 0:
                        affinities.append(aff)
                except ValueError:
                    continue
        return affinities if affinities else None
    except subprocess.TimeoutExpired:
        log("    Vina timeout")
        return None
    except Exception as e:
        log(f"    Vina error: {e}")
        return None


# ── Real docking pipeline ─────────────────────────────────────────────────────
def real_docking(vina, obabel, targets):
    TMP.mkdir(parents=True, exist_ok=True)
    DOCKING_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for _, row in targets.iterrows():
        gene   = str(row["gene"])
        clean  = gene.split("_", 1)[-1] if "_" in gene else gene
        cx     = float(row.get("center_x", 0))
        cy     = float(row.get("center_y", 0))
        cz     = float(row.get("center_z", 0))
        lig_name, lig_cid, ref_aff = LIGANDS.get(clean, DEFAULT_LIGAND)
        comp   = float(row.get("composite_score", 0.7))

        log(f"\n  [{clean}] ligand={lig_name}  center=({cx:.1f},{cy:.1f},{cz:.1f})")

        # Find PDB
        pdbs = list(STRUCTURES_DIR.glob(f"*{clean}.pdb"))
        if not pdbs:
            log(f"    No PDB for {clean} — skip")
            continue
        pdb = pdbs[0]

        # Temp file paths (all in /tmp — no spaces)
        rec_pdb   = TMP / f"{clean}.pdb"
        rec_pdbqt = TMP / f"{clean}.pdbqt"
        lig_sdf   = TMP / f"{lig_name}.sdf"
        lig_pdbqt = TMP / f"{lig_name}.pdbqt"
        out_pdbqt = TMP / f"{clean}_{lig_name}_out.pdbqt"

        shutil.copy(str(pdb), str(rec_pdb))

        # Receptor → PDBQT
        if not rec_pdbqt.exists():
            log("    Preparing receptor...")
            ok = convert(obabel, "pdb", rec_pdb, "pdbqt", rec_pdbqt, ["-xr"])
            if not ok:
                log("    Receptor prep failed — skip")
                continue

        # Ligand SDF
        if not get_ligand_sdf(lig_name, lig_cid, lig_sdf):
            continue

        # Ligand → PDBQT
        if not lig_pdbqt.exists():
            log("    Converting ligand...")
            ok = convert(obabel, "sdf", lig_sdf, "pdbqt", lig_pdbqt, ["--gen3d"])
            if not ok:
                log("    Ligand conversion failed — skip")
                continue

        # Vina
        log("    Running Vina...")
        t0 = time.time()
        affs = run_vina(vina, rec_pdbqt, lig_pdbqt, cx, cy, cz, out_pdbqt)
        elapsed = round(time.time() - t0, 1)

        if not affs:
            log("    Docking failed")
            continue

        best = affs[0]
        log(f"    Done in {elapsed}s  best={best:.2f} kcal/mol  ref={ref_aff:.1f}")

        # Copy result back
        if out_pdbqt.exists():
            shutil.copy(str(out_pdbqt),
                        str(DOCKING_DIR / f"{clean}_{lig_name}.pdbqt"))

        rows.append({
            "gene":            clean,
            "ligand":          lig_name,
            "pubchem_cid":     lig_cid,
            "best_affinity":   round(best, 2),
            "ref_affinity":    ref_aff,
            "delta_vs_ref":    round(best - ref_aff, 2),
            "n_modes":         len(affs),
            "all_modes":       ",".join(f"{a:.2f}" for a in affs),
            "center":          f"({cx:.2f},{cy:.2f},{cz:.2f})",
            "composite_score": comp,
            "simulated":       False,
        })

    return pd.DataFrame(rows)


# ── Simulated fallback ────────────────────────────────────────────────────────
def simulated_docking(targets):
    log("Simulating from published literature values...")
    np.random.seed(42)
    rows = []
    for _, row in targets.iterrows():
        gene  = str(row["gene"])
        clean = gene.split("_", 1)[-1] if "_" in gene else gene
        lig_name, lig_cid, ref = LIGANDS.get(clean, DEFAULT_LIGAND)
        aff   = round(float(np.clip(np.random.normal(ref, 0.8), -12, -3)), 2)
        rows.append({
            "gene":            clean,
            "ligand":          lig_name,
            "pubchem_cid":     lig_cid,
            "best_affinity":   aff,
            "ref_affinity":    ref,
            "delta_vs_ref":    round(aff - ref, 2),
            "n_modes":         9,
            "all_modes":       ",".join(f"{aff+np.random.uniform(0,2):.2f}" for _ in range(8)),
            "center":          f"({row.get('center_x',0):.2f},{row.get('center_y',0):.2f},{row.get('center_z',0):.2f})",
            "composite_score": float(row.get("composite_score", 0.7)),
            "simulated":       True,
        })
    return pd.DataFrame(rows)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_results(df, simulated):
    df = df.sort_values("best_affinity").reset_index(drop=True)
    genes = df["gene"].tolist()
    n     = len(genes)
    x     = np.arange(n)
    w     = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n*0.5)))

    # Left: our vs reference
    axes[0].barh(x - w/2, df["best_affinity"].astype(float),
                 w, color="#4A90D9", label="Our docking", edgecolor="white", linewidth=0.5)
    axes[0].barh(x + w/2, df["ref_affinity"].astype(float),
                 w, color="#B0C4DE", label="Published ref", edgecolor="white",
                 linewidth=0.5, alpha=0.8)
    axes[0].set_yticks(x); axes[0].set_yticklabels(genes, fontsize=10)
    axes[0].set_xlabel("Binding affinity (kcal/mol)")
    axes[0].set_title(f"Docking affinity vs published{'  (simulated)' if simulated else ''}",
                      fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].axvline(-7.0, color="orange", linestyle="--", linewidth=1, alpha=0.7,
                    label="-7 drug-like")
    for i, v in enumerate(df["best_affinity"].astype(float)):
        axes[0].text(v - 0.05, i - w/2, f"{v:.1f}", va="center", ha="right",
                     fontsize=8, color="white", fontweight="bold")

    # Right: delta
    deltas = df["delta_vs_ref"].astype(float).tolist()
    cols   = ["#5BAD8F" if d <= 0 else "#E05C3A" for d in deltas]
    axes[1].barh(range(n), deltas, color=cols, edgecolor="white", linewidth=0.5)
    axes[1].set_yticks(range(n)); axes[1].set_yticklabels(genes, fontsize=10)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Δ vs reference (kcal/mol)")
    axes[1].set_title("Better / worse than reference", fontsize=12)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[
        Patch(color="#5BAD8F", label="Better (more negative)"),
        Patch(color="#E05C3A", label="Weaker than reference"),
    ], fontsize=9)

    plt.suptitle("K. pneumoniae — Molecular Docking Results",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path("visualizations").mkdir(exist_ok=True)
    plt.savefig("visualizations/docking_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("Saved: visualizations/docking_results.png")


def plot_final_ranking(df):
    aff   = df["best_affinity"].astype(float)
    mn, mx = aff.min(), aff.max()
    aff_n = (mn - aff) / (mn - mx + 1e-9)
    comp  = df.get("composite_score", pd.Series([0.7]*len(df))).astype(float)
    df    = df.copy()
    df["final"] = (comp*0.5 + aff_n*0.5).round(4)
    df = df.sort_values("final", ascending=False).reset_index(drop=True)

    genes  = df["gene"].tolist()
    n      = len(genes)
    colors = ["#E05C3A" if v >= 0.7 else "#F5A623" if v >= 0.5 else "#B0C4DE"
              for v in df["final"]]

    fig, ax = plt.subplots(figsize=(10, max(5, n*0.5)))
    bars = ax.barh(range(n), df["final"], color=colors,
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{g}  ({a:.1f} kcal/mol)"
                        for g, a in zip(genes, df["best_affinity"].tolist())],
                       fontsize=10)
    ax.invert_yaxis()
    for i, v in enumerate(df["final"]):
        ax.text(v+0.005, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Final combined score (ML + pocket + docking)")
    ax.set_title("Final drug target ranking — all evidence",
                 fontsize=12, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#E05C3A", label="Priority 1 (≥0.7)"),
        Patch(color="#F5A623", label="Priority 2 (≥0.5)"),
        Patch(color="#B0C4DE", label="Priority 3"),
    ], fontsize=9)

    plt.tight_layout()
    plt.savefig("visualizations/final_ranking.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("Saved: visualizations/final_ranking.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print(f" Week 7 - Molecular Docking  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55 + "\n")

    Path("results").mkdir(exist_ok=True)
    DOCKING_DIR.mkdir(parents=True, exist_ok=True)

    # Load targets
    targets = pd.read_csv("results/final_targets_for_docking.csv")
    log(f"Targets: {', '.join(targets['gene'].tolist())}")

    # Detect tools
    log("\nDetecting tools...")
    vina, obabel = detect_tools()

    # Run docking
    if vina and obabel:
        log("\nBoth tools found — running REAL docking")
        df = real_docking(vina, obabel, targets)
        simulated = False
        if df.empty:
            log("Real docking produced no results — falling back to simulation")
            df = simulated_docking(targets)
            simulated = True
    else:
        log("\nRunning SIMULATED docking")
        df = simulated_docking(targets)
        simulated = True

    df = df.sort_values("best_affinity").reset_index(drop=True)
    df["docking_rank"] = range(1, len(df)+1)

    # Save
    df.to_csv("results/docking_scores.csv", index=False)
    log("Saved: results/docking_scores.csv")

    # Plots
    plot_results(df, simulated)
    plot_final_ranking(df)

    # Report
    with open("results/docking_report.md", "w") as f:
        f.write(f"# Week 7 Docking Report\nMode: {'simulated' if simulated else 'real AutoDock Vina'}\n\n")
        f.write(df[["gene","ligand","best_affinity","ref_affinity","delta_vs_ref"]].to_string())

    # Print table
    print("\n" + "─"*62)
    print(f" RESULTS {'(SIMULATED)' if simulated else '(REAL AUTODOCK VINA)'}")
    print("─"*62)
    print(f"{'Rk':<4} {'Gene':<10} {'Ligand':<16} {'Affinity':>10} {'Ref':>7} {'Δ':>7}")
    print("─"*62)
    for _, r in df.iterrows():
        flag = " ★" if float(r["best_affinity"]) <= float(r["ref_affinity"]) else ""
        print(f"  {int(r['docking_rank']):<3} {r['gene']:<10} {r['ligand']:<16} "
              f"{float(r['best_affinity']):>8.2f}  {float(r['ref_affinity']):>6.1f}  "
              f"{float(r['delta_vs_ref']):>+6.2f}{flag}")

    print(f"\n  ★ = better than published reference")
    print(f"\n  Guide: < -7.0 drug-like  |  < -9.0 very strong")
    print("\n" + "="*55)
    print(f" Week 7 complete! ({'simulated' if simulated else 'real'})")
    print(f" Next: python scripts/08_snakemake_pipeline.py")
    print("="*55)


if __name__ == "__main__":
    main()