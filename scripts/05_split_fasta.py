#!/usr/bin/env python3
"""
Week 5 - Split FASTA for individual ColabFold submission
=========================================================
The official ColabFold batch notebook only processes the FIRST sequence
when given a multi-sequence FASTA. This script splits your combined
FASTA into individual files — one per protein.

Usage:
    python scripts/05_split_fasta.py

Outputs:
    results/colabfold_input/individual/01_ftsI.fasta
    results/colabfold_input/individual/02_pheT.fasta
    ... (one file per target)

    Also prints a submission checklist so you know exactly
    which proteins still need to be run.
"""

import pandas as pd
from pathlib import Path
from Bio import SeqIO
from datetime import datetime


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def split_fasta(
    combined_fasta="results/colabfold_input/combined_targets.fasta",
    scores_csv="results/target_scores.csv",
    out_dir="results/colabfold_input/individual",
    structures_dir="results/structures",
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(structures_dir).mkdir(parents=True, exist_ok=True)

    # Load sequences
    records = list(SeqIO.parse(combined_fasta, "fasta"))
    log(f"Found {len(records)} sequences in {combined_fasta}")

    # Load ML rankings for ordering
    try:
        rankings = pd.read_csv(scores_csv).sort_values(
            "druggability_score", ascending=False
        ).reset_index(drop=True)
        rankings["rank"] = range(1, len(rankings) + 1)
        rank_map = dict(zip(rankings["gene"], rankings["rank"]))
        score_map = dict(zip(rankings["gene"], rankings["druggability_score"]))
    except Exception:
        rank_map = {}
        score_map = {}

    # Check which structures already exist
    existing_pdbs = {p.stem.lower() for p in Path(structures_dir).glob("*.pdb")}

    written = []
    already_done = []

    for rec in records:
        # Parse gene name from header (format: gene|uniprotid|score=X)
        gene = rec.id.split("|")[0]
        rank = rank_map.get(gene, 99)
        score = score_map.get(gene, 0.0)

        # Write individual FASTA — clean header that ColabFold likes
        fname = Path(out_dir) / f"{rank:02d}_{gene}.fasta"
        with open(fname, "w") as f:
            f.write(f">{gene}\n{str(rec.seq)}\n")

        if gene.lower() in existing_pdbs:
            already_done.append((rank, gene, score, len(rec.seq)))
        else:
            written.append((rank, gene, score, len(rec.seq), str(fname)))

    log(f"Wrote {len(written) + len(already_done)} individual FASTA files to {out_dir}/")

    # Print submission checklist
    print("\n" + "=" * 62)
    print(" COLABFOLD SUBMISSION CHECKLIST")
    print("=" * 62)

    if already_done:
        print(f"\n ALREADY DONE ({len(already_done)} structures):")
        for rank, gene, score, length in sorted(already_done):
            print(f"  [{rank:2d}] {gene:<12} score={score:.3f}  length={length} aa  ✓")

    print(f"\n STILL TO RUN ({len(written)} proteins):")
    print(f"  {'Rank':<5} {'Gene':<12} {'Score':<8} {'Length':<10} {'File'}")
    print(f"  {'-'*4} {'-'*11} {'-'*7} {'-'*9} {'-'*30}")
    for rank, gene, score, length, fpath in sorted(written):
        fname = Path(fpath).name
        print(f"  [{rank:2d}]  {gene:<12} {score:.3f}    {length} aa     {fname}")

    print(f"""
HOW TO SUBMIT EACH ONE:
  1. Go to: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb
     (Use the SINGLE sequence notebook, NOT the batch one)
  2. Paste the sequence directly into the 'query_sequence' box
  3. Set 'jobname' to the gene name (e.g. pheT)
  4. num_recycles = 3, num_models = 1, template_mode = none
  5. Runtime → Run all
  6. Download the .pdb file and save it as:
     results/structures/<gene>.pdb
     (e.g. results/structures/pheT.pdb)

PRIORITY ORDER — run these first (Priority 1, highest scores):
""")

    priority1 = [(r, g, s, l) for r, g, s, l, _ in written
                 if s >= 0.75][:10]
    for rank, gene, score, length in priority1:
        est = max(1, length // 100)
        print(f"  [{rank:2d}] {gene:<10} score={score:.3f}  ~{est} min on T4 GPU")

    print(f"""
QUICK SEQUENCE COPY — paste each sequence into ColabFold:
""")
    # Print sequences for easy copy-paste
    for rank, gene, score, length, fpath in sorted(written)[:10]:
        with open(fpath) as f:
            lines = f.readlines()
        seq = lines[1].strip() if len(lines) > 1 else ""
        print(f"  # {gene} (rank {rank}, score={score:.3f}, {length} aa)")
        print(f"  {seq[:80]}{'...' if len(seq) > 80 else ''}")
        print()

    print("=" * 62)
    return written, already_done


def check_existing_structures(structures_dir="results/structures"):
    """Show summary of what's already been predicted."""
    pdbs = list(Path(structures_dir).glob("*.pdb"))
    if not pdbs:
        log("No structures found yet in results/structures/")
        log("After downloading from Colab, save each .pdb as: results/structures/<gene>.pdb")
        return

    log(f"\nExisting structures in {structures_dir}/:")
    for pdb in sorted(pdbs):
        # Quick pLDDT check
        bfactors = []
        try:
            with open(pdb) as f:
                for line in f:
                    if line.startswith("ATOM"):
                        try:
                            bfactors.append(float(line[60:66]))
                        except ValueError:
                            pass
            mean_plddt = sum(bfactors) / len(bfactors) if bfactors else 0
            status = "high" if mean_plddt >= 70 else "low"
            log(f"  {pdb.stem:<15} pLDDT={mean_plddt:.1f}  [{status}]")
        except Exception:
            log(f"  {pdb.stem:<15} (could not parse)")


def main():
    print("=" * 55)
    print(" Week 5 - FASTA Splitter for ColabFold")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55 + "\n")

    # Check existing structures
    check_existing_structures()

    # Split and print checklist
    written, done = split_fasta()

    print(f"\n Summary:")
    print(f"  Done:       {len(done)} proteins")
    print(f"  Remaining:  {len(written)} proteins")
    print(f"  Total:      {len(done) + len(written)} proteins")


if __name__ == "__main__":
    main()