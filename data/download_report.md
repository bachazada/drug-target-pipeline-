# Week 1 Download Report
Generated: 2026-03-29 04:30

## Organism
- Name: Klebsiella pneumoniae subsp. pneumoniae
- Strain: MGH 78578 / ATCC 700721
- UniProt Proteome ID: UP000000265  (CORRECT ID)
- NCBI Taxon ID: 272620

## Downloaded files
| File | Entries | Path |
|------|---------|------|
| Proteome FASTA | 5126 proteins | data/proteome.fasta |
| Essential genes | 154 genes | data/essential_genes.txt |

## Verify with these commands
    grep -c "^>" data/proteome.fasta      # should print ~5126
    wc -l data/essential_genes.txt        # should print ~185

## Next step - Week 2
    python scripts/02_filter_targets.py
