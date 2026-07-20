#!/usr/bin/env bash
#SBATCH --job-name=tex-check
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:12:00
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/tex-check-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/tex-check-%j.out
set -uo pipefail
cd /work/mech-ai-scratch/tirtho/RecusrsiveQFormer/paper
L=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/texchk_full.log
pdflatex -interaction=nonstopmode main.tex >"$L" 2>&1
bibtex main                                 >>"$L" 2>&1
pdflatex -interaction=nonstopmode main.tex >>"$L" 2>&1
pdflatex -interaction=nonstopmode main.tex >>"$L" 2>&1
echo "final pdflatex rc=$?"
echo "=== fatal errors (should be none) ==="; grep -E "^! |Fatal error|no output PDF" "$L" | tail -20 || echo "  none"
echo "=== PDF produced? ==="; ls -l --time-style=+%H:%M main.pdf 2>/dev/null && pdfinfo main.pdf 2>/dev/null | grep -E "Pages"
echo "=== undefined refs/citations warnings ==="; grep -ciE "undefined (reference|citation)" "$L"
