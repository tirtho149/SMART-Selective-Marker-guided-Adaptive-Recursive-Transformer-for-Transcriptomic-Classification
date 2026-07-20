#!/usr/bin/env python
"""Render pm_routing.json into a well-organized LaTeX -> PDF report.

Fixes the overlap/overflow of the auto-markdown version by laying out the
KEPT / DROPPED pathway rankings as two side-by-side compact tables per config.
"""
import json, os, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results/cv5/pm_routing/pm_routing.json")
OUT_TEX = os.path.join(ROOT, "results/cv5/pm_routing/pm_routing_report.tex")
OUT_PDF = os.path.join(ROOT, "results/cv5/pm_routing/pm_routing_report.pdf")

d = json.load(open(SRC))
pathways = d["pathways"]
results = d["results"]
TOPN = 25


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def ranked(mean_depth):
    pairs = list(zip(pathways, mean_depth))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def half_table(rows):
    lines = [r"\begin{tabular}{@{}r l r@{}}", r"\toprule",
             r"\# & Pathway & depth \\", r"\midrule"]
    for i, (pw, dep) in enumerate(rows, 1):
        lines.append(f"{i} & \\texttt{{{esc(pw)}}} & {dep:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


tex = [r"""\documentclass[10pt]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage[colorlinks=true,linkcolor=blue!55!black]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\titlespacing*{\section}{0pt}{12pt}{5pt}
\titleformat{\section}{\large\bfseries\color{blue!35!black}}{}{0pt}{}
\renewcommand{\arraystretch}{1.05}
\begin{document}
"""]

tex.append(r"\begin{center}{\LARGE\bfseries PM Routing Interpretability}\\[2pt]"
           r"{\large which pathways are kept vs.\ dropped}\end{center}")
tex.append(rf"""\vspace{{2pt}}
\noindent\textbf{{Cohort:}} \texttt{{{esc(d['task'])}}} ({esc(d['channels'])}) ---
{d['n_samples']} samples, {d['n_pathways']} Reactome pathway tokens, {d['n_classes']} classes.\\
\textbf{{Protocol:}} unified 5-fold CV, seed {d['seed']} (20\% test / 10\%-of-train val),
identical folds to Table~2. Model: canonical bioMoR (\texttt{{bio\_both}}, learned pathway
graph at both sites, sum-pooled).\\
\textbf{{Keep/drop signal:}} each pathway token's \emph{{mean recursion depth}} over held-out
test folds. Under \textbf{{expert-choice}} a capacity funnel keeps a shrinking top-$k$ each step
(depth $\in[0,K]$ = steps survived); under \textbf{{token-choice}} each pathway self-gates one
depth $\in[1,K]$. Higher depth $=$ router \textbf{{keeps}} compute; the minimum $=$
\textbf{{dropped}} (exits early).
""")

# ---- Summary table -----------------------------------------------------------
tex.append(r"\section*{Summary}")
tex.append(r"\begin{center}\begin{tabular}{@{}l c c c l l@{}}")
tex.append(r"\toprule")
tex.append(r"Routing & $K$ & Macro-F1 & mean depth & active tokens / step & kept-to-$K$ \\")
tex.append(r"\midrule")
for e in results:
    aps = ", ".join(str(round(x)) for x in e["active_per_step"])
    kept = sum(1 for v in e["mean_depth"] if v >= e["K"] - 0.5)
    md = sum(e["mean_depth"]) / len(e["mean_depth"])
    f1 = f"{e['f1_mean']:.1f}$\\pm${e['f1_sd']:.1f}"
    tex.append(f"{e['routing']} & {e['K']} & {f1} & {md:.2f} & {aps} & {kept}/{len(pathways)} \\\\")
tex.append(r"\bottomrule\end{tabular}\end{center}")

# ---- Per-config sections -----------------------------------------------------
for e in results:
    title = f"{e['routing']}-choice, $K={e['K']}$  (Macro-F1 {e['f1_mean']:.1f}$\\pm${e['f1_sd']:.1f})"
    tex.append(rf"\section*{{{title}}}")
    if e["K"] == 1:
        tex.append(r"\emph{$K{=}1$ is a single pass --- every pathway is kept to depth 1, "
                   r"so there is no keep/drop decision.}")
        continue
    pairs = ranked(e["mean_depth"])
    kept = pairs[:TOPN]
    dropped = list(reversed(pairs[-TOPN:]))  # earliest-exit first
    tex.append(r"\filbreak")
    tex.append(r"\begin{minipage}[t]{0.48\textwidth}\centering"
               r"\textbf{\color{green!45!black}KEPT} --- top 25 deepest-routed\\[3pt]")
    tex.append(half_table(kept))
    tex.append(r"\end{minipage}\hfill")
    tex.append(r"\begin{minipage}[t]{0.48\textwidth}\centering"
               r"\textbf{\color{red!60!black}DROPPED} --- 25 earliest-exit\\[3pt]")
    tex.append(half_table(dropped))
    tex.append(r"\end{minipage}")

tex.append(r"\end{document}")

with open(OUT_TEX, "w") as f:
    f.write("\n".join(tex))

# ---- Compile -----------------------------------------------------------------
for _ in range(2):
    p = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                        os.path.basename(OUT_TEX)],
                       cwd=os.path.dirname(OUT_TEX),
                       capture_output=True, text=True)
if p.returncode != 0:
    print(p.stdout[-3000:])
    raise SystemExit(f"xelatex failed rc={p.returncode}")
print("wrote", OUT_PDF)
