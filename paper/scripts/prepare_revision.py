"""Regenerate revision figures and the R01 paired transfer analysis.
The generator reads saved public evidence ledgers and performs deterministic derivations.
"""

from __future__ import annotations

# Embedded TikZ/PGF source is kept line-stable for reproducible generated assets.
# ruff: noqa: E501
import concurrent.futures
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "figures"
D = ROOT / "revision"
T = ROOT / "tables"
D.mkdir(exist_ok=True)
COLORS = {"phase": "fusion", "rgb": "rgb", "event": "event", "garl": "reference"}
PRE = r"""\documentclass[tikz,border=3pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{XCharter}
\usepackage[xcharter]{newtxmath}
\usepackage[scaled=.92]{helvet}
\usepackage{pgfplots}
\usetikzlibrary{arrows.meta,positioning,fit,calc,backgrounds,decorations.pathreplacing,patterns}
\usepgfplotslibrary{groupplots,fillbetween}
\pgfplotsset{compat=1.18}
\definecolor{ink}{HTML}{172536}
\definecolor{rgb}{HTML}{3563B8}
\definecolor{event}{HTML}{17857C}
\definecolor{fusion}{HTML}{76529C}
\definecolor{reference}{HTML}{8994A3}
\definecolor{warn}{HTML}{AC5948}
\definecolor{muted}{HTML}{657183}
\definecolor{rule}{HTML}{D9E0E8}
\definecolor{pale}{HTML}{F5F7FA}
\color{ink}
\tikzset{box/.style={rounded corners=2pt,draw=rule,fill=white,align=center,inner sep=4pt,font=\sffamily\fontsize{9}{10.8}\selectfont,minimum height=9mm},rb/.style={box,draw=rgb!50,fill=rgb!5},eb/.style={box,draw=event!50,fill=event!5},fb/.style={box,draw=fusion!55,fill=fusion!5},arrow/.style={-{Latex[length=1.8mm]},line width=.65pt,draw=muted},note/.style={font=\sffamily\fontsize{8.4}{10.2}\selectfont,align=center,text=muted},heading/.style={font=\sffamily\fontsize{10}{12}\selectfont\bfseries,align=left,text=ink}}
\pgfplotsset{paper/.style={width=15.7cm,height=6cm,axis lines*=left,axis line style={rule},tick style={draw=none},tick label style={font=\sffamily\fontsize{8.8}{10.4}\selectfont,text=muted},label style={font=\sffamily\fontsize{9.4}{11}\selectfont,text=ink},title style={font=\sffamily\fontsize{10}{12}\selectfont\bfseries,text=ink},legend style={draw=none,fill=none,font=\sffamily\fontsize{9}{11}\selectfont,cells={anchor=west}},grid=major,grid style={rule!65,line width=.3pt},minor tick num=0,scaled ticks=false,unbounded coords=jump,clip mode=individual}}
\begin{document}
"""
END = "\n\\end{document}\n"
GENERATED = []


def figure(name: str, body: str) -> None:
    path = F / f"rev_{name}.tex"
    path.write_text(PRE + body + END, encoding="utf-8")
    GENERATED.append(path)


def dat(name: str, header: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    p = D / f"{name}.dat"
    p.write_text(
        " ".join(header) + "\n" + "\n".join(" ".join(str(v) for v in row) for row in rows) + "\n"
    )
    return "revision/" + p.name


def esc(x: object) -> str:
    return str(x).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def symlog(x: float, lin: float = 5) -> float:
    return x / lin if abs(x) <= lin else math.copysign(1 + math.log(abs(x) / lin), x)


def number(x: float) -> str:
    return format(float(x), ".15g")


def coordinates(points: list[tuple[float, float]]) -> str:
    return " ".join(f"({number(x)},{y})" for x, y in points)


# ---- Audit and derive the labeled, paired transfer cohort ----
with (ROOT / "data/transfer_queries.csv").open(newline="") as f:
    allrows = list(csv.DictReader(f))
rows = [r for r in allrows if r["cohort"] == "extended_new" and not r["target_exclusion"]]
assert len(rows) == 468 and len({r["group_id"] for r in rows}) == 32
models = {
    "phase": "PHASE_ONLY_TRAIN40_S7",
    "rgb": "R_CTX_TRAIN40",
    "event": "E_CTX_TRAIN40",
    "garl": "Garl_full",
}
y = np.array([float(r["target_ttc"]) for r in rows])
assert np.all(y > 0)
err = {k: abs(np.array([float(r[m]) for r in rows]) - y) for k, m in models.items()}
stats = {
    k: {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p95": float(np.quantile(v, 0.95, method="linear")),
        "max": float(v.max()),
        "rte": float(np.mean(100 * v / y)),
        "counts_above_s": {str(q): int((v > q).sum()) for q in [1, 3, 5, 10]},
    }
    for k, v in err.items()
}
late = json.loads((ROOT / "data/late_results.json").read_text())["rows"]
for k, m in models.items():
    s = next(r for r in late if r.get("cohort") == "extended_new" and r.get("model") == m)
    for a, b in [
        ("mean", "MAE_s"),
        ("median", "median_AE_s"),
        ("p95", "p95_AE_s"),
        ("rte", "RTE_pct"),
    ]:
        assert np.isclose(stats[k][a], s[b], rtol=1e-12, atol=1e-12), (k, a)
groups = defaultdict(list)
for i, r in enumerate(rows):
    groups[r["group_id"]].append(i)
indices = [np.array(v) for _, v in sorted(groups.items())]
B = 10000
seed = 20261028
rng = np.random.default_rng(seed)
boot = np.empty((B, 3))
for b in range(B):
    ix = np.concatenate([indices[g] for g in rng.integers(0, len(indices), len(indices))])
    a = err["phase"][ix]
    c = err["rgb"][ix]
    boot[b] = [
        a.mean() - c.mean(),
        np.median(a) - np.median(c),
        np.quantile(a, 0.95, method="linear") - np.quantile(c, 0.95, method="linear"),
    ]
ci = np.quantile(boot, [0.025, 0.975], axis=0, method="linear")
delta = np.array([stats["phase"][k] - stats["rgb"][k] for k in ["mean", "median", "p95"]])
record = {
    "scope": "R01 paired post-hoc analysis of saved labeled transfer predictions with fixed models",
    "source": "data/transfer_queries.csv",
    "source_sha256": hashlib.sha256((ROOT / "data/transfer_queries.csv").read_bytes()).hexdigest(),
    "n": 468,
    "groups": 32,
    "metrics": stats,
    "bootstrap": {
        "replicates": B,
        "seed": seed,
        "rng": "numpy.default_rng / PCG64",
        "cluster": "complete physical sequence",
        "estimand": "pooled query mean, median, and 95th percentile, PHASE minus RGB",
        "quantile_method": "linear",
        "interval": "2.5 and 97.5 percentiles",
        "delta": delta.tolist(),
        "ci_low": ci[0].tolist(),
        "ci_high": ci[1].tolist(),
    },
}
(D / "derived_transfer_analysis.json").write_text(json.dumps(record, indent=2))
np.savetxt(
    D / "bootstrap_differences.csv",
    boot,
    delimiter=",",
    header="mean_delta_s,median_delta_s,p95_delta_s",
    comments="",
)
mac = []
for i, n in enumerate(["Mean", "Median", "Pninetyfive"]):
    for suf, v in [("Delta", delta[i]), ("Low", ci[0, i]), ("High", ci[1, i])]:
        mac.append(r"\newcommand{\Revision" + n + suf + "}{" + f"{v:+.3f}" + "}")
(T / "revision_macros.tex").write_text("\n".join(mac) + "\n")
for k, v in err.items():
    values, counts = np.unique(v, return_counts=True)
    lead = np.nextafter(values[0], 0.0)
    cumulative = 0
    curve = [(lead, 100.0, 100.0)]
    for value, count in zip(values, counts, strict=True):
        cumulative += int(count)
        p = 100.0 * (len(v) - cumulative) / len(v)
        curve.append((value, p, "nan" if p == 0 else p))
    assert curve[-1][1] == 0 and len(curve) == len(values) + 1
    dat("exceedance_" + k, ["x", "p", "display_p"], curve)

# ---- Main architecture, with all nine fitted components visible ----
figure(
    "architecture",
    r"""
\begin{tikzpicture}[x=1cm,y=1cm]
\path[use as bounding box] (0,-.3) rectangle (16.2,7.2);
\node[heading,anchor=west] at (.05,6.95) {Causal measurements \(\longrightarrow\) temporal state \(\longrightarrow\) phase correction};
\draw[rule,fill=pale,rounded corners=3pt] (0,5.85) rectangle (16.1,6.55);
\node[note,text=ink] at (8.05,6.2) {Inputs: supplied object regions, synchronized timestamps, and observations available by query time \(t\)};
\draw[event!18,fill=event!3,rounded corners=3pt] (0,3.25) rectangle (10.8,5.55);
\draw[rgb!18,fill=rgb!3,rounded corners=3pt] (0,.5) rectangle (10.8,2.8);
\node[heading,text=event,anchor=west] at (.22,5.25) {EVENT STREAM};
\node[heading,text=rgb,anchor=west] at (.22,2.5) {RGB STREAM};
\node[eb,text width=1.05cm] (ev) at (.9,4.15) {Causal\\events};
\node[eb,text width=1.25cm] (a5e) at (2.9,4.65) {\textbf{1} E-A5};
\node[eb,text width=1.25cm] (c2fe) at (2.9,3.65) {\textbf{2} E-C2F};
\node[eb,text width=1.25cm] (pe) at (4.8,4.65) {\textbf{3} PAIR-E};
\node[eb,text width=1.7cm,minimum height=15mm] (fe) at (6.85,4.15) {PHASE17\\timing + mask};
\node[eb,text width=1.85cm,minimum height=15mm] (ce) at (9.4,4.15) {\textbf{4} E-CTX\\2-layer GRU\\width 160};
\node[rb,text width=1.05cm] (rgbin) at (.9,1.4) {Causal\\RGB};
\node[rb,text width=1.25cm] (a5r) at (2.9,1.9) {\textbf{5} R-A5};
\node[rb,text width=1.25cm] (c2fr) at (2.9,.9) {\textbf{6} R-C2F};
\node[rb,text width=1.25cm] (pr) at (4.8,1.9) {\textbf{7} PAIR-R};
\node[rb,text width=1.7cm,minimum height=15mm] (fr) at (6.85,1.4) {PHASE17\\timing + mask};
\node[rb,text width=1.85cm,minimum height=15mm] (cr) at (9.4,1.4) {\textbf{8} R-CTX\\2-layer GRU\\width 160};
\foreach \i/\a/\b/\p/\f/\c/\col in {ev/a5e/c2fe/pe/fe/ce/event,rgbin/a5r/c2fr/pr/fr/cr/rgb}{
\draw[arrow,\col] (\i.east)--(\a.west);\draw[arrow,\col] (\i.east)--(\b.west);
\draw[arrow,\col] (\a)--(\p);\draw[arrow,\col] (\p.east)--(\f.north west);
\draw[arrow,\col] (\b.east)--(\f.south west);
\draw[arrow,\col] (\a.south)--++(0,-.05)-|(\f.west);
\draw[arrow,\col] (\f)--(\c);
}
\node[fb,text width=2.0cm,minimum height=20mm] (fu) at (12.3,2.75) {\textbf{9} PHASE-TTC\\20 named fields\\RGB-anchored\\fusion head};
\node[fb,text width=1.35cm] (out) at (15.05,2.75) {Signed TTC\\phase bounds};
\draw[arrow,event] (ce.east)--++(.45,0)--(fu.north west);
\draw[arrow,rgb] (cr.east)--++(.45,0)--(fu.south west);
\draw[arrow,fusion] (fu)--(out);
\node[note,text width=4.5cm] at (13.2,.75) {The 1,955-parameter fusion head follows\\eight fitted upstream components.};
\node[note,text width=16cm] at (8,-.08) {Five out-of-fold rotations generate training features; inference executes one nine-component chain.};
\end{tikzpicture}""",
)

figure(
    "fusion",
    r"""
\begin{tikzpicture}[x=1cm,y=1cm]
\path[use as bounding box] (0,-.65) rectangle (16.1,5.6);
\node[heading,anchor=west] at (.1,5.35) {Select named fields, then correct the RGB anchor};
\node[rb,text width=3.6cm,minimum height=14mm] (r) at (2.15,3.8) {\textbf{RGB: 170 coordinates}\\160 hidden + 10 named};
\node[eb,text width=3.6cm,minimum height=14mm] (e) at (6.6,3.8) {\textbf{Events: 170 coordinates}\\160 hidden + 10 named};
\node[fb,text width=3.9cm,minimum height=14mm] (select) at (12.8,3.8) {\textbf{20 selected coordinates}\\5 phase + 4 timing\\+ 1 support / modality};
\draw[arrow,rgb] (r.north)--++(0,.38)-|(select.north);
\draw[arrow,event] (e.east)--(select.west);
\node[note] at (4.4,2.66) {The two 160-dimensional hidden states remain upstream.};
\node[rb,text width=3.2cm,minimum height=14mm] (anchor) at (2.15,1.25) {\textbf{RGB anchor bypass}\\Raw location \(\ell_R\)\\and width logits \(\mathbf w_R\)};
\node[fb,text width=3.4cm,minimum height=14mm] (mlp) at (7.5,1.25) {\textbf{20 \(\to\) 32 \(\to\) 32 \(\to\) 3}\\LayerNorm + SiLU\\zero-initialized final layer};
\node[fb,text width=3.7cm,minimum height=14mm] (o) at (13.15,1.25) {\textbf{Anchored correction}\\RGB location + 2 widths\\\(\to\) phase, TTC, bounds};
\draw[arrow,fusion] (select.south)--++(0,-.6)-|(mlp.north);
\draw[arrow,fusion] (mlp)--(o);
\draw[arrow,rgb] (anchor.south)--++(0,-.7)-|(o.south);
\node[note,text width=15.8cm] at (8,-.52) {The zero-initialized head preserves the RGB anchor. Its two bounded outputs define phase-space interval endpoints.};
\end{tikzpicture}
""",
)

figure(
    "oof",
    r"""
\begin{tikzpicture}[x=1cm,y=1cm]
\path[use as bounding box] (0,-.25) rectangle (16.1,5.2);
\node[heading,anchor=west] at (.1,4.9) {One auxiliary rotation: three disjoint sequence roles};
\node[eb,text width=4.1cm,minimum height=16mm] (p) at (2.45,3.6) {\textbf{24 groups}\\Fit 4 producers + 2 PAIR heads};
\node[rb,text width=4.1cm,minimum height=16mm] (h) at (7.9,3.6) {\textbf{8 other groups}\\Fit 2 context refiners\\and normalizers};
\node[fb,text width=3.4cm,minimum height=16mm] (v) at (13.25,3.6) {\textbf{8 held-out groups}\\Generate OOF features};
\draw[arrow] (p)--(h);\draw[arrow] (h)--(v);
\node[note,text width=14.5cm] at (8,2.3) {Five rotations assign complete sequence groups to producer, context, and held-out feature roles.};
\node[box,text width=5.3cm,minimum height=15mm] (st) at (3.4,1.0) {\textbf{Stitch common-semantic fields}\\PHASE17 \(\to\) context fitting\\PHASE20 \(\to\) fusion fitting};
\node[fb,text width=5.9cm,minimum height=15mm] (dep) at (12.15,1.0) {\textbf{Deploy one all-TRAIN40 chain}\\4 final producers + 5 final heads\\single-chain inference};
\draw[arrow] (v.east)--++(.65,0)--++(0,-1.75)-|(st.north);
\draw[arrow] (st)--(dep);
\node[note,text width=16cm] at (8,-.1) {Targets are attached after OOF stitching. Refitting on 40 rather than 24 producer groups still changes the input distribution.};
\end{tikzpicture}""",
)

figure(
    "history",
    r"""
\begin{tikzpicture}[x=1cm,y=1cm]
\path[use as bounding box] (0,-.25) rectangle (16.1,3.5);
\node[heading,anchor=west] at (.1,3.2) {Past sensor support and output delay are separate};
\fill[pale,rounded corners=2pt] (1.6,.5) rectangle (11.6,2.3);
\draw[arrow] (1.3,.45)--(15.8,.45);
\draw[dashed,ink] (11.6,.2)--(11.6,2.7);
\draw[dashed,fusion] (15,.2)--(15,2.7);
\node[note,anchor=east,text=event] at (1.35,1.65) {Events};
\node[note,anchor=east,text=rgb] at (1.35,.95) {RGB};
\foreach \x in {1.9,2.1,2.5,2.6,3.2,3.4,4.0,4.5,4.6,5.2,5.6,6.3,6.4,7.4,8.0,8.3,8.8,9.0,9.2,9.8,10.3,10.8,11.1,11.4}{\draw[event,line width=.75pt] (\x,1.4)--(\x,1.85);}
\foreach \x in {2.3,4.2,6.0,8.1,9.85,11.3}{\fill[rgb] (\x,.95) circle (2.2pt);}
\draw[decorate,decoration={brace,amplitude=3pt},muted] (1.65,2.35)--(11.55,2.35);
\node[note] at (6.6,2.7) {Up to 650 ms of declared causal support};
\draw[arrow,fusion] (11.75,1.4)--node[above,note,text=fusion]{Response \(L\)}(14.8,1.4);
\node[note] at (11.6,-.08) {Query \(t\)};
\node[note] at (15,-.08) {Host output};
\end{tikzpicture}""",
)

figure(
    "runtime_clock",
    r"""
\begin{tikzpicture}[x=1cm,y=1cm]
\path[use as bounding box] (0,-.35) rectangle (16.1,3.25);
\node[box,text width=2.1cm] (a) at (1.2,1.25) {Scheduled\\arrival};
\node[box,text width=2.25cm] (b) at (5.0,1.25) {RAW data\\available};
\node[box,text width=3.0cm] (c) at (9.6,1.25) {Preparation, queues,\\producer + head work};
\node[fb,text width=2.25cm] (d) at (14.4,1.25) {Complete\\host TTC};
\draw[arrow] (a)--(b);\draw[arrow] (b)--(c);\draw[arrow] (c)--(d);
\draw[decorate,decoration={brace,amplitude=4pt},rgb] (3.9,2.1)--(15.6,2.1);
\node[note,text=rgb] at (9.75,2.58) {Secondary: RAW availability \(\to\) output};
\draw[decorate,decoration={brace,mirror,amplitude=4pt},fusion] (.1,.4)--(15.6,.4);
\node[note,text=fusion] at (7.85,-.1) {Primary deadline response: scheduled arrival \(\to\) every required valid TTC};
\end{tikzpicture}""",
)

# ---- Opening: overall and individual bands, visibly separated ----
off = json.loads((ROOT / "data/official_benchmark.json").read_text())
sc = off["phase_submission"]["metrics"]
ref = off["competitor"]["metrics"]
official_reduction = 100.0 * (ref["overall_MiD"] - sc["overall_MiD"]) / ref["overall_MiD"]
band_order = [("MiDn", 0), ("MiDl", 1), ("MiDs", 2), ("MiDc", 3)]
assert np.isclose(sum(off["benchmark"]["band_weights"].values()), 1.0)
assert np.isclose(
    sum(off["benchmark"]["band_weights"][k] * sc[k] for k, _ in band_order), sc["overall_MiD"]
)
assert sc["num_samples"] == ref["num_samples"] and sc["num_samples"] > 0
assert all(sc[key] < ref[key] for key, _ in band_order)
hero_band_lines = "\n".join(
    rf"\addplot[reference,line width=1.2pt] coordinates {{{coordinates([(sc[key], ypos), (ref[key], ypos)])}}};"
    for key, ypos in band_order
)
hero_phase_points = coordinates([(sc[key], ypos) for key, ypos in band_order])
hero_reference_points = coordinates([(ref[key], ypos) for key, ypos in band_order])
hero = r"""
\begin{tikzpicture}

\begin{axis}[paper,at={(0,0)},anchor=south west,width=7.2cm,height=5.0cm,title={@REDUCTION@\% lower Overall MiD},xbar,bar width=12pt,xmin=0,xmax=53,ymin=-.65,ymax=1.65,ytick={0,1},yticklabels={PHASE-TTC,Garl-TTC},xlabel={Overall MiD \(\downarrow\)},xtick={0,15,30,45},ymajorgrids=false,nodes near coords,nodes near coords style={font=\sffamily\small,text=ink,/pgf/number format/fixed,/pgf/number format/fixed zerofill,/pgf/number format/precision=2},enlarge y limits=false]
\addplot[fill=fusion,draw=none] coordinates {(@PHASE_OVERALL@,0)};
\addplot[fill=reference!70,draw=none] coordinates {(@REFERENCE_OVERALL@,1)};
\end{axis}
\begin{axis}[paper,at={(8.2cm,0)},anchor=south west,width=7.35cm,height=5.0cm,title={Lower in all four TTC bands},xmin=20,xmax=57,ymin=-.65,ymax=3.65,ytick={0,1,2,3},yticklabels={Negative,Large,Small,Crucial},xlabel={Bandwise MiD \(\downarrow\)},xtick={20,30,40,50},ymajorgrids=false]
@BAND_LINES@
\addplot[only marks,mark=*,mark size=2.7pt,fusion] coordinates {@PHASE_BANDS@};
\addplot[only marks,mark=square*,mark size=2.4pt,reference] coordinates {@REFERENCE_BANDS@};
\end{axis}
\end{tikzpicture}"""
hero = (
    hero.replace("@REDUCTION@", f"{official_reduction:.2f}")
    .replace("@PHASE_OVERALL@", number(sc["overall_MiD"]))
    .replace("@REFERENCE_OVERALL@", number(ref["overall_MiD"]))
    .replace("@BAND_LINES@", hero_band_lines)
    .replace("@PHASE_BANDS@", hero_phase_points)
    .replace("@REFERENCE_BANDS@", hero_reference_points)
)
figure("hero", hero)

weights = off["benchmark"]["band_weights"]
contrib = []
for k, w in weights.items():
    contrib.append((k, w * (ref[k] - sc[k])))
# Keys verified against the original official ledger.
record_off = {
    "band_contributions": dict(contrib),
    "source": "data/official_benchmark.json",
    "note": "decomposition of the saved rounded comparator band scores",
}
(D / "weighted_contributions.json").write_text(json.dumps(record_off, indent=2))
weighted_points = coordinates([(dict(contrib)[key], ypos) for key, ypos in band_order])
weighted = r"""
\begin{tikzpicture}
\begin{axis}[paper,width=15.3cm,height=4.8cm,xbar,bar width=10pt,xmin=0,xmax=5.5,ymin=-.55,ymax=3.55,ytick={0,1,2,3},yticklabels={Negative \(w=.1\),Large \(w=.1\),Small \(w=.3\),Crucial \(w=.5\)},xlabel={Contribution to reduction in Overall MiD (points)},xtick={0,1,2,3,4,5},ymajorgrids=false,nodes near coords,nodes near coords style={font=\sffamily\small,text=ink,/pgf/number format/fixed,/pgf/number format/precision=3},enlarge y limits=false]
\addplot[fill=fusion!85,draw=none] coordinates {@POINTS@};
\end{axis}
\end{tikzpicture}""".replace("@POINTS@", weighted_points)
figure("weighted", weighted)

# ---- Signed phase operating region, separated branches without singularity domination ----
figure(
    "phase",
    r"""
\begin{tikzpicture}
\begin{groupplot}[paper,group style={group size=2 by 1,horizontal sep=1.4cm},width=7.7cm,height=5cm,ylabel={Phase \(\phi(T)\)},ytick={-.1,-.05,0,.05,.1},ymin=-.11,ymax=.115]
\nextgroupplot[xmin=-10,xmax=-1,xtick={-10,-6,-3,-1},xlabel={Receding TTC \(T\) (s)},title={Receding: \(T<0\)}]
\addplot[event,very thick,domain=-10:-1,samples=140] {-ln(1-.1/x)};
\nextgroupplot[xmin=1,xmax=10,xtick={1,3,6,10},xlabel={Approaching TTC \(T\) (s)},ylabel={},title={Approaching: \(T>0.1\) s}]
\addplot[rgb,very thick,domain=1:10,samples=140] {-ln(1-.1/x)};
\addplot[only marks,mark=*,rgb] coordinates {(3,.03390155) (6,.01680712) (10,.01005034)};
\end{groupplot}
\end{tikzpicture}""",
)

# ---- Full error distribution, derived directly from every eligible query ----
body = r"""\begin{tikzpicture}
\begin{axis}[paper,width=15.5cm,height=7cm,xmode=log,ymode=log,xmin=.01,xmax=1000,ymin=.18,ymax=110,xtick={.01,.1,1,10,100,1000},xticklabels={0.01,0.1,1,10,100,1000},ytick={.2,1,5,10,25,50,100},yticklabels={0.2,1,5,10,25,50,100},xlabel={Absolute TTC error threshold (s; logarithmic axis)},ylabel={Queries exceeding threshold (\%)},legend style={at={(.5,1.03)},anchor=south,legend columns=4},minor tick num=0,log ticks with fixed point]
\draw[dashed,reference] (axis cs:.01,5)--(axis cs:1000,5);
"""
for k, label in [
    ("garl", "Garl reproduced"),
    ("event", "Event context"),
    ("rgb", "RGB context"),
    ("phase", "PHASE-TTC"),
]:
    body += (
        rf"\addplot[{COLORS[k]},line width={1.15 if k == 'phase' else 0.8}pt,const plot,mark=none] table[x=x,y=display_p]{{revision/exceedance_{k}.dat}};\addlegendentry{{{label}}}"
        + "\n"
    )
body += r"\end{axis}\end{tikzpicture}"
figure("exceedance", body)
# Retain summary pairs, now with direct source values and a labeled derived CI table.
transfer_order = [("p95", 0), ("median", 1), ("mean", 2)]
transfer_segments = "\n".join(
    rf"\addplot[reference,line width=1pt,forget plot] coordinates {{{coordinates([(stats['phase'][key], ypos), (stats['rgb'][key], ypos)])}}};"
    for key, ypos in transfer_order
)
transfer_rgb = coordinates([(stats["rgb"][key], ypos) for key, ypos in transfer_order])
transfer_phase = coordinates([(stats["phase"][key], ypos) for key, ypos in transfer_order])
transfer_pair = r"""
\begin{tikzpicture}
\begin{axis}[paper,width=15.2cm,height=4.9cm,xmin=0,xmax=4.45,ymin=-.6,ymax=2.6,ytick={0,1,2},yticklabels={95th percentile,Median,Mean},xlabel={Absolute TTC error (s)},xtick={0,1,2,3,4},ymajorgrids=false,legend style={at={(.99,.98)},anchor=north east}]
@SEGMENTS@
\addplot[only marks,mark=*,mark size=2.7pt,rgb] coordinates {@RGB@};\addlegendentry{RGB context}
\addplot[only marks,mark=diamond*,mark size=3.2pt,fusion] coordinates {@PHASE@};\addlegendentry{PHASE-TTC}
\node[note,text=fusion,anchor=south] at (axis cs:3.5,.17) {\(@P95_DELTA@\) s};
\node[note,anchor=west] at (axis cs:.7,1) {\(@MEDIAN_DELTA@\) s};
\node[note,anchor=west] at (axis cs:1.33,2) {\(@MEAN_DELTA@\) s};
\end{axis}
\end{tikzpicture}"""
transfer_pair = (
    transfer_pair.replace("@SEGMENTS@", transfer_segments)
    .replace("@RGB@", transfer_rgb)
    .replace("@PHASE@", transfer_phase)
    .replace("@P95_DELTA@", f"{stats['phase']['p95'] - stats['rgb']['p95']:+.3f}")
    .replace("@MEDIAN_DELTA@", f"{stats['phase']['median'] - stats['rgb']['median']:+.3f}")
    .replace("@MEAN_DELTA@", f"{stats['phase']['mean'] - stats['rgb']['mean']:+.3f}")
)
figure("transfer_pair", transfer_pair)
# Exact differences, not rounded-summary subtraction.
lines = [
    r"\begin{table}[H]\centering\small",
    r"\caption{Paired differences in absolute error across 468 labeled queries from 32 sequences. Percentile intervals use 10,000 paired sequence resamples. \evidence{R01; derived from L46}}",
    r"\label{tab:revision-bootstrap}",
    r"\begin{tabular}{@{}lrrr@{}}\toprule",
    r"Metric & RGB (s) & PHASE-TTC (s) & Difference [95\% interval] (s)\\\midrule",
]
for i, (k, label) in enumerate(
    [
        ("mean", "Mean absolute error"),
        ("median", "Median absolute error"),
        ("p95", "95th percentile"),
    ]
):
    lines.append(
        f"{label} & {stats['rgb'][k]:.3f} & {stats['phase'][k]:.3f} & ${delta[i]:+.3f}\\;[{ci[0, i]:+.3f},{ci[1, i]:+.3f}]$ "
        + r"\\"
    )
lines += [r"\bottomrule\end{tabular}", r"\end{table}"]
(T / "revision_bootstrap.tex").write_text("\n".join(lines))
# Full per-sequence comparison, preserving the two estimands.
seq = []
for gid, ix in groups.items():
    rte = {k: float(np.mean(100 * v[ix] / y[ix])) for k, v in err.items()}
    seq.append(
        {
            "group_id": gid,
            "n": len(ix),
            "phase": rte["phase"],
            "rgb": rte["rgb"],
            "garl": rte["garl"],
            "d_rgb": rte["phase"] - rte["rgb"],
            "d_garl": rte["phase"] - rte["garl"],
        }
    )
seq.sort(key=lambda x: (x["d_garl"], x["group_id"]))
assert sum(r["d_garl"] < 0 for r in seq) == 26 and sum(r["d_rgb"] < 0 for r in seq) == 14
(D / "sequence_transfer.json").write_text(json.dumps(seq, indent=2))
labels = [r["group_id"].split("-overlap-")[0] + f" ({r['n']})" for r in seq]
yticks = ",".join(str(i) for i in range(32))
ylabs = ",".join("{" + esc(x) + "}" for x in labels)
body = r"\begin{tikzpicture}" + "\n"
for panel, key, title, xts in [
    (0, "d_garl", r"\shortstack{vs. Garl full\\26/32 lower}", [-1000, -100, -10, 0, 10]),
    (1, "d_rgb", r"\shortstack{vs. RGB context\\14/32 lower}", [-10, 0, 10, 100]),
]:
    v = [r[key] for r in seq]
    lo = min(symlog(x) for x in v) - 0.3
    hi = max(symlog(x) for x in v) + 0.3
    # Axis positions include enough room for the full sequence-label column.
    opts = rf"paper,at={{({panel * 5.3}cm,0)}},anchor=south west,width=5.9cm,height=14.1cm,xmin={lo},xmax={hi},ymin=-.5,ymax=31.5,y dir=reverse,ytick={{{yticks}}},yticklabels={{{ylabs if panel == 0 else ''}}},tick label style={{font=\sffamily\fontsize{{8.2}}{{9.5}}\selectfont,text=muted}},xtick={{{','.join(str(symlog(x)) for x in xts)}}},xticklabels={{{','.join(str(x) for x in xts)}}},xlabel={{RTE difference (pp)}},title={{{title}}},title style={{font=\sffamily\fontsize{{8.5}}{{10}}\selectfont\bfseries}},ymajorgrids=false"
    body += "\\begin{axis}[" + opts + "]\n"
    body += r"\draw[dashed,reference] (axis cs:0,-.5)--(axis cs:0,31.5);" + "\n"
    for i, x in enumerate(v):
        body += (
            rf"\addplot[only marks,mark=*,mark size=2.0pt,{'fusion' if x < 0 else 'warn'}] coordinates {{({symlog(x)},{i})}};"
            + "\n"
        )
    body += r"\end{axis}" + "\n"
body += r"\end{tikzpicture}"
figure("sequence", body)
# Long table retains full IDs in a readable smaller monospaced-free field.
seqtable = [
    r"\begin{longtable}{@{}p{6.0cm}rrrrr@{}}",
    r"\caption{Query counts and mean RTE (\%) for all 32 Extended-new sequences. Differences are in percentage points. \evidence{R01,L46}}\label{tab:sequence-full}\\",
    r"\toprule Sequence & $n$ & PHASE & RGB & Garl & $\Delta_{\rm RGB}$\\\midrule\endfirsthead",
    r"\toprule Sequence & $n$ & PHASE & RGB & Garl & $\Delta_{\rm RGB}$\\\midrule\endhead",
    r"\bottomrule\endfoot",
]
for r in seq:
    seqtable.append(
        esc(r["group_id"])
        + f" & {r['n']} & {r['phase']:.2f} & {r['rgb']:.2f} & {r['garl']:.2f} & {r['d_rgb']:+.2f}"
        + r"\\"
    )
seqtable += [r"\end{longtable}"]
(T / "revision_sequence_full.tex").write_text("\n".join(seqtable))
# Post-hoc extremes, for numerical rather than invented visual case studies.
d = err["phase"] - err["rgb"]
selected = [int(np.argmin(d)), int(np.argmax(d))]
case_lines = [
    r"\begin{table}[H]\centering\small",
    r"\caption{Queries with the largest positive and negative PHASE-minus-RGB absolute-error differences in the labeled transfer set. \evidence{R01,L46}}",
    r"\label{tab:revision-cases}",
    r"\begin{tabularx}{\linewidth}{@{}Xrrrr@{}}\toprule",
    r"Query & Target (s) & PHASE (s) & RGB (s) & $\Delta$ AE (s)\\\midrule",
]
for i in selected:
    r = rows[i]
    short = r["query_id"].replace("EXTENDED:", "")
    case_lines.append(
        r"{\footnotesize\path{"
        + short
        + "}}"
        + f" & {y[i]:.3f} & {float(r[models['phase']]):.3f} & {float(r[models['rgb']]):.3f} & {d[i]:+.3f}"
        + r"\\"
    )
case_lines += [r"\bottomrule\end{tabularx}\end{table}"]
(T / "revision_cases.tex").write_text("\n".join(case_lines))

# ---- Historical paired comparisons, not large nearly-identical bars ----
middle = json.loads((ROOT / "data/middle_results.json").read_text())
h16 = middle["h16_replication"]
h16_rows = [
    ("7", 4, "reference"),
    ("13", 3, "reference"),
    ("23", 2, "reference"),
    ("new13_23", 1, "fusion"),
    ("all7_13_23", 0, "reference"),
]
assert all(key in h16 for key, _, _ in h16_rows)
for key, _, _ in h16_rows:
    assert np.isclose(h16[key]["paired_loss_delta"], h16[key]["h16_mid"] - h16[key]["h8_mid"])
h16_intervals = "\n".join(
    rf"\addplot[{color},line width={'1.9pt' if color == 'fusion' else '1pt'}] coordinates "
    rf"{{{coordinates([(h16[key]['hierarchical_ci95'][0], ypos), (h16[key]['hierarchical_ci95'][1], ypos)])}}};"
    for key, ypos, color in h16_rows
)
h16_reference_points = coordinates(
    [(h16[key]["paired_loss_delta"], ypos) for key, ypos, color in h16_rows if color == "reference"]
)
h16_fusion_points = coordinates(
    [(h16[key]["paired_loss_delta"], ypos) for key, ypos, color in h16_rows if color == "fusion"]
)
h16_body = r"""
\begin{tikzpicture}
\begin{axis}[paper,width=15.0cm,height=5.8cm,xmin=-6.3,xmax=1,ymin=-.65,ymax=4.65,ytick={0,1,2,3,4},yticklabels={All three seeds,New seeds 13/23,Seed 23,Seed 13,Seed 7 observed},xlabel={H16 minus H8: paired MiD difference},xtick={-6,-4,-2,0},ymajorgrids=false]
\draw[dashed,reference] (axis cs:0,-.6)--(axis cs:0,4.6);
@INTERVALS@
\addplot[only marks,mark=*,reference,mark size=2.5pt] coordinates {@REFERENCE_POINTS@};
\addplot[only marks,mark=diamond*,fusion,mark size=3.1pt] coordinates {@FUSION_POINTS@};
\end{axis}\end{tikzpicture}"""
h16_body = (
    h16_body.replace("@INTERVALS@", h16_intervals)
    .replace("@REFERENCE_POINTS@", h16_reference_points)
    .replace("@FUSION_POINTS@", h16_fusion_points)
)
figure("h16", h16_body)

v18_rows = {r["model"]: r for r in late if r.get("version") == "V18" and r.get("cohort") == "V"}
v18_seeds = [(23, 0), (13, 1), (7, 2)]
assert all(f"FUSION-{arm}-S{seed}" in v18_rows for seed, _ in v18_seeds for arm in ["TRUE", "ZERO"])
v18_segments = "\n".join(
    rf"\addplot[reference,line width=1pt,forget plot] coordinates "
    rf"{{{coordinates([(v18_rows[f'FUSION-TRUE-S{seed}']['overall_MiD'], ypos), (v18_rows[f'FUSION-ZERO-S{seed}']['overall_MiD'], ypos)])}}};"
    for seed, ypos in v18_seeds
)
v18_true = coordinates(
    [(v18_rows[f"FUSION-TRUE-S{seed}"]["overall_MiD"], ypos) for seed, ypos in v18_seeds]
)
v18_zero = coordinates(
    [(v18_rows[f"FUSION-ZERO-S{seed}"]["overall_MiD"], ypos) for seed, ypos in v18_seeds]
)
v18_body = r"""
\begin{tikzpicture}
\begin{axis}[paper,width=15.1cm,height=4.9cm,xmin=85,xmax=94,ymin=-.55,ymax=2.55,ytick={0,1,2},yticklabels={Seed 23,Seed 13,Seed 7},xlabel={V18 Overall MiD \(\downarrow\)},xtick={86,88,90,92,94},ymajorgrids=false,legend style={at={(.5,1.04)},anchor=south,legend columns=2,/tikz/every even column/.append style={column sep=6pt}}]
@SEGMENTS@
\addplot[only marks,mark=*,mark size=2.8pt,event] coordinates {@TRUE@};\addlegendentry{TRUE events}
\addplot[only marks,mark=square*,mark size=2.5pt,reference] coordinates {@ZERO@};\addlegendentry{Equal-capacity ZERO}
\end{axis}\end{tikzpicture}"""
v18_body = (
    v18_body.replace("@SEGMENTS@", v18_segments)
    .replace("@TRUE@", v18_true)
    .replace("@ZERO@", v18_zero)
)
figure("v18", v18_body)

runtime_rows = {r["model"]: r for r in late if r.get("version") == "TRAIN40_runtime"}
runtime_metrics = [("max_ms", 0), ("p99_ms", 1), ("p95_ms", 2), ("p50_ms", 3), ("mean_ms", 4)]
assert set(runtime_rows) == {"deadline_response_ms", "raw_availability_response_ms"}
assert (
    runtime_rows["deadline_response_ms"]["count"]
    == runtime_rows["raw_availability_response_ms"]["count"]
)
assert all(row["quantile_method"] == "linear" for row in runtime_rows.values())
runtime_deadline = coordinates(
    [(runtime_rows["deadline_response_ms"][key], ypos) for key, ypos in runtime_metrics]
)
runtime_raw = coordinates(
    [(runtime_rows["raw_availability_response_ms"][key], ypos) for key, ypos in runtime_metrics]
)
runtime_body = r"""
\begin{tikzpicture}
\begin{axis}[paper,width=15.0cm,height=5.9cm,xmin=0,xmax=40,ymin=-.65,ymax=4.65,ytick={0,1,2,3,4},yticklabels={Maximum,99th percentile,95th percentile,Median,Mean},xlabel={Response time (ms)},xtick={0,10,20,30,40},ymajorgrids=false,legend style={at={(.5,1.04)},anchor=south,legend columns=2,/tikz/every even column/.append style={column sep=6pt}}]
\draw[dashed,warn,line width=.8pt] (axis cs:20,-.6)--(axis cs:20,4.6);
\addplot[only marks,mark=diamond*,mark size=3.1pt,fusion] coordinates {@DEADLINE@};\addlegendentry{Scheduled-arrival response}
\addplot[only marks,mark=*,mark size=2.6pt,rgb] coordinates {@RAW@};\addlegendentry{RAW-availability response}
\node[note,text=warn,anchor=east] at (axis cs:19.4,.4) {20 ms budget};
\end{axis}\end{tikzpicture}"""
runtime_body = runtime_body.replace("@DEADLINE@", runtime_deadline).replace("@RAW@", runtime_raw)
figure("runtime", runtime_body)

# ---- Competition data: never labeled as accuracy ----
with (ROOT / "data/evttc_competition_predictions.csv").open(newline="") as f:
    cr = list(csv.DictReader(f))
sequences = [
    "CCRs1-low",
    "CCRs1-medium",
    "CCRs1-high",
    "CCRs2-low",
    "CCRs2-medium",
    "CCRs2-high",
    "CCRm-low",
    "CCRm-medium",
    "Slider-750",
    "Slider-1000",
]
cm = ["phase", "garl_full", "garl_event", "garl_visual"]
ccolors = ["fusion", "reference", "event", "rgb"]
clabels = ["PHASE-TTC", "Garl full", "Garl event-only", "Garl visual-only"]
assert len(cr) == 1736
for m, n in zip(cm, [434, 424, 434, 426], strict=False):
    a = [r for r in cr if r["model"] == m]
    assert len(a) == 434 and sum(r["prediction_ttc_s"] != "" for r in a) == n
body = r"\begin{tikzpicture}[x=1cm,y=1cm]" + "\n"
for j, (_model, label, color) in enumerate(zip(cm, clabels, ccolors, strict=False)):
    body += (
        rf"\node[heading,text={color},text width=2.25cm,align=center] at ({5.0 + j * 2.65},5.5) {{{label}}};"
        + "\n"
    )
for i, s in enumerate(sequences):
    yy = 4.85 - i * 0.47
    N = sum(r["model"] == "phase" and r["sequence_id"] == s for r in cr)
    body += rf"\node[note,anchor=east,text=ink] at (3.45,{yy}) {{{s}}};" + "\n"
    for j, m in enumerate(cm):
        a = [r for r in cr if r["model"] == m and r["sequence_id"] == s]
        n = sum(r["prediction_ttc_s"] != "" for r in a)
        cx = 5.0 + j * 2.65
        body += (
            rf"\node[box,minimum width=2.3cm,minimum height=4.2mm,inner sep=1pt,fill={'pale' if n == N else 'warn!12'},draw={'rule!50' if n == N else 'warn!40'}] at ({cx},{yy}) {{{n}/{N}}};"
            + "\n"
        )
body += r"\node[note,anchor=west] at (.2,-.15) {Counts show finite predictions per sequence; tinted cells mark queries with insufficient causal input.};\end{tikzpicture}"
figure("coverage", body)
# Individual data files retain gaps as NaN, not lines interpolated through missing outputs.
for s in sequences:
    for m in cm:
        a = sorted(
            [r for r in cr if r["sequence_id"] == s and r["model"] == m],
            key=lambda x: int(x["row_index"]),
        )
        dat(
            "comp_" + s + "_" + m,
            ["t", "v", "sy"],
            [
                (
                    float(r["timestamp_text"]),
                    float(r["prediction_ttc_s"]) if r["prediction_ttc_s"] else "nan",
                    symlog(float(r["prediction_ttc_s"]), 2) if r["prediction_ttc_s"] else "nan",
                )
                for r in a
            ],
        )
for scale in ["symlog", "linear"]:
    yopts = (
        "ymin=-4.4,ymax=6.6,ytick={-3.995732,-2.6094379,0,2.6094379,6.298317},yticklabels={-40,-10,0,10,400}"
        if scale == "symlog"
        else "ymin=-55,ymax=420,ytick={0,200,400}"
    )
    body = r"\begin{tikzpicture}" + "\n"
    body += (
        r"\begin{groupplot}[paper,group style={group size=2 by 5,horizontal sep=1.15cm,vertical sep=1.25cm},width=7.9cm,height=3.36cm,"
        + yopts
        + r",tick label style={font=\sffamily\fontsize{8}{9}\selectfont,text=muted},title style={font=\sffamily\fontsize{9}{10.5}\selectfont\bfseries},xlabel={},ylabel={},grid=major]"
        + "\n"
    )
    for i, s in enumerate(sequences):
        N = sum(r["model"] == "phase" and r["sequence_id"] == s for r in cr)
        ts = [
            float(r["timestamp_text"])
            for r in cr
            if r["sequence_id"] == s and r["model"] == "phase"
        ]
        body += (
            rf"\nextgroupplot[title={{{s} / \(n={N}\)}},xmin=0,xmax={max(ts) * 1.02},xlabel={{{'Query time (s)' if i >= 8 else ''}}},ylabel={{{'Signed TTC (s)' if i % 2 == 0 else ''}}}]"
            + "\n"
        )
        for m, c in zip(cm, ccolors, strict=False):
            body += (
                rf"\addplot[{c},line width={1.0 if m == 'phase' else 0.65}pt,mark=none] table[x=t,y={'sy' if scale == 'symlog' else 'v'}]{{revision/comp_{s}_{m}.dat}};"
                + "\n"
            )
    body += r"\end{groupplot}" + "\n"
    body += r"\node[note,anchor=south] at ($(group c1r1.north west)!.5!(group c2r1.north east)+(0,.65)$) {\textcolor{fusion}{\rule{9pt}{1.5pt}} PHASE-TTC\quad\textcolor{reference}{\rule{9pt}{1.5pt}} Garl full\quad\textcolor{event}{\rule{9pt}{1.5pt}} Garl event-only\quad\textcolor{rgb}{\rule{9pt}{1.5pt}} Garl visual-only};"
    body += r"\end{tikzpicture}"
    figure("competition_" + scale, body)

figure_receipt = {
    "schema": "phase_ttc_revision_result_figures_v1",
    "sources": {
        name: hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest()
        for name in [
            "official_benchmark.json",
            "transfer_queries.csv",
            "middle_results.json",
            "late_results.json",
        ]
    },
    "hero": {
        "phase_overall_mid": sc["overall_MiD"],
        "reference_overall_mid": ref["overall_MiD"],
        "relative_reduction_percent": official_reduction,
        "bands": {key: {"phase": sc[key], "reference": ref[key]} for key, _ in band_order},
        "samples": sc["num_samples"],
    },
    "weighted": {
        "contributions": dict(contrib),
        "weights": weights,
        "note": "Reference band values are published to two decimals; decomposition uses those saved values.",
    },
    "transfer_pair": {
        key: {
            "phase": stats["phase"][key],
            "rgb": stats["rgb"][key],
            "phase_minus_rgb": stats["phase"][key] - stats["rgb"][key],
        }
        for key, _ in transfer_order
    },
    "h16": {
        key: {
            "point_delta": h16[key]["paired_loss_delta"],
            "hierarchical_ci95": h16[key]["hierarchical_ci95"],
        }
        for key, _, _ in h16_rows
    },
    "v18": {
        str(seed): {
            "true": v18_rows[f"FUSION-TRUE-S{seed}"]["overall_MiD"],
            "zero": v18_rows[f"FUSION-ZERO-S{seed}"]["overall_MiD"],
        }
        for seed, _ in v18_seeds
    },
    "runtime": {
        model: {key: row[key] for key, _ in runtime_metrics} for model, row in runtime_rows.items()
    },
    "exceedance": {
        "definition": "100 * count(absolute_error > threshold) / 468",
        "ties": "collapsed to unique thresholds",
        "zero_percent": "stored exactly; plotted as NaN on the log axis",
    },
}
(D / "result_figure_inputs.json").write_text(
    json.dumps(figure_receipt, indent=2) + "\n", encoding="utf-8"
)
(D / "generated_figures.json").write_text(
    json.dumps([str(p.relative_to(ROOT)) for p in GENERATED], indent=2)
)
print(
    json.dumps(
        {
            "transfer_metrics": stats,
            "paired_bootstrap": record["bootstrap"],
            "figures": len(GENERATED),
        },
        indent=2,
    )
)

if __name__ == "__main__":

    def compile_one(path: Path) -> None:
        result = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory=figures",
                str(path.relative_to(ROOT)),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        (D / (path.stem + "_build.log")).write_text(stdout + "\n" + stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(path.name + "\n" + stdout[-4000:])
        return path.name

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for n in pool.map(compile_one, GENERATED):
            print("Compiled", n)
