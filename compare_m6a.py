#!/usr/bin/env python3

"""
compare_m6a.py

Run ViennaRNA (RNAsubopt) and/or RNAstructure in m6A mode on:
  - a non-modified (no_mod) sequence
  - an m6A-modified sequence (A→'6' at specified positions)

and compare how m6A affects folding:

Outputs:
  1) Resulting ensembles (dot–bracket lists) for no_mod and m6A, per tool.
  2) Positional Shannon entropy plots (no_mod vs m6A) per tool (HTML).
  3) Positional consensus plots (no_mod vs m6A) per tool (HTML).
  4) If --tool both: extra plots comparing per-position entropy / consensus
     *between tools* on the m6A sequence (Vienna vs RNAstructure).

Usage example:
  python compare_m6a.py \
    -s myseq.fasta \
    -m mods.txt \
    --tool both \
    -n 20 \
    -o compare_m6a_results \
    -c config.yaml
"""

import argparse
import os
import sys
import tempfile
import shutil
from pathlib import Path
from collections import Counter
from math import ceil
from datetime import datetime

from ruamel.yaml import YAML
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import utils  # wrappers & helpers from your codebase

LETTER_MAP = {
    'V': 'RNASubopt',      # ViennaRNA ens via RNAsubopt
    'R': 'RNAStructure',
}

ENSEMBLE_COLORS = {
    'V': 'red',
    'R': "#9b99fc",  # a lighter purple
    'V6': "#f8c9aa",
    'R6': 'violet'
}

# ---------------------------- Helpers ----------------------------

def load_config(path: str):
    yaml = YAML(typ="safe")
    with open(path) as fh:
        return yaml.load(fh)

def load_sequence(path_or_seq: str) -> str:
    p = Path(path_or_seq)
    if p.exists():
        seq = []
        for line in p.read_text().splitlines():
            if line.startswith(">"):
                continue
            seq.append(line.strip())
        return "".join(seq)
    else:
        return path_or_seq.strip()

def read_positions(path: str) -> list:
    pos = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            idx = int(s)
            if idx <= 0:
                raise ValueError
            pos.append(idx)
        except Exception:
            print(f"Warning: skip invalid position '{s}' in {path}")
    return sorted(set(pos))

def make_m6a_sequence(seq: str, positions_1based: list) -> str:
    arr = list(seq)
    L = len(arr)
    for p in positions_1based:
        if 1 <= p <= L:
            # Encode m6A as '6' (as required by your m6A-enabled wrappers)
            arr[p-1] = '6'
        else:
            print(f"Warning: position {p} outside sequence length {L}, skipping.")
    return "".join(arr)

def write_seq_file(seq: str, path: str):
    with open(path, "w") as fh:
        fh.write(seq + "\n")

def _safe_remove(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"Warning: could not remove {path}: {e}")

def generate_ensemble(letter: str, seq_file: str, out_db: str, ens_n: int, cfg: dict, m6a: bool):
    """
    Generate an ensemble using selected tool.
    Removes any intermediate .tmp file after extraction.
    Returns list[str] of dot–bracket strings (trimmed to ens_n).
    """
    tool = LETTER_MAP[letter]
    tool_cfg = cfg[tool]
    exe      = tool_cfg['executable']
    params   = tool_cfg.get('params', {}).copy()
    tmp_db   = out_db + ".tmp"

    if tool == "RNASubopt":
        params['m6A'] = bool(m6a)
        utils.RNASubopt(
            seq_file=seq_file,
            out_file=tmp_db,
            executable=exe,
            n_struc=ens_n,
            method=params.get('method'),
            m6A=params.get('m6A', False)
        )
        full = utils.extract_ensemble(tmp_db)
        _safe_remove(tmp_db)
        trimmed = full[:ens_n]
        with open(out_db, 'w') as fh:
            fh.write("\n".join(trimmed) + "\n")
        return trimmed

    elif tool == "RNAStructure":
        utils.RNAStructure(
            seq_file=seq_file,
            out_file=tmp_db,
            m6A=m6a,
            maxm=params.get('maxm'),
            executable=exe
        )
        full = utils.extract_ensemble(tmp_db)
        _safe_remove(tmp_db)
        if len(full) < ens_n:
            print(f"WARNING: RNAStructure produced only {len(full)} structures (requested {ens_n})")
        trimmed = full[:ens_n]
        with open(out_db, 'w') as fh:
            fh.write("\n".join(trimmed) + "\n")
        return trimmed

    else:
        raise ValueError(f"Unknown tool letter: {letter}")

def positional_stats_with_freq(structs: list):
    """
    For a list of dot–bracket strings, compute per-position:
      - Shannon entropy (utils.shannon_math, base 2)
      - Consensus (fraction of most frequent symbol)
      - Per-symbol frequency dict: {'.': [..], '(': [..], ')': [..]}
    Returns (entropy_list, consensus_list, freq_dict).
    """
    if not structs:
        return [], [], {'.': [], '(': [], ')': []}
    L = len(structs[0])
    ent, cons = [], []
    freq = {'.': [], '(': [], ')': []}
    for i in range(L):
        col = [s[i] for s in structs]
        ent.append(utils.shannon_math(col, unit="shannon"))
        c = Counter(col)
        most, cnt = c.most_common(1)[0]
        cons.append(cnt / len(col))
        for sym in ('.','(',')'):
            freq[sym].append(c.get(sym, 0) / len(col))
    return ent, cons, freq

def plot_series_over_positions(
    seq: str,
    ys_list: list,
    labels: list,
    colors: list,
    freq_list: list,  
    title: str,
    y_title: str,
    out_html: str,
    mods: list = []
):
    """
    Make a 4-row chunked Plotly line plot across positions, with multiple series.
    Hover shows: pos, nt, value, and per-symbol frequencies for that series.
    """
    L = len(seq)
    nrows = 4
    chunk = ceil(L / nrows)
    min_px_per_base = 10
    row_width_px = max(chunk * min_px_per_base, 900)
    mods = mods or []

    fig = make_subplots(
        rows=nrows, cols=1, shared_xaxes=False,
        subplot_titles=[
            f"Positions {r*chunk+1}-{min((r+1)*chunk, L)}"
            for r in range(nrows)
        ]
    )

    for r in range(nrows):
        start = r * chunk
        end   = min(start + chunk, L)
        xs    = list(range(start+1, end+1))
        for ys, lab, col, fq in zip(ys_list, labels, colors, freq_list):
            if "m6A" in lab:
                marker_colors = [
                    "black" if (i in mods) else col
                    for i in xs
                ]
            else:
                marker_colors = [col] * len(xs)
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys[start:end],
                    mode='lines+markers',
                    name=f"{lab} (row {r+1})",
                    line=dict(color=col),
                    marker=dict(color=marker_colors),
                    hovertext=[
                        (
                            (lambda nt_disp:
                                f"pos={i}"
                                f"<br>nt={nt_disp}"
                                f"<br>val={ys[i-1]:.3f}"
                                f"<br>(={fq['('][i-1]:.3f} )={fq[')'][i-1]:.3f} .={fq['.'][i-1]:.3f}"
                            )(
                                "6" if ("m6A" in lab and i in mods) else seq[i-1]
                            )
                        )
                        for i in xs
                    ],
                    hoverinfo="text"
                ),
                row=r+1, col=1
            )
        fig.update_xaxes(range=[start - 0.5, end + 0.5], row=r+1, col=1)

    if "Consensus" in title or "consensus" in y_title.lower():
        y_range = [0, 1.05]
    else:
        m = max(max(ys) if ys else 0.0 for ys in ys_list)
        y_range = [0, max(m, 1.0) * 1.05]

    for r in range(nrows):
        fig.update_yaxes(range=y_range, row=r+1, col=1)

    fig.update_layout(
        title=title,
        xaxis_title="Position",
        yaxis_title=y_title,
        font_family="Courier New",
        width=max(row_width_px, 1075),
        height=max(720, 240 * nrows),
        margin=dict(l=50, r=30, t=150, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0)
    )
    fig.write_html(out_html, auto_open=False)
    print("Wrote plot:", out_html)

def write_output_summary(out_dir: Path, base: str, want_V: bool, want_R: bool, M: int = 20, sequence: str = "input sequence"):
    """
    Create a short text file summarizing all outputs generated by compare_m6a.py.
    """
    summary_path = out_dir / f"{base}_compare_m6a_summary.txt"
    lines = []
    lines.append("compare_m6a.py – Output Summary\n")
    lines.append("=" * 56 + "\n\n")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Output folder: {out_dir.resolve()}\n\n")
    lines.append(f"Sequence:  {sequence}\n")
    lines.append(f"Ensemble size per tool (M): {M}\n")

    if want_V:
        lines.append("[ViennaRNA / RNAsubopt]\n")
        lines.append(f"  - {base}_no_mod_Vienna_ens.db        : Ensemble of non_modified structures\n")
        lines.append(f"  - {base}_m6A_Vienna_ens.db       : Ensemble of m6A-modified structures\n")
        lines.append(f"  - {base}_Vienna_no_mod_vs_m6A_entropy.html : Positional Shannon entropy comparison (no_mod vs m6A) (interactive)\n")
        lines.append(f"  - {base}_Vienna_no_mod_vs_m6A_consensus.html : Positional consensus comparison (no_mod vs m6A) (interactive)\n\n")
        lines.append(f"  - {base}_Vienna_no_mod_vs_m6A_fraction_ssRNA.html : Frequency of unpaired positions ('.') (no_mod vs m6A) (interactive)\n\n")

    if want_R:
        lines.append("[RNAstructure]\n")
        lines.append(f"  - {base}_no_mod_RNAStructure_ens.db  : Ensemble of non_modified structures\n")
        lines.append(f"  - {base}_m6A_RNAStructure_ens.db : Ensemble of m6A-modified structures\n")
        lines.append(f"  - {base}_RNAStructure_no_mod_vs_m6A_entropy.html : Positional Shannon entropy comparison (no_mod vs m6A) (interactive)\n")
        lines.append(f"  - {base}_RNAStructure_no_mod_vs_m6A_consensus.html : Positional consensus comparison (no_mod vs m6A) (interactive)\n\n")
        lines.append(f"  - {base}_RNAStructure_no_mod_vs_m6A_fraction_ssRNA.html : Frequency of unpaired positions ('.') (no_mod vs m6A) (interactive)\n\n")

    if want_V and want_R:
        lines.append("[Cross-tool comparison on m6A sequence]\n")
        lines.append(f"  - {base}_m6A_crossTool_entropy.html   : Entropy comparison between ViennaRNA and RNAstructure (interactive)\n")
        lines.append(f"  - {base}_m6A_crossTool_consensus.html : Consensus comparison between ViennaRNA and RNAstructure (interactive)\n\n")
        lines.append(f"  - {base}_m6A_crossTool_fraction_ssRNA.html   : Frequency of unpaired positions ('.') comparison between ViennaRNA and RNAstructure (interactive)\n\n")


    lines.append("Notes:\n")
    lines.append("Each HTML file is interactive (Plotly). Hover over points for per-nucleotide statistics.\n")
    lines.append("Modified (m6A) positions are highlighted with black markers in the m6A plots.\n")

    with open(summary_path, "w") as fh:
        fh.writelines(lines)

    print(f"Summary file written: {summary_path}")


# ---------------------------- Main ----------------------------

def main():
    p = argparse.ArgumentParser(description="Compare no_mod vs m6A ensembles for ViennaRNA and/or RNAstructure.")
    p.add_argument("-s","--sequence", required=True, help="FASTA path or raw sequence")
    p.add_argument("-m","--mods", required=True, help="TXT file with 1-based positions (one per line) of m6A sites")
    p.add_argument("--tool", choices=["V","R","both"], default="both",
                   help="Which tool(s) to run: V (ViennaRNA/RNAsubopt), R (RNAstructure), or both")
    p.add_argument("-n","--ens_n", type=int, help="Ensemble size (default: global_ensemble_size from config)")
    p.add_argument("-o","--output_folder", default="compare_m6a_results", help="Output folder")
    p.add_argument("-c","--config", default="config.yaml", help="YAML configuration file")

    args = p.parse_args()

    CFG = load_config(args.config)
    env = CFG.get("environment", {})
    if env.get("data_tables"):
        os.environ["DATAPATH"] = os.path.abspath(env["data_tables"])
    if env.get("threads") is not None:
        os.environ["OMP_NUM_THREADS"] = str(env["threads"])

    seq_no_mod  = load_sequence(args.sequence)
    mods    = read_positions(args.mods)
    seq_m6a = make_m6a_sequence(seq_no_mod, mods)

    base = Path(args.sequence).stem
    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmpdir = Path(tempfile.mkdtemp(prefix="m6acmp_"))
    try:
        no_mod_seq_file   = str(tmpdir / "no_mod.seq")
        m6a_seq_file  = str(tmpdir / "m6a.seq")
        write_seq_file(seq_no_mod, no_mod_seq_file)
        write_seq_file(seq_m6a, m6a_seq_file)

        M = args.ens_n if args.ens_n is not None else CFG.get("global_ensemble_size", 20)

        want_V = args.tool in ("V","both")
        want_R = args.tool in ("R","both")

        results = {}  # (tool_letter, variant) -> list of dot-brackets ("no_mod"/"m6A")

        # --- ViennaRNA (RNAsubopt) ---
        if want_V:
            out_db_no_mod  = str(out_dir / f"{base}_no_mod_Vienna_ens.db")
            out_db_m6a = str(out_dir / f"{base}_m6A_Vienna_ens.db")
            res_no_mod  = generate_ensemble('V', no_mod_seq_file,  out_db_no_mod,  M, CFG, m6a=False)
            res_m6a = generate_ensemble('V', m6a_seq_file, out_db_m6a, M, CFG, m6a=True)
            results[('V','no_mod')]  = res_no_mod
            results[('V','m6A')] = res_m6a
            print(f"ViennaRNA ensembles written: {out_db_no_mod}, {out_db_m6a}")

        # --- RNAstructure ---
        if want_R:
            out_db_no_mod  = str(out_dir / f"{base}_no_mod_RNAStructure_ens.db")
            out_db_m6a = str(out_dir / f"{base}_m6A_RNAStructure_ens.db")
            res_no_mod  = generate_ensemble('R', no_mod_seq_file,  out_db_no_mod,  M, CFG, m6a=False)
            res_m6a = generate_ensemble('R', m6a_seq_file, out_db_m6a, M, CFG, m6a=True)
            results[('R','no_mod')]  = res_no_mod
            results[('R','m6A')] = res_m6a
            print(f"RNAstructure ensembles written: {out_db_no_mod}, {out_db_m6a}")

        # ---------- Metrics & Plots ----------
        # For each requested tool: no_mod vs m6A overlays (character breakdown in hover)
        if want_V:
            ent_no_mod,  cons_no_mod,  freq_no_mod  = positional_stats_with_freq(results[('V','no_mod')])
            ent_m6,  cons_m6,  freq_m6  = positional_stats_with_freq(results[('V','m6A')])

            plot_series_over_positions(
                seq=seq_no_mod,
                ys_list=[ent_no_mod, ent_m6],
                labels=["Vienna no_mod", "Vienna m6A"],
                colors=[ENSEMBLE_COLORS['V'], ENSEMBLE_COLORS['V6']],
                freq_list=[freq_no_mod, freq_m6],
                title="Positional Shannon Entropy – ViennaRNA (no_mod vs m6A)",
                y_title="Entropy (shannon)",
                out_html=str(out_dir / f"{base}_Vienna_no_mod_vs_m6A_entropy.html"),
                mods=mods
            )
            plot_series_over_positions(
                seq=seq_no_mod,
                ys_list=[cons_no_mod, cons_m6],
                labels=["Vienna no_mod", "Vienna m6A"],
                colors=[ENSEMBLE_COLORS['V'], ENSEMBLE_COLORS['V6']],
                freq_list=[freq_no_mod, freq_m6],
                title="Positional Consensus – ViennaRNA (no_mod vs m6A)",
                y_title="Consensus",
                out_html=str(out_dir / f"{base}_Vienna_no_mod_vs_m6A_consensus.html"),
                mods=mods
            )


# --- New plot: frequency of unpaired (".") positions ---
            plot_series_over_positions(
                seq=seq_no_mod,
                ys_list=[freq_no_mod['.'], freq_m6['.']],
                labels=["Vienna no_mod", "Vienna m6A"],
                colors=[ENSEMBLE_COLORS['V'], ENSEMBLE_COLORS['V6']],
                freq_list=[freq_no_mod, freq_m6],
                title="Positional ssRNA Frequency ViennaRNA (no_mod vs m6A)",
                y_title="Frequency of ssRNA (unpaired)",
                out_html=str(out_dir / f"{base}_Vienna_no_mod_vs_m6A_fraction_ssRNA.html"),
                mods=mods
            )



        if want_R:
            ent_no_mod,  cons_no_mod,  freq_no_mod  = positional_stats_with_freq(results[('R','no_mod')])
            ent_m6,  cons_m6,  freq_m6  = positional_stats_with_freq(results[('R','m6A')])

            plot_series_over_positions(
                seq=seq_no_mod,
                ys_list=[ent_no_mod, ent_m6],
                labels=["RNAstructure no_mod", "RNAstructure m6A"],
                colors=[ENSEMBLE_COLORS['R'], ENSEMBLE_COLORS['R6']],
                freq_list=[freq_no_mod, freq_m6],
                title="Positional Shannon Entropy – RNAstructure (no_mod vs m6A)",
                y_title="Entropy (shannon)",
                out_html=str(out_dir / f"{base}_RNAStructure_no_mod_vs_m6A_entropy.html"),
                mods=mods
            )
            plot_series_over_positions(
                seq=seq_no_mod,
                ys_list=[cons_no_mod, cons_m6],
                labels=["RNAstructure no_mod", "RNAstructure m6A"],
                colors=[ENSEMBLE_COLORS['R'], ENSEMBLE_COLORS['R6']],
                freq_list=[freq_no_mod, freq_m6],
                title="Positional Consensus – RNAstructure (no_mod vs m6A)",
                y_title="Consensus",
                out_html=str(out_dir / f"{base}_RNAStructure_no_mod_vs_m6A_consensus.html"),
                mods=mods
            )

# --- New plot: frequency of unpaired (".") positions ---
            plot_series_over_positions(
                seq=seq_no_mod,
                ys_list=[freq_no_mod['.'], freq_m6['.']],
                labels=["RNAstructure no_mod", "RNAstructure m6A"],
                colors=[ENSEMBLE_COLORS['R'], ENSEMBLE_COLORS['R6']],
                freq_list=[freq_no_mod, freq_m6],
                title="Positional ssRNA Frequency RNAstructure (no_mod vs m6A)",
                y_title="Frequency of ssRNA (unpaired)",
                out_html=str(out_dir / f"{base}_RNAStructure_no_mod_vs_m6A_fraction_ssRNA.html"),
                mods=mods
            )



        # Cross-tool on m6A sequence (character breakdown shown for each tool)
        if want_V and want_R:
            ent_V_m6, cons_V_m6, freq_V_m6 = positional_stats_with_freq(results[('V','m6A')])
            ent_R_m6, cons_R_m6, freq_R_m6 = positional_stats_with_freq(results[('R','m6A')])

            plot_series_over_positions(
                seq=seq_m6a,
                ys_list=[ent_V_m6, ent_R_m6],
                labels=["Vienna m6A", "RNAstructure m6A"],
                colors=[ENSEMBLE_COLORS['V6'], ENSEMBLE_COLORS['R6']],
                freq_list=[freq_V_m6, freq_R_m6],
                title="Positional Shannon Entropy – m6A (Vienna vs RNAstructure)",
                y_title="Entropy (shannon)",
                out_html=str(out_dir / f"{base}_m6A_crossTool_entropy.html"),
                mods=mods
            )
            plot_series_over_positions(
                seq=seq_m6a,
                ys_list=[cons_V_m6, cons_R_m6],
                labels=["Vienna m6A", "RNAstructure m6A"],
                colors=[ENSEMBLE_COLORS['V6'], ENSEMBLE_COLORS['R6']],
                freq_list=[freq_V_m6, freq_R_m6],
                title="Positional Consensus – m6A (Vienna vs RNAstructure)",
                y_title="Consensus",
                out_html=str(out_dir / f"{base}_m6A_crossTool_consensus.html"),
                mods=mods
            )


            # --- New cross-tool plot: frequency of unpaired (".") positions ---
            plot_series_over_positions(
                seq=seq_m6a,
                ys_list=[freq_V_m6['.'], freq_R_m6['.']],
                labels=["Vienna m6A", "RNAstructure m6A"],
                colors=[ENSEMBLE_COLORS['V6'], ENSEMBLE_COLORS['R6']],
                freq_list=[freq_V_m6, freq_R_m6],
                title="Positional ssRNA freq (m6A Vienna vs m6A RNAstructure)",
                y_title="Frequency of ssRNA (unpaired)",
                out_html=str(out_dir / f"{base}_m6A_crossTool_fraction_ssRNA.html"),
                mods=mods
            )


        write_output_summary(out_dir, base, want_V, want_R, M=M, sequence=args.sequence)
        print("Done. Outputs in:", os.path.abspath(out_dir))

    finally:
        shutil.rmtree(tmpdir)

if __name__ == "__main__":
    main()
