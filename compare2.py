#!/usr/bin/env python3
"""
compare2.py

Generate N structures from two chosen ensemble tools (EternaFold, RNAsubopt,
RNAstructure, or LinearFold), then compute and visualize how they agree:

 1. Produce an interactive N×N Plotly heatmap (HTML) of all‐vs‐all pairwise
    consensus scores between the two ensembles.
 2. Compute and plot positional Shannon entropy (HTML), showing variability
    at each nucleotide position across the two sets.
 3. Compute and plot positional consensus score (HTML), the fraction of
    structures agreeing at each position.
 4. Optionally, extract the top-N best pairs of structures by average consensus and
    write them to a CSV.

Usage example:
    python compare2.py \
      -s myseq.fasta \
      -e1 E -e2 V \
      -n 20 \
      --top_n 5 \
      -c config.yaml \
"""
import argparse
import tempfile
import os
import shutil
from pathlib import Path

from collections import Counter

from ruamel.yaml import YAML
import utils

letter_map = {'E': 'EternaFold',
                'V': 'RNASubopt',
                'R': 'RNAStructure',
                'L': 'LinearFold'}

ENSEMBLE_COLORS = {
    'E': 'blue',
    'V': 'green',
    'R': 'red',
    'L': 'orange',
}

def load_config(path: str):
    yaml = YAML(typ="safe")
    with open(path) as fh:
        return yaml.load(fh)

def load_sequence(path_or_seq: str) -> str:
    """If `path_or_seq` is a file, read FASTA (ignore headers), else return as-is."""
    p = Path(path_or_seq)
    if p.exists():
        seq = []
        for line in p.read_text().splitlines():
            if line.startswith(">"): continue
            seq.append(line.strip())
        return "".join(seq)
    else:
        return path_or_seq.strip()

def generate_ensemble(letter: str, seq: str, out_db: str, ens_n: int, cfg: dict):
    """
    Generate an ensemble of structures using method `letter` (E, V, R, L),
    writing to out_db. Uses the executable & params from cfg.
    """
    tool_name = letter_map[letter]
    tool_cfg  = cfg[tool_name]
    exe        = tool_cfg['executable']
    params     = tool_cfg.get('params', {})

    # write raw results to a temp file
    tmp_db = out_db + ".tmp"

    if letter == 'V':  # RNAsubopt
        utils.RNASubopt(
            seq_file=seq,
            out_file=tmp_db,
            executable=exe,
            n_struc=ens_n,
            method=params.get('method')
        )

    elif letter == 'E':  # EternaFold
        utils.EternaFold(
            seq_file=seq,
            out_file=tmp_db,
            mode="sample",
            executable=exe,
            eternaFold_params=params.get('eternaFold_params'),
            eternaFold_params_shape=params.get('eternaFold_params_shape'),
            nsamples=ens_n 
        )

    elif letter == 'R':  # RNAstructure → warning if < N
        utils.RNAStructure(
            seq_file=seq,
            out_file=tmp_db,
            executable=exe,
            a=params.get('a'),
            maxm=params.get('maxm')
        )
        full_list = utils.extract_ensemble(tmp_db)
        if len(full_list) < ens_n:
            print(f"WARNING: RNAStructure produced only {len(full_list)} structures (requested {ens_n})")
        trimmed = full_list[:ens_n]
        with open(out_db, 'w') as fh:
            fh.write("\n".join(trimmed) + "\n")
        return trimmed

    elif letter == 'L':  # LinearFold → iterative‐delta
        desired       = ens_n
        current_delta = params.get('delta', 5.0)
        while True:
            utils.LinearFold(
                seq_file=seq,
                out_file=tmp_db,
                mode="ensemble",
                executable=exe,
                delta=current_delta
            )
            full_list = utils.extract_ensemble(tmp_db)
            if len(full_list) >= desired:
                break
            current_delta += 1.0

        trimmed = full_list[:desired]
        with open(out_db, 'w') as fh:
            fh.write("\n".join(trimmed) + "\n")
        return trimmed

    else:
        raise ValueError(f"Unknown ensemble letter: {letter}")

    # Extract the dot-bracket lines and trim to top_n
    full_list = utils.extract_ensemble(tmp_db)
    trimmed   = full_list[:ens_n]
    with open(out_db, 'w') as fh:
        fh.write("\n".join(trimmed) + "\n")
    return trimmed

def main():
    p = argparse.ArgumentParser(
        description="Consensus heatmap between two ensemble methods"
    )
    p.add_argument("-s","--sequence",  required=True,
                help="Path to a FASTA (or raw sequence)")
    p.add_argument("-e1","--ensembler1",   required=True, choices=['E','V','R','L'],
                help="First ensemble: E (EternaFold), V (RNAsubopt), R (RNAstructure), L (LinearFold)")
    p.add_argument("-e2","--ensembler2",   required=True, choices=['E','V','R','L'],
                help="Second ensemble method")
    p.add_argument("-n","--ens_n",      type=int,
                help="Number of structures to sample from each ensemble (defaults to global_ensemble_size)")
    p.add_argument("--top_n",           type=int,
                help="Number of best structures (highest average consensus) to return in a separate file")
    p.add_argument("-c","--config",     default="config.yaml",
                help="YAML configuration file")
    args = p.parse_args()

    # ─── load config & env ────────────────────────────────────────────────────────
    CFG = load_config(args.config)
    env = CFG.get("environment", {})
    if env.get("data_tables"):
        os.environ["DATAPATH"] = os.path.abspath(env["data_tables"])
    if env.get("threads") is not None:
        os.environ["OMP_NUM_THREADS"] = str(env["threads"])

    # ─── determine ensemble‐size (ens_n) and best‐structures count (top_n) ─────────
    ens_n = args.ens_n if args.ens_n is not None else CFG.get("global_ensemble_size", 20)
    top_n = args.top_n  # may be None

    seq = load_sequence(args.sequence)
    e1, e2 = args.ensembler1, args.ensembler2

    base = Path(args.sequence).stem

    tmpdir = Path(tempfile.mkdtemp(prefix="enscmp_"))
    try:
        out1 = str(tmpdir / f"ens1_{e1}.db")
        out2 = str(tmpdir / f"ens2_{e2}.db")

        structs1 = generate_ensemble(e1, args.sequence, out1, ens_n, CFG)
        structs2 = generate_ensemble(e2, args.sequence, out2, ens_n, CFG)

        # --- build consensus matrix & hover text ---
        import plotly.graph_objects as go
        z, text = [], []
        rows = [f"{e1}{ens_n-i}" for i in range(ens_n)]
        cols = [f"{e2}{j+1}" for j in range(ens_n)]
        pair_scores = []  

        for i, s1 in enumerate(structs1):
            zrow, trow = [], []
            for j, s2 in enumerate(structs2):
                score = utils.simple_similarity_score(s1, s2)
                zrow.append(score)
                pair_scores.append((rows[i], cols[j], s1, s2, score))
                hover = (
                    f"consensus_score = {score:.3f}<br>"
                    f"seq = {seq[:70]}{'/' if len(seq)>70 else ''}<br>"
                    f"{rows[i]}{' ' if len(rows[i])>2 else '  '}– {s1[:70]}{'/' if len(s1)>70 else ''}<br>"
                    f"{cols[j]}{' ' if len(cols[j])>2 else '  '}– {s2[:70]}{'/' if len(s2)>70 else ''}"
                )
                trow.append(hover)
            z.append(zrow)
            text.append(trow)

        fig = go.Figure(data=go.Heatmap(
            z=z, x=cols, y=rows,
            hoverinfo="text", text=text,
            colorbar=dict(title="Consensus"),
            hoverlabel = dict(
            font=dict(
                family="Courier New",
                color="black",
                size=14
            ),
            bgcolor="white"
        ),
        #font to monospace
        colorscale='Turbo',
            textfont=dict(family="Courier New", size=12)
        ))
        fig.update_layout(
            title=f"Consensus heatmap: {letter_map[e1]} vs {letter_map[e2]} (top {ens_n})",
            xaxis_title=f"{e2} structures",
            yaxis_title=f"{e1} structures"
            # ensure square cells
        )
        fig.update_xaxes(tickangle=-45, tickmode='array', tickvals=cols)
        fig.update_yaxes(tickmode='array', tickvals=rows)
        fig.update_layout(
            width=1000, height=1000,
            margin=dict(l=50, r=50, t=50, b=50),
            autosize=False
        )
        fig.update_layout(font_family="Courier New")

        fig.write_html(f"{base}_compare2_heatmap.html", auto_open=False)
        print("Wrote heatmap to", f"{base}_compare2_heatmap.html")

        # ─── Positional entropy & consensus ───────────────────────────────────────────
        # Compute global entropy for each ensemble
        size1 = len(structs1[0])
        entropy_list1 = [
            utils.shannon_math([s[i] for s in structs1], unit="shannon")
            for i in range(size1)
        ]
        global_ent1 = sum(entropy_list1) / size1

        # Ensemble 2 average positional entropy
        size2 = len(structs2[0])
        entropy_list2 = [
            utils.shannon_math([s[i] for s in structs2], unit="shannon")
            for i in range(size2)
        ]
        global_ent2 = sum(entropy_list2) / size2

        print(
            f"Average positional Shannon entropy: "
            f"{letter_map[e1]} = {global_ent1:.3f}, "
            f"{letter_map[e2]} = {global_ent2:.3f}"
        )

        # Compute per-position entropy and consensus
        seq_len = len(seq)
        pos_ent1 = []
        pos_ent2 = []
        pos_cons1 = []
        pos_cons2 = []

        for i in range(seq_len):
            col1 = [s[i] for s in structs1]
            col2 = [s[i] for s in structs2]
            # entropy
            pos_ent1.append(utils.shannon_math(col1, unit="shannon"))
            pos_ent2.append(utils.shannon_math(col2, unit="shannon"))
            # consensus = fraction of the most common symbol
            c1 = Counter(col1)
            c2 = Counter(col2)
            pos_cons1.append(max(c1.values()) / len(col1))
            pos_cons2.append(max(c2.values()) / len(col2))

        # Split long sequences into chunks of ~50 positions per row
        from math import ceil
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        chunk_size = 50
        nrows = ceil(seq_len / chunk_size)

 # Positional entropy plot
        fig_e = make_subplots(
            rows=nrows, cols=1,
            shared_xaxes=False,
            subplot_titles=[
                f"Positions {r*chunk_size+1}-{min((r+1)*chunk_size, seq_len)}"
                for r in range(nrows)
            ]
        )
        for r in range(nrows):
            start = r * chunk_size
            end   = min(start + chunk_size, seq_len)
            xs    = list(range(start+1, end+1))

            # Ensemble 1 entropy
            fig_e.add_trace(
                go.Scatter(
                    x=xs,
                    y=pos_ent1[start:end],
                    mode='lines+markers',
                    name=f"{letter_map[e1]} entropy",
                    hovertext=[
                        f"pos={i}<br>nt={seq[i-1]}<br>H={pos_ent1[i-1]:.3f}"
                        for i in xs
                    ],
                    hoverlabel = dict(
                        font=dict(
                            family="Courier New",
                            size=14)),
                    line=dict(color=ENSEMBLE_COLORS[e1]),
                    marker=dict(color=ENSEMBLE_COLORS[e1])
                ),
                row=r+1, col=1
            )
            # Ensemble 2 entropy
            fig_e.add_trace(
                go.Scatter(
                    x=xs,
                    y=pos_ent2[start:end],
                    mode='lines+markers',
                    name=f"{letter_map[e2]} entropy",
                    hovertext=[
                        f"pos={i}<br>nt={seq[i-1]}<br>H={pos_ent2[i-1]:.3f}"
                        for i in xs
                    ],
                    hoverlabel = dict(
                        font=dict(
                            family="Courier New",
                            size=14)),
                    line=dict(color=ENSEMBLE_COLORS[e2]),
                    marker=dict(color=ENSEMBLE_COLORS[e2])
                ),
                row=r+1, col=1
            )
        fig_e.update_layout(font_family="Courier New")    

        fig_e.update_layout(
            title_text=(
                f"Positional Shannon Entropy "
                f"(global: {letter_map[e1]}={global_ent1:.3f}, {letter_map[e2]}={global_ent2:.3f})"
            )
        )
        fig_e.write_html(f"{base}_compare2_positional_entropy.html", auto_open=False)
        print("Wrote positional entropy plot to", f"{base}_compare2_positional_entropy.html")


        # Positional consensus plot
        fig_c = make_subplots(
            rows=nrows, cols=1,
            shared_xaxes=False,
            subplot_titles=[
                f"Positions {r*chunk_size+1}-{min((r+1)*chunk_size, seq_len)}"
                for r in range(nrows)
            ]
        )
        for r in range(nrows):
            start = r * chunk_size
            end   = min(start + chunk_size, seq_len)
            xs    = list(range(start+1, end+1))

            # Ensemble 1 consensus
            fig_c.add_trace(
                go.Scatter(
                    x=xs,
                    y=pos_cons1[start:end],
                    mode='lines+markers',
                    name=f"{letter_map[e1]} consensus",
                    hovertext=[
                        f"pos={i}<br>nt={seq[i-1]}<br>cons={pos_cons1[i-1]:.3f}"
                        for i in xs
                    ],
                    line=dict(color=ENSEMBLE_COLORS[e1]),
                    marker=dict(color=ENSEMBLE_COLORS[e1])
                ),
                row=r+1, col=1
            )
            # Ensemble 2 consensus
            fig_c.add_trace(
                go.Scatter(
                    x=xs,
                    y=pos_cons2[start:end],
                    mode='lines+markers',
                    name=f"{letter_map[e2]} consensus",
                    hovertext=[
                        f"pos={i}<br>nt={seq[i-1]}<br>cons={pos_cons2[i-1]:.3f}"
                        for i in xs
                    ],
                    line=dict(color=ENSEMBLE_COLORS[e2]),
                    marker=dict(color=ENSEMBLE_COLORS[e2])
                ),
                row=r+1, col=1
            )
        fig_c.update_layout(font_family="Courier New")
        fig_c.update_layout(title_text="Positional Consensus Score")
        fig_c.write_html(f"{base}_compare2_positional_consensus.html", auto_open=False)
        print("Wrote positional consensus plot to", f"{base}_compare2_positional_consensus.html")


        if top_n:
            # Sort all pairs by consensus score, descending
            pair_scores.sort(key=lambda x: x[4], reverse=True)

            # Take top_n best pairs (no duplicates)
            best_pairs = pair_scores[:min(top_n, len(pair_scores))]
            
            # Write to CSV
            best_file = base + "_compare2_best_pairs.csv"
            with open(best_file, 'w') as fh:
                fh.write(f"{e1}_index,{e2}_index,{e1}_structure,{e2}_structure,consensus_score\n")
                for i, j, s1, s2, score in best_pairs:
                    fh.write(f"{i},{j},{s1},{s2},{score:.3f}\n")

            print(f"Wrote top {len(best_pairs)} best structure pairs to {best_file}")


    finally:
        shutil.rmtree(tmpdir)

    

if __name__ == "__main__":
    main()
    
