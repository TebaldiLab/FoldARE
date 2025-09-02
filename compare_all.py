#!/usr/bin/env python3
# make and aggregate score for TOP N structures
"""
compare_all.py

Generate M structures from each of the four ensemble tools (EternaFold, RNAsubopt,
RNAstructure, LinearFold), then do an all-vs-all pairwise consensus scoring,
aggregate each structure’s total score, rank them, and report:

 1. CSV of all structures sorted by aggregate consensus (highest first)
 2. Top‐N structures CSV (if --top_n given)
 3. Positional consensus score plot (HTML)
 4. Positional Shannon entropy plot (HTML)

 Usage example:
    python compare_all.py \
      -s myseq.fasta \
      -n 20 \
      --top_n 5 \
      -c config.yaml \
"""
import argparse
import os
import tempfile
import shutil
from pathlib import Path
from ruamel.yaml import YAML
import math
from collections import Counter
import utils
import plotly.graph_objects as go
from math import ceil
from plotly.subplots import make_subplots

# ─── Letter‐to‐tool & colors ───────────────────────────────────────────────────
LETTER_MAP = {
    'E': "EternaFold",
    'V': "RNASubopt",
    'R': "RNAStructure",
    'L': "LinearFold",
}

ENSEMBLE_COLORS = {
    'E': 'blue',
    'V': 'green',
    'R': 'red',
    'L': 'orange',
}

# ─── Shannon entropy helper ────────────────────────────────────────────────────
def shannon_math(data, unit="shannon"):
    base = {'shannon': 2.0, 'natural': math.e, 'hartley': 10.0}
    if len(data) <= 1:
        return 0.0
    counts = Counter(data)
    entropy = 0.0
    for count in counts.values():
        p = count / len(data)
        if p > 0.0:
            entropy -= p * math.log(p, base[unit])
    return entropy

# ─── load config & sequence ────────────────────────────────────────────────────
def load_config(path):
    yaml = YAML(typ="safe")
    return yaml.load(Path(path).read_text())

def load_sequence(path_or_seq):
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

# ─── generate M‐member ensemble for a given letter ─────────────────────────────
def generate_ensemble(letter, seq, out_db, M, cfg):
    tool = LETTER_MAP[letter]
    tool_cfg = cfg[tool]
    exe      = tool_cfg['executable']
    params   = tool_cfg.get('params', {})
    tmp_db   = out_db + ".tmp"

    # call the appropriate wrapper
    if letter == 'V':
        utils.RNASubopt(
            seq_file=seq, out_file=tmp_db,
            executable=exe,
            n_struc=M,
            method=params.get('method')
        )
    elif letter == 'E':
        utils.EternaFold(
            seq_file=seq, out_file=tmp_db,
            mode="sample",
            executable=exe,
            eternaFold_params=params.get('eternaFold_params'),
            eternaFold_params_shape=params.get('eternaFold_params_shape'),
            nsamples=M
        )
    elif letter == 'R':
        utils.RNAStructure(
            seq_file=seq, out_file=tmp_db,
            executable=exe,
            a=params.get('a'),
            maxm=params.get('maxm'),
        )
        # warn if fewer than M
        full = utils.extract_ensemble(tmp_db)
        if len(full) < M:
            print(f"WARNING: RNAstructure produced only {len(full)} (requested {M})")
    elif letter == 'L':
        # iterative‐delta until >= M
        delta = params.get('delta', 5.0)
        while True:
            utils.LinearFold(
                seq_file=seq, out_file=tmp_db,
                mode="ensemble",
                executable=exe,
                delta=delta
            )
            full = utils.extract_ensemble(tmp_db)
            if len(full) >= M:
                break
            delta += 1.0
    else:
        raise ValueError(f"Unknown letter: {letter}")

    # extract & trim
    full = utils.extract_ensemble(tmp_db)
    trimmed = full[:M]
    with open(out_db, 'w') as fh:
        fh.write("\n".join(trimmed) + "\n")
    return trimmed

# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-s","--sequence", required=True,
                   help="FASTA path or raw sequence")
    p.add_argument("-n","--ens_n", type=int,
                   help="Number of structures per ensemble (default from config)")
    p.add_argument("--top_n", type=int,
                   help="Number of top structures to output and analyze")
    p.add_argument("-c","--config", default="config.yaml",
                   help="YAML configuration file")
    args = p.parse_args()

    # load config & env
    cfg = load_config(args.config)
    env = cfg.get("environment", {})
    if env.get("data_tables"):
        os.environ["DATAPATH"] = str(Path(env["data_tables"]).absolute())
    if env.get("threads") is not None:
        os.environ["OMP_NUM_THREADS"] = str(env["threads"])

    seq = load_sequence(args.sequence)

    base = Path(args.sequence).stem

    M   = args.ens_n if args.ens_n is not None else cfg.get("global_ensemble_size", 20)
    top_n = args.top_n

    # generate ensembles E,V,R,L
    tmpdir = Path(tempfile.mkdtemp(prefix="cmpall_"))
    try:
        all_structs = []  # list of dicts: {id, letter, index, struct}
        for L in ['E','V','R','L']:
            out_db = str(tmpdir / f"{L}.db")
            members = generate_ensemble(L, args.sequence, out_db, M, cfg)
            for idx, s in enumerate(members, start=1):
                all_structs.append({
                    'id': f"{L}{idx}",
                    'letter': L,
                    'index': idx,
                    'struct': s
                })

        N = len(all_structs)
        # compute pairwise consensus scores
        # initialize aggregate dict
        agg = { entry['id']: 0.0 for entry in all_structs }
        for i in range(N):
            si = all_structs[i]['struct']
            idi= all_structs[i]['id']
            for j in range(N):
                if i == j: 
                    continue
                sj = all_structs[j]['struct']
                score = utils.simple_similarity_score(si, sj)
                agg[idi] += score

        # prepare ranking
        ranked = sorted(all_structs,
                        key=lambda e: agg[e['id']],
                        reverse=True)

        # write full ranking CSV
        full_csv = f"{base}_compare_all_ranked.csv"
        with open(full_csv, 'w') as fh:
            fh.write("id,method,index,structure,agg_score\n")
            for e in ranked:
                fh.write(f"{e['id'][0] + '0' + e['id'][1:] if int(e['id'][1:]) < 10 else e['id']}, {e['letter']},{'0' + str(e['index']) if e['index'] < 10 else e['index']},"
                         f"{e['struct']},{agg[e['id']]:.5f}\n")
        print("Wrote full ranking to", full_csv)

        # if top_n, write top structures CSV
        if top_n:
            top_csv = f"{base}_compare_all_top{top_n}.csv"
            with open(top_csv, 'w') as fh:
                fh.write("id,method,index,structure,agg_score\n")
                for e in ranked[:top_n]:
                    fh.write(f"{e['id'][0] + '0' + e['id'][1:] if int(e['id'][1:]) < 10 else e['id']}, {e['letter']},{'0' + str(e['index']) if e['index'] < 10 else e['index']},"
                         f"{e['struct']},{agg[e['id']]:.5f}\n")
            print(f"Wrote top {top_n} to", top_csv)

            # positional consensus & entropy across top_n
            top_structs = [e['struct'] for e in ranked[:top_n]]
            all_structs = [e['struct'] for e in ranked]
            Lseq = len(seq)
            # consensus freq per position
            cons_scores = []
            entp = []
            freq = {".": [], "(": [], ")": []}
            for i in range(Lseq):
                col = [s[i] for s in top_structs]
                counts = Counter(col)
                most, cnt = counts.most_common(1)[0]
                # consensus score = fraction of structures agreeing on the most common symbol
                cons_scores.append(cnt / len(top_structs))
                # entropy as before
                entp.append(shannon_math(col, unit="shannon"))
                # frequency of all symbols
                for sym in freq:
                    freq[sym].append(counts.get(sym, 0) / len(top_structs))

            entp_all = []
            cons_scores_all = []
            freq_all = {".": [], "(": [], ")": []}
            for i in range(Lseq):
                col = [s[i] for s in all_structs]
                counts = Counter(col)
                most, cnt = counts.most_common(1)[0]
                # consensus score = fraction of structures agreeing on the most common symbol
                cons_scores_all.append(cnt / len(all_structs))
                # entropy as before
                entp_all.append(shannon_math(col, unit="shannon"))
                # frequency of all symbols
                for sym in freq_all:
                    freq_all[sym].append(counts.get(sym, 0) / len(all_structs))

            # Plotly chart for consensus score

            nrows = 4
            chunk_size = ceil(Lseq / nrows)
            min_px_per_base = 9
            row_width_px = max(chunk_size * min_px_per_base, 1000)

            fig_cons = make_subplots(
            rows=nrows, cols=1,
            shared_xaxes=False,
            subplot_titles=[
                f"Positions {r*chunk_size+1}-{min((r+1)*chunk_size, Lseq)}"
                for r in range(nrows)
            ]
        )

            for r in range(nrows):
                start = r * chunk_size
                end   = min(start + chunk_size, Lseq)
                xs    = list(range(start+1, end+1))

                fig_cons.add_trace(
                    go.Scatter(
                        x=xs, y=cons_scores_all[start:end],
                        mode='lines+markers',
                        name=f'Consensus (row {r+1})',
                        hovertext=[
                            f"pos={'0' + str(i) if i < 10 else i}<br>nt={seq[i-1]}<br>cons={cons_scores_all[i-1]:.3f}<br>(={freq_all['('][i-1]:.3f} )={freq_all[')'][i-1]:.3f} .={freq_all['.'][i-1]:.3f}"
                            for i in xs
                        ],
                        hoverlabel=dict(font=dict(family="Courier New", size=14)),
                        line=dict(color='green'),
                        marker=dict(color='green')
                    ),
                    row=r+1, col=1
                )
                fig_cons.update_xaxes(range=[start - 0.5, end + 0.5], row=r + 1, col=1)

            fig_cons.update_layout(
                title="Positional Consensus Score",
                xaxis_title="Position",
                yaxis_title="Consensus Score",
                yaxis=dict(range=[-0.05, 1.05]),
                font_family="Courier New",
                height= max(700, 235 * nrows),
                width=max(row_width_px, 1000),
                margin=dict(l=50, r=20, t=140, b=50),
                legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0)
            )
            cons_html = f"{base}_compare_all_positional_consensus.html"
            fig_cons.write_html(cons_html, auto_open=False)
            print("Wrote positional consensus plot to", cons_html)


            # Plotly chart for positional entropy
            fig_ent = make_subplots(
                rows=nrows, cols=1,
                shared_xaxes=False,
                subplot_titles=[
                    f"Positions {r*chunk_size+1}-{min((r+1)*chunk_size, Lseq)}"
                    for r in range(nrows)
                ]
            )

            for r in range(nrows):
                start = r * chunk_size
                end   = min(start + chunk_size, Lseq)
                xs    = list(range(start+1, end+1))

                fig_ent.add_trace(
                    go.Scatter(
                        x=xs, y=entp_all[start:end],
                        mode='lines+markers',
                        name=f'Entropy (row {r+1})',
                        hovertext=[
                            f"pos={'0' + str(i) if i < 10 else i}<br>nt={seq[i-1]}<br>H={entp_all[i-1]:.3f}<br>(={freq_all['('][i-1]:.3f} )={freq_all[')'][i-1]:.3f} .={freq_all['.'][i-1]:.3f}"
                            for i in xs
                        ],
                        hoverlabel=dict(font=dict(family="Courier New", size=14)),
                        line=dict(color='brown'),
                        marker=dict(color='brown')
                    ),
                    row=r+1, col=1
                )
                fig_ent.update_xaxes(range=[start - 0.5, end + 0.5], row=r + 1, col=1)

            fig_ent.update_layout(
                title="Positional Shannon Entropy",
                xaxis_title="Position",
                yaxis_title="Entropy (shannon)",
                yaxis=dict(range=[-0.05, max(entp_all)*1.05]),
                font_family="Courier New",
                height= max(700, 235 * nrows),
                width=max(row_width_px, 1000),
                margin=dict(l=50, r=20, t=140, b=50),
                legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0)
            )
            ent_html = f"{base}_compare_all_positional_entropy.html"
            fig_ent.write_html(ent_html, auto_open=False)
            print("Wrote positional entropy plot to", ent_html)


    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()
