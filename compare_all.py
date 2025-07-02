#!/usr/bin/env python3
"""
compare_all.py

Generate M structures from each of the four ensemble tools (EternaFold, RNAsubopt,
RNAstructure, LinearFold), then do an all-vs-all pairwise consensus scoring,
aggregate each structure’s total score, rank them, and report:

 1. CSV of all structures sorted by aggregate consensus (highest first)
 2. Top‐N structures CSV (if --top_n given)
 3. Positional consensus string & frequency
 4. Positional Shannon entropy plot (HTML)
"""
import argparse
import os
import tempfile
import shutil
from pathlib import Path
from ruamel.yaml import YAML
import math
from collections import Counter
from compare_utils import simple_similarity_score as consensus_score

import utils_v2  # must have consensus_score, extract_ensemble, RNASubopt, EternaFold, RNAStructure, LinearFold
import plotly.graph_objects as go

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
        utils_v2.RNASubopt(
            seq_file=seq, out_file=tmp_db,
            executable=exe,
            n_struc=M,
            method=params.get('method')
        )
    elif letter == 'E':
        utils_v2.EternaFold(
            seq_file=seq, out_file=tmp_db,
            mode="sample",
            executable=exe,
            eternaFold_params=params.get('eternaFold_params'),
            eternaFold_params_shape=params.get('eternaFold_params_shape'),
            nsamples=M
        )
    elif letter == 'R':
        utils_v2.RNAStructure(
            seq_file=seq, out_file=tmp_db,
            executable=exe,
            a=params.get('a'),
            maxm=params.get('maxm'),
        )
        # warn if fewer than M
        full = utils_v2.extract_ensemble(tmp_db)
        if len(full) < M:
            print(f"WARNING: RNAstructure produced only {len(full)} (requested {M})")
    elif letter == 'L':
        # iterative‐delta until >= M
        delta = params.get('delta', 5.0)
        while True:
            utils_v2.LinearFold(
                seq_file=seq, out_file=tmp_db,
                mode="ensemble",
                executable=exe,
                delta=delta
            )
            full = utils_v2.extract_ensemble(tmp_db)
            if len(full) >= M:
                break
            delta += 1.0
    else:
        raise ValueError(f"Unknown letter: {letter}")

    # extract & trim
    full = utils_v2.extract_ensemble(tmp_db)
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
                score = consensus_score(si, sj)
                agg[idi] += score

        # prepare ranking
        ranked = sorted(all_structs,
                        key=lambda e: agg[e['id']],
                        reverse=True)

        # write full ranking CSV
        full_csv = f"{base}_ranked.csv"
        with open(full_csv, 'w') as fh:
            fh.write("id,method,index,structure,agg_score\n")
            for e in ranked:
                fh.write(f"{e['id']},{e['letter']},{e['index']},"
                         f"{e['struct']},{agg[e['id']]:.5f}\n")
        print("Wrote full ranking to", full_csv)

        # if top_n, write top structures CSV
        if top_n:
            top_csv = f"{base}_top{top_n}.csv"
            with open(top_csv, 'w') as fh:
                fh.write("id,method,index,structure,agg_score\n")
                for e in ranked[:top_n]:
                    fh.write(f"{e['id']},{e['letter']},{e['index']},"
                             f"{e['struct']},{agg[e['id']]:.5f}\n")
            print(f"Wrote top {top_n} to", top_csv)

            # positional consensus & entropy across top_n
            top_structs = [e['struct'] for e in ranked[:top_n]]
            Lseq = len(seq)
            # consensus freq per position
            cons_scores = []
            entp        = []
            for i in range(Lseq):
                col = [s[i] for s in top_structs]
                counts = Counter(col)
                most, cnt = counts.most_common(1)[0]
                # consensus score = fraction of structures agreeing on the most common symbol
                cons_scores.append(cnt / len(top_structs))
                # entropy as before
                entp.append(shannon_math(col, unit="shannon"))

            # build an interactive Plotly chart for consensus score
            fig_cons = go.Figure(
                go.Scatter(
                    x=list(range(1, Lseq+1)),
                    y=cons_scores,
                    mode='lines+markers',
                    name='Positional consensus',
                    hovertext=[
                        f"pos={i}<br>nt={seq[i-1]}<br>cons={cons_scores[i-1]:.3f}"
                        for i in range(1, Lseq+1)
                    ],
                    hoverinfo='text',
                    line=dict(color='purple'),
                    marker=dict(color='purple')
                )
            )
            fig_cons.update_layout(
                title=f"Positional Consensus Score (top {top_n})",
                xaxis_title="Position",
                yaxis_title="Consensus Score"
            )

            cons_html = f"{base}_positional_consensus_all.html"
            fig_cons.write_html(cons_html, auto_open=False)
            print("Wrote positional consensus plot to", cons_html)

    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()
