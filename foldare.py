#!/usr/bin/env python3
"""
foldare.py

2D RNA structure prediction pipeline (ensemble -> SHAPE -> predictor).

Generate up to N RNA secondary structures (an "ensemble") for a single input sequence,
derive per-nucleotide SHAPE reactivities, and then run a final structure prediction.

Workflow:
 1. Ensemble generation (-e/--ensemble, one of E, V, R, L):
    - E (EternaFold): sample N structures
    - V (ViennaRNA/RNASubopt): enumerate N suboptimal structures
    - R (RNAstructure): output the MFE structure (warns if N>1)
    - L (LinearFold): start with Delta=5.0, iteratively increment Delta by 1.0 until >=N structures
    - V6 (ViennaRNA with m6A): as V, but with m6A support
    - R6 (RNAstructure with m6A): as R, but with m6
    * N is controlled by --ens_n or global_ensemble_size in config.yaml
    * The raw ensemble is cleaned and trimmed to exactly N dot-bracket entries

 2. SHAPE reactivity generation
    - Uses create_shape_file() from utils_v2 with thresholds & coefficients from config.yaml

 3. Final prediction (-p/--predictor, one of E, V, R, L):
    - E (EternaFold): converts SHAPE->BPSEQ then predicts
    - V (RNAFold/ViennaRNA): predicts directly with SHAPE input
    - R (RNAstructure): predicts directly with SHAPE input
    - L (LinearFold): predicts directly with SHAPE input
    - V6 (ViennaRNA with m6A): as V, but with m6A support
    - R6 (RNAstructure with m6A): as R, but with m6A support
    * Uses SHAPE file from previous step, or --shape if provided

Outputs (in <output_folder>):
  * <basename>_<ensemble>_ens.db          — N dot-bracket structures
  * <basename>_shape.txt                  — per-nucleotide SHAPE reactivities
  * <basename>_<ensemble>_<predictor>_final.db — final predicted structure
"""
import argparse
import os
from datetime import datetime
from inspect import signature
from pathlib import Path

from ruamel.yaml import YAML

from utils import (
    EternaFold,
    LinearFold,
    RNAFold,
    RNAStructure,
    RNASubopt,
    clean_ensemble_file,
    convert_shape_to_bpseq,
    create_shape_file,    
    create_shape_file_LinFold,
)

LETTER_MAP = {
    "E": "EternaFold",
    "V": "ViennaRNA",
    "R": "RNAStructure",
    "L": "LinearFold",
    "V6": "ViennaRNA_m6A",
    "R6": "RNAStructure_m6A",
}


# -----------------------------------------------------------------------------
# Helper utilities -------------------------------------------------------------
# -----------------------------------------------------------------------------
def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Sequence-level RNA prediction pipeline (ensemble -> SHAPE -> predictor)"
    )
    parser.add_argument(
        "-p",
        "--predictor",
        choices=list(LETTER_MAP.keys()) + list(LETTER_MAP.values()),
        default="RNAStructure",
        help="Tool used for final prediction (E, V, R, L, R6, V6)",
    )
    parser.add_argument(
        "-e",
        "--ensemble",
        choices=list(LETTER_MAP.keys()) + list(LETTER_MAP.values()),
        default="EternaFold",
        help="Ensemble tool (E, V, R, L, R6, V6)",
    )
    parser.add_argument("-s", "--sequence", required=True, help="Input FASTA file")
    parser.add_argument("-o", "--output_folder", default="foldare_results", help="Output folder")
    parser.add_argument("-c", "--config", default="config.yaml", help="YAML configuration file")
    parser.add_argument("--shape", help="Existing SHAPE file – skip ensemble step")
    parser.add_argument("--ens_n", type=int, help="Target ensemble size for all tools")
    #parser.add_argument("--ens_nsamples", type=int)
    #parser.add_argument("--ens_maxm", type=int)
    #parser.add_argument("--ens_par", type=int)
    parser.add_argument("--ens_delta", type=float)
    return parser.parse_args()


def load_config(path: str):
    yaml = YAML(typ="safe")
    with open(path) as fh:
        return yaml.load(fh)


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def select(cfg_dict, cli_val, key):
    """Pick value: config first, overridden by CLI if non-None."""
    return cli_val if cli_val is not None else cfg_dict.get(key)


def filter_kwargs(func, mapping):
    """Drop keys not accepted by func and None-valued entries."""
    sig = signature(func).parameters
    return {k: v for k, v in mapping.items() if k in sig and v is not None}


def clean_in_place(db_path: str):
    """Run clean_ensemble_file safely without input=output overwrite."""
    tmp_path = db_path + ".clean"
    clean_ensemble_file(db_path, tmp_path)
    os.replace(tmp_path, db_path)


def normalize_tool_choice(raw_value: str, role: str):
    """
    Map short flags (E/V/R/L/V6/R6) to full tool names and detect m6A usage.

    To preserve behavior we only normalize when the provided value is a short flag
    (len <= 2), matching the original implementation.
    """
    m6a = False
    resolved = raw_value

    if len(raw_value) <= 2:
        mapped = LETTER_MAP[raw_value]
        if mapped == "ViennaRNA":
            resolved = "RNASubopt" if role == "ensemble" else "RNAFold"
        elif mapped == "ViennaRNA_m6A":
            resolved = "RNASubopt" if role == "ensemble" else "RNAFold"
            m6a = True
        elif mapped == "RNAStructure_m6A":
            resolved = "RNAStructure"
            m6a = True
        else:
            resolved = mapped

    return resolved, m6a


def apply_environment(env_cfg: dict):
    root_dir = os.getcwd() + "/"
    if "data_tables" in env_cfg:
        os.environ["DATAPATH"] = os.path.abspath(os.path.join(root_dir, env_cfg["data_tables"]))
    if env_cfg.get("threads") is not None:
        os.environ["OMP_NUM_THREADS"] = str(env_cfg["threads"])
        print("Using", os.environ["OMP_NUM_THREADS"], "threads for parallel execution.")


def write_ensemblefold_summary(
    out_dir: str,
    base_name: str,
    sequence_path: str,
    ensemble_tool: str,
    predictor_tool: str,
    ens_m6a: bool,
    pred_m6a: bool,
    ensemble_target: int | None,
    nsamples: int | None,
    maxm: int | None,
    par: int | None,
    delta: float | None,
    shape_source: str = "generated from ensemble",
):
    """
    Create <base_name>_foldare_summary.txt describing inputs, params, and outputs.
    Only lists files that actually exist in out_dir.
    """
    out_path = os.path.join(out_dir, f"{base_name}_foldare_summary.txt")

    def _exists(fname: str) -> bool:
        return bool(fname) and os.path.exists(fname)

    ens_db_candidates = [
        os.path.join(out_dir, f"{base_name}_{ensemble_tool}_ens.db"),
        os.path.join(out_dir, f"{base_name}_{ensemble_tool}_ens_m6A.db"),
    ]
    shape_txt = os.path.join(out_dir, f"{base_name}_shape.txt")
    bpseq = os.path.join(out_dir, f"{base_name}_shape.bpseq")

    generated = []
    for path in ens_db_candidates:
        if _exists(path):
            generated.append((os.path.basename(path), "Ensemble of sampled/suboptimal structures"))
    if _exists(shape_txt):
        generated.append((os.path.basename(shape_txt), "Per-nucleotide SHAPE reactivities used for prediction"))
    if _exists(bpseq):
        generated.append((os.path.basename(bpseq), "BPSEQ converted from SHAPE (EternaFold predictor)"))

    for fname in os.listdir(out_dir):
        if fname.startswith(base_name) and fname.endswith("_final.db"):
            generated.append((fname, "Final predicted structure (dot-bracket)"))

    lines = []
    lines.append("FoldARE – Output Summary\n")
    lines.append("=" * 60 + "\n\n")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    lines.append(f"Sequence file:  {os.path.abspath(sequence_path)}\n")
    lines.append(f"Output folder:  {os.path.abspath(out_dir)}\n\n")

    lines.append("Tools\n")
    lines.append(f"  Ensemble tool : {ensemble_tool} {'(m6A)' if ens_m6a else ''}\n")
    lines.append(f"  Predictor tool: {predictor_tool} {'(m6A)' if pred_m6a else ''}\n\n")

    lines.append("Key parameters\n")
    lines.append(f"  global_ensemble_size (target): {ensemble_target}\n")
    #lines.append(f"  nsamples/n_struc (effective)  : {nsamples}\n")
    #if maxm is not None:
    #    lines.append(f"  maxm (RNAStructure)           : {maxm}\n")
    #if par is not None:
    #    lines.append(f"  par  (RNAsubopt)              : {par}\n")
    if delta is not None:
        lines.append(f"  delta (LinearFold)            : {delta}\n")
    lines.append(f"  SHAPE input source            : {shape_source}\n")

    if generated:
        lines.append("Generated files\n")
        for fn, desc in generated:
            lines.append(f"  - {fn:<45} : {desc}\n")
        lines.append("\n")
    else:
        lines.append("Generated files: (none detected)\n\n")

    with open(out_path, "w") as fh:
        fh.writelines(lines)

    print(f"Summary file written: {out_path}")


# -----------------------------------------------------------------------------
# Core steps ------------------------------------------------------------------
# -----------------------------------------------------------------------------
def build_ensemble(
    etool: str,
    base_name: str,
    args,
    ensemble_target: int | None,
    nsamples: int | None,
    maxm: int | None,
    delta: float,
    ens_params_cfg: dict,
    ens_exec: str | None,
    ens_m6a: bool,
):
    ens_db = os.path.join(args.output_folder, f"{base_name}_{etool}_ens.db")
    if ens_m6a:
        ens_db = ens_db.replace(".db", "_m6A.db")
    etool_lower = etool.lower()

    if etool_lower == "rnasubopt":
        desired = ensemble_target or nsamples or 1
        tmp_db = ens_db + ".tmp"
        RNASubopt(
            **filter_kwargs(
                RNASubopt,
                {
                    "seq_file": args.sequence,
                    "out_file": tmp_db,
                    "n_struc": nsamples,
                    "method": ens_params_cfg.get("method"),
                    "executable": ens_exec,
                    **ens_params_cfg,
                },
            )
        )
        clean_in_place(tmp_db)
        with open(tmp_db) as fh:
            lines = fh.readlines()
        if len(lines) > desired:
            with open(tmp_db, "w") as fh:
                fh.writelines(lines[:desired])
        os.replace(tmp_db, ens_db)

    elif etool_lower == "eternafold":
        EternaFold(
            **filter_kwargs(
                EternaFold,
                {
                    "seq_file": args.sequence,
                    "out_file": ens_db,
                    "mode": "sample",
                    "nsamples": nsamples,
                    "executable": ens_exec,
                    **ens_params_cfg,
                },
            )
        )
        clean_in_place(ens_db)

    elif etool_lower == "linearfold":
        desired = ensemble_target or nsamples or 1
        current_delta = delta
        tmp_db = ens_db + ".tmp"
        while True:
            LinearFold(
                **filter_kwargs(
                    LinearFold,
                    {
                        "seq_file": args.sequence,
                        "out_file": tmp_db,
                        "mode": "ensemble",
                        "delta": current_delta,
                        "executable": ens_exec,
                    },
                )
            )
            clean_in_place(tmp_db)
            with open(tmp_db) as fh:
                lines = fh.readlines()
            if len(lines) >= desired:
                break
            current_delta += 1.0
        if len(lines) > desired:
            with open(tmp_db, "w") as fh:
                fh.writelines(lines[:desired])
        os.replace(tmp_db, ens_db)

    elif etool_lower == "rnastructure":
        RNAStructure(
            **filter_kwargs(
                RNAStructure,
                {
                    "seq_file": args.sequence,
                    "out_file": ens_db,
                    "executable": ens_exec,
                    "maxm": maxm,
                    **ens_params_cfg,
                },
            )
        )
        clean_in_place(ens_db)

    else:
        raise ValueError(f"Unsupported ensemble tool: {etool}")

    if etool_lower not in ["linearfold", "rnastructure"] and ensemble_target is not None:
        with open(ens_db) as fh:
            entries = fh.readlines()
        if len(entries) > ensemble_target:
            with open(ens_db, "w") as fh:
                fh.writelines(entries[:ensemble_target])

    return ens_db


def build_shape_file(
    ensemble_db: str,
    predictor_choice: str,
    base_name: str,
    output_folder: str,
    shape_cfg: dict,
):
    shape_file = os.path.join(output_folder, f"{base_name}_shape.txt")
    if predictor_choice=="LinearFold":
        create_shape_file_LinFold(
            ensemble_db,
            shape_file,
            thresholds=shape_cfg.get("thresholds", {}),
            coefficients=shape_cfg.get("coefficients", {}),
            linearfold_style=(predictor_choice.lower() == "linearfold"),
        )
    else:
        create_shape_file(
            ensemble_db,
            shape_file,
            thresholds=shape_cfg.get("thresholds", {}),
            coefficients=shape_cfg.get("coefficients", {}),
            linearfold_style=(predictor_choice.lower() == "linearfold"),
        )
    return shape_file


def predict_structure(
    predictor_choice: str,
    args,
    base_name: str,
    shape_file: str,
    ens_tool: str,
    ens_m6a: bool,
    pred_m6a: bool,
    pred_exec: str | None,
    pred_params_cfg: dict,
):
    ptool = predictor_choice.lower()
    if args.shape is None:
        pred_out = os.path.join(args.output_folder, f"{base_name}_{ens_tool.lower()}_{ptool}_final.db")
        if ens_m6a:
            pred_out = os.path.join(
                args.output_folder, f"{base_name}_{ens_tool.lower()}_m6A_{ptool}_final.db"
            )
        if pred_m6a:
            pred_out = os.path.join(
                args.output_folder, f"{base_name}_{ens_tool.lower()}_m6A_{ptool}_m6A_final.db"
            )
    else:
        pred_out = os.path.join(args.output_folder, f"{base_name}_customShape_{ptool}_final.db")
        if pred_m6a:
            pred_out = os.path.join(
                args.output_folder, f"{base_name}_customShape_{ptool}_m6A_final.db"
            )

    if ptool == "eternafold":
        bpseq = os.path.join(args.output_folder, f"{base_name}_shape.bpseq")
        convert_shape_to_bpseq(shape_file, bpseq, args.sequence)
        kw = filter_kwargs(
            EternaFold,
            {
                "seq_file": bpseq,
                "out_file": pred_out,
                "mode": "predict",
                "executable": pred_exec,
                "eternaFold_params_shape": pred_params_cfg.get("eternaFold_params_shape"),
            },
        )
        EternaFold(**kw)

    elif ptool == "linearfold":
        kw = filter_kwargs(
            LinearFold,
            {
                "seq_file": args.sequence,
                "out_file": pred_out,
                "mode": "predict",
                "executable": pred_exec,
                "coinput_file": shape_file,
            },
        )
        LinearFold(**kw)

    elif ptool == "rnastructure":
        max_str_in_out=1 # conformers in output for RNAstr --> 1 = only best (for possible alternative conformers, set to > 1)
        kw = filter_kwargs(
            RNAStructure,
            {
                "seq_file": args.sequence,
                "out_file": pred_out,
                "coinput_file": shape_file,
                "executable": pred_exec,
                **pred_params_cfg,
                "maxm": max_str_in_out  
            },
        )
        RNAStructure(**kw)
     
    elif ptool == "rnafold":
        kw = filter_kwargs(
            RNAFold,
            {
                "seq_file": args.sequence,
                "out_file": pred_out,
                "method": pred_params_cfg.get("method", "z"),
                "max_bp_span": pred_params_cfg.get("max_bp_span"),
                "executable": pred_exec,
                "coinput_file": shape_file,
                **pred_params_cfg,
            },
        )
        RNAFold(**kw)
        print(pred_params_cfg.get("method", "z"))
    else:
        raise ValueError(f"Unsupported predictor tool: {predictor_choice}")

    return pred_out


# -----------------------------------------------------------------------------
# Entry point -----------------------------------------------------------------
# -----------------------------------------------------------------------------
def main():
    args = parse_cli_args()
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    ensemble_tool, ens_m6a = normalize_tool_choice(args.ensemble, role="ensemble")
    predictor_tool, pred_m6a = normalize_tool_choice(args.predictor, role="predictor")

    cfg = load_config(args.config)
    env_cfg = cfg.get("environment", {})
    apply_environment(env_cfg)

    #ensemble_target = args.ens_n if args.ens_n is not None else cfg.get("global_ensemble_size")
    if args.ens_n:
        ensemble_target = args.ens_n
    else:
        ensemble_target = 75

    ensemble_cfg = cfg.get(ensemble_tool, {})
    predictor_cfg = cfg.get(predictor_tool, {})
    ct2_cfg = cfg.get("ct2dot", {})
    shape_cfg = cfg.get("ShapeConversion", {}).get("params", {})
    delta=cfg.get("LinearFold", {}).get("params", {})["delta"]  # to get delta from cfg file (LinFold only)
    
    
    ens_exec = ensemble_cfg.get("executable")
    pred_exec = predictor_cfg.get("executable")
    ct2_exec = ct2_cfg.get("executable")

    ens_params_cfg = ensemble_cfg.get("params", {}).copy()
    if ens_m6a and "m6A" in ens_params_cfg:
        ens_params_cfg["m6A"] = True
    pred_params_cfg = predictor_cfg.get("params", {}).copy()
    if pred_m6a and "m6A" in pred_params_cfg:
        pred_params_cfg["m6A"] = True
    ct2_params_cfg = ct2_cfg.get("params", {})

    #nsamples = ensemble_target or select(ens_params_cfg, args.ens_nsamples, "n_struc")
    #nsamples = select(ens_params_cfg, args.ens_nsamples, "n_struc") or ensemble_target   # select() first for choosing input over default
    #maxm = ensemble_target or select(ens_params_cfg, args.ens_maxm, "maxm")
    maxm = ensemble_target
    
    nsamples = ensemble_target
    #par = select(ens_params_cfg, args.ens_par, "par")
    par = ens_params_cfg
    #delta = select(ens_params_cfg, args.ens_delta, "delta") or ens_params_cfg.get("delta", 5.0)
    
    ensure_dir(args.output_folder)
    base_name = os.path.splitext(os.path.basename(args.sequence))[0]

    if args.shape is None:
        ensemble_db = build_ensemble(
            ensemble_tool,
            base_name,
            args,
            ensemble_target,
            nsamples,
            maxm,
            delta,
            ens_params_cfg,
            ens_exec,
            ens_m6a,
        )
        shape_file = build_shape_file(
            ensemble_db, predictor_tool, base_name, args.output_folder, shape_cfg
        )
    else:
        shape_file = args.shape

    pred_out = predict_structure(
        predictor_tool,
        args,
        base_name,
        shape_file,
        ensemble_tool,
        ens_m6a,
        pred_m6a,
        pred_exec,
        pred_params_cfg,
    )

    print("Pipeline completed – outputs in", os.path.abspath(args.output_folder))
    print("Result file:", os.path.abspath(pred_out))

    shape_source = (
        "generated from ensemble" if args.shape is None else f"provided file: {os.path.abspath(args.shape)}"
    )
    write_ensemblefold_summary(
        out_dir=args.output_folder,
        base_name=base_name,
        sequence_path=args.sequence,
        ensemble_tool=ensemble_tool.lower(),
        predictor_tool=predictor_tool.lower(),
        ens_m6a=ens_m6a,
        pred_m6a=pred_m6a,
        ensemble_target=ensemble_target,
        nsamples=nsamples,
        maxm=maxm,
        par=par,
        delta=delta,
        shape_source=shape_source,
    )


if __name__ == "__main__":
    main()
