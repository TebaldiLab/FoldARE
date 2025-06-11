import os
import argparse
from inspect import signature
from ruamel.yaml import YAML
from utils_v2 import (
    RNAStructure, RNAFold, ct2dot, RNASubopt,
    clean_ensemble_file, create_shape_file,
    EternaFold, LinearFold, convert_shape_to_bpseq
)

# -----------------------------------------------------------------------------
# Helper utilities -------------------------------------------------------------
# -----------------------------------------------------------------------------

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
    """Drop keys not accepted by *func* and None-valued entries."""
    sig = signature(func).parameters
    return {k: v for k, v in mapping.items() if k in sig and v is not None}


def clean_in_place(db_path: str):
    """Run *clean_ensemble_file* safely without input‑=output overwrite."""
    tmp_path = db_path + ".clean"
    clean_ensemble_file(db_path, tmp_path)
    os.replace(tmp_path, db_path)

# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Sequence-level RNA prediction pipeline (ensemble → SHAPE → predictor)")
parser.add_argument("-p", "--predictor", required=True,
                    choices=["ViennaRNA", "RNAStructure", "EternaFold", "LinearFold"],
                    help="Tool used for final prediction")
parser.add_argument("-e", "--ensemble", required=True,
                    choices=["ViennaRNA", "RNAStructure", "EternaFold", "LinearFold"],
                    help="Tool that generates the ensemble")
parser.add_argument("-s", "--sequence", required=True, help="Input FASTA file")
parser.add_argument("-o", "--output_folder", default=".")
parser.add_argument("-c", "--config", default="config.yaml", help="YAML configuration file")
parser.add_argument("--shape", help="Existing SHAPE file – skip ensemble step")

# Global ensemble size override
parser.add_argument("--ens_n", type=int, help="Target ensemble size for all tools")

# Optional per-tool overrides (take effect only if supplied)
parser.add_argument("--ens_nsamples", type=int)
parser.add_argument("--ens_maxm", type=int)
parser.add_argument("--ens_par", type=int)
parser.add_argument("--ens_delta", type=float)
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Config & environment ---------------------------------------------------------
# -----------------------------------------------------------------------------
CFG = load_config(args.config)

# Environment variables --------------------------------------------------------
env_cfg = CFG.get("environment", {})
root_dir = os.getcwd() + "/"
if "data_tables" in env_cfg:
    os.environ["DATAPATH"] = os.path.abspath(os.path.join(root_dir, env_cfg["data_tables"]))
if env_cfg.get("threads") is not None:
    os.environ["OMP_NUM_THREADS"] = str(env_cfg["threads"])
    print("Using", os.environ["OMP_NUM_THREADS"], "threads for parallel execution.")

# Derived settings -------------------------------------------------------------
ensemble_target = args.ens_n if args.ens_n is not None else CFG.get("global_ensemble_size")
ensemble_cfg = CFG.get(args.ensemble, {})
predictor_cfg = CFG.get(args.predictor, {})
ct2_cfg = CFG.get("ct2dot", {})
shape_cfg = CFG.get("ShapeConversion", {}).get("params", {})

# Executables ------------------------------------------------------------------
ens_exec = ensemble_cfg.get("executable")
pred_exec = predictor_cfg.get("executable")
ct2_exec = ct2_cfg.get("executable")

# Parameter dictionaries -------------------------------------------------------
ens_params_cfg = ensemble_cfg.get("params", {})
pred_params_cfg = predictor_cfg.get("params", {})
ct2_params_cfg = ct2_cfg.get("params", {})

# Merge CLI overrides ----------------------------------------------------------
nsamples = select(ens_params_cfg, args.ens_nsamples, "n_struc") or ensemble_target
maxm     = select(ens_params_cfg, args.ens_maxm, "maxm")
par      = select(ens_params_cfg, args.ens_par, "par")
delta    = select(ens_params_cfg, args.ens_delta, "delta") or ens_params_cfg.get("delta", 5.0)

# -----------------------------------------------------------------------------
# Output prep ------------------------------------------------------------------
# -----------------------------------------------------------------------------
ensure_dir(args.output_folder)
base_name = os.path.splitext(os.path.basename(args.sequence))[0]

# -----------------------------------------------------------------------------
# ENSEMBLE generation ----------------------------------------------------------
# -----------------------------------------------------------------------------
if args.shape is None:
    ens_db = os.path.join(args.output_folder, f"{base_name}_{args.ensemble}_ens.db")
    tool = args.ensemble.lower()

    if tool == "viennarna":
        RNASubopt(**filter_kwargs(RNASubopt, {
            "seq_file": args.sequence,
            "out_file": ens_db,
            "n_struc": nsamples,
            "method": ens_params_cfg.get("method"),
            "executable": ens_exec,
        }))
        clean_in_place(ens_db)

    elif tool == "eternafold":
        EternaFold(**filter_kwargs(EternaFold, {
            "seq_file": args.sequence,
            "out_file": ens_db,
            "mode": "sample",
            "nsamples": nsamples,
            "executable": ens_exec,
            **ens_params_cfg,
        }))
        clean_in_place(ens_db)

    elif tool == "linearfold":
        desired = ensemble_target or nsamples or 1
        current_delta = delta
        tmp_db = ens_db + ".tmp"
        while True:
            LinearFold(**filter_kwargs(LinearFold, {
                "seq_file": args.sequence,
                "out_file": tmp_db,
                "mode": "ensemble",
                "delta": current_delta,
                "executable": ens_exec,
            }))
            clean_in_place(tmp_db)
            with open(tmp_db) as fh:
                lines = fh.readlines()
            if len(lines) >= desired:
                break
            current_delta += 1.0
        # Trim to exactly desired
        if len(lines) > desired:
            with open(tmp_db, "w") as fh:
                fh.writelines(lines[:desired])
        os.replace(tmp_db, ens_db)

    elif tool == "rnastructure":
        RNAStructure(**filter_kwargs(RNAStructure, {
            "seq_file": args.sequence,
            "out_file": ens_db,
            "executable": ens_exec,
            **ens_params_cfg,
        }))
        clean_in_place(ens_db)

    else:
        raise ValueError(f"Unsupported ensemble tool: {args.ensemble}")

    # Post-trim for non-LinearFold multi-structure generators
    if tool not in ["linearfold", "rnastructure"] and ensemble_target is not None:
        with open(ens_db) as fh:
            entries = fh.readlines()
        if len(entries) > ensemble_target:
            with open(ens_db, "w") as fh:
                fh.writelines(entries[:ensemble_target])

    # Shape generation --------------------------------------------------------
    shape_file = os.path.join(args.output_folder, f"{base_name}_shape.txt")
    create_shape_file(
        ens_db,
        shape_file,
        thresholds=shape_cfg.get("thresholds", {}),
        coefficients=shape_cfg.get("coefficients", {}),
    )
else:
    shape_file = args.shape

# -----------------------------------------------------------------------------
# PREDICTION -------------------------------------------------------------------
# -----------------------------------------------------------------------------

ptool = args.predictor.lower()
pred_out = os.path.join(args.output_folder, f"{base_name}_{tool}_{ptool}_final.db")

if ptool == "eternafold":
    bpseq = os.path.join(args.output_folder, f"{base_name}_shape.bpseq")
    convert_shape_to_bpseq(shape_file, bpseq, args.sequence)
    kw = filter_kwargs(EternaFold, {
        "seq_file": bpseq,
        "out_file": pred_out,
        "mode": "predict",
        "executable": pred_exec,
        "eternaFold_params_shape": pred_params_cfg.get("eternaFold_params_shape"),
    })
    EternaFold(**kw)

elif ptool == "linearfold":
    kw = filter_kwargs(LinearFold, {
        "seq_file": args.sequence,
        "out_file": pred_out,
        "mode": "predict",
        "executable": pred_exec,
        "coinput_file": shape_file,
    })
    LinearFold(**kw)

elif ptool == "rnastructure":
    kw = filter_kwargs(RNAStructure, {
        "seq_file": args.sequence,
        "out_file": pred_out,
        "coinput_file": shape_file,
        "executable": pred_exec,
        **pred_params_cfg,
    })
    RNAStructure(**kw)

elif ptool == "viennarna":
    kw = filter_kwargs(RNAFold, {
        "seq_file": args.sequence,
        "out_file": pred_out,
        "max_bp_span": pred_params_cfg.get("max_bp_span"),
        "executable": pred_exec,
        "coinput_file": shape_file,
    })
    RNAFold(**kw)

else:
    raise ValueError(f"Unsupported predictor tool: {args.predictor}")

print("Pipeline completed – outputs in", os.path.abspath(args.output_folder))
print("Result file:", os.path.abspath(pred_out))
