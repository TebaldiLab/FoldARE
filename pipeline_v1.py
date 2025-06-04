#!/usr/bin/env python3
"""
RNA Prediction Pipeline

This script runs the ensemble and predictor steps.
It accepts overrides for shape conversion parameters and ensemble-size parameters.
For ensemble-size overrides, use:
  --ens_maxm for RNAStructure
  --ens_par for RNASubopt
  --ens_nsamples for EternaFold
  --ens_delta for LinearFold

Usage example:
  python3 pipeline.py -p RNAFold -e RNAStructure -s myseq.fa -o results/ \
    --config config.yaml --ens_maxm 50 --shape_threshold_high 0.95 --shape_coeff_high 2.5
"""

import os
import sys
import argparse
import shutil
from ruamel.yaml import YAML
from utils_v2 import (
    RNAStructure, RNAFold, ct2dot, RNASubopt, clean_ensemble_file,
    create_shape_file, EternaFold, LinearFold, convert_shape_to_bpseq
)

def load_config(config_file):
    """Loads the YAML configuration using ruamel.yaml."""
    yaml = YAML(typ='safe')
    try:
        with open(config_file, 'r') as f:
            config = yaml.load(f)
        return config
    except Exception as e:
        sys.exit(f"Error loading configuration file {config_file}: {e}")

# Mapping tool names to functions.
tool_functions = {
    "RNAStructure": RNAStructure,
    "RNAFold": RNAFold,
    "RNASubopt": RNASubopt,
    "EternaFold": EternaFold,
    "LinearFold": LinearFold,
}

def main():
    parser = argparse.ArgumentParser(
        description="RNA prediction pipeline: run ensemble and predictor tools with optional ensemble-size overrides."
    )
    parser.add_argument("-p", "--predictor", required=True,
                        help="Tool to use for final prediction (e.g. RNAFold)")
    parser.add_argument("-e", "--ensemble", required=True,
                        help="Tool to use for ensemble prediction (e.g. RNAStructure or RNAfold)")
    parser.add_argument("-s", "--sequence", required=True,
                        help="Path to the sequence file")
    parser.add_argument("-o", "--output_folder", default=".",
                        help="Folder to save output files (default: current working directory)")
    parser.add_argument("--shape", default=None,
                        help="Path to a SHAPE file. If provided, ensemble predictor is skipped.")
    parser.add_argument("-c", "--config", default="config.yaml",
                        help="Path to the YAML configuration file")
    # Shape conversion override flags.
    parser.add_argument("--shape_threshold_high", type=float,
                        help="Override ShapeConversion threshold for high reactivity")
    parser.add_argument("--shape_threshold_medium", type=float,
                        help="Override ShapeConversion threshold for medium reactivity")
    parser.add_argument("--shape_threshold_low", type=float,
                        help="Override ShapeConversion threshold for low reactivity")
    parser.add_argument("--shape_coeff_high", type=float,
                        help="Override ShapeConversion coefficient for high reactivity")
    parser.add_argument("--shape_coeff_medium", type=float,
                        help="Override ShapeConversion coefficient for medium reactivity")
    parser.add_argument("--shape_coeff_low", type=float,
                        help="Override ShapeConversion coefficient for low reactivity")
    parser.add_argument("--shape_coeff_default", type=float,
                        help="Override ShapeConversion default coefficient")
    # Ensemble size override flags.
    parser.add_argument("--ens_maxm", type=int,
                        help="Override ensemble parameter 'maxm' for RNAStructure")
    parser.add_argument("--ens_par", type=int,
                        help="Override ensemble parameter 'par' for RNASubopt")
    parser.add_argument("--ens_nsamples", type=int,
                        help="Override ensemble parameter 'nsamples' for EternaFold")
    parser.add_argument("--ens_delta", type=float,
                        help="Override ensemble parameter 'delta' for LinearFold")
    args = parser.parse_args()

    # Ensure output folder exists.
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print("Output files will be saved in:", os.path.abspath(output_folder))

    # Load configuration.
    config = load_config(args.config)
    env_config = config.get("environment", {})
    root = os.getcwd() + '/'
    data_tables = env_config.get("data_tables", "../RNAstructureLinuxTextInterfaces64bit/RNAstructure/data_tables/")
    os.environ["DATAPATH"] = root + data_tables
    print("DATAPATH set to:", os.environ["DATAPATH"])

    # Determine tool keys.
    predictor_key = args.predictor
    if args.ensemble.lower() == "rnafold":
        ensemble_key = "RNASubopt"
    else:
        ensemble_key = args.ensemble

    if predictor_key not in config:
        sys.exit(f"Error: predictor tool '{predictor_key}' not found in configuration.")
    if ensemble_key not in config:
        sys.exit(f"Error: Ensemble tool '{ensemble_key}' not found in configuration.")

    predictor_conf = config[predictor_key]
    ensemble_conf = config[ensemble_key]
    ct2dot_conf = config.get("ct2dot", {"executable": "ct2dot", "params": {}})

    predictor_params = predictor_conf.get("params", {})
    ensemble_params = ensemble_conf.get("params", {})
    ct2dot_params = ct2dot_conf.get("params", {})

    predictor_executable = predictor_conf.get("executable", predictor_key)
    ensemble_executable = ensemble_conf.get("executable", ensemble_key)
    ct2dot_executable = ct2dot_conf.get("executable", "ct2dot")

    # Override ensemble parameters if flags are provided.
    if ensemble_key == "RNAStructure" and args.ens_maxm is not None:
        ensemble_params["maxm"] = args.ens_maxm
    if ensemble_key == "RNASubopt" and args.ens_par is not None:
        ensemble_params["n_struc"] = args.ens_par
    if ensemble_key == "EternaFold" and args.ens_nsamples is not None:
        ensemble_params["nsamples"] = args.ens_nsamples
    if ensemble_key == "LinearFold" and args.ens_delta is not None:
        ensemble_params["delta"] = args.ens_delta

    # Set available threads.
    threads = env_config.get("threads", 1)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    print(f"Using {threads} threads.")

    seq_file = args.sequence
    seq_basename = os.path.splitext(os.path.basename(seq_file))[0]
    base = os.path.join(output_folder, seq_basename)

    predictor_raw_out = f"{base}_{predictor_key}_final.raw"
    predictor_db_out = f"{base}_{predictor_key}_final.db"
    ensemble_raw_out = f"{base}_{args.ensemble}_ens.raw"
    ensemble_db_out = f"{base}_{args.ensemble}_ens.db"

    # Determine whether to run the ensemble predictor.
    if args.shape:
        shape_file = args.shape
        print("Using provided SHAPE file:", shape_file)
    else:
        print(f"=== Running ensemble predictor: {args.ensemble} (using {ensemble_key}) ===")
        if ensemble_key not in tool_functions:
            sys.exit(f"Error: Function for ensemble tool '{ensemble_key}' is not implemented.")
        ensemble_func = tool_functions[ensemble_key]

        if ensemble_key == "RNASubopt":
            ensemble_func(seq_file, ensemble_db_out, executable=ensemble_executable, **ensemble_params)
        elif ensemble_key == "EternaFold":
            ensemble_func(seq_file, ensemble_db_out, mode="sample", executable=ensemble_executable, **ensemble_params)
        elif ensemble_key == "LinearFold":
            ensemble_func(seq_file, ensemble_db_out, mode="ensemble", executable=ensemble_executable, **ensemble_params)
        else:
            ensemble_func(seq_file, ensemble_raw_out, executable=ensemble_executable, **ensemble_params)
            if ensemble_key == "RNAStructure":
                print("Converting RNAstructure CT file to DB file for ensemble...")
                ct2dot(ensemble_raw_out, ensemble_db_out, ct2dot_folder=ct2dot_executable, **ct2dot_params)
            else:
                ensemble_db_out = ensemble_raw_out

        print("Cleaning ensemble DB file...")
        ensemble_db_out_copy = ensemble_db_out + ".copy"
        shutil.copy(ensemble_db_out, ensemble_db_out_copy)
        clean_ensemble_file(ensemble_db_out_copy, ensemble_db_out)
        os.remove(ensemble_db_out_copy)

        # Create shape file from ensemble DB file.
        shape_file = f"{base}_shape.txt"
        print("=== Creating shape file from ensemble DB file ===")
        shape_config = config.get("ShapeConversion", {}).get("params", {})
        thresholds = shape_config.get("thresholds", {})
        coefficients = shape_config.get("coefficients", {})

        if args.shape_threshold_high is not None:
            thresholds["high"] = args.shape_threshold_high
        if args.shape_threshold_medium is not None:
            thresholds["medium"] = args.shape_threshold_medium
        if args.shape_threshold_low is not None:
            thresholds["low"] = args.shape_threshold_low
        if args.shape_coeff_high is not None:
            coefficients["high"] = args.shape_coeff_high
        if args.shape_coeff_medium is not None:
            coefficients["medium"] = args.shape_coeff_medium
        if args.shape_coeff_low is not None:
            coefficients["low"] = args.shape_coeff_low
        if args.shape_coeff_default is not None:
            coefficients["default"] = args.shape_coeff_default

        create_shape_file(ensemble_db_out, shape_file, thresholds, coefficients)

    print(f"=== Running predictor tool: {predictor_key} ===")
    if predictor_key not in tool_functions:
        sys.exit(f"Error: Function for predictor tool '{predictor_key}' is not implemented.")
    predictor_func = tool_functions[predictor_key]
    if predictor_key == "EternaFold":
        convert_shape_to_bpseq(shape_file, f"{base}_shape.bpseq", seq_file)
        predictor_func(f"{base}_shape.bpseq", predictor_db_out, mode="predict",
                       executable=predictor_executable, **predictor_params)
    elif predictor_key == "LinearFold":
        predictor_func(seq_file, predictor_db_out, mode="predict",
                       executable=predictor_executable, coinput_file=shape_file, **predictor_params)
    else:
        predictor_func(seq_file, predictor_raw_out, executable=predictor_executable,
                       coinput_file=shape_file, **predictor_params)
        if predictor_key == "RNAStructure":
            print("Converting RNAstructure CT file to DB file for predictor...")
            ct2dot(predictor_raw_out, predictor_db_out, ct2dot_folder=ct2dot_executable, **ct2dot_params)
        else:
            shutil.copy(predictor_raw_out, predictor_db_out)
            os.remove(predictor_raw_out)

    print("=== Pipeline execution completed. ===")
    print(f"Predictor output (DB file): {predictor_db_out}")
    if args.shape:
        print("Ensemble step was skipped due to provided SHAPE file.")
    else:
        print(f"Ensemble output (DB file): {ensemble_db_out}")

if __name__ == "__main__":
    main()
