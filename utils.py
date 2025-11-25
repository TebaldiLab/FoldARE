import os
import sys
import math
from collections import Counter

def ct2dot(ct_file, db_file, st_num=0, ct2dot_folder="dot2ct"):
    """Converts dot-bracket notation to CT format using RNAStrucute tool."""
    command = f"{ct2dot_folder} {ct_file} {st_num} {db_file}"
    print("Running:", command)
    os.system(command)

def RNAStructure(seq_file, out_file, m6A=False, maxm=20, executable="/path/to/RNAStructure/fold-smp", coinput_file=None):
    """
    Runs RNAstructure.
    If coinput_file is provided, the command adds the '--SHAPE <coinput_file>' option.
    """
    if coinput_file:
        command = (f"{executable} {seq_file} {out_file} -k {'-a m6A' if m6A else ''} -m {maxm} "
                   f"--SHAPE {coinput_file}")
    else:
        command = f"{executable} {seq_file} {out_file} -k {'-a m6A' if m6A else ''} -m {maxm}"
    print("Running RNAstructure command:", command)
    os.system(command)

def RNAFold(seq_file, out_file, max_bp_span=600, executable="RNAfold", method="mfe", m6A=False, coinput_file=None):
    """
    Runs RNAfold.
    If coinput_file is provided, the command adds the '--shape <coinput_file>' option.
    """
    if coinput_file:
        command = (f"{executable}{' -m6' if m6A else ''}{' -p' if method == 'p' else ''} --noPS {'--noDP' if method == 'p' else ''} --maxBPspan={max_bp_span} "
                   f"--shape {coinput_file} {seq_file} > {out_file}")
    else:
        command = f"{executable}{' -m6' if m6A else ''}{' -p' if method == 'p' else ''} --noPS {'--noDP' if method == 'p' else ''} --maxBPspan={max_bp_span} {seq_file} > {out_file}"
    print("Running RNAfold command:", command)
    os.system(command)

def RNASubopt(seq_file, out_file, n_struc=20, method='z', m6A=False, executable="RNAsubopt", coinput_file=None):
    if coinput_file:
        command = (f"{executable}{' -m6' if m6A else ''}{' -p' if method == 'p' else ' -z'} {n_struc if method == 'p' else ''} -s < {seq_file} > {out_file} 2>/dev/null")
    else:
        command = f"{executable}{' -m6' if m6A else ''}{' -p' if method == 'p' else ' -z'} {n_struc if method == 'p' else ''} -s < {seq_file} > {out_file} 2>/dev/null"
    print("Running RNAsubopt command:", command)
    os.system(command)

def EternaFold(seq_file, out_file, mode="predict", executable="./../EternaFold-master/src/contrafold", eternaFold_params="../EternaFold-master/parameters/EternaFoldParams.v1", eternaFold_params_shape="../EternaFold-master/parameters/EternaFoldParams_PLUS_POTENTIALS.v1", nsamples=20):
    """
    Runs EnernaFold using contrafold.
    In ensemble mode (mode="sample"), it runs:
      contrafold_run sample <seq_file> --params <enernaFold_params> --nsamples <nsamples> > <out_file>
    In predictor mode (mode="predict"), it runs:
      contrafold_run predict <seq_file> --params <enernaFold_params> > <out_file>
    """
    if mode == "sample":
        command = (f"{executable} sample {seq_file} --params {eternaFold_params} "
                   f"--nsamples {nsamples} > {out_file}")
    else:
        command = f"{executable} predict {seq_file} --evidence --numdatasources 1 --kappa 0.1 --params {eternaFold_params_shape} > {out_file}"
    print("Running EnernaFold command:", command)
    os.system(command)

#./src/contrafold predict test_SHAPE.bpseq --evidence --numdatasources 1 --kappa 0.1 --params parameters/EternaFoldParams_PLUS_POTENTIALS.v1 
def LinearFold(seq_file, out_file, mode="predict", executable="./../LinearFold-master/linearfold", 
               coinput_file=None, delta=2.0):
    """
    Runs LinearFold.
    In ensemble mode (mode="ensemble"), the command uses:
      cat <seq_file> | <executable> -V --zuker --delta 2.0 [--shape <coinput_file>] > <out_file>
    In predictor mode (mode="predict"), the command uses:
      cat <seq_file> | <executable> [ -V --shape <coinput_file> ] > <out_file>
    """
    if mode == "ensemble":
        if coinput_file:
            command = (f"cat {seq_file} | {executable} --shape {coinput_file} "
                       f"--zuker --delta {delta} > {out_file}")
        else:
            command = f"cat {seq_file} | {executable} --zuker --delta {delta} > {out_file}"
    else:  # predictor mode
        if coinput_file:
            command = f"cat {seq_file} | {executable} -V --shape {coinput_file} > {out_file}"
        else:
            command = f"cat {seq_file} | {executable} > {out_file}"
    print("Running LinearFold command:", command)
    os.system(command)


def clean_ensemble_file(input_file, output_file):
    """
    Process an ensemble DB file so that only dot-bracket notations remain.
    Lines starting with '>' or containing any alphabetical characters (e.g. sequence letters)
    are omitted from the output.

    :param input_file: Path to the original ensemble DB file.
    :param output_file: Path to write the cleaned file.
    """
    # Define the set of allowed characters for dot-bracket notation.
    allowed_chars = set(".()<>{}[]")
    
    with open(input_file, 'r') as inf, open(output_file, 'w') as outf:
        for line in inf.readlines():
            line = line.strip()
            line = line.split()[0]  # remove energy values
            # Skip empty lines
            if not line:
                continue
            # Skip lines starting with '>' (titles) or that contain alphabetic characters.
            if line.startswith('>') or any(ch.isalpha() for ch in line):
                continue
            # Optionally, ensure that the line consists solely of allowed characters.
            #if set(line).issubset(allowed_chars):
            outf.write(line + "\n")

def extract_ensemble(file):
    """
    Extract the ensemble from a DB file as a list of dot–bracket strings.
    Only lines that do NOT start with '>' or a letter are retained.
    """
    with open(file) as f:
        lines = f.readlines()
    ensemble = []
    for line in lines:
        line = line.strip()
        line = line.split()[0]  # remove energy values
        if not line:
            continue
        if not line.startswith(">") and not line[0].isalpha():
            ensemble.append(line)
    return ensemble

def count_dots(ens):
    """
    For a list of dot–bracket strings (ensemble), count for each column 
    the fraction of sequences that have a dot '.'.
    Returns a list of probabilities.
    """
    dots = []
    # assume all ensemble strings are of the same length
    for i in range(len(ens[0])):
        dot_count = sum(1 for seq in ens if seq[i] == ".")
        dots.append(dot_count / len(ens))
    return dots

def create_shape_file(ensemble_file, shape_file_out, thresholds, coefficients, linearfold_style=False):
    """
    Create a shape file from the ensemble DB file using configurable thresholds and coefficients.
    The shape file contains one line per column in the format:
      <position> <reactivity>
    
    Reactivity is determined by applying the following conditions:
      - If thresholds["high"] > probability > thresholds["medium"], multiply by coefficients["medium"].
      - If probability >= thresholds["high"], multiply by coefficients["high"].
      - If thresholds["medium"] >= probability > thresholds["low"], multiply by coefficients["low"].
      - Otherwise, use coefficients["default"].
    
    Parameters:
      ensemble_file (str): Path to the ensemble DB file.
      shape_file_out (str): Path to output the shape file.
      thresholds (dict): Dictionary with keys "high", "medium", and "low" for probability boundaries.
      coefficients (dict): Dictionary with keys "high", "medium", "low", and "default" for reactivity multipliers.
    """
    ensemble = extract_ensemble(ensemble_file)
    if not ensemble:
        sys.exit(f"Error: No valid ensemble sequences found in {ensemble_file}.")
    
    dots = count_dots(ensemble)
    shape_lines = ""
    
    for i, prob in enumerate(dots):
        pos = i + 1  # positions are 1-indexed
        
        if thresholds["high"] > prob > thresholds["medium"]:
            value = prob * coefficients["medium"]
        elif prob >= thresholds["high"]:
            value = prob * coefficients["high"]
        elif thresholds["medium"] >= prob > thresholds["low"]:
            value = prob * coefficients["low"]
        else:
            value = coefficients["default"]

        if value == 0:
            if linearfold_style:
                value = "NA"
            else:
                continue  # skip zero values
        
        shape_lines += f"{pos} {value}\n"
    
    with open(shape_file_out, 'w') as f:
        f.write(shape_lines)
    
    print("Shape file created:", shape_file_out)


def convert_shape_to_bpseq(shape_file, bpseq_file, seq_file):
    """
    Converts a (possibly sparse) .shape file to a .bpseq file.

    Accepted .shape formats per line:
        <position> <reactivity>   # sparse or dense, 1-based positions
    Missing positions are filled with 0.0 so the output length matches the sequence.

    Output .bpseq columns:
        <position> <nucleotide> e1 <reactivity_in_scientific_notation>
    """
    # Read nucleotide sequence from the sequence file (FASTA allowed)
    seq_lines = []
    with open(seq_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                continue
            seq_lines.append(line)
    sequence = "".join(seq_lines)
    L = len(sequence)

    # Prepare zero-filled vector
    shape_values = [-1.00E+00] * L

    # Read SHAPE lines: allow sparse input with explicit positions
    with open(shape_file, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            # Expect "<pos> <value>"
            if len(parts) >= 2:
                try:
                    pos = int(parts[0])
                    val = float(parts[1])
                except ValueError:
                    # skip malformed lines silently
                    continue
                if 1 <= pos <= L:
                    shape_values[pos - 1] = val
            else:
                # If only a single value is provided, we can't infer position safely → skip
                # (old dense format without positions is discouraged now)
                continue

    # Write BPSEQ
    with open(bpseq_file, 'w') as out:
        for i, reactivity in enumerate(shape_values, start=1):
            nt = sequence[i - 1]
            formatted_reactivity = "{:.2E}".format(reactivity)
            out.write(f"{i} {nt} e1 {formatted_reactivity}\n")

    print(f"Converted (sparse-aware) {shape_file} to {bpseq_file}.")



def parse_dot_bracket(dot_bracket):
    """
    Parses dot-bracket notation and returns a list where index i corresponds to the partner of nucleotide i.
    If unpaired, partner[i] = -1.
    """
    stack = []
    partner = [-1] * len(dot_bracket)
    pair_dict = {'(':')', '[':']', '{':'}', '<':'>'}
    opening_brackets = pair_dict.keys()
    closing_brackets = pair_dict.values()
    bracket_pairs = {v: k for k, v in pair_dict.items()}
    
    for i, char in enumerate(dot_bracket):
        if char in opening_brackets:
            stack.append((char, i))
        elif char in closing_brackets:
            if stack and stack[-1][0] == bracket_pairs[char]:
                opening_char, j = stack.pop()
                partner[i] = j
                partner[j] = i
            else:
                pass
                #print(f"Warning: Unmatched closing bracket '{char}' at position {i}")
        else:
            # Unpaired nucleotide ('.' or other characters)
            continue
    if stack:
        pass
        #print("Warning: Unmatched opening brackets remain in the structure.")
    return partner

def parse_dot_bracket_agnostic(dot_bracket):
    """
    Parses dot-bracket notation and returns a list where index i corresponds to the partner of nucleotide i.
    If unpaired, partner[i] = -1.
    """
    stack = []
    partner = [-1] * len(dot_bracket)
    pair_dict = {'(':')', '[':']', '{':'}', '<':'>'}
    opening_brackets = pair_dict.keys()
    closing_brackets = pair_dict.values()
    bracket_pairs = {v: k for k, v in pair_dict.items()}
    
    for i, char in enumerate(dot_bracket):
        if char in opening_brackets:
            stack.append((char, i))
        elif char in closing_brackets:
            if stack and stack[-1][0] in opening_brackets:
                opening_char, j = stack.pop()
                partner[i] = j
                partner[j] = i
            else:
                pass
                #print(f"Warning: Unmatched closing bracket '{char}' at position {i}")
        else:
            # Unpaired nucleotide ('.' or other characters)
            continue
    if stack:
        pass
        #print("Warning: Unmatched opening brackets remain in the structure.")
    return partner


def total_difference(structure1, structure2):
    """
    Calculates the total number of positions where the pairing status differs between two structures.
    This includes both paired and unpaired nucleotides.
    """
    partner1 = parse_dot_bracket(structure1)
    partner2 = parse_dot_bracket(structure2)
    
    # Ensure both partner lists are the same length
    min_length = min(len(partner1), len(partner2))
    max_length = max(len(partner1), len(partner2))
    
    # Count differences in the overlapping region
    difference = sum(1 for i in range(min_length) if partner1[i] != partner2[i])
    
    # Account for any extra nucleotides in the longer structure
    difference += max_length - min_length
    
    return difference

def similarity_score(structure1, structure2):
    partner1 = parse_dot_bracket(structure1)
    partner2 = parse_dot_bracket(structure2)
    
    min_length = min(len(partner1), len(partner2))
    max_length = max(len(partner1), len(partner2))
    
    total_positions = max_length
    difference = sum(1 for i in range(min_length) if partner1[i] != partner2[i])
    difference += max_length - min_length  # Account for length differences
    
    similarity = 1 - (difference / total_positions)
    return similarity

def similarity_score_agnostic(structure1, structure2):
    partner1 = parse_dot_bracket_agnostic(structure1)
    partner2 = parse_dot_bracket_agnostic(structure2)
    
    min_length = min(len(partner1), len(partner2))
    max_length = max(len(partner1), len(partner2))
    
    total_positions = max_length
    difference = sum(1 for i in range(min_length) if partner1[i] != partner2[i])
    difference += max_length - min_length  # Account for length differences
    
    similarity = 1 - (difference / total_positions)
    return similarity


def identity_score(struct1, struct2):
    """
    Compute a simple similarity score between two RNA structure strings.
    
    For each position:
      - If the characters are exactly the same, count 1.
      - Otherwise, if both characters are opening brackets (any of "({[<"), count as a match.
      - Otherwise, if both are closing brackets (any of ")}]>"), count as a match.
      - Else, count 0.
    
    The final score is the fraction of positions that match, rounded to two decimals.
    
    :param struct1: First structure string (e.g., "((..))")
    :param struct2: Second structure string (e.g., "{[..]}")
    :return: Similarity score as a float (e.g., 0.90 for 90% matching)
    :raises ValueError: if the two structures are not of the same length.
    """
    if len(struct1) != len(struct2):
        raise ValueError("The two structures must have the same length.")
    
    match_count = 0
    for a, b in zip(struct1, struct2):
        if a == b:
            match_count += 1
        # else, no match at this position
    
    score = match_count / len(struct1)
    return round(score, 5)


def semi_identity_score(struct1, struct2):
    """
    Compute Score B ("semi-identity") between two RNA structures.
    
    For each position i:
      - If the 3-nt window (i-1, i, i+1) is identical in both structures → +1
      - Else, if only the i-th position is identical → +0.5
      - Else → +0

    Final score = sum / len(structure), in [0, 1].

    :param struct1: First structure string (dot-bracket notation)
    :param struct2: Second structure string (dot-bracket notation)
    :return: Semi-identity score (float)
    """
    if len(struct1) != len(struct2):
        raise ValueError("The two structures must have the same length.")

    n = len(struct1)
    score = 0.0

    for i in range(n):
        # define 3-nt window boundaries
        left  = i - 1 if i > 0 else i
        right = i + 1 if i < n - 1 else i

        if struct1[left:right+1] == struct2[left:right+1]:
            score += 1.0
        elif struct1[i] == struct2[i]:
            score += 0.5

    return round(score / n, 5)


def get_similarity_func(method: str):
    """
    Return the similarity scoring function by name.
    method: "identity" for identity (Score A), "semi-identity" for semi-identity (Score B)
    """
    if method.lower() == "identity":
        return identity_score
    elif method.lower() == "semi-identity":
        return semi_identity_score
    else:
        raise ValueError(f"Unknown scoring method: {method}")


def shannon_math(data, unit):
    """
    Compute Shannon entropy (no scipy). 
    data: list of symbols.
    unit: 'shannon', 'natural', or 'hartley'.
    """
    base = {'shannon': 2., 'natural': math.e, 'hartley': 10.}
    if unit not in base:
        raise ValueError(f"Unknown entropy unit '{unit}'")
    if len(data) <= 1:
        return 0.0
    counts = Counter(data)
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / len(data)
        if p > 0:
            entropy -= p * math.log(p, base[unit])
    return entropy
