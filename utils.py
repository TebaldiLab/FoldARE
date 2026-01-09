import math
import os
import sys
from collections import Counter
from typing import Callable, Dict


# -----------------------------------------------------------------------------
# External tool wrappers
# -----------------------------------------------------------------------------
def ct2dot(ct_file: str, db_file: str, st_num: int = 0, ct2dot_folder: str = "dot2ct") -> None:
    """Convert CT to dot-bracket using RNAstructure's dot2ct helper."""
    command = f"{ct2dot_folder} {ct_file} {st_num} {db_file}"
    print("Running:", command)
    os.system(command)


def RNAStructure(  # noqa: N802 - keep historical name
    seq_file: str,
    out_file: str,
    m6A: bool = False,
    maxm: int = 20,
    executable: str = "/path/to/RNAStructure/fold-smp",
    coinput_file: str | None = None,
) -> None:
    """
    Run RNAstructure.

    If coinput_file is provided, adds '--SHAPE <coinput_file>'.
    """
    if coinput_file:
        command = (
            f"{executable} {seq_file} {out_file} -k {'-a m6A' if m6A else ''} -m {maxm} "
            f"--SHAPE {coinput_file}"
        )
    else:
        command = f"{executable} {seq_file} {out_file} -k {'-a m6A' if m6A else ''} -m {maxm}"
    print("Running RNAstructure command:", command)
    os.system(command)


def RNAFold(  # noqa: N802 - keep historical name
    seq_file: str,
    out_file: str,
    max_bp_span: int = 600,
    executable: str = "RNAfold",
    method: str = "mfe",
    m6A: bool = False,
    coinput_file: str | None = None,
) -> None:
    """
    Run RNAfold.

    If coinput_file is provided, adds '--shape <coinput_file>'.
    """
    if coinput_file:
        command = (
            f"{executable}{' -m6' if m6A else ''}{' -p' if method == 'p' else ''} "
            f"--noPS {'--noDP' if method == 'p' else ''} --maxBPspan={max_bp_span} "
            f"--shape {coinput_file} {seq_file} > {out_file}"
        )
    else:
        command = (
            f"{executable}{' -m6' if m6A else ''}{' -p' if method == 'p' else ''} "
            f"--noPS {'--noDP' if method == 'p' else ''} --maxBPspan={max_bp_span} "
            f"{seq_file} > {out_file}"
        )
    print("Running RNAfold command:", command)
    os.system(command)


def RNASubopt(  # noqa: N802 - keep historical name
    seq_file: str,
    out_file: str,
    n_struc: int = 20,
    method: str = "z",
    m6A: bool = False,
    executable: str = "RNAsubopt",
    coinput_file: str | None = None,
) -> None:
    """Run RNAsubopt; supports SHAPE input, m6A flag, and 'p'/'z' methods."""
    method_flag = " -p" if method == "p" else " -z"
    m6_flag = " -m6" if m6A else ""
    n_arg = f"{n_struc if method == 'p' else ''}"
    if coinput_file:
        command = (
            f"{executable}{m6_flag}{method_flag} {n_arg} -s < {seq_file} > {out_file} 2>/dev/null"
        )
    else:
        command = (
            f"{executable}{m6_flag}{method_flag} {n_arg} -s < {seq_file} > {out_file} 2>/dev/null"
        )
    print("Running RNAsubopt command:", command)
    os.system(command)


def EternaFold(  # noqa: N802 - keep historical name
    seq_file: str,
    out_file: str,
    mode: str = "predict",
    executable: str = "./../EternaFold-master/src/contrafold",
    eternaFold_params: str = "../EternaFold-master/parameters/EternaFoldParams.v1",
    eternaFold_params_shape: str = "../EternaFold-master/parameters/EternaFoldParams_PLUS_POTENTIALS.v1",
    nsamples: int = 20,
) -> None:
    """
    Run EternaFold via contrafold.

    mode="sample": contrafold sample ... --nsamples <nsamples>
    mode="predict": contrafold predict ... --evidence --numdatasources 1 --kappa 0.1
    """
    if mode == "sample":
        command = (
            f"{executable} sample {seq_file} --params {eternaFold_params} "
            f"--nsamples {nsamples} > {out_file}"
        )
    else:
        command = (
            f"{executable} predict {seq_file} --evidence --numdatasources 1 --kappa 0.1 "
            f"--params {eternaFold_params_shape} > {out_file}"
        )
    print("Running EnernaFold command:", command)
    os.system(command)


def LinearFold(  # noqa: N802 - keep historical name
    seq_file: str,
    out_file: str,
    mode: str = "predict",
    executable: str = "./../LinearFold-master/linearfold",
    coinput_file: str | None = None,
    delta: float = 2.0,
) -> None:
    """
    Run LinearFold.

    mode="ensemble": cat <seq_file> | <executable> --zuker --delta <delta> [--shape ...]
    mode="predict" : cat <seq_file> | <executable> [-V --shape ...]
    """
    if mode == "ensemble":
        if coinput_file:
            command = (
                f"cat {seq_file} | {executable} --shape {coinput_file} "
                f"--zuker --delta {delta} > {out_file}"
            )
        else:
            command = f"cat {seq_file} | {executable} --zuker --delta {delta} > {out_file}"
    else:
        if coinput_file:
            command = f"cat {seq_file} | {executable} -V --shape {coinput_file} > {out_file}"
        else:
            command = f"cat {seq_file} | {executable} > {out_file}"
    print("Running LinearFold command:", command)
    os.system(command)

# -----------------------------------------------------------------------------
# File processing functions
# -----------------------------------------------------------------------------

def clean_ensemble_file(input_file, output_file):
    """
    Process an ensemble DB file so that only dot-bracket notations remain.
    Lines starting with '>' or containing alphabetic characters are omitted.
    """
    with open(input_file, "r") as inf, open(output_file, "w") as outf:
        for line in inf.readlines():
            line = line.strip()
            line = line.split()[0]  # remove energy values
            if not line:
                continue
            if line.startswith(">") or any(ch.isalpha() for ch in line):
                continue
            outf.write(line + "\n")


def extract_ensemble(file: str) -> list[str]:
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


def count_dots(ens: list[str]) -> list[float]:
    """
    For a list of dot–bracket strings (ensemble), count for each column
    the fraction of sequences that have a dot '.'.
    Returns a list of probabilities.
    """
    if not ens:
        return []
    dots: list[float] = []
    for i in range(len(ens[0])):
        dot_count = sum(1 for seq in ens if seq[i] == ".")
        dots.append(dot_count / len(ens))
    return dots


def create_shape_file(
    ensemble_file: str,
    shape_file_out: str,
    thresholds: Dict[str, float],
    coefficients: Dict[str, float],
    linearfold_style: bool = False,
) -> None:
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
            continue #value = coefficients["default"]

        if value == 0:
            if linearfold_style:
                value = "NA"
            else:
                continue  # skip zero values

        shape_lines += f"{pos} {value}\n"

    with open(shape_file_out, "w") as f:
        f.write(shape_lines)

    print("Shape file created:", shape_file_out)


def convert_shape_to_bpseq(shape_file: str, bpseq_file: str, seq_file: str) -> None:
    """
    Converts a (possibly sparse) .shape file to a .bpseq file.

    Accepted .shape formats per line:
        <position> <reactivity>   # sparse or dense, 1-based positions
    Missing positions are filled with 0.0 so the output length matches the sequence.

    Output .bpseq columns:
        <position> <nucleotide> e1 <reactivity_in_scientific_notation>
    """
    seq_lines = []
    with open(seq_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                continue
            seq_lines.append(line)
    sequence = "".join(seq_lines)
    L = len(sequence)

    shape_values = [-1.00E+00] * L

    with open(shape_file, "r") as f:
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

    with open(bpseq_file, "w") as out:
        for i, reactivity in enumerate(shape_values, start=1):
            nt = sequence[i - 1]
            formatted_reactivity = "{:.2E}".format(reactivity)
            out.write(f"{i} {nt} e1 {formatted_reactivity}\n")

    print(f"Converted (sparse-aware) {shape_file} to {bpseq_file}.")

# -----------------------------------------------------------------------------
# Structure comparison functions
# -----------------------------------------------------------------------------

def parse_dot_bracket(dot_bracket: str) -> list[int]:
    """
    Parses dot-bracket notation and returns a list where index i corresponds to the partner of nucleotide i.
    If unpaired, partner[i] = -1.
    """
    stack = []
    partner = [-1] * len(dot_bracket)
    pair_dict = {"(": ")", "[": "]", "{": "}", "<": ">"}
    opening_brackets = pair_dict.keys()
    closing_brackets = pair_dict.values()
    bracket_pairs = {v: k for k, v in pair_dict.items()}

    for i, char in enumerate(dot_bracket):
        if char in opening_brackets:
            stack.append((char, i))
        elif char in closing_brackets:
            if stack and stack[-1][0] == bracket_pairs[char]:
                _, j = stack.pop()
                partner[i] = j
                partner[j] = i
            else:
                continue
    return partner


def parse_dot_bracket_agnostic(dot_bracket: str) -> list[int]:
    """
    Parses dot-bracket notation and returns a list where index i corresponds to the partner of nucleotide i.
    If unpaired, partner[i] = -1.
    """
    stack = []
    partner = [-1] * len(dot_bracket)
    pair_dict = {"(": ")", "[": "]", "{": "}", "<": ">"}
    opening_brackets = pair_dict.keys()
    closing_brackets = pair_dict.values()
    bracket_pairs = {v: k for k, v in pair_dict.items()}

    for i, char in enumerate(dot_bracket):
        if char in opening_brackets:
            stack.append((char, i))
        elif char in closing_brackets:
            if stack and stack[-1][0] in opening_brackets:
                _, j = stack.pop()
                partner[i] = j
                partner[j] = i
            else:
                continue
    return partner


def total_difference(structure1: str, structure2: str) -> int:
    """
    Calculates the total number of positions where the pairing status differs between two structures.
    This includes both paired and unpaired nucleotides.
    """
    partner1 = parse_dot_bracket(structure1)
    partner2 = parse_dot_bracket(structure2)

    min_length = min(len(partner1), len(partner2))
    max_length = max(len(partner1), len(partner2))

    difference = sum(1 for i in range(min_length) if partner1[i] != partner2[i])
    difference += max_length - min_length
    return difference


#############  scoring systems
## score A -- identity score
def similarity_scoreA(struct1, struct2): 
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
    
    open_set = set("({[<")
    close_set = set(")}]>")
    
    match_count = 0
    for a, b in zip(struct1, struct2):
        if a == b:
            match_count += 1
        elif a in open_set and b in open_set:
            match_count += 1
        elif a in close_set and b in close_set:
            match_count += 1
        # else, no match at this position
    
    score = match_count / len(struct1)
    return round(score, 5)


## score B -- semi-identity score
def similarity_scoreB(struct1,struct2):
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
    p0=struct1[0];r0=struct2[0] # also pn=struct1[-1];rn=struct2[-1]
    consensus=0
    if p0 == r0:
        consensus=consensus+1
    for it in range(1,len(struct1)-1):
        pw=struct1[it-1:it+1] # w x window
        rw=struct2[it-1:it+1]
        pi=struct1[it];ri=struct2[it]
        if pi == ri:
          if pw==rw:
            score=1
            consensus=consensus+score
          else:
             score=0.5
             consensus=consensus+score

    pn=struct1[-1];rn=struct2[-1]
    if pn == rn:
        consensus=consensus+1                                 
    #return consensus
    score = float(consensus)/len(struct1)
    return round(score, 5)


## score C -- similarity score(assess local similarity)
def similarity_scoreC(pred, ref):
    """
    Computes a consensus score between pred and ref with optional window shifts (-1, 0, +1). Window size = 5 nt (-2,0,+2, around a position 0)
    """
    def window_score(pw, rw):
        score = sum(1 for a, b in zip(pw, rw) if a == b)
        delta = len(pw) - score
        return score / (len(pw) + delta)

    def get_window(seq, start, size):
        end = start + size
        return seq[start:end] if start >= 0 and end <= len(seq) else None

    consensus = 0
    length = len(pred)

    # Compare first two elements
    if pred[0] == ref[0]:
        consensus += 1
    if pred[1] == ref[1]:
        consensus += 1

    # Core sliding window with shifts
    window_size = 4  # equivalent to it-2:it+2
    shift_range = [-1, 0, 1]
    consensus_alternatives = []

    for shift in shift_range:
        temp_score = 0
        for i in range(2, length - 2):
            pred_window = get_window(pred, i - 2, window_size)
            ref_index = i + shift
            ref_window = get_window(ref, ref_index - 2, window_size)

            if pred_window is None or ref_window is None:
                continue

            temp_score += window_score(pred_window, ref_window)

        consensus_alternatives.append(temp_score)

    consensus += max(consensus_alternatives, default=0)

    # Compare last two elements
    if pred[-2] == ref[-2]:
        consensus += 1
    if pred[-1] == ref[-1]:
        consensus += 1

    return consensus / len(pred)


## score D -- "matching brackets" (a stricter identity score: for dsRNA positions, local identity is checked for identical pairing
def similarity_scoreD(pred, ref, tolerance=0):
    """
    Calculate F1 score for matching dot-bracket symbols with position tolerance.
    
    Args:
        pred: predicted dot-bracket string
        ref: reference dot-bracket string
        tolerance: allowed position offset (default ±1)
    
    Returns:
        f1_score: harmonic mean of precision and recall
        precision: TP / (TP + FP)
        recall: TP / (TP + FN)
    """
    def extract_pairs(seq):
        """Extract all base pairs from dot-bracket notation."""
        pairs = []
        stack = []
        
        for i, symbol in enumerate(seq):
            if symbol == '(':
                stack.append(i)
            elif symbol == ')':
                if stack:
                    j = stack.pop()
                    pairs.append((j, i))
        
        # Sort pairs for consistent comparison
        pairs.sort()
        return set(pairs)
    
    def get_matched_pairs(pred_pairs, ref_pairs, tolerance):
        """Find matching pairs within tolerance window."""
        matched = set()
        used_pred = set()
        used_ref = set()
        
        # Create copies to avoid modifying originals
        pred_list = sorted(list(pred_pairs))
        ref_list = sorted(list(ref_pairs))
        
        # Try to match each reference pair
        for r_start, r_end in ref_list:
            best_match = None
            best_distance = float('inf')
            
            for p_start, p_end in pred_list:
                if (p_start, p_end) in used_pred:
                    continue
                    
                # Check if positions are within tolerance
                if (abs(p_start - r_start) <= tolerance and 
                    abs(p_end - r_end) <= tolerance):
                    
                    # Calculate total distance
                    distance = abs(p_start - r_start) + abs(p_end - r_end)
                    
                    if distance < best_distance:
                        best_match = (p_start, p_end)
                        best_distance = distance
            
            if best_match:
                matched.add(best_match)
                used_pred.add(best_match)
                used_ref.add((r_start, r_end))
        
        return matched, used_pred, used_ref
    
    # Extract base pairs
    pred_pairs = extract_pairs(pred)
    ref_pairs = extract_pairs(ref)
    
    # Find matches within tolerance
    matched_pairs, used_pred, used_ref = get_matched_pairs(pred_pairs, ref_pairs, tolerance)
    
    # Calculate metrics
    tp = len(matched_pairs)  # True positives
    fp = len(pred_pairs) - len(used_pred)  # False positives
    fn = len(ref_pairs) - len(used_ref)  # False negatives
    
    # Avoid division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    #return f1 , precision, recall, tp, fp, fn
    return f1 


def repair_dot_bracket(db):
    stack = 0
    cleaned = []

    # First pass: remove unmatched ')'
    for c in db:
        if c == '(':
            stack += 1
            cleaned.append(c)
        elif c == ')':
            if stack > 0:
                stack -= 1
                cleaned.append(c)
            # else skip the invalid ')'
        else:
            cleaned.append(c)

    # Second pass: remove excess '(' from the end
    result = []
    excess = stack
    for c in reversed(cleaned):
        if c == '(' and excess > 0:
            excess -= 1
            continue
        result.append(c)

    return ''.join(reversed(result))


def shannon_math(data, unit):
    """
    Compute Shannon entropy (no scipy).
    data: list of symbols.
    unit: 'shannon', 'natural', or 'hartley'.
    """
    base = {"shannon": 2.0, "natural": math.e, "hartley": 10.0}
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


