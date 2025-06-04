# sequence comparison commands

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


def simple_similarity_score(struct1, struct2):
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
