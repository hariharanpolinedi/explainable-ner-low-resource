"""
Validation, span conversion, and BIO consistency checking for Kokborok HITL NER.
"""

from typing import List, Tuple, Dict, Any, Optional

VALID_ENTITY_TYPES = {"PER", "LOC", "ORG", "MISC"}
VALID_TAGS = {"O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"}


def validate_bio_sequence(tokens: List[str], tags: List[str]) -> Tuple[bool, List[str]]:
    """
    Validates that tags match token length, have legal BIO transitions,
    and have no orphan I-tags. Returns (is_valid, list_of_errors).
    """
    errors = []
    if len(tokens) != len(tags):
        errors.append(f"Token length ({len(tokens)}) does not match tag length ({len(tags)}).")
        return False, errors

    prev_tag = "O"
    for i, (tok, tag) in enumerate(zip(tokens, tags)):
        if tag not in VALID_TAGS:
            errors.append(f"Invalid tag '{tag}' at index {i} ('{tok}').")
            continue

        if tag.startswith("I-"):
            ent_type = tag[2:]
            if prev_tag == "O":
                errors.append(f"Orphan tag '{tag}' at index {i} ('{tok}') without prior B-tag.")
            elif prev_tag.startswith("B-") or prev_tag.startswith("I-"):
                prev_type = prev_tag[2:]
                if prev_type != ent_type:
                    errors.append(f"Tag mismatch at index {i}: '{tag}' follows '{prev_tag}'.")

        prev_tag = tag

    return len(errors) == 0, errors


def spans_to_bio(tokens: List[str], spans: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    Converts a list of span dicts [{"start": int, "end": int, "type": str}] to BIO tags.
    start is inclusive, end is exclusive (0-indexed).
    Returns (tags, errors).
    """
    tags = ["O"] * len(tokens)
    errors = []

    # Sort spans by start index
    sorted_spans = sorted(spans, key=lambda s: s["start"])

    prev_end = 0
    for span in sorted_spans:
        s_start = span["start"]
        s_end = span["end"]
        s_type = span["type"].upper()

        if s_type not in VALID_ENTITY_TYPES:
            errors.append(f"Invalid entity type '{s_type}' in span {span}.")
            continue

        if s_start < 0 or s_end > len(tokens) or s_start >= s_end:
            errors.append(f"Out of bounds or invalid span indices: [{s_start}, {s_end}) for token length {len(tokens)}.")
            continue

        if s_start < prev_end:
            errors.append(f"Overlapping span detected at [{s_start}, {s_end}) with previous ending at {prev_end}.")
            continue

        tags[s_start] = f"B-{s_type}"
        for i in range(s_start + 1, s_end):
            tags[i] = f"I-{s_type}"

        prev_end = s_end

    return tags, errors


def bio_to_spans(tokens: List[str], tags: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Converts BIO tags to a list of span dicts:
    [{"start": int, "end": int, "type": str, "text": str}]
    Returns (spans, errors).
    """
    is_valid, errors = validate_bio_sequence(tokens, tags)
    if not is_valid:
        return [], errors

    spans = []
    curr_type = None
    curr_start = None

    for i, (tok, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            if curr_type is not None:
                spans.append({
                    "start": curr_start,
                    "end": i,
                    "type": curr_type,
                    "text": " ".join(tokens[curr_start:i])
                })
            curr_type = tag[2:]
            curr_start = i
        elif tag.startswith("I-"):
            ent_type = tag[2:]
            if curr_type != ent_type:
                # Fallback if somehow invalid
                if curr_type is not None:
                    spans.append({
                        "start": curr_start,
                        "end": i,
                        "type": curr_type,
                        "text": " ".join(tokens[curr_start:i])
                    })
                curr_type = ent_type
                curr_start = i
        else:  # "O"
            if curr_type is not None:
                spans.append({
                    "start": curr_start,
                    "end": i,
                    "type": curr_type,
                    "text": " ".join(tokens[curr_start:i])
                })
                curr_type = None
                curr_start = None

    if curr_type is not None:
        spans.append({
            "start": curr_start,
            "end": len(tokens),
            "type": curr_type,
            "text": " ".join(tokens[curr_start:len(tokens)])
        })

    return spans, []
