"""
preprocessing/fasta_parser.py
══════════════════════════════
Parse and validate FASTA protein sequence files.
"""

import logging
log = logging.getLogger(__name__)


import os

VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def parse_fasta(path: str) -> dict[str, str]:
    """
    Parse a FASTA file into {header_id: sequence} dict.
    Header is the first whitespace-delimited token after '>'.
    Multi-line sequences are concatenated and uppercased.
    """
    sequences: dict[str, str] = {}
    current_id: str | None = None
    current_seq: list[str] = []

    with open(path, "r") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                # Use only the first token as ID (safer for downstream tools)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.upper())

    if current_id is not None:
        sequences[current_id] = "".join(current_seq)

    log.info(f"  Parsed {len(sequences):,} sequences from {path}")
    return sequences


def validate_sequences(sequences: dict[str, str],
                       min_len: int = 10,
                       output_dir: str = "output") -> dict[str, str]:
    """
    Return only sequences that pass all quality gates.
    Every rejected sequence is logged with its reason.

    Gates (in order):
      1. Length ≥ min_len residues
      2. Only standard 20-letter amino acid alphabet
         (X, B, Z, U and other ambiguous chars are rejected here;
          masking later handles low-complexity, not here)

    Parameters
    ----------
    sequences  : raw parsed sequences
    min_len    : minimum sequence length (default 10)
    output_dir : directory to write dropped.tsv

    Returns
    -------
    clean dict[seq_id -> sequence]
    """
    clean: dict[str, str] = {}
    dropped: list[tuple[str, str]] = []

    for seq_id, seq in sequences.items():
        if len(seq) < min_len:
            dropped.append((seq_id,
                            f"too short ({len(seq)} aa; min={min_len})"))
            continue

        bad_chars = sorted({c for c in seq if c not in VALID_AA})
        if bad_chars:
            dropped.append((seq_id,
                            f"invalid characters: {bad_chars}"))
            continue

        clean[seq_id] = seq

    if dropped:
        log.info(f"\n  ⚠  {len(dropped)} sequence(s) dropped during validation:")
        for seq_id, reason in dropped[:15]:
            log.info(f"     DROPPED  {seq_id[:55]:<55}  →  {reason}")
        if len(dropped) > 15:
            log.info(f"     ... and {len(dropped) - 15} more → see output/dropped.tsv")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "dropped.tsv"), "w") as fh:
            fh.write("seq_id\treason\n")
            for seq_id, reason in dropped:
                fh.write(f"{seq_id}\t{reason}\n")

    log.info(f"  Validation: {len(clean):,} passed  |  {len(dropped):,} dropped")
    return clean
