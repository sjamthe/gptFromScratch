#!/usr/bin/env python3
"""
Digit Frequency Distribution Analyzer for RPN Dataset.

Analyzes the frequency of each digit (0-9) in number1 and number2 across
training and test datasets, broken down by digit-length bucket.

This helps diagnose hallucination issues caused by positional/digit bias.

Usage:
    python digit_frequency_analysis.py [--train] [--test] [--val]
                                        [--max-lines N] [--buckets N1,N2,N3]
"""

import re
import sys
import argparse
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

# ── ANSI colors ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Default data files (the mixed-1-22 tens_comp dataset used for the 32k-step run)
DEFAULT_FILES = {
    "train": os.path.join(DATA_DIR, "RPNData-mixed-1-22_tens_comp_train.txt"),
    "val":   os.path.join(DATA_DIR, "RPNData-mixed-1-22_tens_comp_val.txt"),
    "test":  os.path.join(DATA_DIR, "RPNData-mixed-1-22_tens_comp_test.txt"),
}

# ── Parser ────────────────────────────────────────────────────────────────────
# The prompt part of each line is BEFORE the first '=' sign:
#   "  54   12345   + ="
# num1 = first integer token, num2 = second integer token, op = + or -
PROMPT_RE = re.compile(
    r'^\s*(-?\d+)\s+(-?\d+)\s+([+\-])\s*='
)


def parse_line(line: str) -> Optional[Tuple[str, str, str]]:
    """Return (num1_str, num2_str, op) from a dataset line, or None."""
    m = PROMPT_RE.match(line)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


# ── Core analysis ─────────────────────────────────────────────────────────────

def analyze_file(path: str, max_lines: int = 0) -> Dict:
    """
    Return a nested stats dict:
      stats[split_label][num_idx]['digit_dist']  -> Counter of digit chars in all numbers of that position
      stats[split_label][num_idx]['length_dist'] -> Counter of number lengths
      stats[split_label][num_idx]['bucket_digit'] -> dict[bucket] -> Counter of digits
    where num_idx is 0 (num1) or 1 (num2).
    """
    # Per-position counters
    digit_dist   = [Counter(), Counter()]       # [num1, num2]
    length_dist  = [Counter(), Counter()]
    bucket_digit = [defaultdict(Counter), defaultdict(Counter)]
    op_counter   = Counter()
    total = 0
    skipped = 0

    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            parsed = parse_line(line)
            if parsed is None:
                skipped += 1
                continue
            n1, n2, op = parsed
            op_counter[op] += 1
            total += 1

            for idx, num_str in enumerate([n1, n2]):
                # Strip leading minus if any
                digits_only = num_str.lstrip('-')
                L = len(digits_only)
                length_dist[idx][L] += 1

                # Bucket by length (each length is its own bucket here)
                for ch in digits_only:
                    digit_dist[idx][ch] += 1
                    bucket_digit[idx][L][ch] += 1

    return {
        "total": total,
        "skipped": skipped,
        "op_counter": op_counter,
        "digit_dist": digit_dist,
        "length_dist": length_dist,
        "bucket_digit": bucket_digit,
    }


def expected_uniform(total_digits: int) -> float:
    return total_digits / 10.0


def deviation_color(pct_dev: float) -> str:
    """Color code based on % deviation from uniform."""
    if abs(pct_dev) < 3:
        return GREEN
    if abs(pct_dev) < 8:
        return YELLOW
    return RED


# ── Pretty printers ──────────────────────────────────────────────────────────

def print_overall_digit_dist(label: str, digit_dist: List[Counter]):
    """Print side-by-side digit distribution for num1 vs num2."""
    print(f"\n{BOLD}{CYAN}{'─'*80}{RESET}")
    print(f"{BOLD}{CYAN}  [{label}]  Overall Digit Frequency Distribution{RESET}")
    print(f"{BOLD}{CYAN}{'─'*80}{RESET}")
    print(f"{'Digit':>8} | {'num1 Count':>14} {'num1 %':>8} {'Dev':>7} || {'num2 Count':>14} {'num2 %':>8} {'Dev':>7}")
    print(f"{'─'*8}-+-{'─'*14}-{'─'*8}-{'─'*7}-++-{'─'*14}-{'─'*8}-{'─'*7}")

    totals = [sum(d.values()) for d in digit_dist]
    uniform_pcts = [100/10 for _ in range(2)]  # 10%

    for dig in '0123456789':
        row = []
        for idx in range(2):
            cnt = digit_dist[idx].get(dig, 0)
            pct = 100 * cnt / totals[idx] if totals[idx] else 0
            dev = pct - uniform_pcts[idx]
            color = deviation_color(dev)
            row.append((cnt, pct, dev, color))

        # Format
        c1, p1, d1, col1 = row[0]
        c2, p2, d2, col2 = row[1]
        dev1_str = f"{col1}{d1:+.2f}%{RESET}"
        dev2_str = f"{col2}{d2:+.2f}%{RESET}"
        print(f"  {dig:>6}   | {c1:>14,} {p1:>7.3f}% {dev1_str:>18} || {c2:>14,} {p2:>7.3f}% {dev2_str:>18}")

    print(f"  {'Total':>6}   | {totals[0]:>14,} {'100.000%':>8}          || {totals[1]:>14,} {'100.000%':>8}")


def print_length_distribution(label: str, length_dist: List[Counter]):
    """Print number-of-digits distribution for num1 and num2."""
    print(f"\n{BOLD}{CYAN}{'─'*80}{RESET}")
    print(f"{BOLD}{CYAN}  [{label}]  Number Length Distribution (# digits){RESET}")
    print(f"{BOLD}{CYAN}{'─'*80}{RESET}")

    all_lengths = sorted(set(length_dist[0].keys()) | set(length_dist[1].keys()))
    totals = [sum(d.values()) for d in length_dist]

    print(f"{'Length':>8} | {'num1 Count':>14} {'num1 %':>8} || {'num2 Count':>14} {'num2 %':>8}")
    print(f"{'─'*8}-+-{'─'*14}-{'─'*8}-++-{'─'*14}-{'─'*8}")
    for L in all_lengths:
        c1 = length_dist[0].get(L, 0)
        c2 = length_dist[1].get(L, 0)
        p1 = 100 * c1 / totals[0] if totals[0] else 0
        p2 = 100 * c2 / totals[1] if totals[1] else 0
        print(f"  {L:>6}   | {c1:>14,} {p1:>7.3f}% || {c2:>14,} {p2:>7.3f}%")
    print(f"  {'Total':>6}   | {totals[0]:>14,}           || {totals[1]:>14,}")


def print_bucket_analysis(label: str, bucket_digit: List, buckets: List[int]):
    """For each length bucket, print digit distribution of num1 vs num2."""
    print(f"\n{BOLD}{CYAN}{'─'*80}{RESET}")
    print(f"{BOLD}{CYAN}  [{label}]  Digit Distribution Bucketed by Number Length{RESET}")
    print(f"{BOLD}{CYAN}{'─'*80}{RESET}")

    for L in buckets:
        d1 = bucket_digit[0].get(L, Counter())
        d2 = bucket_digit[1].get(L, Counter())
        t1 = sum(d1.values())
        t2 = sum(d2.values())
        if t1 == 0 and t2 == 0:
            continue

        print(f"\n  {BOLD}Length = {L} digit{'s' if L != 1 else ''}{RESET}  "
              f"(num1 total_digits={t1:,}, num2 total_digits={t2:,})")
        print(f"  {'Digit':>5} | {'num1 %':>8} {'Dev':>7} || {'num2 %':>8} {'Dev':>7}")
        print(f"  {'─'*5}-+-{'─'*8}-{'─'*7}-++-{'─'*8}-{'─'*7}")

        for dig in '0123456789':
            p1 = 100 * d1.get(dig, 0) / t1 if t1 else 0
            p2 = 100 * d2.get(dig, 0) / t2 if t2 else 0
            dev1 = p1 - 10.0
            dev2 = p2 - 10.0
            col1, col2 = deviation_color(dev1), deviation_color(dev2)
            d1s = f"{col1}{dev1:+.2f}%{RESET}"
            d2s = f"{col2}{dev2:+.2f}%{RESET}"
            print(f"  {dig:>5}   | {p1:>7.3f}% {d1s:>18} || {p2:>7.3f}% {d2s:>18}")


def print_leading_digit_analysis(label: str, path: str, max_lines: int = 0):
    """Benford's law check: frequency of LEADING digits only."""
    print(f"\n{BOLD}{CYAN}{'─'*80}{RESET}")
    print(f"{BOLD}{CYAN}  [{label}]  Leading Digit Distribution (Benford's Law reference){RESET}")
    print(f"{BOLD}{CYAN}{'─'*80}{RESET}")
    print(f"  Expected uniform (our generator) for leading digit: each digit ~11.1%")

    lead_dist = [Counter(), Counter()]
    total = 0
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            parsed = parse_line(line)
            if parsed is None:
                continue
            n1, n2, _ = parsed
            total += 1
            for idx, ns in enumerate([n1, n2]):
                ds = ns.lstrip('-')
                if ds:
                    lead_dist[idx][ds[0]] += 1

    totals = [sum(d.values()) for d in lead_dist]
    print(f"  {'Digit':>5} | {'num1 Count':>12} {'num1 %':>8} || {'num2 Count':>12} {'num2 %':>8}")
    print(f"  {'─'*5}-+-{'─'*12}-{'─'*8}-++-{'─'*12}-{'─'*8}")
    for dig in '123456789':  # leading digit can't be 0 for positive integers > 0
        c1 = lead_dist[0].get(dig, 0)
        c2 = lead_dist[1].get(dig, 0)
        p1 = 100 * c1 / totals[0] if totals[0] else 0
        p2 = 100 * c2 / totals[1] if totals[1] else 0
        # For uniform-length generation, expected % depends on digit pool
        print(f"  {dig:>5}   | {c1:>12,} {p1:>7.3f}% || {c2:>12,} {p2:>7.3f}%")
    print(f"  {'Total':>5}   | {totals[0]:>12,}          || {totals[1]:>12,}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RPN Dataset Digit Frequency Analyzer")
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--val",   action="store_true", default=False)
    parser.add_argument("--test",  action="store_true", default=False)
    parser.add_argument("--max-lines", type=int, default=500_000,
                        help="Max lines to process per file (0=all). Default=500000")
    parser.add_argument("--buckets", type=str, default="1,2,3,4,5,6,7,8,9,10,11,12,15,22",
                        help="Comma-separated list of digit-lengths to show bucket details for")
    parser.add_argument("--file", type=str, default=None,
                        help="Custom file path to analyze (overrides --train/--val/--test)")
    args = parser.parse_args()

    # Default: analyze train if nothing specified
    if not args.train and not args.val and not args.test and not args.file:
        args.train = True
        args.test = True

    buckets = [int(x) for x in args.buckets.split(",")]

    splits_to_run = []
    if args.file:
        splits_to_run.append(("custom", args.file))
    else:
        if args.train: splits_to_run.append(("TRAIN", DEFAULT_FILES["train"]))
        if args.val:   splits_to_run.append(("VAL",   DEFAULT_FILES["val"]))
        if args.test:  splits_to_run.append(("TEST",  DEFAULT_FILES["test"]))

    for label, path in splits_to_run:
        if not os.path.exists(path):
            print(f"{RED}[ERROR] File not found: {path}{RESET}")
            continue

        print(f"\n{BOLD}{'='*80}{RESET}")
        print(f"{BOLD}  Analyzing: {label}  →  {os.path.basename(path)}{RESET}")
        ml_str = f"{args.max_lines:,}" if args.max_lines else "ALL"
        print(f"  Max lines : {ml_str}")
        print(f"{BOLD}{'='*80}{RESET}")

        stats = analyze_file(path, max_lines=args.max_lines)

        print(f"\n  Lines parsed : {stats['total']:,}")
        print(f"  Lines skipped: {stats['skipped']:,}")
        print(f"  Op distribution: ", end="")
        for op, cnt in stats['op_counter'].most_common():
            print(f"  {op}={cnt:,} ({100*cnt/stats['total']:.1f}%)", end="")
        print()

        print_overall_digit_dist(label, stats['digit_dist'])
        print_length_distribution(label, stats['length_dist'])
        print_bucket_analysis(label, stats['bucket_digit'], buckets)
        print_leading_digit_analysis(label, path, max_lines=args.max_lines)

    print(f"\n{BOLD}{GREEN}Analysis complete.{RESET}\n")


if __name__ == "__main__":
    main()
