"""
probability_analyzer.py  –  RPN model diagnostic tool

Modes:
  1. Autoregressive generation  (default)
  2. Teacher-forcing analysis   (--teacher-forcing)
  3. Side-by-side diagnosis     (--diagnose)     ← shows where AR diverges from TF
  4. Spot-check from test file  (--spot-check N) ← samples N random lines from test file
                                                    and runs REAL autoregressive eval

Usage examples:
  python3 rpn_llm/probability_analyzer.py "54 12345 + ="
  python3 rpn_llm/probability_analyzer.py "54 12345 + =" --diagnose
  python3 rpn_llm/probability_analyzer.py --spot-check 20
  python3 rpn_llm/probability_analyzer.py --spot-check 20 --filter-len 2,5
"""

import torch
import torch.nn.functional as F
import argparse
import os
import random
import re
from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

# ── ANSI colors ───────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

DEFAULT_CKPT      = "rpn_llm/models/rope25M_mixed-1-22_tens_comp_32000.pt"
DEFAULT_TEST_FILE = "rpn_llm/data/RPNData-mixed-1-22_tens_comp_test.txt"

PROMPT_RE = re.compile(r'\((\d+)\)\((\d+)\)([+\-])=')

# ── Model loader (cached across calls in same process) ────────────────────────
_model_cache = {}

def load_model(ckpt_path, device):
    if ckpt_path in _model_cache:
        return _model_cache[ckpt_path]
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    _model_cache[ckpt_path] = model
    return model


# ── 1. Autoregressive generation ──────────────────────────────────────────────

def generate_with_probabilities(model, tokenizer, device, input_prompt, max_new_tokens=256, silent=False):
    """Run greedy AR generation and return (tokens_str, probs, generated_text)."""
    tokens = tokenizer.encode(input_prompt)
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    generated_tokens = []
    generated_probs  = []

    if not silent:
        print("\n" + "="*80)
        print(f"AUTOREGRESSIVE GENERATION")
        print(f"Prompt: {repr(input_prompt)}")
        print("="*80)

    with torch.no_grad():
        for i in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
            logits, _ = model(idx_cond)
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)

            top_prob, top_id = torch.max(probs, dim=-1)
            top_str = tokenizer.decode([top_id.item()])

            generated_tokens.append(top_str)
            generated_probs.append(top_prob.item())

            idx = torch.cat((idx, top_id.view(1, 1)), dim=1)

            if not silent:
                conf_color = GREEN if top_prob.item() >= 0.80 else (YELLOW if top_prob.item() >= 0.30 else RED)
                print(f"Step {i:3} | Generated: {repr(top_str):8} | Confidence: {conf_color}{top_prob.item()*100:6.2f}%{RESET}")

            if top_str == "\n" or top_str == "[UNK]":
                break
            if top_str in "+-" and i > 3:   # stop after echoed operator
                break

    generated_text = "".join(generated_tokens)
    return generated_tokens, generated_probs, generated_text


# ── 2. Teacher-forcing analysis ───────────────────────────────────────────────

def analyze_teacher_forcing(model, tokenizer, device, input_prompt, target_completion, silent=False):
    """Teacher forcing: how confident is the model at each ground-truth position?"""
    full_text = input_prompt + target_completion
    tokens = tokenizer.encode(full_text)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(x, targets=x)
        probabilities = F.softmax(logits, dim=-1)

    if not silent:
        print("\n" + "="*80)
        print(f"TEACHER-FORCING ANALYSIS")
        print(f"Prompt   : {repr(input_prompt)}")
        print(f"Completion: {repr(target_completion)}")
        print("="*80)

    prompt_len = len(tokenizer.encode(input_prompt))
    tf_tokens = []
    tf_probs  = []
    for i in range(prompt_len - 1, len(tokens) - 1):
        actual_id  = tokens[i+1]
        actual_str = tokenizer.decode([actual_id])
        p          = probabilities[0, i, actual_id].item()
        tf_tokens.append(actual_str)
        tf_probs.append(p)
        if not silent:
            conf_color = GREEN if p >= 0.80 else (YELLOW if p >= 0.30 else RED)
            print(f"Pos {i:3} | Target: {repr(actual_str):8} | Confidence: {conf_color}{p*100:6.2f}%{RESET}")

    return tf_tokens, tf_probs


# ── 3. Side-by-side diagnosis ─────────────────────────────────────────────────

def diagnose(model, tokenizer, device, input_prompt, target_completion):
    """
    Run both AR and TF then show a comparison table.
    Highlights the FIRST position where AR diverges from TF ground truth.
    """
    print("\n" + BOLD + "="*80 + RESET)
    print(BOLD + "DIAGNOSIS MODE  –  AR vs Teacher-Forcing" + RESET)
    print(f"Prompt     : {repr(input_prompt)}")
    print(f"Completion : {repr(target_completion)}")
    print(BOLD + "="*80 + RESET)

    # Run both silently first
    ar_tokens, ar_probs, _ = generate_with_probabilities(
        model, tokenizer, device, input_prompt, silent=True)
    tf_tokens, tf_probs = analyze_teacher_forcing(
        model, tokenizer, device, input_prompt, target_completion, silent=True)

    max_len = max(len(ar_tokens), len(tf_tokens))
    gt_toks = tf_tokens   # ground truth token sequence

    print(f"\n{'Pos':>4} | {'GT token':>10} | {'TF conf':>8} | {'AR token':>10} | {'AR conf':>8} | Status")
    print(f"{'─'*4}-+-{'─'*10}-+-{'─'*8}-+-{'─'*10}-+-{'─'*8}-+-{'─'*6}")

    diverged = False
    for i in range(max_len):
        gt_str  = gt_toks[i]  if i < len(gt_toks)  else "—"
        tf_p    = tf_probs[i] if i < len(tf_probs) else 0.0
        ar_str  = ar_tokens[i] if i < len(ar_tokens) else "—"
        ar_p    = ar_probs[i]  if i < len(ar_probs)  else 0.0

        match = (ar_str == gt_str)
        if not match and not diverged:
            status = RED + "DIVERGE!" + RESET
            diverged = True
        elif not match:
            status = RED + "✗" + RESET
        else:
            status = GREEN + "✓" + RESET

        tf_color = GREEN if tf_p >= 0.80 else (YELLOW if tf_p >= 0.30 else RED)
        ar_color = GREEN if ar_p >= 0.80 else (YELLOW if ar_p >= 0.30 else RED)

        print(f"{i:>4} | {repr(gt_str):>10} | "
              f"{tf_color}{tf_p*100:>7.2f}%{RESET} | "
              f"{repr(ar_str):>10} | "
              f"{ar_color}{ar_p*100:>7.2f}%{RESET} | {status}")

    # Summary
    gt_answer  = target_completion.split('>')[-1].strip() if '>' in target_completion else target_completion.strip()
    ar_answer  = "".join(ar_tokens).split('>')[-1].strip() if '>' in "".join(ar_tokens) else "".join(ar_tokens).strip()
    correct    = (ar_answer == gt_answer)
    result_str = GREEN + "CORRECT ✓" + RESET if correct else RED + "WRONG ✗" + RESET

    print(f"\n  Expected answer : {BOLD}{gt_answer}{RESET}")
    print(f"  AR answer       : {BOLD}{ar_answer}{RESET}  →  {result_str}")

    if not correct:
        # Show which TF positions had low confidence — those are the weak spots
        weak = [(i, gt_toks[i], tf_probs[i]) for i in range(len(tf_probs)) if tf_probs[i] < 0.50]
        if weak:
            print(f"\n  {YELLOW}Low-confidence TF positions (model uncertain even with ground truth context):{RESET}")
            for pos, tok, p in weak:
                print(f"    pos {pos}: {repr(tok)} → {p*100:.1f}%")
        else:
            print(f"\n  {CYAN}TF confidences were all high — the model KNOWS the answer when given context.")
            print(f"  This is a pure AUTOREGRESSIVE error propagation problem.{RESET}")


# ── 4. Spot-check from test file ──────────────────────────────────────────────

def spot_check(model, tokenizer, device, test_file, n=20, filter_lens=None):
    """
    Sample N lines from the test file.
    filter_lens: if set, only consider lines where (len(num1), len(num2)) matches one of the pairs.
    Runs real AR evaluation and reports accuracy.
    Format: filter_lens = [(2,5), (2,2), ...] or None for all.
    """
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}SPOT-CHECK  (sampling {n} examples from test file){RESET}")
    if filter_lens:
        print(f"  Filter: num1_len, num2_len ∈ {filter_lens}")
    print(f"{BOLD}{'='*80}{RESET}\n")

    with open(test_file, 'r', encoding='utf-8') as f:
        all_lines = [l for l in f if l.strip() and '=' in l]

    # Filter by digit-length pair if requested
    if filter_lens:
        filtered = []
        for line in all_lines:
            m = PROMPT_RE.match(line)
            if not m:
                continue
            n1, n2 = m.group(1).lstrip('-'), m.group(2).lstrip('-')
            pair = (len(n1), len(n2))
            if pair in filter_lens:
                filtered.append(line)
        all_lines = filtered
        print(f"  Lines matching filter: {len(all_lines):,}")

    if not all_lines:
        print(RED + "  No matching lines found." + RESET)
        return

    sample = random.sample(all_lines, min(n, len(all_lines)))

    nl_id = tokenizer.encode("\n")[0]
    eq_id = tokenizer.encode("=")[0]

    correct = 0
    for idx_s, line in enumerate(sample):
        eq_idx = line.index("=")
        prompt_str = line[:eq_idx + 1]
        full_target = line[eq_idx + 1:].rstrip()

        # AR generation
        tokens = tokenizer.encode(prompt_str)
        prompt_len = len(tokens)
        ten = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        gen_tokens = []
        with torch.no_grad():
            for _ in range(256):
                logits, _ = model(ten)
                next_id = torch.argmax(logits[0, -1, :]).item()
                gen_str = tokenizer.decode([next_id])
                gen_tokens.append(gen_str)
                ten = torch.cat((ten, torch.tensor([[next_id]], device=device)), dim=1)
                if gen_str == "\n" or gen_str == "[UNK]":
                    break

        ar_out  = "".join(gen_tokens)
        gt_ans  = full_target.split('>')[-1].strip() if '>' in full_target else full_target.strip()
        ar_ans  = ar_out.split('>')[-1].strip() if '>' in ar_out else ar_out.strip()
        ar_ans  = ar_ans.split('[UNK]')[0].split('\n')[0].strip()

        ok = (ar_ans == gt_ans)
        if ok:
            correct += 1
        status = GREEN + "✓" + RESET if ok else RED + "✗" + RESET

        m = PROMPT_RE.match(line)
        lens = f"({len(m.group(1).lstrip('-'))}d, {len(m.group(2).lstrip('-'))}d)" if m else "?"
        print(f"  [{idx_s+1:>3}] {lens} Prompt: {prompt_str.strip()[:50]}")
        print(f"        Expected: {gt_ans[:50]}  |  Got: {ar_ans[:50]}  {status}")

    acc = 100 * correct / len(sample)
    color = GREEN if acc >= 95 else (YELLOW if acc >= 80 else RED)
    print(f"\n  {BOLD}Accuracy: {color}{acc:.1f}%{RESET}  ({correct}/{len(sample)})")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPN Model Probability Analyzer")
    parser.add_argument("prompt", nargs="?", default=None,
                        help="Input prompt (e.g. '54 12345 + =')")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE)
    parser.add_argument("--target", default=None,
                        help="Ground-truth completion (required for --diagnose / --teacher-forcing)")
    parser.add_argument("--diagnose", action="store_true",
                        help="Side-by-side AR vs TF comparison")
    parser.add_argument("--teacher-forcing", action="store_true",
                        help="Run teacher-forcing analysis only")
    parser.add_argument("--spot-check", type=int, default=0, metavar="N",
                        help="Sample N random test examples and evaluate AR accuracy")
    parser.add_argument("--filter-len", default=None,
                        help="Comma-separated digit-length pairs for --spot-check, e.g. '2,5 2,2'")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else \
             'mps'  if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    if not os.path.exists(args.ckpt):
        print(RED + f"Checkpoint not found: {args.ckpt}" + RESET)
        exit(1)

    model     = load_model(args.ckpt, device)
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")

    # Parse filter-len
    filter_lens = None
    if args.filter_len:
        pairs = args.filter_len.strip().split()
        filter_lens = []
        for p in pairs:
            parts = p.split(',')
            if len(parts) == 2:
                filter_lens.append((int(parts[0]), int(parts[1])))

    if args.spot_check > 0:
        spot_check(model, tokenizer, device, args.test_file, n=args.spot_check, filter_lens=filter_lens)
    elif args.diagnose:
        if not args.prompt or not args.target:
            print(RED + "--diagnose requires both a prompt and --target <completion>" + RESET)
            exit(1)
        diagnose(model, tokenizer, device, args.prompt, args.target)
    elif args.teacher_forcing:
        if not args.prompt or not args.target:
            print(RED + "--teacher-forcing requires both a prompt and --target <completion>" + RESET)
            exit(1)
        analyze_teacher_forcing(model, tokenizer, device, args.prompt, args.target)
    else:
        prompt = args.prompt or "<54874 818591913 |"
        generate_with_probabilities(model, tokenizer, device, prompt)
