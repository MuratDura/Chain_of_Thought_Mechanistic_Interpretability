# analysis_cot_vs_nocot.py
# Teacher-forcing ile CoT vs No-CoT katman bazlı logit-lens analizi
# - final LN opsiyonu
# - gold rank / probability / margin / entropy / KL
# - katman bazlı grafikler
# - layerwise top-k örnek dökümü

import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from transformer_lens import HookedTransformer
from svamp_dataset import SVAMPDataset, SVAMPQuestion
from tqdm.auto import tqdm
# Removed circular import

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== MODEL YÜKLEME ==========

def load_model():
    model = HookedTransformer.from_pretrained(
            "Qwen/Qwen2.5-0.5b", device=device
        )
    print("Loaded Qwen/Qwen2.5-0.5b via from_pretrained")
    return model

# ========== YARDIMCILAR ==========

def to_ids(model: HookedTransformer, txt: str) -> List[int]:
    return model.tokenizer.encode(txt, add_special_tokens=False)

def run_with_cache_ids(model: HookedTransformer, ids: List[int]):
    toks = torch.tensor([ids], device=device)
    logits, cache = model.run_with_cache(toks)
    return toks, logits, cache

def _anchor_last_token_index(
    model: HookedTransformer,
    prompt_text: str,
    anchor_phrases: Optional[List[str]] = None,
) -> Optional[int]:
    """
    Return the index (in token space) of the last token of the last-matching anchor phrase
    within the prompt. Supports multiple variants (case-insensitive). If not found, return None.
    This computes the position robustly by tokenizing the substring up to the end of the anchor.
    """
    if anchor_phrases is None or len(anchor_phrases) == 0:
        anchor_phrases = [
            "the answer is ",
            "The answer is ",
            "the answer is",
            "The answer is",
        ]

    # Case-insensitive search across provided variants
    lower_prompt = prompt_text.lower()
    best_pos = -1
    best_len = 0
    for phrase in anchor_phrases:
        lower_anchor = phrase.lower()
        char_pos = lower_prompt.rfind(lower_anchor)
        if char_pos > best_pos:
            best_pos = char_pos
            best_len = len(phrase)
    if best_pos == -1:
        return None
    end_char = best_pos + best_len
    prefix_text = prompt_text[:end_char]
    prefix_ids = to_ids(model, prefix_text)
    return len(prefix_ids) - 1 if len(prefix_ids) > 0 else None

def resid_logits_at_position(
    model: HookedTransformer,
    cache,
    pos: int,
    use_final_ln: bool,
) -> Dict[int, torch.Tensor]:
    """
    Her katmanda resid_post'u al, opsiyonel final LN uygula, W_U ile projekte et.
    Dönen: { layer_idx -> logits[vocab] }
    """
    W_U = model.W_U
    out = {}
    for l in range(model.cfg.n_layers):
        r = cache["resid_post", l][0, pos, :]  # [d_model]
        if use_final_ln:
            r = model.ln_final(r)
        logits = r @ W_U  # [vocab]
        out[l] = logits
    return out

# Removed unused functions: entropy_from_logits, kl_pq, cosine_to_gold_unemb

def rank_of_id(logits: torch.Tensor, token_id: int) -> int:
    # 1-based rank: kaç logit daha büyükse +1
    return int((logits > logits[token_id]).sum().item() + 1)

def calculate_entropy(logits: torch.Tensor) -> float:
    """
    Calculate entropy from logits using PyTorch.
    Entropy = -sum(p_i * log(p_i)) where p_i is the probability of token i
    """
    probs = F.softmax(logits, dim=-1)
    # Add small epsilon to avoid log(0)
    probs = probs + 1e-8
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    return float(entropy.item())

def calculate_perplexity(logits: torch.Tensor) -> float:
    """
    Calculate perplexity from logits.
    Perplexity = exp(entropy)
    """
    entropy = calculate_entropy(logits)
    return math.exp(entropy)
def extract_first_number(text: str) -> Optional[float]:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def format_numeric_for_display(value: Optional[float]) -> str:
    """
    If value is an integer-valued float (e.g., 2.0), display as integer ("2").
    Otherwise, display as the float string. If None, return "N/A".
    """
    if value is None:
        return "N/A"
    if float(value).is_integer():
        return str(int(round(value)))
    return str(value)


def numeric_equal(a: Optional[float], b: Optional[float], tol: float = 1e-3) -> bool:
    """
    Equality for numeric answers:
    - If both are near-integers, compare as integers (handles 2 vs 2.0)
    - Else fall back to absolute tolerance on floats
    """
    if a is None or b is None:
        return False
    a_is_int = float(a).is_integer()
    b_is_int = float(b).is_integer()
    if a_is_int and b_is_int:
        return int(round(a)) == int(round(b))
    return abs(a - b) < tol


def greedy_generate_text(
    model: HookedTransformer,
    prompt_text: str,
    *,
    max_new_tokens: int = 16,
    stop_strings: Optional[List[str]] = None,
    stop_after_anchor_next_token: bool = False,
    stop_after_anchor_k_tokens: Optional[int] = None,
    stop_after_anchor_min_digits: Optional[int] = None,
    anchor_phrases: Optional[List[str]] = None,
) -> str:
    ids = to_ids(model, prompt_text)
    generated_ids: List[int] = []

    anchor_phrases = anchor_phrases or [
        "the answer is ",
        "The answer is ",
        "the answer is",
        "The answer is",
    ]
    anchor_lower = [p.lower() for p in anchor_phrases]

    anchor_seen = False
    post_anchor_tokens = 0
    anchor_end_char_idx: Optional[int] = None

    # If K is provided, it overrides next-token behavior
    if stop_after_anchor_k_tokens is not None and stop_after_anchor_k_tokens <= 0:
        stop_after_anchor_k_tokens = None
    use_k = stop_after_anchor_k_tokens is not None or stop_after_anchor_next_token
    k_tokens = (
        stop_after_anchor_k_tokens
        if stop_after_anchor_k_tokens is not None
        else (1 if stop_after_anchor_next_token else None)
    )

    def _find_last_anchor_span(text_lower: str) -> Optional[Tuple[int, int]]:
        best_pos = -1
        best_len = 0
        for a in anchor_lower:
            p = text_lower.rfind(a)
            if p > best_pos:
                best_pos = p
                best_len = len(a)
        if best_pos == -1:
            return None
        return best_pos, best_len

    if use_k:
        lower_prompt = prompt_text.lower()
        last_q = lower_prompt.rfind("q:")
        suffix = lower_prompt[last_q:] if last_q != -1 else lower_prompt
        if any(a in suffix for a in anchor_lower):
            # Prompt already ends with an anchor for the current question
            anchor_seen = True
            span = _find_last_anchor_span(suffix)
            if span is not None:
                # Compute index within the generated text space (prompt has no generated yet)
                anchor_end_char_idx = len(model.tokenizer.decode([]))  # 0 for generated part

    # Also initialize anchor state for digit-based stopping when anchor is in the prompt
    if stop_after_anchor_min_digits is not None and stop_after_anchor_min_digits > 0 and not anchor_seen:
        lower_prompt = prompt_text.lower()
        last_q = lower_prompt.rfind("q:")
        suffix = lower_prompt[last_q:] if last_q != -1 else lower_prompt
        span = _find_last_anchor_span(suffix)
        if span is not None:
            anchor_seen = True
            anchor_end_char_idx = 0

    for _ in range(max_new_tokens):
        toks = torch.tensor([ids + generated_ids], device=device)
        logits = model(toks)
        next_id = int(torch.argmax(logits[0, -1, :]).item())
        generated_ids.append(next_id)

        # Priority 1: digit-based stopping after anchor
        if stop_after_anchor_min_digits is not None and stop_after_anchor_min_digits > 0:
            txt = model.tokenizer.decode(generated_ids)
            lower_txt = txt.lower()
            if not anchor_seen:
                span = _find_last_anchor_span(lower_txt)
                if span is not None:
                    anchor_seen = True
                    anchor_end_char_idx = span[0] + span[1]
            if anchor_seen and anchor_end_char_idx is not None:
                after = txt[anchor_end_char_idx:]
                digit_count = sum(ch.isdigit() for ch in after)
                if digit_count >= stop_after_anchor_min_digits:
                    return txt
            continue

        if use_k:
            if anchor_seen:
                post_anchor_tokens += 1
                if k_tokens is not None and post_anchor_tokens >= k_tokens:
                    return model.tokenizer.decode(generated_ids)
            else:
                txt_lower = model.tokenizer.decode(generated_ids).lower()
                span = _find_last_anchor_span(txt_lower)
                if span is not None:
                    anchor_seen = True
                    post_anchor_tokens = 0
            continue

        if stop_strings:
            txt_lower = model.tokenizer.decode(generated_ids).lower()
            for s in stop_strings:
                if s and s.lower() in txt_lower:
                    return model.tokenizer.decode(generated_ids)

    return model.tokenizer.decode(generated_ids)


def build_prompts(question: SVAMPQuestion) -> Tuple[str, str]:
    q = question.question.strip()
    # Lower-case anchor used for alignment downstream
    nocot = f"""Q: Roger has 5 tennis balls. He buys 2 more cans of
tennis balls. Each can has 3 tennis balls. How many
tennis balls does he have now?
A: The answer is 11.
Q:There were 30 pencils in a box. Ashley took 8 pencils from the box.
 Ashley added 6 pencils to the box. How many pencils are in the box now?
A: The answer is 28.
Q: {q}\nA: The answer is """
    cot = (
        f"""Q: Roger has 5 tennis balls. He buys 2 more cans of
tennis balls. Each can has 3 tennis balls. How many
tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls
each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
Q:There were 30 pencils in a box. Ashley took 8 pencils from the box.
 Ashley added 6 pencils to the box. How many pencils are in the box now?
A: Ashley has 30 pencils. She took 8 pencils from the box. 30 - 8 = 22. She added 6 pencils to the box. 22 + 6 = 28. The answer is 28. Q: {q}\nA:"""
    )
    return nocot, cot


def build_prompts_from_prefix(
    question: SVAMPQuestion,
    nocot_prefix: str,
    cot_prefix: str,
) -> Tuple[str, str]:
    """
    Few-shot promptlarını kullanarak yeni soru için CoT/No-CoT promptları üretir.
    Beklenen: prefix'ler "... The answer is 3\nQ: " ile biter.
    """
    q = question.question.strip()
    # No-CoT: few-shot + yeni soru, direkt cevap formatı
    nocot = f"{nocot_prefix}\n"
    # CoT: few-shot + yeni soru, adım adım + cevap
    cot = (
        f"{cot_prefix}\n"
    )
    return nocot, cot


def extract_answer_number_from_generation(
    prompt_text: str, completion_text: str
) -> Optional[float]:
    """
    Heuristic: Prefer the number immediately following the last occurrence of
    'the answer is' (any case). Ignores trailing punctuation like '.'
    If not present, fallback to last number after the last 'Q:' marker.
    """
    combined = f"{prompt_text}{completion_text}"
    region_start = combined.rfind("Q:")
    region = combined[region_start:] if region_start != -1 else combined
    lower_region = region.lower()
    anchor = "the answer is "
    a_pos = lower_region.rfind(anchor)
    if a_pos != -1:
        after = region[a_pos + len(anchor):]
        # Strictly parse the number right after the anchor, ignoring trailing '.' etc.
        m = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)", after)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    # Fallback: choose the last number in the region
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", region)
    if nums:
        try:
            return float(nums[-1])
        except Exception:
            return None
    return None


def run_svamp_experiment(
    model: HookedTransformer,
    *,
    json_path: str = "SVAMP.json",
    limit: Optional[int] = None,
    anchor_phrase: str = "The answer is ",
    use_final_ln: bool = True,
    show_plots: bool = True,
    nocot_prefix: Optional[str] = None,
    cot_prefix: Optional[str] = None,
    print_generations: bool = True,
) -> Dict[str, Any]:
    ds = SVAMPDataset(json_path)
    n_total = len(ds) if limit is None else min(limit, len(ds))

    correct_nocot = 0
    correct_cot = 0

    # Aggregates per layer for first answer token metrics
    n_layers = model.cfg.n_layers
    sum_delta: List[float] = [0.0] * n_layers
    sum_p_no: List[float] = [0.0] * n_layers
    sum_p_cot: List[float] = [0.0] * n_layers
    sum_ent_no: List[float] = [0.0] * n_layers
    sum_ent_cot: List[float] = [0.0] * n_layers
    sum_ppx_no: List[float] = [0.0] * n_layers
    sum_ppx_cot: List[float] = [0.0] * n_layers
    count_items = 0

    # Aggregates per layer per answer-token-index (for multi-digit answers)
    per_token_sums: Dict[int, Dict[str, List[float]]] = {}
    per_token_counts: Dict[int, int] = {}

    # Correct-only (CoT-correct) aggregates
    sum_delta_correct: List[float] = [0.0] * n_layers
    sum_p_no_correct: List[float] = [0.0] * n_layers
    sum_p_cot_correct: List[float] = [0.0] * n_layers
    sum_ent_no_correct: List[float] = [0.0] * n_layers
    sum_ent_cot_correct: List[float] = [0.0] * n_layers
    sum_ppx_no_correct: List[float] = [0.0] * n_layers
    sum_ppx_cot_correct: List[float] = [0.0] * n_layers
    count_items_correct = 0

    per_token_sums_correct: Dict[int, Dict[str, List[float]]] = {}
    per_token_counts_correct: Dict[int, int] = {}

    def _ensure_token_slot(tok_index: int) -> None:
        if tok_index not in per_token_sums:
            per_token_sums[tok_index] = {
                "sum_delta": [0.0] * n_layers,
                "sum_p_no": [0.0] * n_layers,
                "sum_p_cot": [0.0] * n_layers,
                "sum_ent_no": [0.0] * n_layers,
                "sum_ent_cot": [0.0] * n_layers,
                "sum_ppx_no": [0.0] * n_layers,
                "sum_ppx_cot": [0.0] * n_layers,
            }
        if tok_index not in per_token_counts:
            per_token_counts[tok_index] = 0

    for idx in tqdm(range(n_total), total=n_total, desc="SVAMP two-ops"):
        item = ds[idx]
        if nocot_prefix is not None and cot_prefix is not None:
            nocot_prompt, cot_prompt = build_prompts_from_prefix(item, nocot_prefix, cot_prefix)
        else:
            nocot_prompt, cot_prompt = build_prompts(item)

        # Decide digit count from gold answer before generation
        gold = extract_first_number(item.answer)
        gold_str = format_numeric_for_display(gold) if gold is not None else ""
        min_digits = sum(ch.isdigit() for ch in gold_str) if gold_str else 1
        if min_digits <= 0:
            min_digits = 1

        # Greedy prediction for No-CoT with min-digit stopping
        cont_no = greedy_generate_text(
            model,
            nocot_prompt,
            max_new_tokens=100,
            stop_after_anchor_min_digits=min_digits,
            anchor_phrases=[anchor_phrase, anchor_phrase.strip(), "The answer is ", "the answer is "],
        )
        pred_no = extract_answer_number_from_generation(nocot_prompt, cont_no)

        # Greedy prediction for CoT with min-digit stopping
        cont_cot = greedy_generate_text(
            model,
            cot_prompt,
            max_new_tokens=200,
            stop_after_anchor_min_digits=min_digits,
            anchor_phrases=[anchor_phrase, anchor_phrase.strip(), "The answer is ", "the answer is "],
        )
        pred_cot = extract_answer_number_from_generation(cot_prompt, cont_cot)

        if print_generations:
            print("-" * 80)
            print(f"[{idx+1}/{n_total}] Q: {item.question}")
            print(f"  Gold: {format_numeric_for_display(gold)}")
            print(f"  NoCoT → {cont_no.strip()}")
            print(f"  CoT   → {cont_cot.strip()}")

        if numeric_equal(pred_no, gold):
            correct_nocot += 1
        if numeric_equal(pred_cot, gold):
            correct_cot += 1

        # Mechanistic experiment metrics per layer
        # Use teacher forcing with gold answer text; include CoT reasoning up to anchor
        gold_text = format_numeric_for_display(gold) if gold is not None else item.answer

        cont_cot_lower = cont_cot.lower()
        anchor_lower = anchor_phrase.lower()
        a_pos = cont_cot_lower.rfind(anchor_lower)
        cot_prefix_until_anchor = cont_cot[: a_pos + len(anchor_phrase)] if a_pos != -1 else ""
        cot_prompt_tf = cot_prompt + cot_prefix_until_anchor

        metrics, _ = analyze_pair(
            model,
            nocot_prompt,
            cot_prompt_tf,
            gold_text,
            use_final_ln=use_final_ln,
            topk_debug_layer=None,
            show_plots=False,
            anchor_phrase=anchor_phrase,
        )
        # First-token aggregates (backward compat with older plots)
        ms_first = [m for m in metrics if m.tok_index_in_answer == 0]
        if len(ms_first) > 0:
            for l in range(n_layers):
                m = ms_first[l]
                sum_delta[l] += m.delta_logit_gold
                sum_p_no[l] += m.p_gold_no
                sum_p_cot[l] += m.p_gold_cot
                sum_ent_no[l] += m.entropy_no
                sum_ent_cot[l] += m.entropy_cot
                sum_ppx_no[l] += m.perplexity_no
                sum_ppx_cot[l] += m.perplexity_cot
            count_items += 1

            # If CoT was correct, also add to correct-only aggregates
            if numeric_equal(pred_cot, gold):
                for l in range(n_layers):
                    m = ms_first[l]
                    sum_delta_correct[l] += m.delta_logit_gold
                    sum_p_no_correct[l] += m.p_gold_no
                    sum_p_cot_correct[l] += m.p_gold_cot
                    sum_ent_no_correct[l] += m.entropy_no
                    sum_ent_cot_correct[l] += m.entropy_cot
                    sum_ppx_no_correct[l] += m.perplexity_no
                    sum_ppx_cot_correct[l] += m.perplexity_cot
                count_items_correct += 1

        # Per-token aggregates across all answer tokens
        # Group metrics by tok_index_in_answer
        tok_to_layers: Dict[int, Dict[int, LayerMetrics]] = {}
        for m in metrics:
            d = tok_to_layers.setdefault(m.tok_index_in_answer, {})
            d[m.layer] = m
        for tok_idx, layer_map in tok_to_layers.items():
            _ensure_token_slot(tok_idx)
            for l in range(n_layers):
                m = layer_map.get(l)
                if m is None:
                    continue
                per_token_sums[tok_idx]["sum_delta"][l] += m.delta_logit_gold
                per_token_sums[tok_idx]["sum_p_no"][l] += m.p_gold_no
                per_token_sums[tok_idx]["sum_p_cot"][l] += m.p_gold_cot
                per_token_sums[tok_idx]["sum_ent_no"][l] += m.entropy_no
                per_token_sums[tok_idx]["sum_ent_cot"][l] += m.entropy_cot
                per_token_sums[tok_idx]["sum_ppx_no"][l] += m.perplexity_no
                per_token_sums[tok_idx]["sum_ppx_cot"][l] += m.perplexity_cot
            per_token_counts[tok_idx] += 1

            # Correct-only per-token aggregates (CoT correct)
            if numeric_equal(pred_cot, gold):
                if tok_idx not in per_token_sums_correct:
                    per_token_sums_correct[tok_idx] = {
                        "sum_delta": [0.0] * n_layers,
                        "sum_p_no": [0.0] * n_layers,
                        "sum_p_cot": [0.0] * n_layers,
                        "sum_ent_no": [0.0] * n_layers,
                        "sum_ent_cot": [0.0] * n_layers,
                        "sum_ppx_no": [0.0] * n_layers,
                        "sum_ppx_cot": [0.0] * n_layers,
                    }
                if tok_idx not in per_token_counts_correct:
                    per_token_counts_correct[tok_idx] = 0
                for l in range(n_layers):
                    m = layer_map.get(l)
                    if m is None:
                        continue
                    per_token_sums_correct[tok_idx]["sum_delta"][l] += m.delta_logit_gold
                    per_token_sums_correct[tok_idx]["sum_p_no"][l] += m.p_gold_no
                    per_token_sums_correct[tok_idx]["sum_p_cot"][l] += m.p_gold_cot
                    per_token_sums_correct[tok_idx]["sum_ent_no"][l] += m.entropy_no
                    per_token_sums_correct[tok_idx]["sum_ent_cot"][l] += m.entropy_cot
                    per_token_sums_correct[tok_idx]["sum_ppx_no"][l] += m.perplexity_no
                    per_token_sums_correct[tok_idx]["sum_ppx_cot"][l] += m.perplexity_cot
                per_token_counts_correct[tok_idx] += 1

    acc_no = correct_nocot / n_total if n_total > 0 else 0.0
    acc_cot = correct_cot / n_total if n_total > 0 else 0.0

    avg_delta = [v / max(count_items, 1) for v in sum_delta]
    avg_p_no = [v / max(count_items, 1) for v in sum_p_no]
    avg_p_cot = [v / max(count_items, 1) for v in sum_p_cot]
    avg_ent_no = [v / max(count_items, 1) for v in sum_ent_no]
    avg_ent_cot = [v / max(count_items, 1) for v in sum_ent_cot]
    avg_ppx_no = [v / max(count_items, 1) for v in sum_ppx_no]
    avg_ppx_cot = [v / max(count_items, 1) for v in sum_ppx_cot]

    # Correct-only averages
    avg_delta_correct = [v / max(count_items_correct, 1) for v in sum_delta_correct]
    avg_p_no_correct = [v / max(count_items_correct, 1) for v in sum_p_no_correct]
    avg_p_cot_correct = [v / max(count_items_correct, 1) for v in sum_p_cot_correct]
    avg_ent_no_correct = [v / max(count_items_correct, 1) for v in sum_ent_no_correct]
    avg_ent_cot_correct = [v / max(count_items_correct, 1) for v in sum_ent_cot_correct]
    avg_ppx_no_correct = [v / max(count_items_correct, 1) for v in sum_ppx_no_correct]
    avg_ppx_cot_correct = [v / max(count_items_correct, 1) for v in sum_ppx_cot_correct]

    if show_plots:
        import numpy as np
        layers = np.arange(n_layers)

        def plot_line(values, title, ylabel):
            plt.figure()
            plt.plot(layers, values, marker="o")
            plt.title(title)
            plt.xlabel("Layer")
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.tight_layout()

        plot_line(avg_delta, "Avg Delta Logit(gold) (CoT - NoCoT)", "Delta logit")
        # Correct-only plots
        plot_line(avg_delta_correct, "[Correct Only] Avg Delta Logit(gold) (CoT - NoCoT)", "Delta logit")

        plt.figure()
        plt.plot(layers, avg_p_no, label="p(gold)_noCoT")
        plt.plot(layers, avg_p_cot, label="p(gold)_CoT")
        plt.title("Avg Gold Probability by Layer")
        plt.xlabel("Layer"); plt.ylabel("prob")
        plt.legend(); plt.grid(True); plt.tight_layout()

        plt.figure()
        plt.plot(layers, avg_p_no_correct, label="p(gold)_noCoT")
        plt.plot(layers, avg_p_cot_correct, label="p(gold)_CoT")
        plt.title("[Correct Only] Avg Gold Probability by Layer")
        plt.xlabel("Layer"); plt.ylabel("prob")
        plt.legend(); plt.grid(True); plt.tight_layout()

        plt.figure()
        plt.plot(layers, avg_ent_no, label="entropy_noCoT")
        plt.plot(layers, avg_ent_cot, label="entropy_CoT")
        plt.title("Avg Entropy by Layer")
        plt.xlabel("Layer"); plt.ylabel("entropy")
        plt.legend(); plt.grid(True); plt.tight_layout()

        plt.figure()
        plt.plot(layers, avg_ent_no_correct, label="entropy_noCoT")
        plt.plot(layers, avg_ent_cot_correct, label="entropy_CoT")
        plt.title("[Correct Only] Avg Entropy by Layer")
        plt.xlabel("Layer"); plt.ylabel("entropy")
        plt.legend(); plt.grid(True); plt.tight_layout()

        plt.figure()
        plt.plot(layers, avg_ppx_no, label="perplexity_noCoT")
        plt.plot(layers, avg_ppx_cot, label="perplexity_CoT")
        plt.title("Avg Perplexity by Layer")
        plt.xlabel("Layer"); plt.ylabel("perplexity")
        plt.legend(); plt.grid(True); plt.tight_layout()

        plt.figure()
        plt.plot(layers, avg_ppx_no_correct, label="perplexity_noCoT")
        plt.plot(layers, avg_ppx_cot_correct, label="perplexity_CoT")
        plt.title("[Correct Only] Avg Perplexity by Layer")
        plt.xlabel("Layer"); plt.ylabel("perplexity")
        plt.legend(); plt.grid(True); plt.tight_layout()

        plt.show()

        # Additional per-token plots (for multi-token answers)
        if len(per_token_sums) > 0:
            for tok_idx in sorted(per_token_sums.keys()):
                c = max(per_token_counts.get(tok_idx, 0), 1)
                avg_delta_tok = [v / c for v in per_token_sums[tok_idx]["sum_delta"]]
                avg_p_no_tok = [v / c for v in per_token_sums[tok_idx]["sum_p_no"]]
                avg_p_cot_tok = [v / c for v in per_token_sums[tok_idx]["sum_p_cot"]]
                avg_ent_no_tok = [v / c for v in per_token_sums[tok_idx]["sum_ent_no"]]
                avg_ent_cot_tok = [v / c for v in per_token_sums[tok_idx]["sum_ent_cot"]]
                avg_ppx_no_tok = [v / c for v in per_token_sums[tok_idx]["sum_ppx_no"]]
                avg_ppx_cot_tok = [v / c for v in per_token_sums[tok_idx]["sum_ppx_cot"]]

                plot_line(avg_delta_tok, f"Avg Delta Logit(gold) (CoT - NoCoT) [tok {tok_idx}]", "Delta logit")

                plt.figure()
                plt.plot(layers, avg_p_no_tok, label="p(gold)_noCoT")
                plt.plot(layers, avg_p_cot_tok, label="p(gold)_CoT")
                plt.title(f"Avg Gold Probability by Layer [tok {tok_idx}]")
                plt.xlabel("Layer"); plt.ylabel("prob")
                plt.legend(); plt.grid(True); plt.tight_layout()

                plt.figure()
                plt.plot(layers, avg_ent_no_tok, label="entropy_noCoT")
                plt.plot(layers, avg_ent_cot_tok, label="entropy_CoT")
                plt.title(f"Avg Entropy by Layer [tok {tok_idx}]")
                plt.xlabel("Layer"); plt.ylabel("entropy")
                plt.legend(); plt.grid(True); plt.tight_layout()

                plt.figure()
                plt.plot(layers, avg_ppx_no_tok, label="perplexity_noCoT")
                plt.plot(layers, avg_ppx_cot_tok, label="perplexity_CoT")
                plt.title(f"Avg Perplexity by Layer [tok {tok_idx}]")
                plt.xlabel("Layer"); plt.ylabel("perplexity")
                plt.legend(); plt.grid(True); plt.tight_layout()

                # Correct-only per-token plots
                if tok_idx in per_token_sums_correct:
                    cc = max(per_token_counts_correct.get(tok_idx, 0), 1)
                    avg_delta_tok_c = [v / cc for v in per_token_sums_correct[tok_idx]["sum_delta"]]
                    avg_p_no_tok_c = [v / cc for v in per_token_sums_correct[tok_idx]["sum_p_no"]]
                    avg_p_cot_tok_c = [v / cc for v in per_token_sums_correct[tok_idx]["sum_p_cot"]]
                    avg_ent_no_tok_c = [v / cc for v in per_token_sums_correct[tok_idx]["sum_ent_no"]]
                    avg_ent_cot_tok_c = [v / cc for v in per_token_sums_correct[tok_idx]["sum_ent_cot"]]
                    avg_ppx_no_tok_c = [v / cc for v in per_token_sums_correct[tok_idx]["sum_ppx_no"]]
                    avg_ppx_cot_tok_c = [v / cc for v in per_token_sums_correct[tok_idx]["sum_ppx_cot"]]

                    plot_line(avg_delta_tok_c, f"[Correct Only] Avg Delta Logit(gold) (CoT - NoCoT) [tok {tok_idx}]", "Delta logit")

                    plt.figure()
                    plt.plot(layers, avg_p_no_tok_c, label="p(gold)_noCoT")
                    plt.plot(layers, avg_p_cot_tok_c, label="p(gold)_CoT")
                    plt.title(f"[Correct Only] Avg Gold Probability by Layer [tok {tok_idx}]")
                    plt.xlabel("Layer"); plt.ylabel("prob")
                    plt.legend(); plt.grid(True); plt.tight_layout()

                    plt.figure()
                    plt.plot(layers, avg_ent_no_tok_c, label="entropy_noCoT")
                    plt.plot(layers, avg_ent_cot_tok_c, label="entropy_CoT")
                    plt.title(f"[Correct Only] Avg Entropy by Layer [tok {tok_idx}]")
                    plt.xlabel("Layer"); plt.ylabel("entropy")
                    plt.legend(); plt.grid(True); plt.tight_layout()

                    plt.figure()
                    plt.plot(layers, avg_ppx_no_tok_c, label="perplexity_noCoT")
                    plt.plot(layers, avg_ppx_cot_tok_c, label="perplexity_CoT")
                    plt.title(f"[Correct Only] Avg Perplexity by Layer [tok {tok_idx}]")
                    plt.xlabel("Layer"); plt.ylabel("perplexity")
                    plt.legend(); plt.grid(True); plt.tight_layout()

    summary = {
        "n_total": n_total,
        "accuracy_nocot": acc_no,
        "accuracy_cot": acc_cot,
        "count_items_for_mech": count_items,
    }
    print("\n" + "=" * 80)
    print("SVAMP Two-Op Experiment Summary")
    print(f"n_total={n_total} | acc_no={acc_no:.3f} | acc_cot={acc_cot:.3f}")
    print("=" * 80)
    return summary



@dataclass
class LayerMetrics:
    layer: int
    tok_index_in_answer: int
    delta_logit_gold: float
    p_gold_no: float
    p_gold_cot: float
    rank_gold_no: int
    rank_gold_cot: int
    entropy_no: float
    entropy_cot: float
    perplexity_no: float
    perplexity_cot: float

# ========== ANA ANALİZ ==========

def analyze_pair(
    model: HookedTransformer,
    nocot_prompt: str,
    cot_prompt: str,
    gold_answer_text: str,
    use_final_ln: bool = True,
    topk_debug_layer: Optional[int] = None,
    show_plots: bool = True,
    anchor_phrase: str = "the answer is ",
) -> Tuple[List[LayerMetrics], Dict[str, Any]]:
    """
    Teacher-forcing ile:
      - nocot_prompt + gold_answer
      - cot_prompt  + gold_answer
    kurup, gold'un üretildiği adım(lar)da katman bazlı metrikleri döndürür.
    """

    # Tokenization
    ids_nop = to_ids(model, nocot_prompt)
    ids_cop = to_ids(model, cot_prompt)
    gold_ids = to_ids(model, gold_answer_text)
    assert len(gold_ids) >= 1, "Gold cevabın token'ları çıkarılamadı."

    # Sequences (teacher forcing)
    seq_no  = ids_nop + gold_ids
    seq_cot = ids_cop + gold_ids

    _, _, cache_no  = run_with_cache_ids(model, seq_no)
    _, _, cache_cot = run_with_cache_ids(model, seq_cot)

    # İlk cevap token'ını tahmin eden adımlar: anchor'a göre hizala
    # Eğer anchor bulunamazsa, önceki davranışa (prompt sonu) geri dön
    pos_no_first_anchor = _anchor_last_token_index(model, nocot_prompt, [anchor_phrase, "The answer is "])
    pos_cot_first_anchor = _anchor_last_token_index(model, cot_prompt, [anchor_phrase, "The answer is "])

    pos_no_first  = pos_no_first_anchor if pos_no_first_anchor is not None else len(ids_nop) - 1
    pos_cot_first = pos_cot_first_anchor if pos_cot_first_anchor is not None else len(ids_cop) - 1
    positions_no  = [pos_no_first + i for i in range(len(gold_ids))]
    positions_cot = [pos_cot_first + i for i in range(len(gold_ids))]

    all_layer_metrics: List[LayerMetrics] = []

    # Her cevap token'ı için (çok-token cevaplarda ortalama/medyan da alınabilir)
    for tok_idx, (p_no, p_cot) in enumerate(zip(positions_no, positions_cot)):
        gold_id = gold_ids[tok_idx]

        proj_no  = resid_logits_at_position(model, cache_no,  p_no, use_final_ln)
        proj_cot = resid_logits_at_position(model, cache_cot, p_cot, use_final_ln)

        for l in range(model.cfg.n_layers):
            ln = proj_no[l]
            lc = proj_cot[l]

            # Metrikler
            delta_logit_gold = float((lc[gold_id] - ln[gold_id]).item())

            # p(gold) ve rank(gold)
            p_no_gold  = float(F.softmax(ln, dim=-1)[gold_id].item())
            p_cot_gold = float(F.softmax(lc, dim=-1)[gold_id].item())
            r_no  = rank_of_id(ln, gold_id)
            r_cot = rank_of_id(lc, gold_id)

            # Entropy ve perplexity hesaplamaları
            entropy_no = calculate_entropy(ln)
            entropy_cot = calculate_entropy(lc)
            perplexity_no = calculate_perplexity(ln)
            perplexity_cot = calculate_perplexity(lc)

            all_layer_metrics.append(
                LayerMetrics(
                    layer=l,
                    tok_index_in_answer=tok_idx,
                    delta_logit_gold=delta_logit_gold,
                    p_gold_no=p_no_gold,
                    p_gold_cot=p_cot_gold,
                    rank_gold_no=r_no,
                    rank_gold_cot=r_cot,
                    entropy_no=entropy_no,
                    entropy_cot=entropy_cot,
                    perplexity_no=perplexity_no,
                    perplexity_cot=perplexity_cot,
                )
            )

    debug: Dict[str, Any] = {}
    # Top-k debug dump (isteğe bağlı)
    if topk_debug_layer is not None:
        l = topk_debug_layer
        # ilk cevap tokenı için
        p_no = positions_no[0]
        p_co = positions_cot[0]
        ln_map = resid_logits_at_position(model, cache_no,  p_no, use_final_ln)
        lc_map = resid_logits_at_position(model, cache_cot, p_co, use_final_ln)
        ln = ln_map[l]; lc = lc_map[l]
        topk_no  = torch.topk(ln, k=10)
        topk_cot = torch.topk(lc, k=10)
        ids_no   = topk_no.indices.tolist()
        ids_cot  = topk_cot.indices.tolist()
        toks_no  = [model.tokenizer.decode([tid]) for tid in ids_no]
        toks_cot = [model.tokenizer.decode([tid]) for tid in ids_cot]
        debug["topk_layer"] = l
        debug["nocot_topk_tokens"] = toks_no
        debug["cot_topk_tokens"]   = toks_cot
        debug["nocot_topk_logits"] = [float(v.item()) for v in topk_no.values]
        debug["cot_topk_logits"]   = [float(v.item()) for v in topk_cot.values]

    # Grafikler
    if show_plots:
        import numpy as np
        L = model.cfg.n_layers
        # Tek cevap tokenı varsayımıyla kolay çizim:
        # (çok-token ise aynı layer için ortalama al)
        # Burada ilk tok_index=0 filtreliyoruz
        ms = [m for m in all_layer_metrics if m.tok_index_in_answer == 0]
        layers = np.array([m.layer for m in ms])

        def plot_line(values, title, ylabel):
            plt.figure()
            plt.plot(layers, values, marker="o")
            plt.title(title)
            plt.xlabel("Layer")
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.tight_layout()

        plot_line([m.delta_logit_gold for m in ms], "Delta Logit(gold) (CoT - NoCoT)", "Delta logit")

        # Rank ve p(gold) ayrı eksenler:
        plt.figure()
        plt.plot(layers, [m.rank_gold_no  for m in ms], label="rank_noCoT")
        plt.plot(layers, [m.rank_gold_cot for m in ms], label="rank_CoT")
        plt.title("Gold Rank by Layer")
        plt.xlabel("Layer"); plt.ylabel("1=best")
        plt.legend(); plt.grid(True); plt.tight_layout()

        plt.figure()
        plt.plot(layers, [m.p_gold_no  for m in ms], label="p(gold)_noCoT")
        plt.plot(layers, [m.p_gold_cot for m in ms], label="p(gold)_CoT")
        plt.title("Gold Probability by Layer")
        plt.xlabel("Layer"); plt.ylabel("prob")
        plt.legend(); plt.grid(True); plt.tight_layout()

        # Entropy plots
        plt.figure()
        plt.plot(layers, [m.entropy_no  for m in ms], label="entropy_noCoT")
        plt.plot(layers, [m.entropy_cot for m in ms], label="entropy_CoT")
        plt.title("Entropy by Layer")
        plt.xlabel("Layer"); plt.ylabel("entropy")
        plt.legend(); plt.grid(True); plt.tight_layout()

        # Perplexity plots
        plt.figure()
        plt.plot(layers, [m.perplexity_no  for m in ms], label="perplexity_noCoT")
        plt.plot(layers, [m.perplexity_cot for m in ms], label="perplexity_CoT")
        plt.title("Perplexity by Layer")
        plt.xlabel("Layer"); plt.ylabel("perplexity")
        plt.legend(); plt.grid(True); plt.tight_layout()

        plt.show()

    return all_layer_metrics, debug

# ========== ATTENTION ANALİZİ ==========

def analyze_attention_patterns(
    model: HookedTransformer,
    nocot_prompt: str,
    cot_prompt: str,
    gold_answer_text: str,
    use_final_ln: bool = True,
    anchor_phrase: str = "the answer is ",
) -> Dict[str, Any]:
    """
    Her katmanda hangi tokenlara en fazla attention verildiğini analiz eder.
    CoT ve No-CoT için karşılaştırmalı analiz yapar.
    """
    
    # Tokenization
    ids_nop = to_ids(model, nocot_prompt)
    ids_cop = to_ids(model, cot_prompt)
    gold_ids = to_ids(model, gold_answer_text)
    
    # Sequences (teacher forcing)
    seq_no  = ids_nop + gold_ids
    seq_cot = ids_cop + gold_ids
    
    # Token'ları decode et
    tokens_no = [model.tokenizer.decode([tid]) for tid in seq_no]
    tokens_cot = [model.tokenizer.decode([tid]) for tid in seq_cot]
    
    # Cache'leri al
    _, _, cache_no  = run_with_cache_ids(model, seq_no)
    _, _, cache_cot = run_with_cache_ids(model, seq_cot)
    
    # İlk cevap token'ının pozisyonu: anchor'a göre hizala
    pos_no_first_anchor = _anchor_last_token_index(model, nocot_prompt, [anchor_phrase, "The answer is "])
    pos_cot_first_anchor = _anchor_last_token_index(model, cot_prompt, [anchor_phrase, "The answer is "])

    pos_no_first  = pos_no_first_anchor if pos_no_first_anchor is not None else len(ids_nop) - 1
    pos_cot_first = pos_cot_first_anchor if pos_cot_first_anchor is not None else len(ids_cop) - 1
    
    results = {
        "nocot": {"tokens": tokens_no, "attention_data": {}},
        "cot": {"tokens": tokens_cot, "attention_data": {}}
    }
    
    # Her katman için attention analizi
    for layer in range(model.cfg.n_layers):
        # No-CoT attention pattern
        attn_no = cache_no["attn", layer][0]  # [n_heads, seq_len, seq_len]
        # CoT attention pattern  
        attn_cot = cache_cot["attn", layer][0]  # [n_heads, seq_len, seq_len]
        
        # İlk cevap token'ına gelen attention'ları al
        target_pos_no = pos_no_first + len(gold_ids) - 1  # Son cevap token'ı
        target_pos_cot = pos_cot_first + len(gold_ids) - 1
        
        # Her head için attention değerlerini al ve ortalama
        attn_to_target_no = attn_no[:, target_pos_no, :].mean(dim=0)  # [seq_len]
        attn_to_target_cot = attn_cot[:, target_pos_cot, :].mean(dim=0)  # [seq_len]
        
        # En yüksek 5 attention değerini bul
        top5_indices_no = torch.topk(attn_to_target_no, k=5).indices
        top5_indices_cot = torch.topk(attn_to_target_cot, k=5).indices
        
        # No-CoT için top-5
        top5_no = []
        for i, idx in enumerate(top5_indices_no):
            token_idx = idx.item()
            attention_val = attn_to_target_no[idx].item()
            token_text = tokens_no[token_idx] if token_idx < len(tokens_no) else f"<pos_{token_idx}>"
            top5_no.append({
                "rank": i + 1,
                "token": token_text,
                "attention": round(attention_val, 4),
                "position": token_idx
            })
        
        # CoT için top-5
        top5_cot = []
        for i, idx in enumerate(top5_indices_cot):
            token_idx = idx.item()
            attention_val = attn_to_target_cot[idx].item()
            token_text = tokens_cot[token_idx] if token_idx < len(tokens_cot) else f"<pos_{token_idx}>"
            top5_cot.append({
                "rank": i + 1,
                "token": token_text,
                "attention": round(attention_val, 4),
                "position": token_idx
            })
        
        results["nocot"]["attention_data"][layer] = top5_no
        results["cot"]["attention_data"][layer] = top5_cot
    
    return results

def print_attention_analysis(attention_results: Dict[str, Any]):
    """
    Attention analiz sonuçlarını terminale yazdırır.
    """
    print("=" * 80)
    print("ATTENTION PATTERN ANALIZI")
    print("=" * 80)
    
    for layer in range(len(attention_results["nocot"]["attention_data"])):
        print(f"\nLAYER {layer}")
        print("-" * 50)
        
        # No-CoT sonuçları
        print("No-CoT (Direkt Cevap):")
        nocot_data = attention_results["nocot"]["attention_data"][layer]
        for item in nocot_data:
            print(f"  {item['rank']}. Token: '{item['token']}' | Attention: {item['attention']:.4f} | Pos: {item['position']}")
        
        print()
        
        # CoT sonuçları
        print("CoT (Adim Adim):")
        cot_data = attention_results["cot"]["attention_data"][layer]
        for item in cot_data:
            print(f"  {item['rank']}. Token: '{item['token']}' | Attention: {item['attention']:.4f} | Pos: {item['position']}")
        
        print("\n" + "="*50)

# ========== ÖRNEK KULLANIM ==========

if __name__ == "__main__":
    model = load_model()

    run_svamp_experiment(
        model,
        json_path="SVAMP.json",
        limit=250,                   # şimdilik 10 soru
        show_plots=True,
        nocot_prefix=None,
        cot_prefix=None,
        print_generations=True,
    )
