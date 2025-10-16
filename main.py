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
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from transformer_lens import HookedTransformer
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

    # İlk cevap token'ını tahmin eden adımlar
    pos_no_first  = len(ids_nop) - 1
    pos_cot_first = len(ids_cop) - 1
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
    
    # İlk cevap token'ının pozisyonu
    pos_no_first  = len(ids_nop) - 1
    pos_cot_first = len(ids_cop) - 1
    
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

    no_cot_prompt = (
       """Q: Lena bought an iced tea for $2 and 8 candy bars.
       She spent a total of $26.\nHow much did each candy bar cost?\n
       A: The answer is 3\n
       Q:Benny bought a soft drink for 2 dollars and 5 candy bars.
        He spent a total of 27\ndollars. How much did each candy bar cost?"""
        "the answer is "
    )
    cot_prompt = (
        """Q: Lena bought an iced tea for $2 and 8 candy bars.
         She spent a total of $26.\nHow much did each candy bar cost?
         \nA: let p be the price; 2 + 8p = 26 ⇒ 8p = 24 ⇒ p = 24/8 = $3.
          The answer is 3\nQ:Benny bought a soft drink for 2 dollars and 5 candy bars.
           He spent a total of 27\ndollars. How much did each candy bar cost?
           \nA: Let p be the price; 2 + 5p = 27 ⇒ 5p = 25 ⇒ p = 25/5 = $5.""" 
        "the answer is "
    )

    # Önemli: tokenizer'a uygun space/newline ile yaz
    gold_answer_text = "5"  # çoğu BPE'de boşluklu olur; 

    # Attention pattern analizi
    print("Attention Pattern Analizi Baslatiliyor...")
    attention_results = analyze_attention_patterns(
        model,
        no_cot_prompt,
        cot_prompt,
        gold_answer_text,
        use_final_ln=True,
    )
    
    # Attention sonuçlarını yazdır
    print_attention_analysis(attention_results)
    
    print("\n" + "="*80)
    print("LOGIT-LENS ANALİZİ")
    print("="*80)
    
    metrics, dbg = analyze_pair(
        model,
        no_cot_prompt,
        cot_prompt,
        gold_answer_text,
        use_final_ln=True,        # Final LN ile lens (daha okunur)
        topk_debug_layer=23,      # örnek: 17. katmanda top-k dökümü
        show_plots=True,
    )

    # Konsola kısa özet
    first_tok = [m for m in metrics if m.tok_index_in_answer == 0]
    print("First answer token summary for first 5 layers:")
    for m in first_tok[:]:
        print({
            "layer": m.layer,
            "delta_logit_gold": round(m.delta_logit_gold, 4),
            "p_gold_no": round(m.p_gold_no, 4),
            "p_gold_cot": round(m.p_gold_cot, 4),
            "rank_no": m.rank_gold_no,
            "rank_cot": m.rank_gold_cot,
            "entropy_no": round(m.entropy_no, 4),
            "entropy_cot": round(m.entropy_cot, 4),
            "perplexity_no": round(m.perplexity_no, 2),
            "perplexity_cot": round(m.perplexity_cot, 2),
        })

    if "topk_layer" in dbg:
        print(f"\nLayer {dbg['topk_layer']} top-10 (No-CoT): {dbg['nocot_topk_tokens'][:10]}")
        print(f"Layer {dbg['topk_layer']} top-10 (CoT):   {dbg['cot_topk_tokens'][:10]}")
