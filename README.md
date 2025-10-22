# Chain of Thoughts - Mechanistic Interpretability

Bu proje, Chain of Thought (CoT) ve No-CoT yaklaşımları arasındaki farkları mekanik interpretability teknikleri kullanarak analiz eder.

## Özellikler

- **Logit-Lens Analizi**: Her katmanda `resid_post` → (opsiyonel) `ln_final` → `W_U` ile projeksiyon ve logit analizi
- **Attention Pattern Analizi**: Katman/head bazında hedef cevaba giden attention yoğunlukları
- **Katman Bazlı Metrikler**: 
  - Gold token'ın rank'i ve probability'si
  - Entropy ve perplexity hesaplamaları
  - Delta logit analizi (CoT - NoCoT)
- **Görselleştirme**: 
  - Tüm örneklerin ortalaması için grafikler
  - Sadece CoT’un doğru bildiği örnekler (correct-only) için ek grafikler
  - Çok basamaklı cevaplar için token-bazlı (tok_index=0,1,...) grafikler

## Jenerasyon ve Durdurma Mantığı

- Model, ankordan sonra üretimi durdurmak üzere yapılandırılır. Anchor varsayılanı: `"The answer is "` (boşluk dahil).
- Çok basamaklı cevaplar için: ankordan sonra gold cevabın basamak sayısı kadar rakam üretilene kadar devam eder (digit-bazlı durdurma).
- No-CoT promptu anchor ile biter; CoT promptunda model akıl yürütmeyi üretir, anchor’a ulaşınca cevap rakamlarını üretir ve durur.

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```python
python main.py
```

Parametreler (başlıcaları `run_svamp_experiment` içinde):
- `json_path`: `SVAMP.json` yolu
- `limit`: kaç örnek üzerinde koşulacağı
- `use_final_ln`: logit-lens'te final layer norm kullanımı
- `show_plots`: grafiklerin çizilip çizilmeyeceği
- `anchor_phrase`: anchor ifadesi (sonunda boşluk olacak şekilde)

Veri filtresi CLI (tam iki operatör):
```bash
python swamp_dataset.py --json SVAMP.json --save SVAMP_two_ops.json
```

## Analiz Edilen Metrikler

1. **Delta Logit Gold**: CoT ve No-CoT arasındaki gold token logit farkı
2. **Gold Probability**: Her katmanda gold token'ın olasılığı
3. **Gold Rank**: Gold token'ın sıralaması (1=en iyi)
4. **Entropy**: Her katmandaki belirsizlik ölçüsü
5. **Perplexity**: Entropy'nin üstel değeri
6. **Attention Patterns**: Her katmanda en çok attention alan tokenlar

Ek Notlar:
- CoT analizinde teacher-forcing bağlamı, modelin ürettiği CoT akıl yürütmesi (anchor’a kadar) ile genişletilerek yapılır. Böylece metrikler doğru bağlamda hesaplanır.
- GPU otomatik seçilir: `cuda` mevcutsa GPU, aksi halde CPU. Hızlı kontrol için `nvidia-smi` veya kodda `print(device)` kullanılabilir.

## Model

Proje Qwen2.5-0.5B modelini kullanır ve transformer_lens kütüphanesi ile analiz yapar.

## Örnek Çıktı

Analiz sonuçları hem terminal çıktısı hem de matplotlib grafikleri olarak sunulur. 
- Tüm örnekler için grafikler
- Sadece doğru olan örnekler için ek grafikler
- Çok token’lı cevaplar için her cevap token’ına özel grafikler
