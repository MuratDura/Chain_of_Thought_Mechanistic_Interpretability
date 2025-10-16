# Chain of Thoughts - Mechanistic Interpretability

Bu proje, Chain of Thought (CoT) ve No-CoT yaklaşımları arasındaki farkları mekanik interpretability teknikleri kullanarak analiz eder.

## Özellikler

- **Logit-Lens Analizi**: Her katmanda resid_post'u alıp W_U ile projekte ederek logit analizi
- **Attention Pattern Analizi**: Her katmanda hangi tokenlara en fazla attention verildiğini analiz eder
- **Katman Bazlı Metrikler**: 
  - Gold token'ın rank'i ve probability'si
  - Entropy ve perplexity hesaplamaları
  - Delta logit analizi (CoT - NoCoT)
- **Görselleştirme**: Katman bazlı grafikler ve analiz sonuçları

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```python
python main.py
```

## Analiz Edilen Metrikler

1. **Delta Logit Gold**: CoT ve No-CoT arasındaki gold token logit farkı
2. **Gold Probability**: Her katmanda gold token'ın olasılığı
3. **Gold Rank**: Gold token'ın sıralaması (1=en iyi)
4. **Entropy**: Her katmandaki belirsizlik ölçüsü
5. **Perplexity**: Entropy'nin üstel değeri
6. **Attention Patterns**: Her katmanda en çok attention alan tokenlar

## Model

Proje Qwen2.5-0.5B modelini kullanır ve transformer_lens kütüphanesi ile analiz yapar.

## Örnek Çıktı

Analiz sonuçları hem terminal çıktısı hem de matplotlib grafikleri olarak sunulur. Her katman için detaylı metrikler ve attention pattern analizi yapılır.
