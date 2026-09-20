# Methodology (nguồn cho Report R4)

> Đồng bộ nội dung chính sang `03_Report/R4_Methodology.md` sau khi chốt. File R4 là khung thesis; file này là bản làm việc tuần W06.

## 1. Overview pipeline

```text
Data prep + OCR + frozen split
        → Reproduce 1 VLM baseline
        → Cải tiến PEFT-only (LoRA/QLoRA · OCR-prompt · RAG · prompt-opt)
        → Eval EM / VQA-Acc / ANLS + ablation + error + hallucination
        → Web prototype
```

## 2. Data

- Datasets: ViVQA (general), ViTextVQA (scene-text)
- OCR: PaddleOCR / VietOCR — chiến lược: _…_
- Split + seed: _…_
- Question-type tagging: _…_

## 3. Baseline

- Đúng 1 model: _…_
- Protocol: xem W05
- Mốc số liệu: _…_

## 4. Proposed improvement (≥ 1)

| Thành phần | Có triển khai? | Chi tiết |
|------------|----------------|----------|
| LoRA / QLoRA 4/8-bit | | target modules, rank, alpha |
| OCR-enhanced prompting | | template |
| RAG | | index, k |
| Prompt optimization | | |

> **Ràng buộc cứng (PEFT-only):** CHỈ được dùng PEFT — LoRA hoặc QLoRA (4-bit / 8-bit).
> **KHÔNG** full fine-tuning toàn bộ VLM. Mọi cải tiến phải so với baseline đã reproduce, **cùng dataset split, cùng seed, cùng hardware protocol**.
> Runtime mục tiêu: Google Colab Pro/Premium hoặc GPU server trường.


## 5. Training (nếu có PEFT)

| Hyperparameter | Value |
|----------------|-------|
| rank / alpha / dropout | |
| lr / scheduler | |
| epochs / max steps | |
| batch / accum | |
| quant 4/8-bit | |
| seed | |

**Cấm:** full fine-tuning toàn bộ vision encoder + LM.

## 6. Evaluation protocol

Metric chính: EM, VQA Accuracy, ANLS. Phụ: BLEU, CIDEr.  
So sánh **cùng** split, seed, decode, hardware class.

## 7. Ablation plan

- Zero-shot vs prompt-opt vs LoRA/QLoRA
- Có vs không OCR
- Có vs không RAG
- (Optional) backbone khác — chỉ nếu còn GPU; không phá ràng buộc "1 baseline chính"

## 8. Error & hallucination plan

Xem W12. Định nghĩa nhãn: _…_

## 9. Prototype

Upload ảnh · câu hỏi vi · answer · confidence · OCR/evidence.
