# R1 — Introduction

> Khung thesis. Điền dần từ W01–W02 và hoàn thiện W14. Không thay file `.docx` có sẵn trong thư mục này.

## 1. Motivation

Visual Question Answering tiếng Việt: ảnh + câu hỏi → câu trả lời. VLM mã nguồn mở thiên về tiếng Anh; scene-text và hallucination còn khó.

## 2. Problem statement

```text
Vietnamese image + Vietnamese question → VLM → Vietnamese answer
```

## 3. Research gap (trỏ W02)

G1 English-centric · G2 Vietnamese VQA/scene-text · G3 PEFT cho VLM vi · G4 hallucination.

**Gap nhóm chọn:** _…_

## 4. Objectives & contribution

Reproduce **đúng 1** baseline (BLIP-2 / Florence-2 / Qwen2.5-VL). Đề xuất ≥ 1 cải tiến **PEFT-only** (LoRA/QLoRA 4/8-bit, OCR-prompt, RAG, prompt-opt), so sánh **cùng điều kiện**, kèm ablation, error analysis, hallucination analysis.

## 5. Scope

In: ViVQA, ViTextVQA, Colab Pro/Premium hoặc GPU trường. Out: full fine-tuning.

## 6. Thesis structure

R3 existing systems · R4 methodology · R5 implementation · R6 results · R7 discussion.
