# R4 — Methodology

> Bản thesis. Bản làm việc chi tiết: `docs/weekly/W06/Methodology.md`.

## 1. Overview

Pipeline: data+OCR+split → 1 VLM baseline → cải tiến PEFT-only → eval (EM, VQA-Acc, ANLS) → prototype.

## 2. Datasets and splits

ViVQA · ViTextVQA · seed `_` · question types.

## 3. Baseline

Một trong: BLIP-2 / Florence-2 / Qwen2.5-VL. Protocol W04–W05.

## 4. Proposed method

> **Ràng buộc cứng (PEFT-only):** CHỈ được dùng PEFT — LoRA hoặc QLoRA (4-bit / 8-bit).
> **KHÔNG** full fine-tuning toàn bộ VLM. Mọi cải tiến phải so với baseline đã reproduce, **cùng dataset split, cùng seed, cùng hardware protocol**.
> Runtime mục tiêu: Google Colab Pro/Premium hoặc GPU server trường.


Chi tiết LoRA/QLoRA / OCR-prompt / RAG.

## 5. Training setup

Hyperparameters · hardware · thời gian. Cấm full fine-tuning.

## 6. Evaluation protocol

Metric chính vs phụ · fair comparison · ablation design · error/hallucination protocol.

## 7. Ethics

Privacy, bias, reliability (CLO7).
