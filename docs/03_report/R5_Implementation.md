# R5 — Implementation

## 1. Repository layout

`src/data` preprocessing/OCR/tagging · `src/baseline` · `src/adaptation` PEFT/prompt/RAG · `src/eval` · `src/prototype`.  
`data/` · `configs/` · `adapters/` · `experiments/`.

## 2. Data pipeline

Raw → processed → OCR → splits (seed).

## 3. Baseline inference

Thư viện: PyTorch, HuggingFace Transformers. Version: `_`.

## 4. PEFT adaptation

PEFT library · bitsandbytes · target modules · lưu adapter (không full weights).

## 5. Evaluation scripts

EM · VQA Accuracy · ANLS · export pred.

## 6. Prototype

UI, GPU runtime, OCR/evidence visualization.

## 7. Reproducibility

Map sang checklist W15. Lưu ý: thư mục `src/production-agentic-rag-course/` là code đề tài cũ, không phải pipeline VQA.
