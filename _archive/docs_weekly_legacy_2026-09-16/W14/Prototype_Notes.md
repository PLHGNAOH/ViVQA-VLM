# Prototype Notes

## Yêu cầu (Master Instruction §11)

- [ ] Upload ảnh tiếng Việt
- [ ] Nhập câu hỏi tiếng Việt
- [ ] Sinh câu trả lời
- [ ] Hiện **confidence**
- [ ] Hiện **OCR text / evidence retrieve** khi liên quan
- [ ] Disclaimer hallucination & privacy

## Stack (điền)

| Lớp | Công nghệ | Path |
|-----|-----------|------|
| UI | Gradio / Streamlit / other | `src/prototype/` |
| Inference | HF + PEFT adapter | `src/baseline/` + `src/adaptation/` |
| OCR | PaddleOCR / VietOCR | `src/data/` |
| Runtime | Colab / server trường | |

## Luồng

```text
image + question → [OCR] → [optional retrieve] → VLM(+LoRA) → answer + scores
```

## Giới hạn demo

VRAM `_` · timeout `_` · max image size `_` · không lưu ảnh user mặc định: _có/không_

## Test thủ công

| Case | Ảnh | Câu hỏi | Kỳ vọng | Pass? |
|------|-----|---------|---------|-------|
| 1 general | | | | |
| 2 scene-text | | | | |
| 3 empty/OCR fail | | | | |
| 4 tiếng Việt dấu | | | | |
