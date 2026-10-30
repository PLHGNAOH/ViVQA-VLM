# W01_Pilot — Pilot QLoRA trên VLM (smoke test)

> Owner: SV3 · Mục tiêu: kiểm chứng pipeline QLoRA chạy được end-to-end trên Colab trước khi reproduce baseline chính thức (W04). **Không** dùng số liệu pilot làm kết quả chính.

## Mục tiêu pilot
- Load VLM ứng viên (Qwen2.5-VL-3B-Instruct) ở NF4 + double quant.
- Gắn LoRA (r=16, all-linear LLM, freeze ViT), train 200 mẫu ViVQA × 1 epoch.
- Đo: VRAM đỉnh, thời gian/step, loss giảm, 20 câu trả lời mẫu có tiếng Việt đúng dấu.

## Cấu trúc thư mục (điền khi chạy)

```text
W01_Pilot/
├── README.md              # file này
├── config.yaml            # cấu hình QLoRA (copy từ configs/)
├── run_log.md             # log theo mẫu bên dưới
├── metrics.json           # EM/VQA-Acc trên 50 mẫu val (chỉ để smoke test)
└── samples.md             # 20 ví dụ (ảnh, câu hỏi, đáp án, dự đoán)
```

## Mẫu run log

| Trường | Giá trị |
|--------|---------|
| Ngày | |
| Người chạy | SV3 |
| Hardware | Colab _T4 / L4 / A100_ · VRAM __ GB |
| Model | Qwen/Qwen2.5-VL-3B-Instruct |
| Quantization | NF4 + DQ, compute _bf16/fp16_ |
| LoRA | r=16, α=32, dropout=0.05, targets=q,k,v,o,gate,up,down |
| Data | ViVQA train subset N=200, seed=42 |
| LR / optimizer | 2e-4 / paged_adamw_32bit |
| Batch × grad-accum | 2 × 8 |
| Epoch / steps | 1 / __ |
| Thời gian train | __ phút |
| VRAM đỉnh | __ GB |
| Loss đầu → cuối | __ → __ |
| Thời gian infer / mẫu | __ s |
| Ghi chú / lỗi | |

## Checklist
- [ ] Pipeline chạy không OOM
- [ ] Adapter lưu được (`adapters/pilot_qlora_r16/`, kích cỡ __ MB)
- [ ] Câu trả lời tiếng Việt đúng dấu (kiểm tra 20 mẫu)
- [ ] Ghi kết quả vào `docs/W01/W01_Team_Report.md` / W04 report
