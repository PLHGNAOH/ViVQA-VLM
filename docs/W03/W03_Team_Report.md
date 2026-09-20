# W03 — Team Weekly Report

Nguồn: công việc automatable W03 (2026-09-17) + khung `docs/templates/Weekly_Report_Template.md` + plan `docs/W01/Project_Plan_15_Weeks.md`. Không thêm số liệu model. Ô chờ hội đồng → `[CHỜ SAU BUỔI REVIEW]`.

## Metadata

| Trường | Nội dung |
|--------|----------|
| **Tuần** | W03 — REVIEW 1 |
| **Nhóm** | ViVQA-VLM (4 SV) |
| **GVHD** | Assoc.Prof. Đặng Ngọc Minh Đức |
| **Ngày** | 2026-09-17 (đóng pack tài liệu; ngày họp hội đồng = ____) |

## Mục tiêu

Tổng hợp W1–W2 thành câu chuyện Review 1: gap G1–G4, **RQ1–RQ4**, **H1–H8**, dataset stats seed 42, candidate methods, **đề xuất** baseline (chưa CHỐT), kế hoạch 15 tuần, risk. Trả nợ kỹ thuật W2 (numbering, Gap×Paper, link chết). **Không train. Không tải model/dataset.**

## Đã làm (per-SV)

| SV | Vai trò | Việc | Trạng thái | Blocker |
|----|---------|------|------------|---------|
| SV1 | VLM / Baseline | Bảng so sánh 3B vs 7B vs Florence-2; khuyến nghị 3B (nhãn đề xuất) | **Xong khung** — `docs/W03/Baseline_Decision.md` | Chưa CHỐT; GPU/P100 chưa xác nhận với thầy; licence 3B ⚠️REVIEW |
| SV2 | Dataset / OCR | Kiểm tra split/licence trên đĩa (không tách val) | **Xong kiểm tra** | ViVQA & ViTextVQA: không có file local + không có val trên mirror; licence paper ❌ TODO. ViVQA-X có train/val/test + `LICENSE` (MIT *repo*, không phải licence ViVQA) |
| SV3 | PEFT / Training (**leader**) | Canonical RQ/H; reconciliation; remap Gap×Paper; khung metric; conventions run-name; pack W03 | **Xong automatable** | Ô Gap×Paper ⚠️REVIEW (L7/L8/L10, L9×G4); metric paper TODO-human; H1 ngưỡng +5 chưa đo |
| SV4 | Evaluation / Deploy | Outline slide Review 1 (9 mục pack); nhắc EM/VQA-Acc/ANLS, không số model | **Xong outline** | Deck `.pptx` **không** render ở Cursor (Claude skill). Zero-shot sanity chưa có run |

**Team:** pack `docs/W03/` + `docs/04_slides/review1/review1_outline.md` + canonical literature.

## Feedback hội đồng Review 1

[CHỜ SAU BUỔI REVIEW]

| Câu hỏi / góp ý | Ai nói | Việc follow-up | Owner |
|-----------------|--------|----------------|-------|
| [CHỜ SAU BUỔI REVIEW] | | | |

## Kết quả buổi review

[CHỜ SAU BUỔI REVIEW]

- Baseline CHỐT? ____
- Gap/RQ được chấp nhận? ____
- Thay đổi kế hoạch W4–W7? ____

## Kết quả / số liệu

Không có thí nghiệm model trong W03. Số dataset = run W02 seed 42 (không đo lại).

| Mục | Giá trị |
|-----|---------|
| Dataset split + seed | seed **42**. ViVQA train 11,999 / test 3,001 (không val). ViTextVQA train 35,159 / test 10,028 (val trống). Official split paper = TODO-human |
| Model / baseline | **ĐỀ XUẤT** Qwen2.5-VL-3B-Instruct; backup Florence-2; **CHỐT bởi nhóm: ____** |
| Method | tài liệu + remap; chưa LoRA/QLoRA |
| **Exact Match (EM)** | chưa chạy (W5) |
| **VQA Accuracy** | chưa chạy (W5) |
| **ANLS** | chưa chạy (W5) |
| Hardware | W03 = CPU docs only |

## Blocker

| Blocker | Impact | Owner | Next step |
|---------|--------|-------|-----------|
| Baseline chưa CHỐT + GPU chưa xác nhận | High | SV1 + leader | Thầy Đức; điền `Baseline_Decision.md` |
| Mirror ViTextVQA (và ViVQA) thiếu val | Medium | SV2 | **Không tự tách.** Đề xuất lệnh seed=42 — mục dưới. Chờ xác nhận tỉ lệ |
| 0/10 PDF paper | Medium | cả nhóm | điền metric TODO-human |
| Ô Gap×Paper ⚠️REVIEW | Medium | leader | xác nhận L7/L8/L10/L9 |
| `experiments.zip` / `docs/W02.zip` không còn ở root **và** không thấy trong `_archive/` | Low | SV3 | snapshot 14/9 đã mất khỏi workspace; bản sống `experiments/` + `docs/W02/` vẫn đủ |

## Đề xuất tách val (KHÔNG chạy ở W03)

Mirror ViTextVQA: JSON chỉ `train`/`test` (`vitextvqa_stats.json`). Dataset_Analysis: validation empty. **Không** tạo file dưới `data/splits/` trong W03.

Tỉ lệ val: ⚠️REVIEW (không có official ratio trên đĩa). Gợi ý *chờ xác nhận*: 10% **image_id** của train, seed=42, toàn bộ QA của ảnh val đi cùng (tránh leakage).

```text
# ĐỀ XUẤT — copy chạy ở W04 SAU KHI nhóm chốt tỉ lệ. Không execute W03.
# Input: danh sách image_id unique của split train (từ HF mirror nhonhoccode/ViTextVQA).
# Output đề xuất: data/splits/vitextvqa_val_image_ids_seed42.txt  (+ jsonl QA)

python -c "import random; random.seed(42); ids=sorted(set(train_image_ids)); random.shuffle(ids); n=max(1,int(round(0.10*len(ids)))); val=set(ids[:n]); print(len(val))"
```

Cùng protocol cho ViVQA nếu paper không có val (mirror `minhnguyent546/ViVQA`: train/test only).

## Kế hoạch W4

Bám `docs/W01/Project_Plan_15_Weeks.md` (W04 Environment + baseline startup):

- [ ] SV1: env (Colab / GPU trường / Kaggle P100 — **sau khi thầy xác nhận**); pin `transformers`/`peft`/`bitsandbytes`; zero-shot 1 batch trên baseline đề xuất
- [ ] SV2: xác nhận nguồn official + licence; **nếu** được phép — tách val seed=42, commit `data/splits/`; chưa tải ảnh hàng loạt nếu chưa có GPU
- [ ] SV3: pilot QLoRA ~200 mẫu (`experiments/W01_Pilot/` theo plan); đặt tên run `W04_qlora_qwen3b_seed42` (nếu baseline 3B được chốt)
- [ ] SV4: script eval EM/VQA-Acc/ANLS dry-run trên 10 mẫu; log template

## Link experiment log

- W02 stats (vẫn sống): `experiments/W02_dataset_stats/`
- W03: không có run model
- Convention: `docs/00_admin/CONVENTIONS.md`

> **Ràng buộc cứng (PEFT-only):** CHỈ LoRA hoặc QLoRA (4-bit / 8-bit). **KHÔNG** full fine-tuning. Cùng split, seed, hardware so với baseline.
