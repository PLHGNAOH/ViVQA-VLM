# SV3 Paper Review — PEFT / Training (LoRA · QLoRA · PEFT cho VLM)

> **Đề tài:** ViVQA-VLM — *Efficient Vietnamese Visual Question Answering using Vision-Language Models*
> **Vai trò:** SV3 — PEFT / Training (LoRA, QLoRA, prompting, multimodal prompting, RAG)
> **Tuần:** W01 · **Ngày:** 2026-09-06 · **Reviewer:** SV3
> **Ràng buộc cứng (nhắc lại):** KHÔNG full fine-tuning — chỉ PEFT (LoRA / QLoRA 4-bit hoặc 8-bit). Runtime mục tiêu: Google Colab Pro/Premium hoặc GPU server trường. Mọi cải tiến phải so với **đúng 1** baseline đã reproduce (BLIP-2 / Florence-2 / Qwen2.5-VL) trên **cùng split, cùng seed, cùng hardware protocol**.

---

## 0. Tổng quan & tiêu chí chọn paper

Mảng của SV3: *thích nghi một VLM English-centric cho Vietnamese VQA mà không được full fine-tuning và phải chạy trên GPU ~16–48 GB.* Bốn paper dưới đây được chọn theo 3 nhóm bắt buộc, cộng ba tiêu chí sàng lọc: **(i)** công trình gốc/kinh điển, được trích dẫn nhiều nhất trong nhóm; **(ii)** có số liệu định lượng đủ để làm căn cứ thiết kế thí nghiệm; **(iii)** có mã nguồn công khai, đã tích hợp vào HuggingFace `peft` / `bitsandbytes` — tức **dùng được ngay** trong W04–W09.

| # | Nhóm bắt buộc | Paper | Venue | Vì sao chọn |
|---|---------------|-------|-------|-------------|
| **P1** | **LoRA** | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* | ICLR 2022 (arXiv:2106.09685) | Paper gốc định nghĩa LoRA; nền tảng toán học cho mọi biến thể sau. |
| **P2** | **QLoRA** | Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* | NeurIPS 2023, Oral (arXiv:2305.14314) | Paper gốc QLoRA; lý do trực tiếp khiến đề tài chạy được trên Colab. |
| **P3** | **PEFT / VLM adaptation** | Sung, Cho & Bansal, *VL-Adapter: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks* | CVPR 2022, pp. 5227–5237 (arXiv:2112.06825) | Benchmark PEFT có hệ thống **đầu tiên trực tiếp trên VQA**; kết luận chuyển thẳng sang thiết kế ablation. |
| **P4** | **PEFT / VLM adaptation** (bổ sung) | Liu et al., *Improved Baselines with Visual Instruction Tuning* (LLaVA-1.5) | CVPR 2024 (arXiv:2310.03744) | Recipe VLM hiện đại (ViT → MLP projector → LLM) + bằng chứng công khai **LoRA ≈ full FT** trên VLM 7B/13B; kiến trúc đồng dạng Qwen2.5-VL. |

Mỗi paper phân tích theo **10 câu hỏi cốt lõi** + mục bắt buộc *"Paper này liên quan gì đến đề tài ViVQA-VLM?"*. Mục 5 tổng hợp thành khuyến nghị kỹ thuật.

> **Nguồn số liệu.** Các con số trích từ bảng chính của paper (bản ICLR/NeurIPS/CVPR hoặc arXiv tương ứng) và từ Model Zoo chính thức của LLaVA. Bảng LoRA-vs-full-FT (P4) và các trang xuất bản (P3) đã được đối chiếu lại với nguồn gốc; số nhớ gần đúng đánh dấu "≈".

---

## P1. LoRA: Low-Rank Adaptation of Large Language Models

| Trường | Giá trị |
|--------|---------|
| Title | LoRA: Low-Rank Adaptation of Large Language Models |
| Author(s) | Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (Microsoft) |
| Year / Venue | 2022, ICLR (arXiv:2106.09685, 06/2021) |
| Code | https://github.com/microsoft/LoRA (tích hợp trong HF `peft`) |

### 1. Vấn đề paper giải quyết
Chi phí **adaptation** LLM cho downstream task. Full fine-tuning buộc mỗi task lưu và phục vụ một bản sao *toàn bộ* tham số (GPT-3 175B ≈ 350 GB ở FP16). Chi phí nằm ở ba chỗ: (a) bộ nhớ GPU khi train (tham số + gradient + trạng thái Adam ≈ 3× số tham số), (b) lưu trữ checkpoint theo task, (c) chuyển task khi serving nhiều task. Các giải pháp trước — adapter layers (Houlsby et al., 2019) và prefix/prompt tuning (Li & Liang, 2021; Lester et al., 2021) — hoặc **thêm độ trễ suy luận** (adapter là lớp tuần tự không gộp được), hoặc **chiếm mất độ dài chuỗi** và **khó tối ưu**, và thường **thua** full FT về chất lượng.

### 2. Vì sao quan trọng
- Khi tham số tiến tới $10^{11}$, full FT vượt khả năng của phần lớn phòng lab/nhóm SV; vấn đề này quyết định *ai* được phép nghiên cứu adaptation.
- Serving nhiều task đồng thời cần hot-swap tham số nhanh; checkpoint hàng trăm GB/task khiến điều đó bất khả thi.
- Với ViVQA-VLM: đây chính là ràng buộc cứng "không full fine-tuning" (`Master_Instruction.md` §4.3). Paper là **cơ sở lý thuyết** cho phép ràng buộc đó **không** làm giảm chất lượng.

### 3. Research gap
Trước LoRA, chưa phương pháp adaptation nào thỏa đồng thời 4 tính chất: (1) tham số huấn luyện nhỏ hơn full FT nhiều bậc; (2) **không tăng độ trễ suy luận**; (3) **không giảm độ dài chuỗi khả dụng**; (4) chất lượng **ngang hoặc hơn** full FT. Về lý thuyết, các công trình *intrinsic dimensionality* (Aghajanyan et al., 2020) gợi ý mô hình over-parameterized nằm trên đa tạp thấp chiều, nhưng chưa ai khai thác trực tiếp giả thuyết "**cập nhật trọng số $\Delta W$ có hạng thấp**" để thiết kế phương pháp.

### 4. Phương pháp
Đóng băng toàn bộ trọng số pretrained $W_0 \in \mathbb{R}^{d\times k}$ và tham số hóa **phần cập nhật** bằng tích hai ma trận hạng thấp:

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x, \qquad B \in \mathbb{R}^{d\times r},\ A \in \mathbb{R}^{r\times k},\ r \ll \min(d,k).$$

- **Khởi tạo:** $A \sim \mathcal{N}(0,\sigma^2)$, $B = 0$ ⇒ tại bước 0, $\Delta W = 0$, mô hình hành xử đúng như pretrained (không phá knowledge ban đầu).
- **Scaling $\alpha/r$:** cho phép đổi $r$ mà không phải chỉnh lại learning rate.
- **Vị trí áp dụng:** paper chỉ áp lên ma trận attention ($W_q, W_k, W_v, W_o$); thực nghiệm chính dùng $W_q, W_v$; MLP giữ nguyên.
- **Suy luận không độ trễ:** sau train gộp $W = W_0 + BA$; đổi task chỉ cần trừ $BA$ cũ, cộng $B'A'$ mới.
- **Số tham số huấn luyện:** $|\Theta| = 2 \times \hat{L}_{\text{LoRA}} \times d_{\text{model}} \times r$ — với GPT-3 175B, $r{=}4$ trên $\{W_q, W_v\}$ chỉ **4.7M** tham số.

### 5. Ý tưởng kỹ thuật cốt lõi
**Giả thuyết hạng thấp của $\Delta W$**: mô hình over-parameterized sau pretraining chỉ cần điều chỉnh một không gian con rất nhỏ để giải task mới. Paper kiểm chứng qua 3 phân tích (Section 7):

1. **$r$ cực nhỏ vẫn đủ:** trên GPT-3 175B, $r{=}1$ khi adapt cả $W_q, W_v$ đã cạnh tranh; tăng lên $r{=}64$ không cải thiện đáng kể ⇒ $\Delta W$ thực sự hạng thấp.
2. **Chồng lấp không gian con:** các singular directions top của $\Delta W_{r=8}$ và $\Delta W_{r=64}$ trùng nhau mạnh (Grassmann similarity cao ở top-1/top-2), phần còn lại là nhiễu.
3. **$\Delta W$ khuếch đại hướng mà $W_0$ không nhấn mạnh:** $\Delta W$ không sao chép các hướng chính của $W_0$ mà **khuếch đại (≈21.5× với $r{=}4$)** các hướng vốn trọng lượng nhỏ trong $W_0$ nhưng quan trọng với task ⇒ adaptation là "bật lên" đặc trưng đã có sẵn.

> **Hệ quả thiết kế:** phân bổ tham số cho **nhiều ma trận với $r$ nhỏ** tốt hơn **ít ma trận với $r$ lớn**.

### 6. Dataset
| Mô hình | Dataset | Loại task |
|--------|---------|-----------|
| RoBERTa-base/large, DeBERTa-XXL | GLUE (MNLI, SST-2, MRPC, CoLA, QNLI, QQP, RTE, STS-B) | NLU / phân loại |
| GPT-2 Medium/Large | E2E NLG, WebNLG, DART | NLG data-to-text |
| GPT-3 175B | WikiSQL (NL→SQL), MNLI-matched, SAMSum (tóm tắt hội thoại) | Sinh có cấu trúc, NLU, tóm tắt |

Không có dataset thị giác — paper thuần NLP.

### 7. Metric
Accuracy (GLUE, MNLI, WikiSQL), Matthews corr. (CoLA), Pearson (STS-B), BLEU/NIST/METEOR/ROUGE-L/CIDEr (E2E NLG), ROUGE-1/2/L (SAMSum). Ngoài ra đo **số tham số huấn luyện**, **bộ nhớ GPU**, **kích cỡ checkpoint**, **độ trễ suy luận** so với adapter.

### 8. Kết quả chính
- **GPT-3 175B (Table 4):** LoRA 4.7M tham số đạt WikiSQL **73.4** / MNLI-m **91.7** / SAMSum **53.8-29.8-45.9**, so với full FT (175B tham số) 73.8 / 89.5 / 52.0-28.0-44.5 ⇒ **ngang hoặc hơn** full FT với **ít hơn ~10.000× tham số huấn luyện**, giảm ~3× bộ nhớ GPU, checkpoint 350 GB → ~35 MB.
- **RoBERTa-large (GLUE):** LoRA (0.8M) avg ≈89.0 vs full FT (355M) ≈88.9; DeBERTa-XXL LoRA ≈91.3 vs FT ≈91.8 (≈).
- **GPT-2 Medium (E2E NLG):** LoRA (0.35M) BLEU **70.4** vs full FT 68.2; vượt cả adapter và prefix-tuning cùng ngân sách tham số.
- **Độ trễ:** với batch nhỏ / chuỗi ngắn (serving online), adapter tăng độ trễ 20–30%; LoRA sau gộp = 0%.
- **Prefix tuning** giảm hiệu năng khi tăng token đặc biệt (>256) — không đơn điệu, khó tối ưu; LoRA ổn định hơn nhiều theo số tham số.

### 9. Hạn chế
1. **Batching nhiều task khác nhau** trong một forward pass không thẳng thắn nếu đã gộp $BA$ (phải giữ tách và tra cứu adapter theo mẫu ⇒ có độ trễ nhỏ).
2. **Chọn ma trận áp LoRA còn heuristic**; paper chỉ khảo sát attention, chưa khảo sát MLP (QLoRA/LLaVA về sau cho thấy áp lên **tất cả linear layer** quan trọng).
3. Chỉ chứng minh trên **LLM thuần văn bản, tiếng Anh**; không có bằng chứng cho vision encoder, projector, hay ngôn ngữ ít tài nguyên.
4. Chưa có lý thuyết chặt vì sao $\Delta W$ hạng thấp; phân tích chủ yếu thực nghiệm.
5. Vẫn cần toàn bộ $W_0$ trong VRAM ở FP16/BF16 ⇒ 7B ≈ 14 GB **chỉ riêng trọng số** — vẫn quá tải Colab T4 16 GB nếu không lượng tử hóa (đây chính là gap mà QLoRA lấp).

### 10. Học/dùng được gì cho đề tài? (BẮT BUỘC)
- **Biện minh cho ràng buộc PEFT-only:** trích Table 4 khi hội đồng hỏi "không full FT thì có mất chất lượng không?" — với LLM, LoRA ≥ full FT trên 3/3 task ở quy mô 175B.
- **Nguyên tắc phân bổ rank:** $r$ nhỏ (4–16) trên **nhiều module** > $r$ lớn trên ít module ⇒ ablation nên xoay quanh *target modules* trước, *rank* sau.
- **Khởi tạo $B{=}0$** ⇒ baseline zero-shot và LoRA-bước-0 trùng nhau ⇒ mọi khác biệt đo được đến từ huấn luyện, không từ thay đổi kiến trúc — thuận cho **so sánh công bằng với baseline** (Master Instruction §14).
- **Gộp trọng số sau train** ⇒ prototype web (SV4) suy luận với độ trễ bằng baseline; adapter chỉ vài chục MB ⇒ nộp `adapters/` trong reproducibility package dễ dàng.
- **Rank sweep** ($r \in \{4,8,16,32,64\}$) trên tập ViVQA nhỏ để kiểm chứng giả thuyết hạng thấp có còn đúng cho adaptation *đa phương thức + đổi ngôn ngữ* — có thể là quan sát nhỏ nhưng mới trong thesis.
- **Cấu hình xuất phát** (paper + thực hành cộng đồng): `r=16, alpha=32, dropout=0.05, target = q,k,v,o (+ MLP theo QLoRA)`.

> ### Paper này liên quan gì đến đề tài ViVQA-VLM?
> LoRA là **kỹ thuật bắt buộc số 1** nêu tường minh trong ràng buộc §4.3. Với ViVQA-VLM, LoRA sẽ áp lên **khối LLM (decoder) của Qwen2.5-VL / BLIP-2 / Florence-2** để dạy mô hình sinh câu trả lời tiếng Việt đúng dấu, đúng định dạng ngắn (EM/VQA-Acc) mà không đụng vào hàng tỷ tham số pretrained. LoRA cho phép lưu **một adapter cho ViVQA (general)** và **một adapter cho ViTextVQA (scene-text)** trên cùng một backbone — trực tiếp phục vụ ablation "adapter chung vs. adapter theo dataset". Cuối cùng, phân tích "$\Delta W$ khuếch đại hướng có sẵn" gợi ý một *giả thuyết nghiên cứu*: **kiến thức thị giác đã đủ trong VLM; phần thiếu là ánh xạ ngôn ngữ Việt — thứ hạng thấp hoàn toàn có thể học được.**

---

## P2. QLoRA: Efficient Finetuning of Quantized LLMs

| Trường | Giá trị |
|--------|---------|
| Title | QLoRA: Efficient Finetuning of Quantized LLMs |
| Author(s) | Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (Univ. of Washington) |
| Year / Venue | 2023, NeurIPS (Oral) (arXiv:2305.14314) |
| Code | https://github.com/artidoro/qlora · `bitsandbytes` · HF `peft` |

### 1. Vấn đề paper giải quyết
LoRA giảm tham số huấn luyện nhưng **không giảm bộ nhớ chứa trọng số gốc**: fine-tune LLaMA-65B ở 16-bit cần **>780 GB** VRAM (trọng số + activation + optimizer). Bài toán: **fine-tune 33B–65B trên MỘT GPU 24–48 GB mà không mất hiệu năng so với 16-bit fine-tuning.** Các phương pháp lượng tử hóa trước đó chỉ phục vụ *inference*; lượng tử hóa rồi *huấn luyện* thường làm sụt chất lượng.

### 2. Vì sao quan trọng
- Dân chủ hóa nghiên cứu LLM: đưa fine-tuning 65B từ cụm 16×A100 xuống một GPU đơn.
- Với nhóm chỉ có Colab (T4 16 GB, L4 24 GB, A100 40 GB), QLoRA là **điều kiện cần** để fine-tune bất kỳ VLM 3B–8B nào. Không có QLoRA, ngay LoRA trên Qwen2.5-VL-7B (≈15 GB trọng số BF16) cũng tràn VRAM T4.
- Master Instruction §12 liệt kê rủi ro "GPU không đủ → Mitigation: QLoRA / quantization" — paper là bằng chứng cho phương án đó.

### 3. Research gap
1. Lượng tử hóa 4-bit hiện có (Int4, FP4) dùng lưới **không tối ưu cho phân bố trọng số ~ chuẩn** của mạng neural ⇒ mất thông tin.
2. Lượng tử hóa theo block cần lưu hằng số (absmax) ở 32-bit ⇒ **overhead ~0.5 bit/tham số**, đáng kể ở 4-bit.
3. **Memory spikes** khi gradient checkpointing với chuỗi dài gây OOM dù bộ nhớ trung bình vẫn đủ.
4. Chưa có bằng chứng thực nghiệm quy mô lớn rằng **fine-tuning trên base 4-bit ≈ 16-bit**.

### 4. Phương pháp
QLoRA = **base model lượng tử hóa 4-bit đóng băng** + **LoRA adapter ở BF16** + backprop qua trọng số lượng tử hóa (dequantize on-the-fly về BF16 khi tính). Ba thành phần mới:

1. **4-bit NormalFloat (NF4):** kiểu *quantile quantization* xây trên các phân vị của $\mathcal{N}(0,1)$; vì trọng số pretrained ≈ Gaussian tâm 0, mỗi bin chứa số giá trị bằng nhau ⇒ **tối ưu về lý thuyết thông tin**. Lưới bất đối xứng để biểu diễn đúng số 0. Block size 64.
2. **Double Quantization (DQ):** lượng tử hóa lần hai chính các hằng số lượng tử hóa: 32-bit absmax → 8-bit (block 256) ⇒ tiết kiệm **≈0.373 bit/tham số** (≈3 GB với 65B).
3. **Paged Optimizers:** dùng NVIDIA Unified Memory tự động phân trang trạng thái optimizer sang CPU RAM khi GPU đầy, tránh OOM tại đỉnh bộ nhớ.

Công thức một lớp linear:

$$Y^{\text{BF16}} = X^{\text{BF16}}\,\mathrm{doubleDequant}\!\left(c_1^{\text{FP32}}, c_2^{\text{8-bit}}, W^{\text{NF4}}\right) + X^{\text{BF16}} L_1^{\text{BF16}} L_2^{\text{BF16}}$$

Gradient chỉ tính cho $L_1, L_2$ (LoRA); $W$ chỉ dùng để lan truyền $\partial E/\partial X$.

> **Cấu hình Guanaco:** LoRA trên **mọi linear layer**, $r{=}64$, $\alpha{=}16$, dropout 0.1 (≤13B) / 0.05 (33B, 65B), LR 2e-4 (7B/13B) / 1e-4 (33B/65B), paged AdamW 32-bit, max grad norm 0.3, batch 16, NF4 + DQ, BF16 compute.

### 5. Ý tưởng kỹ thuật cốt lõi
Tách vai trò: **trọng số gốc chỉ cần đủ chính xác để "đọc"** (forward + lan truyền gradient qua) — nên nén cực mạnh về 4-bit — trong khi **toàn bộ khả năng học nằm ở adapter BF16 nhỏ**. Điều kiện để không sụt chất lượng: (a) NF4 hầu như không mất thông tin với phân bố trọng số thực; (b) **áp LoRA lên tất cả linear layer** (không chỉ attention) để **bù sai số lượng tử hóa** và khớp 16-bit full FT — **số lượng module quan trọng hơn rank $r$**.

### 6. Dataset
- **Huấn luyện:** 8 bộ instruction — OASST1 (~9k mẫu chọn lọc), HH-RLHF, Alpaca (52k), Self-Instruct, Unnatural Instructions, FLAN v2 (mẫu 450k), Chip2, Longform.
- **Kiểm chứng lượng tử hóa:** GLUE (RoBERTa-large), Super-NaturalInstructions (T5 60M–3B), LLaMA 7B–65B trên Alpaca / FLAN v2.
- **Đánh giá chatbot:** Vicuna benchmark (80 prompt), OpenAssistant benchmark (953 prompt).
- **Bias:** CrowS-Pairs. Quy mô: >1.000 mô hình được fine-tune.

### 7. Metric
- **MMLU** 5-shot accuracy (kiến thức học thuật).
- **Elo rating** từ so sánh cặp bởi **GPT-4** và bởi **người chấm (MTurk)**; % so với ChatGPT theo thang 10 điểm.
- Accuracy (GLUE), ROUGE-L (SuperNI).
- Bộ nhớ GPU (GB), thời gian train (giờ/GPU đơn), footprint 4-bit. Bias: CrowS-Pairs.

### 8. Kết quả chính
- **Guanaco-65B** đạt **99.3% hiệu năng ChatGPT** trên Vicuna benchmark (GPT-4 judge), train **24 h trên một GPU 48 GB**; Guanaco-33B 97.8% với 12 h trên GPU 24 GB; Guanaco-7B chạy ~5 GB VRAM.
- **NF4 + DQ khớp 16-bit:** trên MMLU 5-shot (LLaMA 7B→65B, Alpaca/FLAN v2), chênh giữa QLoRA (NF4+DQ) và LoRA BF16 nằm trong ±0.5–1 điểm; tương tự trên GLUE (RoBERTa) và SuperNI (T5) so với 16-bit full FT ⇒ **không mất hiệu năng**.
- **NF4 > FP4 ≈ Int4** nhất quán (zero-shot acc / perplexity trên LLaMA/OPT/BLOOM/Pythia 125M–65B).
- **Target modules > rank:** LoRA chỉ trên attention thua rõ; LoRA trên mọi linear layer mới khớp full FT; $r$ từ 8→256 ảnh hưởng nhỏ khi đã áp đủ layer.
- **Chất lượng dữ liệu > kích cỡ:** 9k mẫu OASST1 cho chatbot tốt hơn 450k FLAN v2; MMLU cao **không** đồng nghĩa chatbot tốt.
- **Bộ nhớ:** 65B từ >780 GB xuống <48 GB (giảm ~16×); DQ tiết kiệm ≈3 GB ở 65B.
- **Đánh giá:** GPT-4 có **thiên lệch thứ tự** (thiên vị đáp án trước) ⇒ cần chấm cả hai thứ tự; đồng thuận GPT-4 – người ở mức trung bình.

### 9. Hạn chế
1. Chưa kiểm chứng QLoRA khớp **full 16-bit FT ở 33B/65B** (quá tốn), chỉ chứng minh ở ≤13B/T5-3B và so với LoRA-16bit.
2. Đánh giá chatbot chủ yếu bằng GPT-4 judge trên benchmark nhỏ (80 prompt) ⇒ nhiễu, thiên lệch; MMLU không phản ánh khả năng thực tế.
3. Không khảo sát 3-bit hay phối hợp với pruning/distillation.
4. **Tốc độ:** dequantize on-the-fly khiến mỗi bước train **chậm hơn** LoRA 16-bit (cộng đồng đo ≈30–40% tùy phần cứng); paged optimizer khi kích hoạt chậm đáng kể.
5. **Thuần văn bản, thuần tiếng Anh**; không đụng vision encoder/projector; không có kết quả cho ngôn ngữ ít tài nguyên.
6. Phụ thuộc `bitsandbytes` (CUDA); tương thích Windows/AMD/TPU hạn chế.

### 10. Học/dùng được gì cho đề tài? (BẮT BUỘC)
- **Cấu hình huấn luyện xuất phát:** `load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bfloat16` (T4 không hỗ trợ BF16 ⇒ dùng FP16 trên T4; BF16 trên A100/L4). Optimizer `paged_adamw_32bit` / `paged_adamw_8bit`, `max_grad_norm=0.3`, gradient checkpointing.
- **Target modules = tất cả linear của LLM** (`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` với Qwen2.5-VL) — ablation "attention-only vs. all-linear" nên chạy đầu tiên (W08).
- **Ước lượng VRAM (mục tiêu W04):** Qwen2.5-VL-3B ở NF4 ≈ 2–2.5 GB trọng số, 7B ≈ 4.5–5.5 GB; cộng activation với ảnh 448–768 px, chuỗi ≤1024 token, batch 1–2 + grad-accum ⇒ **7B khả thi trên T4 16 GB (chậm), thoải mái trên A100 40 GB**. Phải đo và ghi thật vào `experiments/`.
- **Bài học dữ liệu:** ~9k mẫu chất lượng cao đủ đổi hành vi mô hình ⇒ với ViVQA (~15k QA) và ViTextVQA, nhóm không cần lo "ít dữ liệu"; ưu tiên **làm sạch câu trả lời tiếng Việt** (chuẩn hóa NFC, chữ thường, bỏ dấu câu) hơn tăng số mẫu.
- **Bài học đánh giá:** không dùng một benchmark duy nhất; báo cáo **cả EM/VQA-Acc/ANLS** + **hallucination**; nếu dùng LLM-as-judge cho câu trả lời mở, phải hoán vị thứ tự để khử thiên lệch.
- **Rủi ro vào plan:** QLoRA chậm hơn LoRA 16-bit ⇒ phải log **thời gian PEFT fine-tuning** (Master Instruction §7.5); cân nhắc 8-bit / LoRA-BF16 trên A100 nếu VRAM cho phép — ablation "4-bit vs 8-bit vs 16-bit LoRA" là đóng góp *efficiency* đo được.

> ### Paper này liên quan gì đến đề tài ViVQA-VLM?
> QLoRA là **cầu nối giữa ràng buộc kỹ thuật (PEFT-only) và ràng buộc hạ tầng (Colab)**. Master Instruction cho phép "QLoRA (quantize 4-bit hoặc 8-bit)" chính vì paper này chứng minh NF4 + LoRA không làm mất chất lượng. Với ViVQA-VLM, QLoRA cho phép chọn baseline **lớn hơn và mạnh hơn** (Qwen2.5-VL-7B thay vì 3B) mà vẫn fine-tune được, tăng cơ hội cải thiện EM/ANLS. Hai kết luận thực nghiệm — *all-linear targets* và *data quality > size* — trở thành hai giả thuyết phụ (**H3, H4** trong `Hypotheses.md`) SV3 kiểm chứng ở W08–W12. Paper cũng cảnh báo QLoRA mới chứng minh trên LLM thuần văn bản tiếng Anh: **kiểm chứng NF4 + LoRA trên VLM cho tiếng Việt là một khoảng trống nhỏ nhưng thật** mà đề tài có thể đóng góp bằng chứng.

---

## P3. VL-Adapter: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks

| Trường | Giá trị |
|--------|---------|
| Title | VL-Adapter: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks |
| Author(s) | Yi-Lin Sung, Jaemin Cho, Mohit Bansal (UNC Chapel Hill) |
| Year / Venue | 2022, CVPR, pp. 5227–5237 (arXiv:2112.06825) |
| Code | https://github.com/ylsung/VL_adapter |

### 1. Vấn đề paper giải quyết
Các mô hình V&L (VL-T5 / VL-BART) được fine-tune **toàn bộ** cho từng task (VQA, captioning, reasoning). Khi backbone ngôn ngữ lớn dần, việc này tốn bộ nhớ, lưu trữ, không chia sẻ được giữa task. Câu hỏi: **các kỹ thuật PEFT thành công trong NLP (Adapter, Hyperformer, Compacter, LoRA, Prompt-tuning) có chuyển được sang V&L — đặc biệt VQA — mà vẫn giữ hiệu năng full FT không? Kỹ thuật nào tốt nhất? Chia sẻ tham số giữa task có lợi không?**

### 2. Vì sao quan trọng
- Trước paper này, chưa nghiên cứu nào áp dụng PEFT cho V&L khó như VQA/captioning một cách có hệ thống. Kết luận NLP không hiển nhiên chuyển sang đa phương thức vì mô hình V&L phải hợp nhất hai phân bố đầu vào.
- Với ViVQA-VLM: đây là **paper duy nhất trong 4 paper trực tiếp đo PEFT trên VQA** với metric giống nhóm (VQA accuracy). Nó quyết định nhóm nên đặt kỳ vọng thế nào cho LoRA/adapter trên VQA.

### 3. Research gap
1. Chưa có benchmark PEFT có hệ thống trên V&L; các so sánh NLP trước dùng cấu hình khác nhau, không công bằng.
2. Chưa rõ **có cần fine-tune vision encoder** (CLIP) khi làm PEFT.
3. Chưa rõ **chia sẻ adapter giữa task** (multi-task) có giúp task ít dữ liệu.
4. Chưa rõ **prompt-tuning** — rẻ nhất — có hoạt động trên V&L.

### 4. Phương pháp
Kiến trúc **CLIP-BART / CLIP-T5**: CLIP (ResNet-101 cho ảnh; ViT-B/32 cho video) → visual projection layer → encoder-decoder LM (BART-base / T5-base); mọi task đưa về **sinh văn bản** với task prefix (theo VL-T5). Trên nền đó **benchmark có kiểm soát** trong thiết lập **multi-task** (train chung 4 task):

- **Adapter** (bottleneck $d \to d_i \to d$, GELU, cập nhật thêm LayerNorm), ba biến thể chia sẻ: *Multiple* (mỗi task một adapter), *Half-shared* (chia sẻ lớp up-sampling), *Single* (một adapter chung mọi task).
- **Hyperformer:** hyper-network chung sinh trọng số adapter theo (task, layer).
- **Compacter:** adapter với lớp PHM (Kronecker) — thấy rằng chia sẻ ma trận + low-rank trong PHM làm hại V&L nên bỏ low-rank.
- **LoRA** (Single / Multiple) và **Prompt-tuning** (Single / Multiple, chỉ ở encoder).
- Vision encoder CLIP **đóng băng** trong mọi thiết lập PEFT (chỉ train projection + module PEFT).

### 5. Ý tưởng kỹ thuật cốt lõi
**(a)** Chia sẻ trọng số adapter giữa task để vừa giảm tham số vừa chuyển kiến thức sang task ít dữ liệu (NLVR2); **(b)** với V&L, chỉ cần adapt phần *ngôn ngữ* + một lớp chiếu, **không cần đụng vision encoder** — kiến thức thị giác từ CLIP đã đủ, vấn đề là *giao diện* giữa hai phương thức. Đây là tiền đề tư duy của LLaVA sau này (frozen ViT + projector + LLM).

### 6. Dataset
- **Image-text (multi-task):** VQAv2 (Karpathy split), GQA (test-dev), NLVR2 (test-P), MSCOCO Captioning (Karpathy).
- **Video-text (VALUE):** TVQA, How2QA (video QA), TVC, YC2C (video captioning).
- Ảnh resize 224×224, grid feature 7×7 → pool 6×6.

### 7. Metric
VQA accuracy (VQAv2, GQA), accuracy (NLVR2), **CIDEr** (COCO caption), **% tham số cập nhật** (trục hiệu quả), trung bình 4 task (Avg.). Thêm leaderboard test-std cho VQA/GQA.

### 8. Kết quả chính
Bảng chính — **CLIP-BART, multi-task (Table 1)**:

| Phương pháp | % tham số | VQA | GQA | NLVR2 | COCO CIDEr | Avg |
|-------------|-----------|-----|-----|-------|------------|-----|
| Full fine-tuning | 100.00 | 67.6 | 56.7 | 73.0 | 112.9 | **77.6** |
| Multiple Adapters | 12.22 | 65.4 | 54.0 | 69.8 | 114.3 | 75.9 |
| Half-shared Adapters | 8.36 | 65.2 | 53.4 | 71.2 | 113.7 | 75.9 |
| **Single Adapter** | **4.18** | 65.9 | 54.5 | **74.2** | 114.9 | **77.4** |
| Hyperformer | 5.79 | 65.1 | 53.4 | 72.3 | 114.6 | 76.4 |
| Single Compacter | 2.70 | 64.2 | 53.3 | 71.7 | 114.1 | 75.8 |
| Multiple LoRA | 17.72 | 65.5 | 53.0 | 62.8 | 115.4 | 74.2 |
| **Single LoRA** | 5.93 | 65.2 | 53.6 | 71.9 | 115.3 | 76.5 |
| Single Prompt | 2.00 | 44.0 | 36.3 | 51.8 | 103.9 | 59.0 |

Kết luận:
- **Single Adapter (4.18% tham số) khớp full FT** (77.4 vs 77.6); trên CLIP-T5 tương tự (77.4 vs 78.1). Video-text: 3.39% tham số, ngang full FT.
- **Chia sẻ giữa task giúp task ít dữ liệu:** NLVR2 tăng 69.8 (Multiple) → 74.2 (Single), cao hơn cả full FT.
- **LoRA cạnh tranh** (Single LoRA 76.5, tốt nhất CIDEr) nhưng **Multiple LoRA sụp trên NLVR2 (62.8)** — LoRA riêng cho task nhỏ dễ overfit / thiếu dữ liệu.
- **Prompt-tuning thất bại** trên V&L (59.0), nhạy với khởi tạo và kích cỡ mô hình.
- **Đóng băng CLIP** chỉ mất ≈0.9 điểm VQA ⇒ đáng đổi để tiết kiệm tham số.
- Compacter/Hyperformer — phức tạp hơn — **không** tốt hơn adapter vanilla trên V&L.

### 9. Hạn chế
1. Backbone nhỏ (BART-base/T5-base ~140–220M) và **không có V&L pretraining quy mô lớn**; kết luận có thể không giữ ở LLM 7B+ (LLaVA-1.5 phần nào xác nhận vẫn giữ).
2. Chỉ **tiếng Anh**, ảnh thông thường; **không có scene-text VQA / OCR**, không có ngôn ngữ ít tài nguyên.
3. Độ phân giải thấp (224 px) + feature grid 6×6 ⇒ không đủ đọc chữ trong ảnh — hạn chế trực tiếp với ViTextVQA.
4. Prompt-tuning không được tune kỹ (tác giả thừa nhận) ⇒ kết luận "prompt yếu" chưa phải kết luận cuối.
5. Adapter thêm lớp tuần tự ⇒ có độ trễ suy luận, khác LoRA gộp được.
6. Chưa khảo sát lượng tử hóa / ngân sách VRAM — hiệu quả chỉ đo bằng % tham số.

### 10. Học/dùng được gì cho đề tài? (BẮT BUỘC)
- **Kỳ vọng thực tế:** PEFT ≈ 4–6% tham số có thể **khớp** full FT trên VQA ⇒ ràng buộc PEFT-only không phải "chấp nhận mất chất lượng" mà là lựa chọn hợp lý về khoa học.
- **Đóng băng vision encoder** (ViT của Qwen2.5-VL / Q-Former+ViT của BLIP-2 / DaViT của Florence-2) là mặc định; chỉ adapt (i) projector/merger và (ii) LLM. Có thể thêm 1 ablation "LoRA thêm vào ViT" cho ViTextVQA vì scene-text tiếng Việt có dấu là phân bố thị giác mới.
- **Multi-task sharing:** train **một adapter chung ViVQA + ViTextVQA** thay vì hai adapter riêng — paper cho thấy task nhỏ (ViTextVQA, hoặc loại câu hỏi hiếm như counting) được lợi. Ablation rẻ và có tính "novelty" trong ngữ cảnh tiếng Việt.
- **Không đầu tư soft prompt-tuning**; thay bằng **prompt engineering rời (discrete) + OCR-enhanced prompting** — phù hợp danh mục cải tiến §4.2.
- **Không kỳ vọng Compacter/Hyperformer**; giữ pipeline đơn giản: LoRA (gộp được, không độ trễ) là lựa chọn số 1, adapter vanilla là dự phòng nếu LoRA không hội tụ.
- **Trục hiệu quả:** báo cáo % tham số huấn luyện, VRAM đỉnh, thời gian train **bên cạnh** EM/ANLS — như Table 1 — để hội đồng thấy trade-off rõ ràng.

> ### Paper này liên quan gì đến đề tài ViVQA-VLM?
> VL-Adapter là bằng chứng gần nhất về **PEFT trên đúng bài toán VQA** — cùng loại metric (VQA accuracy) và cùng câu hỏi "PEFT có ngang full FT không". Nó cho ba quyết định thiết kế mà ViVQA-VLM kế thừa trực tiếp: **đóng băng vision encoder**, **PEFT trên phần ngôn ngữ + giao diện**, **chia sẻ adapter đa task**. Đồng thời, các hạn chế của nó (tiếng Anh, 224 px, không scene-text) **định vị chính xác khoảng trống G2–G3** trong `Research_Gap.md`: PEFT cho VLM chưa kiểm chứng trên **tiếng Việt** và **text-in-image**. Kết quả Multiple LoRA sụp ở NLVR2 là cảnh báo sớm: nếu subset ViTextVQA của nhóm nhỏ, LoRA riêng có thể kém ⇒ cân nhắc adapter chung hoặc trộn dữ liệu.

---

## P4. Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)

| Trường | Giá trị |
|--------|---------|
| Title | Improved Baselines with Visual Instruction Tuning |
| Author(s) | Haotian Liu, Chunyuan Li, Yuheng Li, Yong Jae Lee (UW–Madison, Microsoft Research) |
| Year / Venue | 2024, CVPR (arXiv:2310.03744, 10/2023) |
| Code / Weights | https://github.com/haotian-liu/LLaVA · Model Zoo có checkpoint **LoRA** và **full FT** |

### 1. Vấn đề paper giải quyết
Sau LLaVA (2023), cộng đồng chưa rõ **recipe tốt nhất để huấn luyện Large Multimodal Model (LMM)**: LLaVA mạnh hội thoại nhưng yếu VQA ngắn; InstructBLIP/Qwen-VL mạnh VQA nhưng cần resampler (Q-Former) train trên hàng trăm triệu–tỷ cặp ảnh-văn bản. Câu hỏi có kiểm soát: **những thay đổi tối thiểu nào (dữ liệu, connector, độ phân giải, LLM) đủ để tạo baseline VLM mạnh, tái lập được, train trong ~1 ngày trên 8×A100 với chỉ ~1.2M mẫu công khai?** Phần Model Zoo trả lời thêm: **LoRA có thay được full FT ở giai đoạn instruction tuning không?**

### 2. Vì sao quan trọng
- Baseline **đơn giản, công khai, rẻ** là điều kiện để nghiên cứu VLM tái lập và so sánh công bằng — đúng tinh thần "reproduce trước, cải tiến sau".
- Kiến trúc *ViT → MLP projector → LLM decoder* của LLaVA-1.5 là **mẫu hình** Qwen2.5-VL (baseline ứng viên số 1) cũng theo ⇒ kết luận về LoRA, dữ liệu VQA và response-format prompt chuyển sang Qwen2.5-VL với ít rủi ro.
- Paper mở rộng về **hallucination** và **độ phân giải** — hai trục phân tích bắt buộc (Master Instruction §7.2, §7.4).

### 3. Research gap
1. Chưa có nghiên cứu **có kiểm soát** về lựa chọn thiết kế LMM (connector, dữ liệu, độ phân giải, kích cỡ LLM); các so sánh trước lẫn nhiều biến.
2. Chưa rõ vì sao LLaVA yếu ở VQA trả lời ngắn — thiếu dữ liệu VQA hay do **định dạng câu trả lời** không được chỉ dẫn.
3. Chưa có bằng chứng công khai (cùng dữ liệu, cùng lịch huấn luyện) rằng **LoRA ≈ full FT cho VLM 7B/13B**.

### 4. Phương pháp
Giữ khung LLaVA + các cải tiến "trực giao":
- **Vision encoder:** CLIP ViT-L/14 **336 px** (thay 224 px) — tăng chi tiết ảnh.
- **Connector:** **MLP 2 lớp** (thay 1 lớp linear).
- **LLM:** Vicuna-v1.5 7B / 13B.
- **Dữ liệu:** 558K LCS (pretrain projector) + **665K** instruction gồm LLaVA-Instruct 158K, ShareGPT 40K (thuần văn bản), và các bộ **VQA học thuật**: VQAv2, GQA, OKVQA, A-OKVQA, **OCR-VQA, TextCaps**, RefCOCO, Visual Genome.
- **Response-format prompt:** thêm chỉ dẫn *"Answer the question using a single word or phrase."* vào dữ liệu VQA ngắn ⇒ mô hình học phân biệt khi nào trả lời ngắn/dài **mà không cần ChatGPT hậu xử lý**.
- **Huấn luyện:** 2 giai đoạn, ~6 h pretrain + ~20 h instruction tuning (13B) trên 8×A100; greedy decoding khi đánh giá.
- **LoRA (`finetune_lora.sh`):** `lora_r=128, lora_alpha=256`, LR 2e-4 cho LoRA, LR riêng **2e-5 cho projector**, 1 epoch; vision encoder đóng băng; chạy 13B trên 8×A100-40G / 8×A6000, 7B trên 8×RTX3090.
- **LLaVA-1.5-HD:** chia ảnh thành lưới ô 336 px + ảnh toàn cục thu nhỏ để đạt 448 px+ mà không train lại ViT.

### 5. Ý tưởng kỹ thuật cốt lõi
**Connector đơn giản (MLP) đủ mạnh và cực tiết kiệm dữ liệu nếu (a) độ phân giải đủ và (b) dữ liệu instruction có định dạng câu trả lời rõ.** Phần "yếu VQA" của LLaVA không phải lỗi kiến trúc mà là lỗi **định dạng** — chỉ thêm một câu prompt định dạng, MME tăng **809.6 → 1323.8** (chỉ với VQAv2). Về PEFT: vì ViT đóng băng và projector đã pretrain, phần cần học ở giai đoạn 2 chủ yếu là **ánh xạ chỉ dẫn → hành vi** trong LLM — bản chất hạng thấp — nên LoRA r=128 trên LLM + projector full-train là đủ.

### 6. Dataset
- **Huấn luyện:** LCS-558K (LAION/CC/SBU caption); 665K mixture (§4).
- **Đánh giá (12 benchmark):** VQAv2, GQA, VizWiz, ScienceQA-IMG, **TextVQA** (text-rich), **POPE** (hallucination), MME, MMBench, MMBench-CN, SEED-Bench, LLaVA-Bench-in-the-Wild, MM-Vet.

### 7. Metric
VQA accuracy (VQAv2, GQA, VizWiz, TextVQA), accuracy (SQA, MMBench, SEED), **F1 trên POPE** (random/popular/adversarial — object hallucination), điểm MME-Perception, GPT-4-judge (LLaVA-Wild, MM-Vet). Thêm: thời gian train, số mẫu, kích cỡ LLM, độ phân giải.

### 8. Kết quả chính
- **SOTA 11/12 benchmark** với 1.2M mẫu công khai (vs. hàng trăm triệu của Qwen-VL/InstructBLIP). 13B: VQAv2 80.0, GQA 63.3, TextVQA 61.3, POPE 85.9, MME 1531.3, MMBench 67.7.
- **Ablation lộ trình (Table 2, 7B):** +VQAv2 → MME 809.6→1197.0; +format prompt → 1323.8; +MLP → 1355.2; +OKVQA/OCR → 1377.6; +336 px → 1450; +GQA → 1469.2; +ShareGPT → 1510.7; 13B → 1531.3.
- **LoRA vs full FT (Model Zoo chính thức, cùng dữ liệu, 1 epoch):**

| Mô hình | Lịch train | VQAv2 | GQA | VizWiz | SQA | TextVQA | POPE | MME | MMBench |
|---------|-----------|-------|-----|--------|-----|---------|------|-----|---------|
| LLaVA-1.5-7B | full FT | 78.5 | 62.0 | 50.0 | 66.8 | 58.2 | 85.9 | 1510.7 | 64.3 |
| LLaVA-1.5-7B | **LoRA** | 79.1 | 63.0 | 47.8 | 68.4 | 58.2 | 86.4 | 1476.9 | 66.1 |
| LLaVA-1.5-13B | full FT | 80.0 | 63.3 | 53.6 | 71.6 | 61.3 | 85.9 | 1531.3 | 67.7 |
| LLaVA-1.5-13B | **LoRA** | 80.0 | 63.3 | 58.9 | 71.2 | 60.2 | 86.7 | 1541.7 | 68.5 |

  ⇒ LoRA **ngang hoặc hơn** full FT trên phần lớn benchmark (kể cả **POPE — hallucination thấp hơn**), chỉ thấp hơn nhẹ ở TextVQA (−1.1 ở 13B) và MME (7B).
- **Hallucination giảm khi tăng độ phân giải** (448 px, LLaVA-1.5-HD): mô hình bớt "bịa" chi tiết không nhìn thấy; nhiễu dữ liệu nhỏ không phải nguyên nhân chính.
- **Đa ngôn ngữ nổi lên (emergent):** chỉ nhờ ShareGPT (văn bản đa ngữ), LLaVA-1.5 trả lời được nhiều ngôn ngữ dù không có dữ liệu ảnh-đa-ngữ; vẫn còn lỗi ở một số ngôn ngữ.

### 9. Hạn chế
1. **Dùng toàn bộ patch ảnh** ⇒ chuỗi dài, mỗi bước train chậm; chưa có resampler hiệu quả tương đương.
2. **Không xử lý nhiều ảnh** trong một lượt; hạn chế ở in-context / tài liệu nhiều trang.
3. **Hallucination vẫn tồn tại**, đặc biệt khi độ phân giải không đủ so với độ chi tiết dữ liệu huấn luyện.
4. Kết quả LoRA nằm ở **Model Zoo/GitHub, không có trong bản CVPR chính thức** ⇒ khi trích dẫn phải ghi rõ nguồn; không có phân tích rank/target-module.
5. **Tiếng Anh là trung tâm**; TextVQA tiếng Anh; đa ngữ chỉ là hiện tượng nổi lên chưa đo định lượng — không có gì cho tiếng Việt / dấu thanh.
6. Cần 8×A100 để tái lập đầy đủ — **vượt tài nguyên nhóm**; nhóm chỉ tái lập ở quy mô nhỏ hoặc dùng checkpoint công khai.

### 10. Học/dùng được gì cho đề tài? (BẮT BUỘC)
- **Bằng chứng LoRA ≈ full FT trên VLM 7B/13B** — bảng trên là lý lẽ mạnh nhất để hội đồng chấp nhận PEFT-only không giảm giá trị khoa học. Đặc biệt **POPE tăng** với LoRA gợi ý PEFT còn *giảm* hallucination (giữ prior của LLM gốc) — giả thuyết đáng kiểm chứng trên Vietnamese VQA (**H5**).
- **Response-format prompt tiếng Việt:** thêm *"Trả lời bằng một từ hoặc cụm từ ngắn."* vào mọi mẫu ViVQA/ViTextVQA khi fine-tune và khi eval ⇒ trực tiếp cải thiện **EM/VQA-Acc** mà không tốn tham số. Đây là *prompt optimization* trong §4.2.
- **Hai learning rate:** LoRA trên LLM ~2e-4, projector/merger ~2e-5 (hoặc đóng băng). Với Qwen2.5-VL: cân nhắc mở `visual.merger` LR nhỏ, đóng băng ViT.
- **Độ phân giải là đòn bẩy cho scene-text + hallucination:** với ViTextVQA, dùng độ phân giải cao nhất VRAM cho phép (Qwen2.5-VL hỗ trợ dynamic resolution qua `min_pixels/max_pixels`) — ablation "độ phân giải × OCR-prompting" hợp cho W12.
- **Trộn dữ liệu văn bản (ShareGPT-style) tiếng Việt** vào tập fine-tune có thể nâng chất lượng tiếng Việt của câu trả lời mà không cần thêm ảnh — rẻ, nên thử như một ablation.
- **Greedy decoding khi eval** để tái lập; báo cáo POPE-style yes/no probing tiếng Việt cho hallucination (phối hợp SV4).

> ### Paper này liên quan gì đến đề tài ViVQA-VLM?
> LLaVA-1.5 là **bản thiết kế tham chiếu** cho baseline VLM hiện đại nhóm sẽ dùng (Qwen2.5-VL chia sẻ khung ViT → merger → LLM), và là **bằng chứng thực nghiệm gần nhất với bối cảnh đề tài** rằng LoRA đủ để adapt VLM 7B ở giai đoạn instruction tuning. Paper cung cấp ba "đòn bẩy rẻ" phù hợp ràng buộc PEFT-only: (1) LoRA $r$ lớn trên LLM + projector LR nhỏ, (2) response-format prompt tiếng Việt để tối ưu EM, (3) tăng độ phân giải để cải thiện scene-text và giảm hallucination. Hạn chế "tiếng Anh trung tâm, đa ngữ chỉ nổi lên" của LLaVA-1.5 chính là **gap G1** trong `Research_Gap.md`: alignment ảnh–ngôn ngữ chưa tối ưu cho tiếng Việt; nhóm sẽ đo xem **LoRA + dữ liệu Việt** lấp được phần nào gap đó.

---

## 5. Tổng hợp so sánh & khuyến nghị kỹ thuật cho ViVQA-VLM

### 5.1 Bảng so sánh 4 paper

| Tiêu chí | P1 LoRA | P2 QLoRA | P3 VL-Adapter | P4 LLaVA-1.5 |
|----------|---------|----------|---------------|--------------|
| Đối tượng | LLM (GPT-3, RoBERTa, GPT-2) | LLM (LLaMA 7–65B, T5) | V&L nhỏ (CLIP-BART/T5) | VLM 7B/13B |
| Đóng góp chính | $\Delta W$ hạng thấp, 0 độ trễ | NF4 + DQ + paged optim; 65B trên 1 GPU | Benchmark PEFT trên VQA; adapter chia sẻ | Recipe VLM đơn giản; LoRA ≈ full FT |
| % tham số train | ~0.01–0.1% | ~0.5–2% ($r{=}64$ all-linear) | 4.18% (adapter) / 5.93% (LoRA) | LoRA $r{=}128$ (~vài %) |
| Bằng chứng ≈ full FT | Có (LLM) | Có (LLM, ≤13B) | Có (VQA, mô hình nhỏ) | Có (VLM 7B/13B) |
| Có VQA? | Không | Không | **Có** | **Có** |
| Có scene-text/OCR? | Không | Không | Không | TextVQA (tiếng Anh) |
| Có tiếng Việt? | Không | Không | Không | Không |
| Có hallucination? | Không | Không | Không | **POPE** |
| Dùng được qua HF `peft`? | Có | Có (`bitsandbytes`) | Có (LoRA/adapter) | Có (script LoRA) |

**Khoảng trống chung của cả 4 paper** (evidence cho `Research_Gap.md`): không paper nào kiểm chứng PEFT/QLoRA cho **VLM trên ngôn ngữ ít tài nguyên có dấu thanh (tiếng Việt)**, và không paper nào đo PEFT trên **scene-text VQA ngoài tiếng Anh**. Đây chính là gap **G3** của đề tài, có evidence.

### 5.2 Recipe PEFT đề xuất cho baseline Qwen2.5-VL (áp dụng tương tự BLIP-2 / Florence-2)

```yaml
# configs/peft_qlora_qwen25vl.yaml (đề xuất, hiện thực ở W04–W06)
base_model: Qwen/Qwen2.5-VL-3B-Instruct      # fallback: 7B nếu A100 40GB
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4                    # QLoRA (P2)
  bnb_4bit_use_double_quant: true             # QLoRA (P2)
  bnb_4bit_compute_dtype: bfloat16            # fp16 nếu T4
lora:
  r: 16                                       # sweep {8,16,32,64}; P1: r nhỏ đủ
  alpha: 32                                   # alpha = 2r
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]  # all-linear (P2)
  modules_to_save: []                         # ablation: thêm visual.merger (P4: projector LR nhỏ)
freeze_vision_tower: true                     # P3, P4
train:
  lr: 2.0e-4                                  # P2/P4
  projector_lr: 2.0e-5                        # P4 (nếu mở merger)
  epochs: 1-3
  batch_size: 2
  grad_accum: 8
  max_grad_norm: 0.3                          # P2
  optimizer: paged_adamw_32bit                # P2
  gradient_checkpointing: true
  seed: 42
prompt:
  response_format_vi: "Trả lời bằng một từ hoặc cụm từ ngắn."   # P4
  ocr_enhanced: optional                      # phối hợp SV2
eval:
  decoding: greedy                            # P4
  metrics: [EM, VQA-Acc, ANLS]                # Master Instruction §7.1
  hallucination: POPE-style (vi)              # P4 + SV4
```

### 5.3 Ablation do SV3 chịu trách nhiệm (W08–W12), xếp theo ưu tiên

| # | Ablation | Xuất phát từ | Giả thuyết |
|---|----------|--------------|-----------|
| A1 | Zero-shot vs prompt-format-vi vs LoRA/QLoRA | P4 | Format prompt đã tăng EM; LoRA tăng thêm |
| A2 | Target modules: attention-only vs all-linear | P1, P2 | All-linear > attention-only ở cùng ngân sách |
| A3 | Rank $r \in \{8,16,32,64\}$ | P1 | Hiệu năng bão hòa sớm ($r \le 16$) |
| A4 | 4-bit NF4 vs 8-bit vs 16-bit LoRA | P2 | Chênh ≤ 1 điểm EM; VRAM/thời gian khác nhiều |
| A5 | Adapter chung (ViVQA+ViTextVQA) vs adapter riêng | P3 | Chung tốt hơn cho tập nhỏ / câu hỏi hiếm |
| A6 | Mở merger/projector vs đóng băng | P3, P4 | Mở với LR nhỏ giúp scene-text |
| A7 | Có vs không OCR-enhanced prompting (với SV2) | P4 (TextVQA) | Tăng ANLS trên ViTextVQA |
| A8 | Hallucination (POPE-vi) trước/sau LoRA (với SV4) | P4 | LoRA không làm tăng, có thể giảm |

### 5.4 Rủi ro riêng của mảng PEFT & phương án

| Rủi ro | Dấu hiệu | Mitigation | Backup |
|--------|----------|-----------|--------|
| OOM trên T4 với 7B | CUDA OOM ở bước đầu | Giảm `max_pixels`, batch 1, grad-accum, dùng 3B | Colab A100 / GPU trường |
| QLoRA train chậm | > 2 h / epoch trên ViVQA | Subset train (ghi rõ N), 8-bit / BF16 LoRA nếu VRAM đủ | Prompt-opt không train |
| Không hội tụ / catastrophic forgetting | Loss không giảm, trả lời tiếng Anh | Giảm LR (1e-4), thêm text-only tiếng Việt, kiểm tra tokenizer | Prompt engineering + RAG |
| Adapter overfit trên tập nhỏ | Train EM ≫ val EM | Dropout 0.1, adapter chung (P3), early stopping | — |
| `bitsandbytes` không tương thích | Import lỗi | Pin phiên bản, dùng Linux/Colab | 8-bit / BF16 |

### 5.5 Đóng góp vào Literature Matrix (4 hàng — đã chép sang `Literature_Matrix.md`)

| # | Author / Year / Venue | Problem | Method | Dataset / Metric | Kết quả chính | Limitation | What can we use? | SV |
|---|------------------------|---------|--------|------------------|---------------|------------|------------------|----|
| 6 | Hu et al. / 2022 / ICLR | Full FT LLM quá tốn | LoRA ($\Delta W{=}BA$) | GLUE, E2E, WikiSQL, MNLI, SAMSum / Acc, BLEU, ROUGE | ≈/＞ full FT với 10.000× ít tham số, 0 độ trễ | Chỉ LLM, tiếng Anh; chọn module heuristic | Kỹ thuật bắt buộc; $r$ nhỏ nhiều module; gộp trọng số cho demo | SV3 |
| 7 | Dettmers et al. / 2023 / NeurIPS | Fine-tune 65B trên 1 GPU | QLoRA: NF4 + DQ + paged optim | MMLU, Vicuna, GLUE, SuperNI | 99.3% ChatGPT; NF4 ≈ 16-bit | Chưa full-FT 65B; chậm hơn 16-bit; tiếng Anh | Fit Colab; all-linear targets; data quality > size | SV3 |
| 11 | Sung et al. / 2022 / CVPR | PEFT cho V&L | VL-Adapter (adapter chia sẻ, LoRA, prompt) | VQAv2, GQA, NLVR2, COCO / Acc, CIDEr | 4.18% tham số ≈ full FT; prompt-tuning yếu | Mô hình nhỏ, 224 px, tiếng Anh, không OCR | Freeze ViT; adapter chung đa task; bỏ soft-prompt | SV3 |
| 12 | Liu et al. / 2024 / CVPR | Recipe VLM mạnh, rẻ | LLaVA-1.5 (MLP, 336 px, VQA data, format prompt); LoRA | 12 benchmark incl. TextVQA, POPE | SOTA 11/12; LoRA ≈ full FT 7B/13B | Tiếng Anh; 8×A100; hallucination còn | Format prompt vi; 2 LR; độ phân giải; POPE | SV3 |

---

## 6. Tài liệu tham khảo

1. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. arXiv:2106.09685.
2. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023 (Oral). arXiv:2305.14314.
3. Sung, Y.-L., Cho, J., & Bansal, M. (2022). *VL-Adapter: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks.* CVPR 2022, pp. 5227–5237. arXiv:2112.06825.
4. Liu, H., Li, C., Li, Y., & Lee, Y. J. (2024). *Improved Baselines with Visual Instruction Tuning.* CVPR 2024. arXiv:2310.03744. Model Zoo: github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md.
5. Houlsby, N. et al. (2019). *Parameter-Efficient Transfer Learning for NLP.* ICML 2019. (adapter gốc — trích dẫn qua P1/P3)
6. Aghajanyan, A., Zettlemoyer, L., & Gupta, S. (2020). *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.* arXiv:2012.13255. (cơ sở giả thuyết hạng thấp của P1)
7. Bai, S. et al. (2025). *Qwen2.5-VL Technical Report.* arXiv:2502.13923. (baseline ứng viên — do SV1 review)

**Đọc bổ sung cho W02 (chưa phân tích đầy đủ):** DoRA (Liu et al., ICML 2024 — tách magnitude/direction, cải thiện LoRA ở $r$ thấp); LLaMA-Adapter V2 (Gao et al., 2023 — visual instruction model tham số-hiệu quả); *Aurora — Parameter-efficient Tuning of Large-scale Multimodal Foundation Model* (Wang et al., NeurIPS 2023).

---

## 7. Checklist tự kiểm (SV3)

- [x] ≥ 1 paper LoRA (P1)
- [x] ≥ 1 paper QLoRA (P2)
- [x] ≥ 1 paper PEFT/VLM adaptation (P3, P4)
- [x] Mỗi paper trả lời đủ 10 câu hỏi, câu 10 bắt buộc
- [x] Mỗi paper có mục "Paper này liên quan gì đến đề tài ViVQA-VLM?"
- [x] Số liệu có nguồn (bảng chính của paper / Model Zoo); chỗ nhớ gần đúng đánh dấu "≈"
- [x] Đối chiếu venue/tác giả/số liệu then chốt với công bố gốc (LLaVA Model Zoo, VL-Adapter CVPR pp. 5227–5237)
- [x] Chuyển 4 hàng vào `Literature_Matrix.md`
- [x] Đề xuất recipe + ablation + rủi ro cho `Candidate_Methods.md`, `Hypotheses.md`, `Project_Plan_15_Weeks.md`
- [ ] Tải PDF 4 paper vào `docs/W02/02_References/` và đối chiếu lại các số "≈" còn lại (VL-Adapter Table 3, RoBERTa/DeBERTa GLUE) — W02
