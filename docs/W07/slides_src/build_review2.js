// Build ViVQA-VLM Review 2 deck (EN or VI). Usage (needs pptxgenjs + the pptx skill apply_theme.js via SKILL_DIR):
//   SKILL_DIR=<pptx skill dir> node docs/W07/slides_src/build_review2.js en docs/W07/ViVQA-VLM_Review2_EN.pptx
// Every number on the slides comes from experiments/ (W07 runs) — see notes per slide.
const pptxgen = require("pptxgenjs");
const path = require("path");
const { applyTheme } = require(process.env.SKILL_DIR + "/scripts/apply_theme.js");

const LANG = process.argv[2] || "en";
const OUT = process.argv[3] || `ViVQA-VLM_Review2_${LANG.toUpperCase()}.pptx`;
const L = (en, vi) => (LANG === "en" ? en : vi);
// VI: decimal point -> comma for numbers like 22.1 / 0.86 (not 10.799, not Qwen2.5)
const FMT = (c) => (LANG === "en" ? c : c.replace(/(?<![A-Za-z\d])(\d+)\.(\d{1,2})(?!\d)/g, "$1,$2"));
const FIG = path.join(__dirname, "..", "figures");
const FIGS = LANG === "en"
  ? { a6: "Fig6A_ViVQA_pipeline_noise_EN.png", a7: "Fig7A_Qwen25VL3B_architecture_EN.png", b7: "Fig7B_evaluation_protocol_EN.png" }
  : { a6: "Fig6A_ViVQA_pipeline_noise_TV.png", a7: "Fig7A_Qwen25VL3B_architecture_TV.png", b7: "Fig7B_evaluation_protocol_TV.png" };
const SIZE = { a6: [2104, 816], a7: [2104, 1200], b7: [2104, 410] };

const HEX = { ink: "13294B", teal: "1C7293", amber: "E8A33D", text: "243044", muted: "6B7280", wash: "F3F5F8",
              tealt: "E7F0F3", ambert: "FBF0DA", line: "E2E7EE", red: "B42318", white: "FFFFFF" };
const THEME = {
  name: "ViVQA-VLM Review",
  headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: HEX.text, lt1: HEX.white, dk2: HEX.ink, lt2: HEX.wash, accent1: HEX.teal, accent2: HEX.amber,
            accent3: HEX.muted, accent4: HEX.tealt, accent5: HEX.ambert, accent6: HEX.red, hlink: HEX.teal, folHlink: HEX.ink },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "ViVQA-VLM Review 2";
pres.author = "Phạm Lê Huy Hoàng";
const C = pres.SchemeColor;
const W = 13.333, M = 0.6, CW = W - 2 * M;
const FOOT = L("ViVQA-VLM · Review 2 · GFA26AI35 · FPT University", "ViVQA-VLM · Review 2 · GFA26AI35 · Đại học FPT");

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE_DARK", background: { color: HEX.ink },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 1.7, w: CW, h: 1.2, fontFace: "Cambria", fontSize: 54, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 3.0, w: CW, h: 3.6, fontFace: "Calibri", fontSize: 18, color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: HEX.white },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "eyebrow", type: "body", x: M, y: 0.32, w: 9, h: 0.3, fontFace: "Calibri", fontSize: 11, bold: true, color: C.accent1, charSpacing: 2, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.62, w: CW, h: 0.7, fontFace: "Cambria", fontSize: 32, bold: true, color: C.text2, valign: "top", align: "left", margin: 0 }, text: "" } },
    { text: { text: FOOT, options: { x: M, y: 7.05, w: 8, h: 0.3, fontFace: "Calibri", fontSize: 10, color: C.accent3, margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.05, w: 0.6, h: 0.3, fontFace: "Calibri", fontSize: 10, color: HEX.muted, align: "right" },
});

// ---------- helpers ----------
let n = 0;
const oid = (s) => `${s}_${++n}`;
function content(section, eyebrow, title, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(eyebrow, { placeholder: "eyebrow" });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function card(s, x, y, w, h, fill = C.background2) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill, width: 0 }, objectName: oid("card") });
}
function txt(s, text, o) {
  s.addText(text, { isTextBox: true, margin: 0, fontFace: "Calibri", color: C.text1, valign: "top", objectName: oid("text"), ...o });
}
function bullets(s, items, o) {
  const arr = items.map((it, i) => ({ text: it, options: { bullet: { indent: 14 }, breakLine: i < items.length - 1 } }));
  s.addText(arr, { isTextBox: true, margin: 0, fontFace: "Calibri", fontSize: 15, color: C.text1, valign: "top", paraSpaceAfter: 6, objectName: oid("list"), ...o });
}
function stat(s, x, y, w, big, label, color = C.text2) {
  txt(s, big, { x, y, w, h: 0.9, fontFace: "Cambria", fontSize: 40, bold: true, color });
  txt(s, label, { x, y: y + 0.95, w, h: 0.9, fontSize: 14, color: C.text1 });
}
function fig(s, key, x, y, maxW, maxH) {
  const [pw, ph] = SIZE[key];
  let w = maxW, h = (maxW * ph) / pw;
  if (h > maxH) { h = maxH; w = (maxH * pw) / ph; }
  s.addImage({ path: path.join(FIG, FIGS[key]), x: x + (maxW - w) / 2, y, w, h, objectName: oid("figure"),
               altText: key });
  return { w, h };
}
const TH = (t) => ({ text: t, options: { bold: true, color: HEX.white, fill: { color: HEX.ink }, fontSize: 13 } });
function table(s, rows, o) {
  s.addTable(rows, { fontFace: "Calibri", fontSize: 13, color: HEX.text, border: { type: "solid", pt: 0.75, color: HEX.line },
                     valign: "middle", margin: [3, 6, 3, 6], objectName: oid("table"), ...o });
}
const chartBase = (extra) => ({
  catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 12, dataLabelColor: HEX.text,
  valGridLine: { color: HEX.line, size: 0.75 }, catGridLine: { style: "none" }, showValue: true, dataLabelPosition: "outEnd",
  titleFontFace: "+mn-lt", titleFontSize: 14, titleColor: HEX.text, legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: HEX.text,
  ...extra,
});

// ================= SLIDES =================
// 1. Title
pres.addSection({ title: L("Opening", "Mở đầu") });
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: L("Opening", "Mở đầu") });
  s.addText("ViVQA-VLM", { placeholder: "title" });
  s.addText([
    { text: "Efficient Vietnamese Visual Question Answering using Vision-Language Models", options: { fontSize: 20, breakLine: true } },
    { text: " ", options: { fontSize: 10, breakLine: true } },
    { text: L("Review 2  ·  Baseline reproduction and first PEFT results", "Review 2  ·  Tái lập baseline và kết quả PEFT đầu tiên"), options: { fontSize: 18, bold: true, color: HEX.amber, breakLine: true } },
    { text: " ", options: { fontSize: 18, breakLine: true } },
    { text: L("Supervisors:  ", "GVHD:  "), options: { bold: true, fontSize: 15 } },
    { text: "Đặng Ngọc Minh Đức · Phạm Minh Trí", options: { fontSize: 15, breakLine: true } },
    { text: L("Team:  ", "Nhóm:  "), options: { bold: true, fontSize: 15 } },
    { text: L("Phạm Lê Huy Hoàng (Leader) · Văn Quang Trường · Trần Quốc Huân · Hồ Đắc Toàn", "Phạm Lê Huy Hoàng (Trưởng nhóm) · Văn Quang Trường · Trần Quốc Huân · Hồ Đắc Toàn"), options: { fontSize: 15, breakLine: true } },
    { text: " ", options: { fontSize: 15, breakLine: true } },
    { text: L("Group GFA26AI35  ·  FPT University  ·  October 2026", "Nhóm GFA26AI35  ·  Đại học FPT  ·  Tháng 10/2026"), options: { fontSize: 13, color: "A8C4E0" } },
  ], { placeholder: "body" });
  s.addNotes("Chào hội đồng. Nhóm trình bày Review 2: tái lập baseline Qwen2.5-VL-3B và kết quả cải tiến đầu tiên bằng QLoRA. Toàn bộ số liệu đều có log và có thể chạy lại.");
}

// 2. Summary
pres.addSection({ title: L("Summary", "Tóm tắt") });
{
  const s = content(L("Summary", "Tóm tắt"), L("SUMMARY", "TÓM TẮT"), L("Three results of this milestone", "Ba kết quả chính của mốc này"),
    "Ba con số cần nhớ. Một: pipeline đánh giá của nhóm tái lập đúng số công bố của Qwen2.5-VL-3B trên TextVQA (79,6 so với 79,3), nên các số tiếng Việt phía sau đáng tin. Hai: QLoRA chỉ train 0,2% tham số nhưng tăng EM rất mạnh trên cả hai bộ dữ liệu, có kiểm định thống kê. Ba: trên toàn bộ test ViTextVQA, B2 vượt SOTA của paper ViTextBLIP-2.");
  const cw = (CW - 2 * 0.4) / 3, y = 1.75, h = 2.9;
  const items = [
    ["79.6", L("Pipeline check — TextVQA val, Qwen2.5-VL-3B; the official report gives 79.3", "Kiểm chứng pipeline — TextVQA val, Qwen2.5-VL-3B; báo cáo chính thức ghi 79,3"), C.accent1],
    ["+21.1 EM", L("QLoRA over zero-shot on the full ViTextVQA test (+51.7 EM on ViVQA); McNemar p ≈ 0", "QLoRA so với zero-shot trên full test ViTextVQA (+51,7 EM trên ViVQA); McNemar p ≈ 0"), C.accent2],
    ["50.3 vs 25.5", L("EM on the full ViTextVQA test vs the published SOTA ViTextBLIP-2", "EM trên full test ViTextVQA so với SOTA công bố ViTextBLIP-2"), C.text2],
  ];
  items.forEach(([big, label, col], i) => {
    const x = M + i * (cw + 0.4);
    card(s, x, y, cw, h);
    stat(s, x + 0.35, y + 0.5, cw - 0.7, FMT(big), label, col);
  });
  card(s, M, 5.0, CW, 1.05, C.accent4);
  txt(s, L("Efficient by design: QLoRA trains 7.37 M parameters (0.196%), a 29.5 MB adapter, on a single NVIDIA L4 GPU — no full fine-tuning.",
           "Hiệu quả theo thiết kế: QLoRA chỉ train 7,37 triệu tham số (0,196%), adapter 29,5 MB, trên một GPU NVIDIA L4 — không full fine-tuning."),
      { x: M + 0.35, y: 5.0, w: CW - 0.7, h: 1.05, fontSize: 16, valign: "middle", color: C.text2 });
}

// 3. Feedback
pres.addSection({ title: L("Feedback", "Góp ý") });
{
  const s = content(L("Feedback", "Góp ý"), L("PART 01 · RESPONSE TO FEEDBACK", "PHẦN 01 · XỬ LÝ GÓP Ý"), L("Advisor feedback (07/10) and what we did", "Góp ý của GVHD (07/10) và việc nhóm đã làm"),
    "Sau buổi họp ngày 07/10, thầy Trí yêu cầu năm việc trước khi bàn cải tiến. Bảng này cho thấy mỗi yêu cầu được xử lý ở slide nào. Các góp ý của reviewer Review 1 đã được sửa trong bản báo cáo Review 1 và ghi trong checklist.");
  const rows = [
    [TH(L("Feedback", "Góp ý")), TH(L("Action", "Việc đã làm")), TH(L("Slide", "Slide"))],
    [L("No baseline architecture diagram", "Chưa có sơ đồ kiến trúc baseline"), L("Component-level diagram of Qwen2.5-VL-3B with LoRA placement (Fig. 7A)", "Sơ đồ từng thành phần của Qwen2.5-VL-3B và vị trí gắn LoRA (Hình 7A)"), "5"],
    [L("Explain the dataset and its noise", "Giải thích dataset và nhiễu"), L("ViVQA build pipeline + noise points N1–N4 (Fig. 6A); train–test leakage check; 400-sample manual audit", "Quy trình tạo ViVQA + điểm nhiễu N1–N4 (Hình 6A); kiểm tra rò rỉ train–test; duyệt tay 400 mẫu"), "4, 14"],
    [L("Re-run the baseline with a stated evaluation method", "Chạy lại baseline với phương pháp đánh giá rõ ràng"), L("Frozen protocol (Fig. 7B); pipeline validated on a published number (R0); zero-shot B1 on both datasets", "Giao thức cố định (Hình 7B); kiểm chứng pipeline bằng số đã công bố (R0); B1 zero-shot trên 2 dataset"), "6–8"],
    [L("Know the SOTA", "Nắm SOTA"), L("ViTextBLIP-2 compared on the full test; Table 7 LMM row checked with Qwen2-VL-7B (R1)", "So với ViTextBLIP-2 trên full test; kiểm dòng LMM ở Table 7 bằng Qwen2-VL-7B (R1)"), "9, 12"],
    [L("Run experiments before discussing improvements", "Chạy thí nghiệm rồi mới bàn cải tiến"), L("QLoRA pilot (B2) + three controls: leakage strata, few-shot prompt, backbone", "QLoRA sơ bộ (B2) + ba đối chứng: phân tầng rò rỉ, few-shot prompt, backbone"), "10–11"],
  ].map((r, i) => i === 0 ? r : r.map((c) => ({ text: c, options: { fontSize: 14 } })));
  table(s, rows, { x: M, y: 1.65, w: CW, colW: [3.6, 7.3, 1.23], rowH: 0.62 });
}

// 4. Dataset & noise
pres.addSection({ title: L("Data & baseline", "Dữ liệu & baseline") });
{
  const s = content(L("Data & baseline", "Dữ liệu & baseline"), L("PART 02 · DATASET UNDERSTANDING", "PHẦN 02 · HIỂU DỮ LIỆU"), L("ViVQA is machine-translated — noise is built in", "ViVQA được dịch máy — nhiễu nằm sẵn trong dữ liệu"),
    "ViVQA được tạo từ COCO-QA, câu hỏi sinh tự động rồi dịch bằng Google Translate. Vì vậy có các điểm nhiễu N1 đến N4, ví dụ purple bị dịch thành tía. Nhóm cũng kiểm tra rò rỉ: ở ViVQA, 46% ảnh test xuất hiện trong train; ở ViTextVQA thì không có ảnh trùng nào, nên đây là phép thử tổng quát hoá sạch hơn.");
  const f = fig(s, "a6", M, 1.5, CW, 4.2);
  const y = 1.5 + f.h + 0.3, cw = (CW - 0.4) / 2;
  card(s, M, y, cw, 6.85 - y);
  txt(s, [
    { text: "ViVQA  ", options: { bold: true, color: HEX.ink } },
    { text: L("1,274 / 2,789 test images also appear in train; 99.4% of test answers belong to the 333 train answers (closed vocabulary)",
              "1.274 / 2.789 ảnh test cũng có trong train; 99,4% đáp án test thuộc 333 đáp án của train (từ vựng đóng)") },
  ], { x: M + 0.3, y: y + 0.15, w: cw - 0.6, h: 6.85 - y - 0.3, fontSize: 14, valign: "middle" });
  card(s, M + cw + 0.4, y, cw, 6.85 - y, C.accent4);
  txt(s, [
    { text: "ViTextVQA  ", options: { bold: true, color: HEX.ink } },
    { text: L("0 images shared between train/dev and test; 46% of test answers occur in train → a cleaner test of generalization",
              "0 ảnh trùng giữa train/dev và test; 46% đáp án test có trong train → phép thử tổng quát hoá sạch hơn") },
  ], { x: M + cw + 0.7, y: y + 0.15, w: cw - 0.6, h: 6.85 - y - 0.3, fontSize: 14, valign: "middle" });
}

// 5. Architecture
{
  const s = content(L("Data & baseline", "Dữ liệu & baseline"), L("PART 03 · BASELINE ARCHITECTURE", "PHẦN 03 · KIẾN TRÚC BASELINE"), "Qwen2.5-VL-3B-Instruct",
    "Đây là sơ đồ baseline mà thầy yêu cầu. Ảnh đi qua vision encoder 32 khối, được gộp còn 256 đến 512 token, rồi ghép với câu hỏi đưa vào decoder 36 lớp. Toàn bộ trọng số gốc đóng băng và lượng tử 4-bit. Phần duy nhất được train là LoRA ở 4 phép chiếu q, k, v, o của self-attention — ô màu cam.");
  const f = fig(s, "a7", M, 1.45, 8.9, 5.45);
  const x = M + f.w + 0.4, w = CW - f.w - 0.4;
  card(s, x, 1.45, w, 2.45, C.accent5);
  txt(s, L("Trained (QLoRA)", "Được train (QLoRA)"), { x: x + 0.25, y: 1.6, w: w - 0.5, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  bullets(s, [L("LoRA r = 16, α = 32 on q, k, v, o", "LoRA r = 16, α = 32 trên q, k, v, o"),
              L("7,372,800 params = 0.196%", "7.372.800 tham số = 0,196%"),
              L("Adapter 29.5 MB", "Adapter 29,5 MB")], { x: x + 0.25, y: 2.05, w: w - 0.5, h: 1.75, fontSize: 14 });
  card(s, x, 4.1, w, 2.75);
  txt(s, L("Frozen", "Đóng băng"), { x: x + 0.25, y: 4.25, w: w - 0.5, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  bullets(s, [L("ViT (32 blocks) + patch merger", "ViT (32 khối) + patch merger"),
              L("LLM decoder, 36 layers, 4-bit NF4", "LLM decoder 36 lớp, 4-bit NF4"),
              L("256–512 visual tokens per image", "256–512 token ảnh mỗi ảnh")], { x: x + 0.25, y: 4.7, w: w - 0.5, h: 2.0, fontSize: 14 });
}

// 6. Evaluation protocol
{
  const s = content(L("Data & baseline", "Dữ liệu & baseline"), L("PART 03 · EVALUATION PROTOCOL", "PHẦN 03 · GIAO THỨC ĐÁNH GIÁ"), L("Every run follows one frozen protocol", "Mọi run theo một giao thức cố định"),
    "Mọi run dùng chung một giao thức: split và seed cố định, cùng prompt, cấu hình khoá trong file config, cùng hàm chuẩn hoá và metric, và log đầy đủ commit, GPU, thời gian. So sánh hai run luôn dùng kiểm định ghép cặp McNemar cho EM và bootstrap cho ANLS. Luật kết luận được chốt trước khi chạy để tránh chọn kết quả có lợi.");
  const f = fig(s, "b7", M, 1.45, CW, 2.4);
  const y = 1.45 + f.h + 0.35, cw = (CW - 2 * 0.4) / 3, h = 6.85 - y;
  const cards = [
    [L("Frozen configuration", "Cấu hình cố định"), [L("bf16 compute, 4-bit NF4 + double quant", "Tính bf16, 4-bit NF4 + double quant"), L("512 visual tokens max", "Tối đa 512 token ảnh"), L("Greedy decoding, 32 new tokens", "Greedy, 32 token sinh"), L("Seed 42 · NVIDIA L4", "Seed 42 · NVIDIA L4")]],
    [L("Metrics", "Metric"), [L("Primary: EM, VQA-Acc, ANLS", "Chính: EM, VQA-Acc, ANLS"), L("Secondary: token-F1 (to compare with the paper)", "Phụ: token-F1 (để so với paper)"), L("Symmetric Vietnamese normalization, tone marks kept", "Chuẩn hoá tiếng Việt đối xứng, giữ dấu thanh")]],
    [L("Statistics", "Thống kê"), [L("Paired McNemar exact test (EM)", "Kiểm định McNemar chính xác, ghép cặp (EM)"), L("Paired bootstrap CI95 (ANLS)", "CI95 bootstrap ghép cặp (ANLS)"), L("Decision rules fixed before each run", "Luật kết luận chốt trước mỗi run")]],
  ];
  cards.forEach(([t, items], i) => {
    const x = M + i * (cw + 0.4);
    card(s, x, y, cw, h);
    txt(s, t, { x: x + 0.25, y: y + 0.15, w: cw - 0.5, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
    bullets(s, items, { x: x + 0.25, y: y + 0.6, w: cw - 0.5, h: h - 0.7, fontSize: 14 });
  });
}

// 7. Pipeline validation
pres.addSection({ title: L("Results", "Kết quả") });
{
  const s = content(L("Results", "Kết quả"), L("PART 04 · PIPELINE VALIDATION (R0)", "PHẦN 04 · KIỂM CHỨNG PIPELINE (R0)"), L("Our pipeline reproduces the published number", "Pipeline của nhóm tái lập được số đã công bố"),
    "Vấn đề: chưa paper nào công bố Qwen2.5-VL-3B trên ViVQA hay ViTextVQA, nên không thể so trực tiếp. Nhóm kiểm chứng pipeline bằng một con số đã công bố: TextVQA val trong báo cáo Qwen2.5-VL là 79,3. Nhóm chạy được 79,6, nằm trong ngưỡng ±2 điểm đã chốt. Cấu hình tiết kiệm của nhóm (4-bit, 512 token) mất 4,4 điểm nhưng giảm VRAM 4,3 lần.");
  s.addChart(pres.charts.BAR, [{ name: "VQA-Acc", labels: [L("Qwen2.5-VL report", "Báo cáo Qwen2.5-VL"), L("R0 · bf16, full res.", "R0 · bf16, đủ độ phân giải"), L("R0b · 4-bit, 512 tok.", "R0b · 4-bit, 512 token")], values: [79.3, 79.6, 75.21] }],
    chartBase({ x: M, y: 1.5, w: 7.2, h: 5.3, barDir: "col", chartColors: [HEX.teal], valAxisMinVal: 60, valAxisMaxVal: 85, showLegend: false,
                showTitle: true, title: L("TextVQA val (5,000 questions), VQA accuracy", "TextVQA val (5.000 câu), VQA accuracy"), dataLabelFormatCode: "0.0", objectName: oid("chart") }));
  const x = M + 7.6, w = CW - 7.6;
  card(s, x, 1.5, w, 2.45, C.accent4);
  txt(s, L("Rule fixed before the run", "Luật chốt trước khi chạy"), { x: x + 0.3, y: 1.65, w: w - 0.6, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  txt(s, L("Reproduced if |ours − 79.3| ≤ 2.0 on the full val set → 79.6 (+0.3): reproduced.", "Tái lập được nếu |nhóm − 79,3| ≤ 2,0 trên toàn bộ val → 79,6 (+0,3): đạt."),
      { x: x + 0.3, y: 2.1, w: w - 0.6, h: 1.7, fontSize: 15 });
  card(s, x, 4.15, w, 2.65);
  txt(s, L("Cost of the efficient setup", "Cái giá của cấu hình tiết kiệm"), { x: x + 0.3, y: 4.3, w: w - 0.6, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  bullets(s, [L("−4.4 points accuracy", "−4,4 điểm accuracy"), L("VRAM 10.9 → 2.5 GB (4.3× less)", "VRAM 10,9 → 2,5 GB (ít hơn 4,3 lần)"),
              L("H2 (≥ 97% kept): not met — 94.5%", "H2 (giữ ≥ 97%): chưa đạt — 94,5%")], { x: x + 0.3, y: 4.75, w: w - 0.6, h: 1.9, fontSize: 15 });
}

// 8. Zero-shot B1
{
  const s = content(L("Results", "Kết quả"), L("PART 04 · ZERO-SHOT BASELINE (B1)", "PHẦN 04 · BASELINE ZERO-SHOT (B1)"), L("Zero-shot fails mostly on answer format", "Zero-shot sai chủ yếu ở định dạng câu trả lời"),
    "B1 là baseline zero-shot chính thức. Lỗi lớn nhất là định dạng: model trả lời dài hơn đáp án chuẩn, ví dụ 'Lịch tết năm 2021' trong khi đáp án là '2021'. Trên ViTextVQA có 19% câu model đọc ra đúng đáp án nhưng EM vẫn tính sai. Theo loại câu hỏi, ViVQA rất yếu ở vị trí và đồ vật.");
  const rows = [
    [TH(L("Test set", "Tập test")), TH("EM"), TH("ANLS"), TH("F1")],
    [L("ViVQA (3,001)", "ViVQA (3.001)"), "22.1", "28.6", "38.1"],
    [L("ViTextVQA subset (2,000)", "ViTextVQA tập con (2.000)"), "28.8", "47.5", "53.8"],
    [L("ViTextVQA full (10,028)", "ViTextVQA đầy đủ (10.028)"), "29.1", "47.6", "53.9"],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: FMT(c), options: { fontSize: 14, align: j ? "center" : "left" } })));
  table(s, rows, { x: M, y: 1.55, w: 6.6, colW: [3.3, 1.1, 1.1, 1.1], rowH: 0.5 });
  card(s, M, 3.85, 6.6, 3.0);
  txt(s, L("ViVQA EM by question type", "ViVQA EM theo loại câu hỏi"), { x: M + 0.3, y: 4.0, w: 6, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  s.addChart(pres.charts.BAR, [{ name: "EM", labels: [L("number", "số lượng"), L("color", "màu"), L("object", "đồ vật"), L("location", "vị trí")], values: [69.1, 44.8, 5.7, 0.7] }],
    chartBase({ x: M + 0.2, y: 4.4, w: 6.2, h: 2.35, barDir: "bar", chartColors: [HEX.amber], showLegend: false, valAxisMaxVal: 80, valAxisHidden: true, valGridLine: { style: "none" }, dataLabelFormatCode: "0.0", objectName: oid("chart") }));
  const x = M + 7.0, w = CW - 7.0;
  card(s, x, 1.55, w, 5.3, C.accent5);
  txt(s, L("Main failure: verbose answers", "Lỗi chính: trả lời dài dòng"), { x: x + 0.3, y: 1.7, w: w - 0.6, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  bullets(s, [
    L("Words per answer: 2.9 vs gold 1.6 (ViVQA); 5.5 vs 3.7 (ViTextVQA)", "Số từ mỗi câu trả lời: 2,9 so với gold 1,6 (ViVQA); 5,5 so với 3,7 (ViTextVQA)"),
    L("19% of ViTextVQA answers contain the gold answer but fail EM", "19% câu trả lời ViTextVQA chứa đáp án gold nhưng vẫn trượt EM"),
    L("“Lịch tết năm 2021” vs gold “2021”", "“Lịch tết năm 2021” so với gold “2021”"),
    L("Also: code-switching (“truck”), hallucination, noisy gold (“tía”)", "Ngoài ra: chen tiếng Anh (“truck”), hallucination, gold nhiễu (“tía”)"),
  ], { x: x + 0.3, y: 2.2, w: w - 0.6, h: 4.5, fontSize: 15 });
}

// 9. Backbone ablation R1
{
  const s = content(L("Results", "Kết quả"), L("PART 04 · BACKBONE ABLATION (R1)", "PHẦN 04 · ABLATION BACKBONE (R1)"), L("A larger model is not better here", "Model lớn hơn không tốt hơn ở bài này"),
    "Table 7 của paper ViTextVQA có dòng QwenVL-7b đạt EM 28 trên 100 câu. Nhóm chạy Qwen2-VL-7B trong đúng điều kiện của baseline. Kết quả 7B kém 3B 12,6 điểm EM, p rất nhỏ, chậm hơn và tốn VRAM hơn. Nguyên nhân chính là 7B trả lời dài, trung bình 8,2 từ. Table 7 chỉ có 100 câu nên sai số khoảng ±9 điểm, không đủ để chọn giữa 3B và 7B. Vì vậy nhóm giữ Qwen2.5-VL-3B.");
  const fmt = FMT;
  const rows = [
    [TH(L("Run (ViTextVQA, 2,000 q.)", "Run (ViTextVQA, 2.000 câu)")), TH("EM"), TH("F1"), TH(L("words", "số từ")), TH(L("s / q", "s / câu")), TH("VRAM GB")],
    [L("B1 · Qwen2.5-VL-3B · 4-bit · 512 tok.", "B1 · Qwen2.5-VL-3B · 4-bit · 512 token"), "28.8", "53.8", "5.5", "0.86", "≈ 2.5"],
    [L("R1b · Qwen2-VL-7B · 4-bit · 512 tok.", "R1b · Qwen2-VL-7B · 4-bit · 512 token"), "16.2", "41.3", "8.0", "1.02", "6.2"],
    [L("R1 · Qwen2-VL-7B · bf16 · full res.", "R1 · Qwen2-VL-7B · bf16 · đủ độ phân giải"), "16.7", "43.5", "8.2", "2.85", "20.3"],
    [L("Paper Table 7 · “QwenVL-7b” (100 q.)", "Paper Table 7 · “QwenVL-7b” (100 câu)"), "28.0", "53.8", "–", "–", "–"],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: fmt(c), options: { fontSize: 14, align: j ? "center" : "left", bold: i === 1 } })));
  table(s, rows, { x: M, y: 1.55, w: CW, colW: [5.03, 1.2, 1.2, 1.4, 1.4, 1.9], rowH: 0.52 });
  const y = 4.5, cw = (CW - 2 * 0.4) / 3, h = 1.95;
  [[L("7B < 3B under identical conditions", "7B < 3B trong cùng điều kiện"), L("−12.6 EM, McNemar p ≈ 10⁻³³; higher resolution does not help (R1 = R1b, p = 0.45)", "−12,6 EM, McNemar p ≈ 10⁻³³; tăng độ phân giải không giúp (R1 = R1b, p = 0,45)")],
   [L("Why: instruction following", "Vì sao: tuân thủ chỉ dẫn"), L("7B writes 8.2 words per answer vs gold 3.7 — EM fails even when the text is read correctly", "7B trả lời 8,2 từ so với gold 3,7 — EM trượt dù đọc đúng chữ")],
   [L("Table 7 cannot decide", "Table 7 không đủ để quyết định"), L("n = 100 → ±9 EM; B1 is already consistent with it. Decision: keep Qwen2.5-VL-3B", "n = 100 → ±9 EM; B1 đã nhất quán với nó. Quyết định: giữ Qwen2.5-VL-3B")]]
    .forEach(([t, b], i) => {
      const x = M + i * (cw + 0.4);
      card(s, x, y, cw, h, i === 2 ? C.accent4 : C.background2);
      txt(s, t, { x: x + 0.25, y: y + 0.15, w: cw - 0.5, h: 0.45, fontSize: 16, bold: true, color: C.text2 });
      txt(s, b, { x: x + 0.25, y: y + 0.65, w: cw - 0.5, h: h - 0.8, fontSize: 14 });
    });
}

// 10. QLoRA B2
pres.addSection({ title: L("Improvement", "Cải tiến") });
{
  const s = content(L("Improvement", "Cải tiến"), L("PART 05 · QLoRA (B2)", "PHẦN 05 · QLoRA (B2)"), L("QLoRA, one epoch: large and significant gains", "QLoRA một epoch: tăng lớn và có ý nghĩa thống kê"),
    "B2 khác B1 đúng một biến: có adapter LoRA. Chỉ một epoch, EM ViVQA tăng từ 22 lên 74, ViTextVQA từ 29 lên 50 trên toàn bộ test. Cả hai đạt luật chốt trước: EM tăng và McNemar p nhỏ hơn 0,05. Chi phí rất thấp: 3 giờ train cho ViVQA, 12 giờ cho ViTextVQA, VRAM dưới 7 GB.");
  s.addChart(pres.charts.BAR, [
    { name: L("B1 zero-shot", "B1 zero-shot"), labels: ["ViVQA", L("ViTextVQA (2,000)", "ViTextVQA (2.000)"), L("ViTextVQA full", "ViTextVQA đầy đủ")], values: [22.09, 28.8, 29.12] },
    { name: L("B2 QLoRA 1 epoch", "B2 QLoRA 1 epoch"), labels: ["ViVQA", L("ViTextVQA (2,000)", "ViTextVQA (2.000)"), L("ViTextVQA full", "ViTextVQA đầy đủ")], values: [73.78, 49.7, 50.26] },
  ], chartBase({ x: M, y: 1.5, w: 7.4, h: 5.35, barDir: "col", barGrouping: "clustered", chartColors: [HEX.muted, HEX.teal], showLegend: true, legendPos: "b",
                 valAxisMaxVal: 90, valAxisMinVal: 0, showTitle: true, title: L("Exact Match (%)", "Exact Match (%)"), dataLabelFormatCode: "0.0", objectName: oid("chart") }));
  const x = M + 7.8, w = CW - 7.8;
  const fmt = FMT;
  const rows = [
    [TH(L("Cost", "Chi phí")), TH("ViVQA"), TH("ViTextVQA")],
    [L("Train samples", "Mẫu train"), LANG === "en" ? "10,799" : "10.799", LANG === "en" ? "35,159" : "35.159"],
    [L("Train time (L4)", "Thời gian train (L4)"), "3.0 h", "12.0 h"],
    [L("Peak VRAM", "VRAM đỉnh"), "6.6 GB", "6.9 GB"],
    [L("Inference", "Suy luận"), L("0.57 s / q", "0,57 s / câu"), L("1.0 s / q", "1,0 s / câu")],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: fmt(c), options: { fontSize: 14, align: j ? "center" : "left" } })));
  table(s, rows, { x, y: 1.5, w, colW: [2.0, 1.36, 1.37], rowH: 0.5 });
  card(s, x, 4.25, w, 2.6, C.accent4);
  txt(s, L("Rule fixed before the run", "Luật chốt trước khi chạy"), { x: x + 0.25, y: 4.4, w: w - 0.5, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  txt(s, L("Improvement if EM rises and McNemar p < 0.05 → met on both datasets (ViTextVQA full: 2,543 vs 423 questions only one model gets right).",
           "Cải thiện nếu EM tăng và McNemar p < 0,05 → đạt trên cả hai dataset (ViTextVQA đầy đủ: 2.543 so với 423 câu chỉ một model đúng)."),
      { x: x + 0.25, y: 4.85, w: w - 0.5, h: 1.9, fontSize: 14 });
}

// 11. Is the gain real?
{
  const s = content(L("Improvement", "Cải tiến"), L("PART 05 · CONTROLS", "PHẦN 05 · ĐỐI CHỨNG"), L("The gain is real, not leakage and not just format", "Mức tăng là thật, không do rò rỉ, không chỉ do định dạng"),
    "Ba phép đối chứng. Một: ở ViVQA, trên phần test sạch chưa từng xuất hiện trong train, EM vẫn tăng từ 25 lên 69. Hai: ViTextVQA không có ảnh trùng mà vẫn tăng 21 điểm. Ba: đưa 5 ví dụ đáp án vào prompt (few-shot) không giúp gì, thậm chí giảm ở ViVQA. Vậy QLoRA học được nhiều hơn là cách trả lời ngắn.");
  const cw = 3.9;
  card(s, M, 1.5, cw, 2.55);
  txt(s, L("ViVQA leakage strata", "Phân tầng rò rỉ ViVQA"), { x: M + 0.25, y: 1.62, w: cw - 0.5, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  txt(s, [
    { text: L("Clean subset (n = 1,219): ", "Tập sạch (n = 1.219): "), options: { breakLine: false } },
    { text: L("EM 25.0 → 69.1", "EM 25,0 → 69,1"), options: { bold: true, color: HEX.teal, breakLine: true } },
    { text: L("Answer-prior baseline: 14.4", "Mốc đoán theo phân bố đáp án: 14,4") },
  ], { x: M + 0.25, y: 2.1, w: cw - 0.5, h: 1.85, fontSize: 14 });
  card(s, M, 4.3, cw, 2.55);
  txt(s, L("ViTextVQA: no shared images", "ViTextVQA: không trùng ảnh"), { x: M + 0.25, y: 4.42, w: cw - 0.5, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  txt(s, [
    { text: L("0 train–test image overlap, ", "0 ảnh trùng train–test, "), options: {} },
    { text: L("yet +21.1 EM on the full test", "vẫn +21,1 EM trên full test"), options: { bold: true, color: HEX.teal } },
  ], { x: M + 0.25, y: 4.9, w: cw - 0.5, h: 1.85, fontSize: 14 });
  const x = M + cw + 0.4, w = CW - cw - 0.4;
  card(s, x, 1.5, w, 5.35);
  txt(s, L("Few-shot control: 5 text examples in the prompt (FS5)", "Đối chứng few-shot: 5 ví dụ chữ trong prompt (FS5)"), { x: x + 0.3, y: 1.62, w: w - 0.6, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  s.addChart(pres.charts.BAR, [
    { name: "B1", labels: ["ViVQA", L("ViTextVQA (2,000)", "ViTextVQA (2.000)")], values: [22.09, 28.8] },
    { name: "FS5", labels: ["ViVQA", L("ViTextVQA (2,000)", "ViTextVQA (2.000)")], values: [20.03, 27.6] },
    { name: "B2", labels: ["ViVQA", L("ViTextVQA (2,000)", "ViTextVQA (2.000)")], values: [73.78, 49.7] },
  ], chartBase({ x: x + 0.15, y: 2.05, w: w - 0.3, h: 3.55, barDir: "col", barGrouping: "clustered", chartColors: [HEX.muted, HEX.amber, HEX.teal], showLegend: true, legendPos: "r",
                 valAxisMaxVal: 90, valAxisMinVal: 0, dataLabelFormatCode: "0.0", objectName: oid("chart") }));
  txt(s, L("Few-shot shortens answers but loses content (EM −2.1 on ViVQA, p = 0.004; −1.2 on ViTextVQA, n.s.). B2 beats FS5 on both (p ≈ 0) → QLoRA learns more than answer format.",
           "Few-shot làm câu trả lời ngắn lại nhưng mất nội dung (EM −2,1 ở ViVQA, p = 0,004; −1,2 ở ViTextVQA, không ý nghĩa). B2 vượt FS5 ở cả hai (p ≈ 0) → QLoRA học nhiều hơn định dạng."),
      { x: x + 0.3, y: 5.65, w: w - 0.6, h: 1.1, fontSize: 14 });
}

// 12. vs SOTA
{
  const s = content(L("Improvement", "Cải tiến"), L("PART 05 · COMPARISON WITH SOTA", "PHẦN 05 · SO VỚI SOTA"), L("Full ViTextVQA test: above the published SOTA", "Full test ViTextVQA: cao hơn SOTA đã công bố"),
    "Trên toàn bộ 10.028 câu test, B2 đạt EM 50,3 và F1 71,2, cao hơn ViTextBLIP-2 là SOTA của paper, với cận dưới khoảng tin cậy vẫn cao hơn số công bố. B1 zero-shot chỉ cao hơn về EM, còn F1 thì ngang. Chạy lại trên 2.000 câu chung cho kết quả trùng 100%. Giới hạn: paper không công bố cách chuẩn hoá đáp án.");
  const fmt = FMT;
  const rows = [
    [TH(L("Model (full test, 10,028 q.)", "Model (full test, 10.028 câu)")), TH(L("EM [CI95]", "EM [CI95]")), TH(L("F1 [CI95]", "F1 [CI95]")), TH(L("vs SOTA (rule: CI95 lower bound > published)", "so với SOTA (luật: cận dưới CI95 > số công bố)"))],
    [L("ViTextBLIP-2 · paper Table 6 (trained)", "ViTextBLIP-2 · paper Table 6 (có train)"), "25.48", "53.95", "–"],
    [L("B1 · Qwen2.5-VL-3B zero-shot", "B1 · Qwen2.5-VL-3B zero-shot"), "29.12 [28.23, 29.99]", "53.89 [53.11, 54.66]", L("higher on EM; F1 indistinguishable", "cao hơn về EM; F1 ngang")],
    [L("B2 · + QLoRA, 1 epoch", "B2 · + QLoRA, 1 epoch"), "50.26 [49.27, 51.23]", "71.16 [70.42, 71.89]", L("higher on EM and F1", "cao hơn cả EM và F1")],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: fmt(c).replace(/\[(\d+,\d+), (\d)/, "[$1; $2"), options: { fontSize: 14, align: j && j < 3 ? "center" : "left", bold: i === 3, color: i === 3 ? HEX.ink : HEX.text, fill: i === 3 ? { color: HEX.tealt } : undefined } })));
  table(s, rows, { x: M, y: 1.55, w: CW, colW: [4.0, 2.55, 2.55, 3.03], rowH: 0.62 });
  const y = 4.3, cw = (CW - 2 * 0.4) / 3, h = 1.95;
  [[L("Reproducible", "Tái lập được"), L("Full-test predictions match the earlier subset runs on all 2,000 shared questions (100%).", "Dự đoán full test trùng 100% với các run tập con trên 2.000 câu chung.")],
   [L("Fair in kind", "Công bằng về loại"), L("Both B2 and ViTextBLIP-2 are trained on the ViTextVQA train split.", "Cả B2 và ViTextBLIP-2 đều được train trên tập train ViTextVQA.")],
   [L("Limitation", "Giới hạn"), L("The paper does not publish its answer normalization; the comparison assumes equivalent scoring.", "Paper không công bố cách chuẩn hoá đáp án; phép so giả định cách chấm tương đương.")]]
    .forEach(([t, b], i) => {
      const x = M + i * (cw + 0.4);
      card(s, x, y, cw, h, i === 2 ? C.accent5 : C.background2);
      txt(s, t, { x: x + 0.25, y: y + 0.15, w: cw - 0.5, h: 0.45, fontSize: 16, bold: true, color: C.text2 });
      txt(s, b, { x: x + 0.25, y: y + 0.65, w: cw - 0.5, h: h - 0.8, fontSize: 14 });
    });
}

// 13. Hypotheses scoreboard
pres.addSection({ title: L("Analysis & plan", "Phân tích & kế hoạch") });
{
  const s = content(L("Analysis & plan", "Phân tích & kế hoạch"), L("PART 06 · HYPOTHESES", "PHẦN 06 · GIẢ THUYẾT"), L("What the evidence says so far", "Bằng chứng đến thời điểm này"),
    "Đối chiếu với các giả thuyết đã nêu ở Review 1. H1 được ủng hộ mạnh. H2 chỉ đạt một nửa: VRAM giảm nhiều nhưng độ chính xác giữ 94,5%, dưới mức 97%. H7 chưa được ủng hộ với dạng few-shot đã thử. H8 về hallucination đang chờ kết quả duyệt tay. H3 đến H6 thuộc kế hoạch W8 đến W12.");
  const OK = L("Supported", "Ủng hộ"), PART = L("Partly", "Một phần"), NO = L("Not supported", "Chưa ủng hộ"), WAIT = L("Pending", "Đang chờ");
  const col = { [OK]: HEX.teal, [PART]: HEX.amber, [NO]: HEX.red, [WAIT]: HEX.muted };
  const rows = [
    [TH(L("Hypothesis (Review 1)", "Giả thuyết (Review 1)")), TH(L("Evidence", "Bằng chứng")), TH(L("Status", "Trạng thái"))],
    [L("H1  QLoRA ≥ +5 EM over zero-shot", "H1  QLoRA ≥ +5 EM so với zero-shot"), L("+51.7 (ViVQA), +21.1 (ViTextVQA full), p ≈ 0", "+51,7 (ViVQA), +21,1 (ViTextVQA đầy đủ), p ≈ 0"), OK],
    [L("H2  4-bit keeps ≥ 97% accuracy, VRAM −40%", "H2  4-bit giữ ≥ 97% độ chính xác, VRAM −40%"), L("Keeps 94.5%; VRAM −77% (R0 vs R0b)", "Giữ 94,5%; VRAM −77% (R0 so với R0b)"), PART],
    [L("H7  Vietnamese format prompt ≥ +3 EM", "H7  Prompt định dạng tiếng Việt ≥ +3 EM"), L("Few-shot variant: −2.1 / −1.2 EM", "Dạng few-shot: −2,1 / −1,2 EM"), NO],
    [L("H8  PEFT does not increase hallucination", "H8  PEFT không làm tăng hallucination"), L("Manual audit of B2 errors in progress", "Đang duyệt tay lỗi của B2"), WAIT],
    [L("H3–H6  LoRA config, shared adapter, OCR prompting", "H3–H6  Cấu hình LoRA, adapter dùng chung, OCR prompting"), L("Planned W8–W12", "Kế hoạch W8–W12"), WAIT],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c, options: { fontSize: 14, bold: j === 2, color: j === 2 ? col[c] : HEX.text } })));
  table(s, rows, { x: M, y: 1.55, w: CW, colW: [5.2, 4.93, 2.0], rowH: 0.62 });
  card(s, M, 5.6, CW, 1.25, C.accent4);
  txt(s, L("RQ1 answered: zero-shot reaches EM 22.1 (ViVQA) and 29.1 (ViTextVQA).   RQ2 answered: QLoRA adds +51.7 and +21.1 EM.",
           "RQ1 đã trả lời: zero-shot đạt EM 22,1 (ViVQA) và 29,1 (ViTextVQA).   RQ2 đã trả lời: QLoRA tăng +51,7 và +21,1 EM."),
      { x: M + 0.35, y: 5.6, w: CW - 0.7, h: 1.25, fontSize: 15, valign: "middle", color: C.text2 });
}

// 14. Error analysis
{
  const s = content(L("Analysis & plan", "Phân tích & kế hoạch"), L("PART 06 · ERROR & HALLUCINATION ANALYSIS", "PHẦN 06 · PHÂN TÍCH LỖI & HALLUCINATION"), L("Where B2 still fails", "B2 còn sai ở đâu"),
    "Phân loại tự động 4.988 câu B2 còn sai trên full test ViTextVQA. 9% chỉ sai dấu thanh, ví dụ thủy và thúy — đây là thách thức riêng của tiếng Việt. 17% chỉ lệch một hai ký tự, thường là đọc chữ hoặc số. Khoảng 27% sai về độ dài: thừa hoặc thiếu chữ so với gold. 47% còn lại sai về nội dung hoặc vùng ảnh, cần duyệt tay để tách hallucination; nhóm đang duyệt tay 400 mẫu.");
  s.addChart(pres.charts.BAR, [{ name: L("% of B2 errors", "% lỗi của B2"),
      labels: [L("other: content / region", "khác: nội dung / vùng ảnh"), L("1–2 characters off", "lệch 1–2 ký tự"), L("answer too long", "trả lời thừa chữ"), L("answer too short", "trả lời thiếu chữ"), L("tone marks only", "chỉ sai dấu thanh")],
      values: [47.4, 16.8, 13.6, 13.2, 9.0] }],
    chartBase({ x: M, y: 1.45, w: 7.0, h: 3.05, barDir: "bar", chartColors: [HEX.teal], showLegend: false, valAxisHidden: true, valGridLine: { style: "none" }, valAxisMaxVal: 60,
                showTitle: true, title: L("Automatic buckets, B2 errors on full ViTextVQA test (n = 4,988)", "Phân loại tự động lỗi B2, full test ViTextVQA (n = 4.988)"), dataLabelFormatCode: "0.0", catAxisOrientation: "maxMin", objectName: oid("chart") }));
  const ex = [
    [L("tone marks", "dấu thanh"), "đặng việt thủy", "đặng việt thúy"],
    [L("1–2 chars", "lệch ký tự"), "372 . 568 . 177", "372 . 566 . 177"],
    [L("too short", "thiếu chữ"), "nhân dân quận 2", "ủy ban nhân dân quận 2"],
  ];
  const rows = [[TH(L("Bucket", "Nhóm")), TH(L("B2 answer", "B2 trả lời")), TH("Gold")]]
    .concat(ex.map((r) => r.map((c) => ({ text: c, options: { fontSize: 14 } }))));
  table(s, rows, { x: M, y: 4.75, w: 7.0, colW: [1.8, 2.45, 2.75], rowH: 0.5 });
  const x = M + 7.4, w = CW - 7.4;
  card(s, x, 1.5, w, 5.35);
  txt(s, L("Manual audit (400 samples, in progress)", "Duyệt tay (400 mẫu, đang làm)"), { x: x + 0.3, y: 1.65, w: w - 0.6, h: 0.75, fontSize: 16, bold: true, color: C.text2 });
  bullets(s, [
    L("200 B2 errors: format, synonym, OCR, tone marks, wrong region / hallucination, code-switching, counting, noisy gold", "200 lỗi của B2: định dạng, đồng nghĩa, OCR, dấu thanh, sai vùng / hallucination, chen tiếng Anh, đếm, gold nhiễu"),
    L("200 train labels: correct / wrong / ambiguous + noise type", "200 nhãn train: đúng / sai / mơ hồ + loại nhiễu"),
    L("Separates hallucination inside the 47% “other” bucket → evidence for H8", "Tách hallucination trong nhóm “khác” 47% → bằng chứng cho H8"),
    L("Deterministic sample (seed 42), images attached", "Chọn mẫu tất định (seed 42), kèm ảnh"),
  ], { x: x + 0.3, y: 2.5, w: w - 0.6, h: 4.2, fontSize: 14 });
}

// 15. Next steps
{
  const s = content(L("Analysis & plan", "Phân tích & kế hoạch"), L("PART 07 · NEXT STEPS", "PHẦN 07 · BƯỚC TIẾP THEO"), L("Plan to Review 3 (week 13)", "Kế hoạch đến Review 3 (tuần 13)"),
    "Từ tuần 8 đến tuần 13: chạy QLoRA chính thức nhiều epoch và chọn epoch trên tập dev, thử làm sạch nhãn nhiễu, OCR-enhanced prompting cho ViTextVQA, ablation cấu hình LoRA, adapter dùng chung cho hai dataset, đo hallucination có hệ thống, rồi dựng prototype web cho Review 3.");
  const steps = [
    ["W8", L("Main QLoRA run: 3 epochs, epoch chosen on dev; label-noise cleaning ablation", "QLoRA chính thức: 3 epoch, chọn epoch trên dev; ablation làm sạch nhãn nhiễu")],
    ["W9–10", L("OCR-enhanced prompting (H6); LoRA rank and target-module ablations (H3, H4)", "OCR-enhanced prompting (H6); ablation rank và module gắn LoRA (H3, H4)")],
    ["W11", L("Shared adapter for both datasets (H5); systematic hallucination evaluation (H8)", "Adapter dùng chung cho 2 dataset (H5); đo hallucination có hệ thống (H8)")],
    ["W12–13", L("Web prototype (upload image, ask in Vietnamese); Review 3", "Prototype web (tải ảnh, hỏi tiếng Việt); Review 3")],
  ];
  const cw = (CW - 3 * 0.35) / 4, y = 1.75, h = 3.4;
  steps.forEach(([wk, t], i) => {
    const x = M + i * (cw + 0.35);
    card(s, x, y, cw, h, i === 0 ? C.accent4 : C.background2);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.3, w: 0.95, h: 0.95, fill: { color: i === 0 ? HEX.teal : HEX.ink }, line: { color: HEX.white, width: 0 }, objectName: oid("dot") });
    txt(s, wk, { x: x + 0.3, y: y + 0.3, w: 0.95, h: 0.95, fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle" });
    txt(s, t, { x: x + 0.3, y: y + 1.5, w: cw - 0.6, h: h - 1.7, fontSize: 15 });
  });
  txt(s, L("All runs keep the frozen protocol, PEFT only, and pre-registered decision rules.", "Mọi run giữ giao thức cố định, chỉ PEFT, và luật kết luận chốt trước."),
      { x: M, y: 5.5, w: CW, h: 0.45, fontSize: 14, color: C.accent3 });
}

// 16. Thank you
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: L("Analysis & plan", "Phân tích & kế hoạch") });
  s.addText(L("Thank you", "Cảm ơn hội đồng"), { placeholder: "title" });
  s.addText([
    { text: L("We welcome your feedback.", "Nhóm rất mong nhận được góp ý."), options: { fontSize: 20, breakLine: true } },
    { text: " ", options: { fontSize: 12, breakLine: true } },
    { text: L("Code, configs and every run log: github.com/PLHGNAOH/ViVQA-VLM", "Code, config và log mọi run: github.com/PLHGNAOH/ViVQA-VLM"), options: { fontSize: 15, color: "A8C4E0" } },
  ], { placeholder: "body" });
  s.addNotes("Cảm ơn hội đồng đã lắng nghe. Nhóm sẵn sàng trả lời câu hỏi.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
