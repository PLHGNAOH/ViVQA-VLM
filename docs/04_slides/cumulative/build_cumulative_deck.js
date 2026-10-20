// Cumulative ViVQA-VLM review deck (EN or VI) — ONE deck that grows at every review:
//   Part A = Review 1 (revised after reviewer feedback 24/09/2026), Part B = Review 2 (W07–W09 results), later parts added at R3/final.
// Usage (needs pptxgenjs + the pptx skill apply_theme.js via SKILL_DIR):
//   SKILL_DIR=<pptx skill dir> node docs/04_slides/cumulative/build_cumulative_deck.js en docs/04_slides/cumulative/ViVQA-VLM_Review2_Cumulative_EN.pptx
// Part A content: docs/W03 Review 1 report (revised, tag review1-report). Part B numbers: experiments/ (W07–W09) — see notes per slide.
// Hypotheses use the revised numbering H1–H4 everywhere (old R1 H6→H2, H7→H3, H8→H4; old H2 → efficiency measurement; old H3/H4 → ablations A1/A2; old H5 → optional).
const pptxgen = require("pptxgenjs");
const path = require("path");
const { applyTheme } = require(process.env.SKILL_DIR + "/scripts/apply_theme.js");

const LANG = process.argv[2] || "en";
const OUT = process.argv[3] || `ViVQA-VLM_Review2_Cumulative_${LANG.toUpperCase()}.pptx`;
const L = (en, vi) => (LANG === "en" ? en : vi);
// VI: decimal point -> comma for numbers like 22.1 / 0.86 (not 10.799, not Qwen2.5)
const FMT = (c) => (LANG === "en" ? c : c.replace(/(?<![A-Za-z\d])(\d+)\.(\d{1,2})(?!\d)/g, "$1,$2"));
const FIG = path.join(__dirname, "..", "..", "W07", "figures");
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
pres.title = "ViVQA-VLM Review 2 (cumulative)";
pres.author = "Phạm Lê Huy Hoàng";
const C = pres.SchemeColor;
const W = 13.333, M = 0.6, CW = W - 2 * M;
const FOOT = L("ViVQA-VLM · Review 2 (cumulative: R1 revised + R2) · GFA26AI35 · FPT University", "ViVQA-VLM · Review 2 (cộng dồn: R1 đã sửa + R2) · GFA26AI35 · Đại học FPT");

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
    { placeholder: { options: { name: "eyebrow", type: "body", x: M, y: 0.32, w: CW, h: 0.3, fontFace: "Calibri", fontSize: 11, bold: true, color: C.accent1, charSpacing: 2, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.62, w: CW, h: 0.7, fontFace: "Cambria", fontSize: 32, bold: true, color: C.text2, valign: "top", align: "left", margin: 0 }, text: "" } },
    { text: { text: FOOT, options: { x: M, y: 7.05, w: 10, h: 0.3, fontFace: "Calibri", fontSize: 10, color: C.accent3, margin: 0 } } },
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
function stat(s, x, y, w, big, label, color = C.text2, size = 40) {
  txt(s, big, { x, y, w, h: 0.9, fontFace: "Cambria", fontSize: size, bold: true, color });
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
function footnote(s, text, y = 6.5, h = 0.48) {
  txt(s, text, { x: M, y, w: CW, h, fontSize: 10, color: C.accent3, valign: "bottom" });
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
    { text: L("Review 2 (cumulative)  ·  Part A: Review 1, revised  ·  Part B: baseline, QLoRA and the proposed OCR-aware QLoRA", "Review 2 (cộng dồn)  ·  Phần A: Review 1 đã sửa  ·  Phần B: baseline, QLoRA và đề xuất QLoRA nhận biết OCR"), options: { fontSize: 18, bold: true, color: HEX.amber, breakLine: true } },
    { text: " ", options: { fontSize: 18, breakLine: true } },
    { text: L("Supervisors:  ", "GVHD:  "), options: { bold: true, fontSize: 15 } },
    { text: "Đặng Ngọc Minh Đức · Phạm Minh Trí", options: { fontSize: 15, breakLine: true } },
    { text: L("Team:  ", "Nhóm:  "), options: { bold: true, fontSize: 15 } },
    { text: L("Phạm Lê Huy Hoàng (Leader) · Văn Quang Trường · Trần Quốc Huân · Hồ Đắc Toàn", "Phạm Lê Huy Hoàng (Trưởng nhóm) · Văn Quang Trường · Trần Quốc Huân · Hồ Đắc Toàn"), options: { fontSize: 15, breakLine: true } },
    { text: " ", options: { fontSize: 15, breakLine: true } },
    { text: L("Group GFA26AI35  ·  FPT University  ·  October 2026", "Nhóm GFA26AI35  ·  Đại học FPT  ·  Tháng 10/2026"), options: { fontSize: 13, color: "A8C4E0" } },
  ], { placeholder: "body" });
  s.addNotes("Chào hội đồng. Đây là bộ slide cộng dồn của nhóm. Phần A là nội dung Review 1, đã sửa theo góp ý của phản biện. Phần B là Review 2: tái lập baseline Qwen2.5-VL-3B, kết quả QLoRA, và phương pháp đề xuất là QLoRA nhận biết OCR. Toàn bộ số liệu đều có log và có thể chạy lại.");
}

// ================= PART A — REVIEW 1 (revised after reviewer feedback, 24/09/2026) =================
// Source of content: docs/W03/index_TV.html at tag review1-report (74fa721, "cập nhật theo góp ý reviewer"),
// facts that changed later (T4 → L4, mirror → official ViTextVQA, pilot zero-shot numbers) are updated or dropped
// so Part A never contradicts Part B. Hypotheses use the revised numbering H1–H4 throughout the deck.
const A = {};            // slide numbers of Part A, filled as slides are added (used by the feedback table)
const here = () => pres.slides.length;

// A-contents slide
pres.addSection({ title: L("Contents", "Nội dung") });
const contentsSlide = content(L("Contents", "Nội dung"), L("CONTENTS", "NỘI DUNG"), L("One deck, every review", "Một bộ slide cho mọi mốc review"),
  "Bộ slide này cộng dồn qua các mốc. Phần A là nội dung Review 1, đã sửa theo mười góp ý của giảng viên phản biện ngày 24/09. Phần B là nội dung Review 2: tái lập baseline, QLoRA và phương pháp đề xuất. Mỗi mốc sau sẽ thêm một phần mới và sửa các phần trước theo góp ý.");

pres.addSection({ title: L("A · Review 1 (revised)", "A · Review 1 (đã sửa)") });

// A0. Response to Review 1 feedback — table filled at the end of Part A (needs slide numbers)
const fbSlide = content(L("A · Review 1 (revised)", "A · Review 1 (đã sửa)"), L("REVIEW 1 · RESPONSE TO REVIEWER FEEDBACK (24/09/2026)", "REVIEW 1 · TIẾP THU GÓP Ý PHẢN BIỆN (24/09/2026)"),
  L("Ten reviewer comments and how Part A addresses them", "Mười góp ý của phản biện và cách Phần A xử lý"),
  "Ngày 24/09, giảng viên phản biện đưa ra mười góp ý. Bảng này ghi lại từng góp ý, việc nhóm đã sửa và slide tương ứng trong Phần A. Ba điểm lớn nhất: phát biểu bài toán với input và output rõ ràng; giả thuyết rút còn H1 đến H4, mỗi giả thuyết gắn với một vấn đề riêng của tiếng Việt; và chốt một phương pháp đề xuất kèm sơ đồ pipeline.");
A.fb = here();

// A1. Problem statement + I/O
{
  const s = content(L("A · Review 1 (revised)", "A · Review 1 (đã sửa)"), L("REVIEW 1 · 01 · PROBLEM", "REVIEW 1 · 01 · BÀI TOÁN"), L("Problem statement and input / output", "Phát biểu bài toán và input / output"),
    "Bài toán trong một câu: cho một ảnh và một câu hỏi tiếng Việt, sinh ra câu trả lời tiếng Việt ngắn, đúng nội dung ảnh và đúng chính tả có dấu. Bài toán nghiên cứu là thích nghi một VLM mã nguồn mở, Qwen2.5-VL-3B, chỉ bằng QLoRA và chạy được trên một GPU Colab. Bảng bên dưới đặc tả input, output và cách đánh giá cho hai bộ dữ liệu. Đây là phần sửa cho góp ý số 1.");
  A.problem = here();
  card(s, M, 1.5, CW, 1.05, C.accent4);
  txt(s, [
    { text: L("Task  ", "Bài toán  "), options: { bold: true, color: HEX.ink } },
    { text: L("Given an image and a Vietnamese question about it, generate a short Vietnamese answer that is faithful to the image and correctly spelled with tone marks — adapting an open VLM (Qwen2.5-VL-3B) with PEFT (QLoRA) only, on a single Colab GPU.",
              "Cho một ảnh và một câu hỏi tiếng Việt về ảnh, sinh câu trả lời tiếng Việt ngắn, đúng nội dung ảnh và đúng chính tả có dấu — bằng cách thích nghi một VLM mở (Qwen2.5-VL-3B) chỉ với PEFT (QLoRA), trên một GPU Colab.") },
  ], { x: M + 0.3, y: 1.5, w: CW - 0.6, h: 1.05, fontSize: 15, valign: "middle" });
  // flow
  const boxes = [L("Image + Vietnamese question", "Ảnh + câu hỏi tiếng Việt"), L("Qwen2.5-VL-3B (4-bit) + QLoRA adapter", "Qwen2.5-VL-3B (4-bit) + adapter QLoRA"), L("Short Vietnamese answer", "Câu trả lời tiếng Việt ngắn")];
  const bw = 3.5, gap = (CW - 3 * bw) / 2;
  boxes.forEach((b, i) => {
    const x = M + i * (bw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.8, w: bw, h: 0.7, rectRadius: 0.08, fill: { color: i === 1 ? HEX.ambert : HEX.wash }, line: { color: i === 1 ? HEX.amber : HEX.line, width: 1 }, objectName: oid("box") });
    txt(s, b, { x: x + 0.1, y: 2.8, w: bw - 0.2, h: 0.7, fontSize: 14, bold: true, color: C.text2, align: "center", valign: "middle" });
    if (i < 2) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bw + gap / 2 - 0.3, y: 3.0, w: 0.6, h: 0.3, fill: { color: HEX.teal }, line: { color: HEX.teal, width: 0 }, objectName: oid("arrow") });
  });
  const rows = [
    [TH(L("Component", "Thành phần")), TH(L("Definition", "Định nghĩa")), TH("ViVQA"), TH("ViTextVQA")],
    [L("Input 1 · Image", "Input 1 · Ảnh"), L("RGB image, resized under max_pixels", "Ảnh RGB, resize theo max_pixels"), L("Everyday photos (MS COCO), not text-centric", "Ảnh đời thường (MS COCO), ít chữ"), L("Vietnamese scene text: signs, menus, packaging", "Ảnh có chữ tiếng Việt: biển hiệu, menu, bao bì")],
    [L("Input 2 · Question", "Input 2 · Câu hỏi"), L("Vietnamese text, Unicode NFC", "Văn bản tiếng Việt, chuẩn NFC"), L("Short, semi-automatically generated", "Ngắn, sinh bán tự động"), L("Human-written, 9.59 words on average", "Do người viết, trung bình 9,59 từ")],
    [L("Output · Answer", "Output · Câu trả lời"), L("Free-form Vietnamese text with tone marks", "Văn bản tiếng Việt có dấu, sinh tự do"), L("99.83% of answers ≤ 3 words", "99,83% đáp án ≤ 3 từ"), L("4.18 words on average; a span of text in the image", "Trung bình 4,18 từ; là đoạn chữ trong ảnh")],
    [L("Evaluation", "Đánh giá"), L("Compared with gold after the same normalization", "So với đáp án chuẩn sau cùng một bước chuẩn hoá"), "EM · VQA-Acc", "ANLS · EM"],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: FMT(c), options: { fontSize: 13, bold: j === 0 } })));
  table(s, rows, { x: M, y: 3.75, w: CW, colW: [2.2, 3.6, 3.0, 3.33], rowH: 0.56 });
}

// A2. Vietnamese-specific problems V1–V4 + preprocessing vs research problem
{
  const s = content(L("A · Review 1 (revised)", "A · Review 1 (đã sửa)"), L("REVIEW 1 · 01 · WHY VIETNAMESE IS DIFFERENT", "REVIEW 1 · 01 · TIẾNG VIỆT KHÁC Ở ĐÂU"), L("Four Vietnamese-specific problems (V1–V4)", "Bốn vấn đề riêng của tiếng Việt (V1–V4)"),
    "Góp ý số 1 và số 2: phải nói được tiếng Việt khác tiếng Anh ở đâu, và tách tiền xử lý ra khỏi vấn đề nghiên cứu. Bốn vấn đề: V1 đọc chữ có dấu trong ảnh — dấu là nét rất nhỏ, đọc sai là đổi nghĩa; V2 trả lời sai ngôn ngữ hoặc sai định dạng; V3 thứ tự chữ và ranh giới từ, vì khoảng trắng tiếng Việt tách âm tiết chứ không tách từ; V4 hallucination. Chuẩn hoá NFC, chữ thường, dấu câu chỉ là tiền xử lý bắt buộc, không phải đóng góp.");
  A.viet = here();
  const v = [
    ["V1", L("Reading tone-marked text in images", "Đọc chữ có dấu trong ảnh"), L("Tone marks are tiny strokes; a wrong mark changes the word (má ≠ mà). A pixel-level perception error that text normalization cannot fix. Evidence: removing diacritics lowers F1 of every ViTextVQA baseline (Nguyen et al., 2024, §7.1).", "Dấu là nét rất nhỏ; sai dấu là đổi nghĩa (má ≠ mà). Lỗi nhận thức ở mức pixel, chuẩn hoá văn bản không sửa được. Bằng chứng: bỏ dấu làm giảm F1 mọi baseline ViTextVQA (Nguyen et al., 2024, §7.1).")],
    ["V2", L("Wrong language or wrong format", "Sai ngôn ngữ hoặc sai định dạng"), L("VLMs pretrained mostly on English/Chinese answer in English, mix languages (code-switching) or answer verbosely → EM = 0 even when the content is right.", "VLM pretrain chủ yếu trên tiếng Anh/Trung dễ trả lời tiếng Anh, trộn Anh–Việt hoặc dài dòng → EM = 0 dù hiểu đúng.")],
    ["V3", L("Text order and word boundaries", "Thứ tự chữ và ranh giới từ"), L("Spaces separate syllables, not words (“thời gian” is one word). Text read in the wrong order breaks words. Evidence: top-left → bottom-right OCR order helps most models (Nguyen et al., 2024, §6.4).", "Khoảng trắng tách âm tiết, không tách từ (“thời gian” là một từ). Đọc sai thứ tự làm vỡ từ. Bằng chứng: sắp OCR trái-trên → phải-dưới giúp đa số mô hình (Nguyen et al., 2024, §6.4).")],
    ["V4", L("Hallucination", "Hallucination"), L("Fluent answers about objects or text that are not in the image. No POPE-style hallucination evaluation exists yet for Vietnamese VQA (gap G4).", "Trả lời trôi chảy nhưng nội dung không có trong ảnh. Chưa có đánh giá hallucination kiểu POPE cho VQA tiếng Việt (G4).")],
  ];
  const cw = (CW - 3 * 0.3) / 4, y = 1.5, h = 3.45;
  v.forEach(([id, t, d], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, y, cw, h, i === 0 ? C.accent5 : C.background2);
    txt(s, id, { x: x + 0.25, y: y + 0.15, w: 1, h: 0.5, fontFace: "Cambria", fontSize: 24, bold: true, color: i === 0 ? HEX.amber : HEX.teal });
    txt(s, t, { x: x + 0.25, y: y + 0.65, w: cw - 0.5, h: 0.65, fontSize: 14, bold: true, color: C.text2 });
    txt(s, d, { x: x + 0.25, y: y + 1.3, w: cw - 0.5, h: h - 1.4, fontSize: 13 });
  });
  const y2 = 5.15, cw2 = (CW - 0.4) / 2, h2 = 1.7;
  card(s, M, y2, cw2, h2);
  txt(s, L("Preprocessing — required engineering, not a contribution", "Tiền xử lý — bước kỹ thuật bắt buộc, không phải đóng góp"), { x: M + 0.25, y: y2 + 0.12, w: cw2 - 0.5, h: 0.4, fontSize: 14, bold: true, color: C.text2 });
  bullets(s, [L("Unicode NFC, lowercase, strip punctuation, collapse spaces", "Chuẩn NFC, chữ thường, bỏ dấu câu thừa, gộp khoảng trắng"), L("Same normalization for gold and prediction before scoring", "Cùng một hàm chuẩn hoá cho đáp án chuẩn và đầu ra trước khi chấm"), L("Resize images under max_pixels", "Resize ảnh theo max_pixels")],
    { x: M + 0.25, y: y2 + 0.55, w: cw2 - 0.5, h: h2 - 0.6, fontSize: 12.5, paraSpaceAfter: 2 });
  card(s, M + cw2 + 0.4, y2, cw2, h2, C.accent4);
  txt(s, L("Research problem — the focus of the thesis", "Vấn đề nghiên cứu — trọng tâm đề tài"), { x: M + cw2 + 0.65, y: y2 + 0.12, w: cw2 - 0.5, h: 0.4, fontSize: 14, bold: true, color: C.text2 });
  bullets(s, [L("Read tone-marked text correctly from the image (V1, V3)", "Đọc đúng chữ có dấu từ ảnh (V1, V3)"), L("Answer in the right language and format (V2); do not hallucinate (V4)", "Trả lời đúng ngôn ngữ, đúng định dạng (V2); không bịa (V4)"), L("Achieve this with PEFT only, on one Colab GPU", "Đạt được điều đó chỉ bằng PEFT, trên một GPU Colab")],
    { x: M + cw2 + 0.65, y: y2 + 0.55, w: cw2 - 0.5, h: h2 - 0.6, fontSize: 12.5, paraSpaceAfter: 2 });
}

// A3. Research gaps
{
  const s = content(L("A · Review 1 (revised)", "A · Review 1 (đã sửa)"), L("REVIEW 1 · 02 · RESEARCH GAPS", "REVIEW 1 · 02 · KHOẢNG TRỐNG NGHIÊN CỨU"), L("Four gaps — G3 is the focus", "Bốn khoảng trống — G3 là trọng tâm"),
    "Bốn khoảng trống từ văn liệu. G1: alignment của VLM tối ưu cho tiếng Anh. G2: VQA tiếng Việt còn yếu, nhất là scene-text — SOTA công bố trên ViTextVQA mới đạt EM 25,48 trong khi con người đạt 88. G3 là trọng tâm: chưa có nghiên cứu hệ thống nào áp dụng LoRA hay QLoRA cho VLM trên VQA tiếng Việt — trong ba paper VQA tiếng Việt nhóm khảo sát, không paper nào dùng PEFT. G4: chưa có đánh giá hallucination kiểu POPE cho tiếng Việt.");
  A.gaps = here();
  const g = [
    ["G1", L("English-biased pretraining", "Tiền huấn luyện thiên lệch tiếng Anh"), L("VLM alignment is optimized for English; transfer to Vietnamese is under-studied.", "Alignment của VLM tối ưu cho tiếng Anh; chuyển giao sang tiếng Việt chưa được nghiên cứu kỹ."), "LLaVA-1.5 (Liu et al., CVPR 2024)"],
    ["G2", L("Weak Vietnamese VQA, especially scene text", "VQA tiếng Việt yếu, nhất là scene-text"), L("Best published ViTextVQA EM is 25.48 vs 88.00 for humans; OCR, tone marks and code-switching remain open.", "EM công bố cao nhất trên ViTextVQA là 25,48 so với 88,00 của con người; OCR, dấu thanh, code-switching còn mở."), "ViTextVQA (Nguyen et al., 2024)"],
    ["G3", L("PEFT for Vietnamese VLMs is unexplored", "PEFT cho VLM tiếng Việt chưa được khai thác"), L("No systematic study applies LoRA/QLoRA to a VLM for Vietnamese VQA — none of the three Vietnamese VQA papers we surveyed uses PEFT.", "Chưa có nghiên cứu hệ thống áp dụng LoRA/QLoRA cho VLM trên VQA tiếng Việt — ba paper VQA tiếng Việt nhóm khảo sát đều không dùng PEFT."), L("Literature Matrix (W2)", "Literature Matrix (W2)")],
    ["G4", L("Hallucination not measured for Vietnamese", "Chưa đo hallucination cho tiếng Việt"), L("No POPE-style hallucination benchmark has been evaluated on Vietnamese VQA data.", "Chưa có benchmark hallucination kiểu POPE nào được đánh giá trên dữ liệu VQA tiếng Việt."), "POPE (Li et al., EMNLP 2023)"],
  ];
  const cw = (CW - 3 * 0.3) / 4, y = 1.5, h = 4.6;
  g.forEach(([id, t, d, src], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, y, cw, h, i === 2 ? C.accent5 : C.background2);
    txt(s, id + (i === 2 ? L("  · FOCUS", "  · TRỌNG TÂM") : ""), { x: x + 0.25, y: y + 0.2, w: cw - 0.5, h: 0.5, fontFace: "Cambria", fontSize: 22, bold: true, color: i === 2 ? HEX.amber : HEX.teal });
    txt(s, t, { x: x + 0.25, y: y + 0.8, w: cw - 0.5, h: 0.85, fontSize: 15, bold: true, color: C.text2 });
    txt(s, FMT(d), { x: x + 0.25, y: y + 1.7, w: cw - 0.5, h: 2.1, fontSize: 14.5 });
    txt(s, src, { x: x + 0.25, y: y + h - 0.65, w: cw - 0.5, h: 0.5, fontSize: 10.5, italic: true, color: C.accent3, valign: "bottom" });
  });
  txt(s, L("How the deck answers them: G2 → OCR-aware QLoRA on ViTextVQA (Part B · 06) · G3 → QLoRA on Qwen2.5-VL-3B (Part B · 05) · G4 → hallucination audit and H4 (Part B · 05, 07).",
           "Cách bộ slide trả lời: G2 → QLoRA nhận biết OCR trên ViTextVQA (Phần B · 06) · G3 → QLoRA trên Qwen2.5-VL-3B (Phần B · 05) · G4 → duyệt hallucination và H4 (Phần B · 05, 07)."),
      { x: M, y: 6.3, w: CW, h: 0.55, fontSize: 13, color: C.accent1 });
}

// A4. Prior work & reference baselines
{
  const s = content(L("A · Review 1 (revised)", "A · Review 1 (đã sửa)"), L("REVIEW 1 · 03 · PRIOR WORK", "REVIEW 1 · 03 · CÔNG TRÌNH TRƯỚC"), L("Prior work and reference baselines", "Công trình trước và baseline tham chiếu"),
    "Góp ý số 7: cần ba baseline và state-of-the-art cho tiếng Việt. Nhóm dùng hai loại mốc. Baseline tái lập là mô hình nhóm tự chạy lại và dùng để kết luận cải tiến — Qwen2.5-VL-3B. Baseline từ công trình trước là số trích từ paper, không chạy lại, chỉ dùng để định vị. Trên ViTextVQA có BLIP-2, PreSTU và ViTextBLIP-2 là SOTA công bố với EM 25,48. Trên ViVQA, cao nhất nhóm khảo sát được là AViVQA-TranConI với accuracy 71,04. Lưu ý: các mốc này dùng metric và thiết lập khác nhóm nên chỉ để định vị.");
  A.prior = here();
  const r1 = [
    [TH(L("ViTextVQA test · work", "ViTextVQA test · công trình")), TH(L("Approach", "Hướng tiếp cận")), TH(L("Trained params", "Tham số train")), TH("F1"), TH("EM")],
    [L("BLIP-2 large (Li et al., ICML 2023)", "BLIP-2 large (Li và cs., ICML 2023)"), L("Frozen image encoder + LM, train Q-Former; OCR SwinTextSpotter, LM ViT5", "Encoder ảnh + LM đóng băng, train Q-Former; OCR SwinTextSpotter, LM ViT5"), "≈ 8% / 1.4B", "38.47", "15.58"],
    [L("PreSTU large (Kil et al., ICCV 2023)", "PreSTU large (Kil và cs., ICCV 2023)"), L("T5 pretrained for scene text, layout-ordered OCR", "T5 pretrain cho scene-text, OCR theo bố cục"), "≈ 100% / 860M", "44.93", "22.64"],
    [L("ViTextBLIP-2 (Nguyen et al., 2024) — SOTA", "ViTextBLIP-2 (Nguyen và cs., 2024) — SOTA"), L("BLIP-2 + OCR + caption, train Q-Former only", "BLIP-2 + OCR + caption, chỉ train Q-Former"), "≈ 5% / 2.5B", "53.95", "25.48"],
    [L("QwenVL-7B zero-shot (same paper)", "QwenVL-7B zero-shot (cùng paper)"), L("Multilingual VLM, no training; 100 random questions", "VLM đa ngôn ngữ, không train; 100 câu ngẫu nhiên"), "0%", "53.80", "28.00"],
    [L("Human (same paper)", "Con người (cùng paper)"), "—", "—", "96.42", "88.00"],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: FMT(c), options: { fontSize: 12, bold: i === 3 && j > 2 } })));
  table(s, r1, { x: M, y: 1.45, w: CW, colW: [3.6, 5.13, 1.6, 0.9, 0.9], rowH: 0.42 });
  const r2 = [
    [TH(L("ViVQA test · work", "ViVQA test · công trình")), TH(L("Approach", "Hướng tiếp cận")), TH(L("Result", "Kết quả"))],
    [L("AViVQA-TranConI (Nguyen et al., Comput. Electr. Eng. 2024)", "AViVQA-TranConI (Nguyen và cs., Comput. Electr. Eng. 2024)"), L("Frozen BLIP-2 + EfficientNet, picks an answer from a fixed label set", "BLIP-2 đóng băng + EfficientNet, chọn đáp án từ tập nhãn cố định"), L("Accuracy 71.04", "Accuracy 71,04")],
    [L("Improved Multi-Vision Contextual Attention (Bui et al., 2025)", "Multi-Vision Contextual Attention cải tiến (Bui và cs., 2025)"), L("Attention over several visual features", "Attention trên nhiều đặc trưng thị giác"), L("Accuracy 62.41 (original 60)", "Accuracy 62,41 (gốc 60)")],
  ].map((r, i) => i === 0 ? r : r.map((c) => ({ text: c, options: { fontSize: 12 } })));
  table(s, r2, { x: M, y: 4.25, w: CW, colW: [4.6, 5.0, 2.53], rowH: 0.42 });
  card(s, M, 5.65, CW, 1.2, C.accent5);
  txt(s, [
    { text: L("Caution  ", "Lưu ý  "), options: { bold: true, color: HEX.ink } },
    { text: L("These numbers use other metrics and settings (token-F1; classification accuracy over a fixed label set) — used to position our results, not as head-to-head comparisons. The baseline we reproduce and improve on is Qwen2.5-VL-3B (Part B). Lesson kept: parameter-efficient ViTextBLIP-2 beats full fine-tuning → supports QLoRA.",
              "Các số này dùng metric và thiết lập khác (F1 theo token; accuracy phân loại trên tập nhãn cố định) — chỉ để định vị, không so trực tiếp. Baseline nhóm tái lập và cải tiến là Qwen2.5-VL-3B (Phần B). Bài học giữ lại: ViTextBLIP-2 hiệu quả tham số vẫn thắng full fine-tuning → ủng hộ QLoRA.") },
  ], { x: M + 0.3, y: 5.65, w: CW - 0.6, h: 1.2, fontSize: 13, valign: "middle" });
}

// A5. Research questions
{
  const s = content(L("A · Review 1 (revised)", "A · Review 1 (đã sửa)"), L("REVIEW 1 · 04 · RESEARCH QUESTIONS", "REVIEW 1 · 04 · CÂU HỎI NGHIÊN CỨU"), L("Four research questions — RQ2 is central", "Bốn câu hỏi nghiên cứu — RQ2 là trung tâm"),
    "Bốn câu hỏi nghiên cứu. RQ1: zero-shot đạt bao nhiêu và lỗi tiếng Việt nào chiếm ưu thế. RQ2 là trung tâm: QLoRA cải thiện bao nhiêu so với zero-shot mà không full fine-tuning. RQ3: đưa chữ OCR có dấu vào prompt có giúp đọc chữ không và có cộng hưởng với QLoRA không. RQ4: sau QLoRA, model bịa nhiều hơn hay ít hơn.");
  A.rq = here();
  const q = [
    ["RQ1", L("Zero-shot & prompting", "Zero-shot & prompting"), L("How well does the VLM do zero-shot on ViVQA / ViTextVQA, which Vietnamese error types (V1–V3) dominate, and how far does Vietnamese prompting help?", "VLM đạt bao nhiêu khi zero-shot trên ViVQA / ViTextVQA, lỗi tiếng Việt nào (V1–V3) chiếm ưu thế, prompt tiếng Việt giúp được đến đâu?"), "EM · VQA-Acc · ANLS"],
    ["RQ2", L("PEFT (QLoRA) vs zero-shot — central", "PEFT (QLoRA) so với zero-shot — trung tâm"), L("How much does QLoRA improve over zero-shot / prompting, without full fine-tuning?", "QLoRA cải thiện bao nhiêu so với zero-shot / prompting, không full fine-tuning?"), "EM · VQA-Acc · ANLS"],
    ["RQ3", L("Vietnamese OCR × QLoRA on ViTextVQA", "OCR tiếng Việt × QLoRA trên ViTextVQA"), L("Does tone-marked OCR text, ordered top-left → bottom-right, help read text in the image, and does it work together with QLoRA?", "Chữ OCR có dấu, sắp trái-trên → phải-dưới, có giúp đọc chữ trong ảnh không, và có cộng hưởng với QLoRA không?"), "ANLS · EM"],
    ["RQ4", L("QLoRA and hallucination", "QLoRA và hallucination"), L("After QLoRA adaptation, does the model invent more or less content that is not in the image?", "Sau khi thích nghi bằng QLoRA, model bịa nội dung nhiều hơn hay ít hơn?"), L("POPE-style (vi) + manual audit", "Kiểu POPE (vi) + duyệt tay")],
  ];
  const cw = (CW - 0.4) / 2, h = 2.4;
  q.forEach(([id, t, d, m], i) => {
    const x = M + (i % 2) * (cw + 0.4), y = 1.5 + Math.floor(i / 2) * (h + 0.3);
    card(s, x, y, cw, h, i === 1 ? C.accent5 : C.background2);
    txt(s, id, { x: x + 0.3, y: y + 0.2, w: 1.3, h: 0.55, fontFace: "Cambria", fontSize: 26, bold: true, color: i === 1 ? HEX.amber : HEX.teal });
    txt(s, t, { x: x + 1.6, y: y + 0.25, w: cw - 1.9, h: 0.5, fontSize: 16, bold: true, color: C.text2, valign: "middle" });
    txt(s, d, { x: x + 0.3, y: y + 0.9, w: cw - 0.6, h: 1.0, fontSize: 14 });
    txt(s, L("Measured by: ", "Đo bằng: ") + m, { x: x + 0.3, y: y + h - 0.5, w: cw - 0.6, h: 0.35, fontSize: 12, italic: true, color: C.accent3 });
  });
}

// A6. Hypotheses H1–H4
{
  const s = content(L("A · Review 1 (revised)", "A · Review 1 (đã sửa)"), L("REVIEW 1 · 05 · HYPOTHESES", "REVIEW 1 · 05 · GIẢ THUYẾT"), L("Four hypotheses, each tied to a Vietnamese problem", "Bốn giả thuyết, mỗi giả thuyết gắn một vấn đề tiếng Việt"),
    "Góp ý số 3: giả thuyết 4-bit hay VRAM không giải quyết vấn đề tiếng Việt. Nhóm rút còn bốn giả thuyết H1 đến H4, mỗi cái gắn một vấn đề V1 đến V4 và được kiểm chứng trên cùng model, cùng split, cùng seed. Hiệu quả bộ nhớ của 4-bit là đặc tính đã biết của QLoRA nên chuyển thành số đo báo cáo ở mỗi run. Cấu hình LoRA chuyển thành ablation. Ngưỡng là ngưỡng tham khảo từ văn liệu, chốt trước khi chạy.");
  A.hyp = here();
  const rows = [
    [TH("ID"), TH(L("Hypothesis (threshold fixed before running)", "Giả thuyết (ngưỡng chốt trước khi chạy)")), TH(L("Answers", "Trả lời"))],
    ["H1", L("QLoRA on Qwen2.5-VL-3B improves EM by ≥ +5 over the same model zero-shot", "QLoRA trên Qwen2.5-VL-3B tăng EM ≥ +5 so với chính model đó zero-shot"), "RQ2 · V1–V3"],
    ["H2", L("Tone-marked OCR text (top-left → bottom-right) in the prompt adds ≥ +3 ANLS on ViTextVQA; combined with QLoRA it gives the best result", "Chữ OCR có dấu (trái-trên → phải-dưới) trong prompt tăng ≥ +3 ANLS trên ViTextVQA; kết hợp QLoRA cho kết quả cao nhất"), "RQ3 · V1, V3"],
    ["H3", L("A Vietnamese format prompt (short answer, with tone marks) adds ≥ +3 EM and reduces wrong-language / missing-diacritic answers", "Prompt định dạng tiếng Việt (trả lời ngắn, có dấu) tăng ≥ +3 EM và giảm câu trả lời sai ngôn ngữ / mất dấu"), "RQ1 · V2"],
    ["H4", L("QLoRA does not increase the hallucination rate compared with zero-shot", "QLoRA không làm tăng tỉ lệ hallucination so với zero-shot"), "RQ4 · V4"],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c, options: { fontSize: 13.5, bold: j === 0, color: j === 0 ? HEX.teal : HEX.text } })));
  table(s, rows, { x: M, y: 1.5, w: CW, colW: [0.8, 9.33, 2.0], rowH: 0.62 });
  const y = 4.85, cw = (CW - 0.4) / 2, h = 2.0;
  card(s, M, y, cw, h);
  txt(s, L("No longer hypotheses → measured or ablated", "Không còn là giả thuyết → đo lường hoặc ablation"), { x: M + 0.25, y: y + 0.12, w: cw - 0.5, h: 0.4, fontSize: 14, bold: true, color: C.text2 });
  bullets(s, [L("4-bit vs 16-bit memory: a known QLoRA property (Dettmers et al., NeurIPS 2023) → VRAM, time and trained parameters are reported for every run", "4-bit so với 16-bit: đặc tính đã biết của QLoRA (Dettmers và cs., NeurIPS 2023) → báo cáo VRAM, thời gian, số tham số train ở mỗi run"),
              L("LoRA target modules and rank → ablations A1, A2", "Module gắn LoRA và rank → ablation A1, A2"), L("Shared adapter across datasets → optional", "Adapter dùng chung cho 2 dataset → tuỳ chọn")],
    { x: M + 0.25, y: y + 0.55, w: cw - 0.5, h: h - 0.6, fontSize: 12, paraSpaceAfter: 2 });
  card(s, M + cw + 0.4, y, cw, h, C.accent4);
  txt(s, L("Mapping from the first Review 1 version", "Đối chiếu với bản Review 1 đầu tiên"), { x: M + cw + 0.65, y: y + 0.12, w: cw - 0.5, h: 0.4, fontSize: 14, bold: true, color: C.text2 });
  txt(s, L("H1 → H1 · H6 → H2 · H7 → H3 · H8 → H4\nH2 (4-bit) → efficiency measurement\nH3, H4 (LoRA config) → ablations A1, A2 · H5 → optional",
           "H1 → H1 · H6 → H2 · H7 → H3 · H8 → H4\nH2 (4-bit) → số đo hiệu quả\nH3, H4 (cấu hình LoRA) → ablation A1, A2 · H5 → tuỳ chọn"),
      { x: M + cw + 0.65, y: y + 0.6, w: cw - 0.5, h: h - 0.7, fontSize: 13.5, color: C.text1 });
}

// A7. Data
{
  const s = content(L("A · Review 1 (revised)", "A · Review 1 (đã sửa)"), L("REVIEW 1 · 06 · DATA", "REVIEW 1 · 06 · DỮ LIỆU"), L("Two benchmarks, fixed splits, preprocessing before the split", "Hai bộ benchmark, split cố định, tiền xử lý trước khi chia"),
    "Hai bộ dữ liệu. ViVQA có 15.000 cặp hỏi–đáp trên 10.328 ảnh; nhóm tách 1.200 câu val từ train bằng seed 42, còn 10.799 câu train và 3.001 câu test. Đáp án gần như luôn ngắn nên dùng EM và VQA-Acc. ViTextVQA dùng repo chính thức: 35.159 train, 5.155 dev, 10.028 test; đáp án dài hơn và là chữ trong ảnh nên dùng thêm ANLS. Góp ý số 9: tiền xử lý áp dụng giống hệt cho mọi mẫu, rồi mới cố định split; chọn cấu hình trên dev hoặc val, chỉ báo cáo trên test.");
  A.data = here();
  const cw = (CW - 0.4) / 2, y = 1.5, h = 3.75;
  const ds = [
    ["ViVQA", "Tran et al., PACLIC 2021", [["15,000", L("QA pairs", "cặp hỏi–đáp")], ["10,328", L("images (MS COCO)", "ảnh (MS COCO)")]],
     L("Split (seed 42): train 10,799 · val 1,200 (taken from train) · test 3,001", "Split (seed 42): train 10.799 · val 1.200 (tách từ train) · test 3.001"),
     L("99.83% of answers ≤ 3 words → EM / VQA-Acc", "99,83% đáp án ≤ 3 từ → EM / VQA-Acc")],
    ["ViTextVQA", "Nguyen et al., 2024 (official repo)", [["50,342", L("QA pairs", "cặp hỏi–đáp")], ["16,762", L("images with Vietnamese text", "ảnh có chữ tiếng Việt")]],
     L("Official split: train 35,159 · dev 5,155 · test 10,028", "Split chính thức: train 35.159 · dev 5.155 · test 10.028"),
     L("Answers 4.18 words on average, read from the image → ANLS (tolerates character errors)", "Đáp án trung bình 4,18 từ, đọc từ ảnh → ANLS (chịu lỗi ký tự)")],
  ];
  ds.forEach(([name, src, nums, split, note], i) => {
    const x = M + i * (cw + 0.4);
    card(s, x, y, cw, h, i === 1 ? C.accent4 : C.background2);
    txt(s, name, { x: x + 0.3, y: y + 0.15, w: cw - 0.6, h: 0.5, fontFace: "Cambria", fontSize: 22, bold: true, color: C.text2 });
    txt(s, src, { x: x + 0.3, y: y + 0.65, w: cw - 0.6, h: 0.3, fontSize: 11, italic: true, color: C.accent3 });
    nums.forEach(([big, lab], j) => stat(s, x + 0.3 + j * (cw - 0.6) / 2, y + 1.0, (cw - 0.6) / 2 - 0.1, LANG === "en" ? big : big.replace(",", "."), lab, C.text2, 30));
    txt(s, split, { x: x + 0.3, y: y + 2.45, w: cw - 0.6, h: 0.55, fontSize: 13.5, bold: true, color: C.text1 });
    txt(s, FMT(note), { x: x + 0.3, y: y + 3.0, w: cw - 0.6, h: 0.65, fontSize: 13 });
  });
  // processing order
  const steps = [L("Preprocess (NFC, normalize)", "Tiền xử lý (NFC, chuẩn hoá)"), L("Fix split (seed 42)", "Cố định split (seed 42)"), L("Tune on dev / val", "Chọn cấu hình trên dev / val"), L("Report on test", "Báo cáo trên test")];
  const bw = (CW - 3 * 0.55) / 4, y2 = 5.6;
  steps.forEach((t, i) => {
    const x = M + i * (bw + 0.55);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y2, w: bw, h: 0.75, rectRadius: 0.08, fill: { color: i === 3 ? HEX.ambert : HEX.tealt }, line: { color: i === 3 ? HEX.amber : HEX.teal, width: 1 }, objectName: oid("box") });
    txt(s, t, { x: x + 0.1, y: y2, w: bw - 0.2, h: 0.75, fontSize: 14, bold: true, color: C.text2, align: "center", valign: "middle" });
    if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bw + 0.1, y: y2 + 0.23, w: 0.35, h: 0.3, fill: { color: HEX.teal }, line: { color: HEX.teal, width: 0 }, objectName: oid("arrow") });
  });
  footnote(s, L("Test data is never used to choose a configuration. Train–test overlap is checked in Part B · 02.", "Dữ liệu test không bao giờ dùng để chọn cấu hình. Mức trùng train–test được kiểm ở Phần B · 02."), 6.45, 0.4);
}

// A8. Proposed method & pipeline
{
  const s = content(L("A · Review 1 (revised)", "A · Review 1 (đã sửa)"), L("REVIEW 1 · 07 · PROPOSED METHOD", "REVIEW 1 · 07 · PHƯƠNG PHÁP ĐỀ XUẤT"), L("One proposed method, an 8-step pipeline", "Một phương pháp đề xuất, pipeline 8 bước"),
    "Góp ý số 4: chưa có phương pháp đề xuất và chưa có hình pipeline. Nhóm chốt một phương pháp: QLoRA trên Qwen2.5-VL-3B-Instruct với prompt tiếng Việt có chữ OCR cho ViTextVQA. Mô hình gốc đóng băng ở 4-bit, chỉ train adapter LoRA. Pipeline tám bước từ dữ liệu, tiền xử lý, OCR, chia tập cố định, ghép prompt, train và suy luận, chấm điểm, đến prototype web. Mọi thành phần khác là ablation. Ở Review 2, phương pháp này được hiện thực thành QLoRA nhận biết OCR, B3 ở Phần B mục 06.");
  A.method = here();
  card(s, M, 1.45, CW, 0.85, C.accent5);
  txt(s, [
    { text: L("Proposed method  ", "Phương pháp đề xuất  "), options: { bold: true, color: HEX.ink } },
    { text: L("QLoRA on Qwen2.5-VL-3B-Instruct + Vietnamese prompt with OCR text (ViTextVQA). Base model frozen in 4-bit NF4; only LoRA adapters are trained. Realised as OCR-aware QLoRA (B3) in Part B · 06.",
              "QLoRA trên Qwen2.5-VL-3B-Instruct + prompt tiếng Việt có chữ OCR (ViTextVQA). Model gốc đóng băng 4-bit NF4; chỉ train adapter LoRA. Hiện thực thành QLoRA nhận biết OCR (B3) ở Phần B · 06.") },
  ], { x: M + 0.3, y: 1.45, w: CW - 0.6, h: 0.85, fontSize: 14, valign: "middle" });
  const st = [
    [L("Data", "Dữ liệu"), L("ViVQA + ViTextVQA (official)", "ViVQA + ViTextVQA (chính thức)")],
    [L("Preprocess", "Tiền xử lý"), L("NFC, lowercase, normalize answers; resize", "NFC, chữ thường, chuẩn hoá đáp án; resize")],
    [L("OCR (ViTextVQA)", "OCR (ViTextVQA)"), L("Read text, order top-left → bottom-right, cache once", "Đọc chữ, sắp trái-trên → phải-dưới, cache một lần")],
    [L("Fixed split", "Chia tập cố định"), L("Official splits; ViVQA val 1,200; seed 42", "Split chính thức; val ViVQA 1.200; seed 42")],
    [L("Prompt builder", "Ghép prompt"), L("Image + question (+ OCR line); OCR capped by a token budget", "Ảnh + câu hỏi (+ dòng OCR); OCR giới hạn theo ngân sách token")],
    [L("Qwen2.5-VL-3B 4-bit + LoRA", "Qwen2.5-VL-3B 4-bit + LoRA"), L("Train: loss on answer tokens only; infer: greedy, short", "Train: loss chỉ trên token đáp án; suy luận: greedy, ngắn")],
    [L("Score & analyse", "Chấm & phân tích"), L("EM, VQA-Acc, ANLS (+F1); hallucination; error types V1–V4; log seed, VRAM, time", "EM, VQA-Acc, ANLS (+F1); hallucination; lỗi V1–V4; log seed, VRAM, thời gian")],
    [L("Web prototype", "Prototype web"), L("Upload image, ask in Vietnamese, show answer + OCR text", "Tải ảnh, hỏi tiếng Việt, hiện câu trả lời + chữ OCR")],
  ];
  const bw = (CW - 3 * 0.35) / 4, bh = 1.45;
  st.forEach(([t, d], i) => {
    const r = Math.floor(i / 4), c = i % 4;
    const x = M + c * (bw + 0.35), y = 2.5 + r * (bh + 0.3);
    card(s, x, y, bw, bh, i === 5 ? C.accent4 : C.background2);
    txt(s, `${i + 1} · ${t}`, { x: x + 0.2, y: y + 0.1, w: bw - 0.4, h: 0.45, fontSize: 13.5, bold: true, color: C.text2 });
    txt(s, FMT(d), { x: x + 0.2, y: y + 0.55, w: bw - 0.4, h: bh - 0.6, fontSize: 11.5 });
    if (c < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bw + 0.05, y: y + bh / 2 - 0.12, w: 0.25, h: 0.24, fill: { color: HEX.teal }, line: { color: HEX.teal, width: 0 }, objectName: oid("arrow") });
  });
  footnote(s, L("Steps 1–4 run once and are cached; steps 5–7 are identical for every run (frozen protocol, Part B · 03). Final prompt form: Part B · 06 (one OCR line, same prompt for training and testing).",
                "Bước 1–4 chạy một lần và được cache; bước 5–7 giống hệt cho mọi run (giao thức cố định, Phần B · 03). Dạng prompt cuối: Phần B · 06 (một dòng OCR, cùng prompt cho train và test)."), 6.35, 0.5);
}

// A9. QLoRA + model roles
{
  const s = content(L("A · Review 1 (revised)", "A · Review 1 (đã sửa)"), L("REVIEW 1 · 07 · QLoRA AND MODEL ROLES", "REVIEW 1 · 07 · QLoRA VÀ VAI TRÒ MÔ HÌNH"), L("Why QLoRA only — and which model is used for what", "Vì sao chỉ dùng QLoRA — và mô hình nào dùng để làm gì"),
    "Góp ý số 5 và số 6. QLoRA: trọng số gốc nén 4-bit NF4 và đóng băng; mỗi lớp được gắn hai ma trận nhỏ A và B, chỉ hai ma trận này được train; kết quả là một adapter khoảng 30 MB. Nhóm chỉ dùng QLoRA vì đó là điều kiện để chạy được trên GPU Colab, không phải biến cần so sánh, nên không so LoRA với QLoRA. Góp ý số 6: không so mô hình khác thế hệ hay khác kích thước để kết luận cải tiến; mọi kết luận chỉ trên cùng backbone Qwen2.5-VL-3B. Các mô hình khác chỉ để định vị hoặc làm ablation backbone.");
  A.qlora = here();
  const cw = 5.4;
  card(s, M, 1.5, cw, 2.75);
  txt(s, L("How QLoRA works", "QLoRA hoạt động thế nào"), { x: M + 0.25, y: 1.6, w: cw - 0.5, h: 0.4, fontSize: 15, bold: true, color: C.text2 });
  bullets(s, [L("Base weights quantized to 4-bit NF4 and frozen", "Trọng số gốc nén 4-bit NF4 và đóng băng"),
              L("Each target layer gets two small matrices A·B (rank r); only A and B are trained", "Mỗi lớp đích gắn hai ma trận nhỏ A·B (hạng r); chỉ A và B được train"),
              L("Result: a ~30 MB adapter loaded on top of the base model", "Kết quả: adapter ~30 MB gắn lên model gốc khi suy luận")],
    { x: M + 0.25, y: 2.05, w: cw - 0.5, h: 1.9, fontSize: 13 });
  txt(s, "Hu et al., LoRA, ICLR 2022 · Dettmers et al., QLoRA, NeurIPS 2023", { x: M + 0.25, y: 3.85, w: cw - 0.5, h: 0.3, fontSize: 10, italic: true, color: C.accent3 });
  card(s, M, 4.45, cw, 2.4, C.accent5);
  txt(s, L("Why only QLoRA (no LoRA vs QLoRA study)", "Vì sao chỉ QLoRA (không so LoRA với QLoRA)"), { x: M + 0.25, y: 4.55, w: cw - 0.5, h: 0.4, fontSize: 15, bold: true, color: C.text2 });
  txt(s, L("3B weights take ~7–8 GB in 16-bit before image activations; ~2–3 GB in 4-bit. At Review 1 the GPU was a Colab T4 (16 GB, no bf16); from W7 an L4 (24 GB). QLoRA is the condition for running on this hardware, not a variable — the only comparison used for conclusions is QLoRA vs zero-shot on the same model.",
           "Trọng số 3B ở 16-bit ~7–8 GB chưa tính activation của ảnh; ở 4-bit ~2–3 GB. Lúc Review 1 GPU là Colab T4 (16 GB, không bf16); từ W7 là L4 (24 GB). QLoRA là điều kiện để chạy được, không phải biến so sánh — phép so dùng để kết luận chỉ là QLoRA so với zero-shot trên cùng model."),
      { x: M + 0.25, y: 5.0, w: cw - 0.5, h: 1.8, fontSize: 13 });
  const x = M + cw + 0.4, w = CW - cw - 0.4;
  const rows = [
    [TH(L("Model", "Mô hình")), TH(L("Role", "Vai trò")), TH(L("Used to claim improvement?", "Dùng để kết luận cải tiến?"))],
    ["Qwen2.5-VL-3B-Instruct", L("Reproduced baseline + base of the proposed method", "Baseline tái lập + nền của phương pháp đề xuất"), L("Yes — same model, split, seed, hardware", "Có — cùng model, split, seed, phần cứng")],
    ["Qwen2-VL-7B-Instruct", L("Backbone ablation, zero-shot (Part B · 04)", "Ablation backbone, zero-shot (Phần B · 04)"), L("No — other generation and size", "Không — khác thế hệ và kích thước")],
    ["Florence-2", L("Fallback if the baseline failed (not activated)", "Dự phòng nếu baseline lỗi (không kích hoạt)"), L("No", "Không")],
    ["BLIP-2 · PreSTU · ViTextBLIP-2", L("Prior work — numbers from the papers", "Công trình trước — số từ paper"), L("No — positioning only", "Không — chỉ định vị")],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c, options: { fontSize: 12.5, bold: j === 0 } })));
  table(s, rows, { x, y: 1.5, w, colW: [2.15, 2.55, w - 4.7], rowH: 0.72 });
  card(s, x, 5.35, w, 1.5, C.accent4);
  txt(s, L("Principle: claim an improvement only on the same backbone, size, split, seed and hardware. Comparing models of another generation or size does not measure the method.",
           "Nguyên tắc: chỉ kết luận “cải thiện” khi cùng backbone, kích thước, split, seed và phần cứng. So mô hình khác thế hệ hay kích thước không đo được hiệu quả của phương pháp."),
      { x: x + 0.3, y: 5.35, w: w - 0.6, h: 1.5, fontSize: 13.5, valign: "middle", color: C.text2 });
}

// A10. Token budget + RAG
{
  const s = content(L("A · Review 1 (revised)", "A · Review 1 (đã sửa)"), L("REVIEW 1 · 07 · TOKEN BUDGET & RAG", "REVIEW 1 · 07 · NGÂN SÁCH TOKEN & RAG"), L("Token budget per sample, and where RAG fits", "Ngân sách token mỗi mẫu, và vị trí của RAG"),
    "Góp ý số 8: khuyến nghị RAG và cần tính số token. Mỗi câu hỏi là một lượt độc lập gồm một ảnh và một câu hỏi, không cần bộ nhớ hội thoại. Phần lớn token là token ảnh, tối đa 512 theo cấu hình cố định. Tổng ước tính dưới khoảng một nghìn token mỗi mẫu, nhỏ hơn nhiều so với context window; giới hạn thật là VRAM. Chữ OCR trong prompt đóng vai trò truy xuất bằng chứng từ chính ảnh. RAG với kho tri thức ngoài chỉ làm nếu phân tích lỗi cho thấy nhiều câu sai do thiếu tri thức ngoài ảnh — đến Review 2, duyệt tay lỗi chưa cho thấy điều đó.");
  A.tokens = here();
  const rows = [
    [TH(L("Component", "Thành phần")), TH(L("How it is counted / capped", "Cách tính / giới hạn")), TH(L("Note", "Ghi chú"))],
    [L("Visual tokens", "Token ảnh"), L("≈ (H/28)·(W/28) after resize; capped by max_pixels = 401,408 → ≤ 512", "≈ (cao/28)·(rộng/28) sau resize; chặn bởi max_pixels = 401.408 → ≤ 512"), L("Frozen config (Part B · 03)", "Cấu hình cố định (Phần B · 03)")],
    [L("Question", "Câu hỏi"), L("A few dozen tokens", "Vài chục token"), L("ViTextVQA: 9.59 words on average", "ViTextVQA: trung bình 9,59 từ")],
    [L("OCR text", "Chữ OCR"), L("One line, capped at N tokens (logged per run)", "Một dòng, tối đa N token (log ở mỗi run)"), L("ViTextVQA only", "Chỉ ViTextVQA")],
    [L("Instruction + template", "Chỉ dẫn + template"), L("Fixed", "Cố định"), "—"],
    [L("Output", "Đầu ra"), L("max_new_tokens = 32", "max_new_tokens = 32"), L("Answers: 4.18 words on average", "Đáp án: trung bình 4,18 từ")],
    [L("Total (estimate)", "Tổng (ước tính)"), L("≲ 1,000 tokens per sample", "≲ 1.000 token mỗi mẫu"), L("Far below the context window; the real limit is VRAM", "Nhỏ hơn nhiều context window; giới hạn thật là VRAM")],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: FMT(c), options: { fontSize: 12.5, bold: i === 6 || j === 0 } })));
  table(s, rows, { x: M, y: 1.5, w: 8.0, colW: [2.0, 3.4, 2.6], rowH: 0.62 });
  const x = M + 8.4, w = CW - 8.4;
  card(s, x, 1.5, w, 5.35, C.accent4);
  txt(s, L("RAG — optional, conditional", "RAG — tuỳ chọn, có điều kiện"), { x: x + 0.3, y: 1.65, w: w - 0.6, h: 0.45, fontSize: 16, bold: true, color: C.text2 });
  bullets(s, [L("OCR in the prompt already retrieves evidence from the image itself", "OCR trong prompt đã là “truy xuất bằng chứng” từ chính ảnh"),
              L("External-knowledge RAG (embeddings + FAISS, top-k) only if the error analysis shows many errors need knowledge outside the image", "RAG tri thức ngoài (embedding + FAISS, top-k) chỉ làm nếu phân tích lỗi cho thấy nhiều câu sai do thiếu tri thức ngoài ảnh"),
              L("If triggered: specify knowledge base, embedding model, retrieval and token budget", "Nếu kích hoạt: đặc tả kho tri thức, mô hình embedding, cách truy xuất và ngân sách token"),
              L("Status at Review 2: the 200-error audit (Part B · 05) shows reading/format errors, not missing knowledge → not triggered", "Trạng thái Review 2: duyệt 200 lỗi (Phần B · 05) cho thấy lỗi đọc chữ/định dạng, không phải thiếu tri thức → chưa kích hoạt")],
    { x: x + 0.3, y: 2.15, w: w - 0.6, h: 4.6, fontSize: 13 });
}

// A11. 15-week plan
{
  const s = content(L("A · Review 1 (revised)", "A · Review 1 (đã sửa)"), L("REVIEW 1 · 10 · PLAN", "REVIEW 1 · 10 · KẾ HOẠCH"), L("15-week plan with verifiable outputs", "Kế hoạch 15 tuần với đầu ra kiểm chứng được"),
    "Góp ý số 10: kế hoạch chưa chi tiết. Bảng ghi công việc từng tuần và đầu ra kiểm chứng được. Bốn mốc: Review 1 tuần 3, Review 2 tuần 7, hội đồng cấp Khoa tuần 13, bảo vệ cuối kỳ tuần 15. Tiến độ thực tế và các thay đổi so với kế hoạch được trình bày ở Phần B, slide bước tiếp theo.");
  A.plan = here();
  const p = [
    ["W1", L("Read the proposal, roles, 10 core papers", "Đọc đề cương, phân vai, 10 paper nền"), L("W1 report, 15-week plan", "Báo cáo W1, kế hoạch 15 tuần")],
    ["W2", L("Literature matrix; analyse ViVQA / ViTextVQA", "Literature Matrix; phân tích ViVQA / ViTextVQA"), "Literature_Matrix.md, Dataset_Analysis.md"],
    ["W3", L("RQs, hypotheses, baseline proposal — Review 1", "Câu hỏi NC, giả thuyết, đề xuất baseline — Review 1"), L("Review 1 report + slides", "Báo cáo + slide Review 1")],
    ["W4", L("Colab setup; zero-shot pilot on ViTextVQA", "Môi trường Colab; zero-shot thử trên ViTextVQA"), L("Zero-shot notebook", "Notebook zero-shot")],
    ["W5", L("Official ViTextVQA; NFC preprocessing; fixed split + 2,000-question test subset", "ViTextVQA chính thức; tiền xử lý NFC; cố định split + test subset 2.000 câu"), L("Official data + split log", "Dữ liệu chính thức + log split")],
    ["W6", L("Official zero-shot on both datasets; V1–V4 error types; prior-work table", "Zero-shot chính thức 2 dataset; phân loại lỗi V1–V4; bảng công trình trước"), L("Baseline table; Methodology (Report 4)", "Bảng baseline; Methodology (Report 4)")],
    ["W7", L("Review 2: baseline + proposed method + QLoRA pilot", "Review 2: baseline + phương pháp đề xuất + QLoRA thử"), L("Review 2 slides; pilot run logs", "Slide Review 2; log pilot")],
    ["W8", L("Full QLoRA on ViVQA; log VRAM, time, trained params", "QLoRA đầy đủ trên ViVQA; log VRAM, thời gian, số tham số"), "Adapter v1; experiments/W08_*"],
    ["W9", L("OCR pipeline + Vietnamese prompt; QLoRA on ViTextVQA", "OCR pipeline + prompt tiếng Việt; QLoRA trên ViTextVQA"), L("OCR cache; adapter v2", "OCR cache; adapter v2")],
    ["W10", L("Main runs: zero-shot / prompt / QLoRA / QLoRA + OCR, same split and seed", "Chạy chính: zero-shot / prompt / QLoRA / QLoRA + OCR, cùng split, seed"), L("Main results table", "Bảng kết quả chính")],
    ["W11", L("Full evaluation (EM, VQA-Acc, ANLS, F1); hallucination", "Đánh giá đầy đủ (EM, VQA-Acc, ANLS, F1); hallucination"), L("Metric + hallucination report", "Báo cáo metric + hallucination")],
    ["W12", L("Ablations (OCR, OCR order, rank / target modules); error analysis", "Ablation (OCR, thứ tự OCR, rank / module); phân tích lỗi"), L("Ablation table; failure cases", "Bảng ablation; failure cases")],
    ["W13", L("Faculty-level review", "Hội đồng cấp Khoa"), L("Slides + Report 6–7", "Slide + Report 6–7")],
    ["W14", L("Gradio prototype; write the thesis", "Prototype Gradio; viết thesis"), L("Working demo; thesis draft", "Demo chạy được; bản thesis")],
    ["W15", L("Final defense; reproducibility package", "Bảo vệ cuối kỳ; gói tái lập"), L("Code, adapters, configs, inference script, README, logs", "Code, adapter, config, script suy luận, README, log")],
  ];
  const milestone = { W3: 1, W7: 1, W13: 1, W15: 1 };
  const rows = [[TH(L("Week", "Tuần")), TH(L("Planned work", "Công việc")), TH(L("Verifiable output", "Đầu ra kiểm chứng được"))]]
    .concat(p.map(([w, t, o]) => [w, t, o].map((c, j) => ({ text: c, options: { fontSize: 10.5, bold: j === 0 || !!milestone[w], fill: milestone[w] ? { color: HEX.ambert } : undefined } }))));
  table(s, rows, { x: M, y: 1.4, w: CW, colW: [0.8, 6.4, 4.93], rowH: 0.335, margin: [1, 6, 1, 6] });
}

// A12. Risks + constraints
{
  const s = content(L("A · Review 1 (revised)", "A · Review 1 (đã sửa)"), L("REVIEW 1 · 11–12 · RISKS & CONSTRAINTS", "REVIEW 1 · 11–12 · RỦI RO & RÀNG BUỘC"), L("Risks (status at Review 2) and constraints", "Rủi ro (trạng thái ở Review 2) và ràng buộc"),
    "Ma trận rủi ro từ Review 1, cột trạng thái cập nhật đến Review 2. Phần lớn rủi ro cao đã được khử: model chạy 4-bit trên L4, checkpoint và resume đã kiểm, baseline đã tái lập đúng số công bố, dữ liệu ViTextVQA dùng repo chính thức. Bên phải là các ràng buộc bắt buộc: chỉ PEFT, tái lập một baseline trước, cùng split, seed và phần cứng, và hiệu quả là ràng buộc thiết kế.");
  A.risk = here();
  const H = L("High", "Cao"), Md = L("Medium", "TB");
  const rows = [
    [TH(L("Risk", "Rủi ro")), TH(L("Level", "Mức")), TH(L("Mitigation", "Giảm thiểu")), TH(L("Status at Review 2", "Trạng thái ở Review 2"))],
    [L("Model too large / OOM", "Model quá lớn / OOM"), H, L("QLoRA 4-bit, 3B model", "QLoRA 4-bit, model 3B"), L("Resolved: training ≤ 7 GB VRAM on L4", "Đã khử: train ≤ 7 GB VRAM trên L4")],
    [L("Colab interruption", "Colab bị gián đoạn"), H, L("Drive mount, frequent checkpoints", "Mount Drive, checkpoint thường xuyên"), L("Resolved: resume verified on L4 (W5)", "Đã khử: resume đã kiểm trên L4 (W5)")],
    [L("Baseline hard to reproduce", "Baseline khó tái lập"), H, L("Official repo, fixed seed", "Repo gốc, seed cố định"), L("Resolved: TextVQA 79.6 vs 79.3 reported (R0)", "Đã khử: TextVQA 79,6 so với 79,3 công bố (R0)")],
    [L("Training does not converge", "Train không hội tụ"), H, L("Fall back to prompting / OCR", "Quay về prompting / OCR"), L("Not observed (loss logs, B2, B3)", "Không xảy ra (log loss, B2, B3)")],
    [L("Dataset / licence", "Dữ liệu / licence"), Md, L("Check licence; preprocessing", "Kiểm licence; tiền xử lý"), L("Resolved: official ViTextVQA repo (W5)", "Đã khử: repo ViTextVQA chính thức (W5)")],
    [L("Slow QLoRA training", "QLoRA train chậm"), Md, L("Pilot subsets, gradient accumulation", "Subset thử, gradient accumulation"), L("Managed: fixed 8k subset for B3 (≈ 25% time)", "Đã xử lý: tập con 8k cố định cho B3 (≈ 25% thời gian)")],
    [L("Poor OCR", "OCR kém"), Md, L("Swap PaddleOCR ↔ VietOCR", "Đổi PaddleOCR ↔ VietOCR"), L("Open: OCR errors remain a main error source", "Còn mở: lỗi OCR vẫn là nguồn lỗi chính")],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: FMT(c), options: { fontSize: 11.5, bold: j === 1, color: j === 1 ? (c === H ? HEX.red : HEX.amber) : HEX.text } })));
  table(s, rows, { x: M, y: 1.5, w: 8.6, colW: [2.0, 0.95, 2.45, 3.2], rowH: 0.6 });
  const x = M + 9.0, w = CW - 9.0;
  card(s, x, 1.5, w, 5.35, C.accent4);
  txt(s, L("Hard constraints", "Ràng buộc bắt buộc"), { x: x + 0.25, y: 1.62, w: w - 0.5, h: 0.45, fontSize: 16, bold: true, color: C.text2 });
  bullets(s, [L("PEFT only (LoRA / QLoRA) — no full fine-tuning", "Chỉ PEFT (LoRA / QLoRA) — không full fine-tuning"),
              L("Reproduce one baseline first, then improve", "Tái lập một baseline trước, rồi mới cải tiến"),
              L("Same split, seed and hardware for every comparison", "Cùng split, seed, phần cứng cho mọi so sánh"),
              L("Efficiency is a design constraint: VRAM and time logged every run", "Hiệu quả là ràng buộc thiết kế: log VRAM, thời gian mỗi run"),
              L("Runtime: Google Colab (Pro for training), Drive mounted, GPU logged every run", "Runtime: Google Colab (Pro cho training), mount Drive, log GPU mỗi run")],
    { x: x + 0.25, y: 2.15, w: w - 0.5, h: 4.6, fontSize: 13 });
}

// A13. Deliverables, value, team
{
  const s = content(L("A · Review 1 (revised)", "A · Review 1 (đã sửa)"), L("REVIEW 1 · 08–09–13 · OUTPUTS, VALUE, TEAM", "REVIEW 1 · 08–09–13 · SẢN PHẨM, GIÁ TRỊ, NHÓM"), L("Expected outputs, value and team", "Sản phẩm dự kiến, giá trị và nhóm"),
    "Ba nhóm sản phẩm: một cải tiến PEFT đã được kiểm chứng so với baseline kèm ablation, phân tích lỗi và phân tích hallucination; một prototype web; và luận văn kèm gói tái lập gồm code, adapter, config và log. Giá trị thực tiễn là đọc biển hiệu, nhãn hàng, hỗ trợ người khiếm thị. Ý nghĩa khoa học là nghiên cứu PEFT có hệ thống cho VLM tiếng Việt và đo hallucination cho tiếng Việt.");
  A.outputs = here();
  const items = [
    [L("Research contribution", "Đóng góp nghiên cứu"), L("≥ 1 validated PEFT improvement over the reproduced baseline, with ablation, error and hallucination analysis", "≥ 1 cải tiến PEFT đã kiểm chứng so với baseline tái lập, kèm ablation, phân tích lỗi và hallucination")],
    [L("Web prototype", "Prototype web"), L("Upload a Vietnamese image, ask in Vietnamese, get the answer with the OCR text as evidence", "Tải ảnh, hỏi tiếng Việt, nhận câu trả lời kèm chữ OCR làm bằng chứng")],
    [L("Thesis + reproducibility package", "Luận văn + gói tái lập"), L("Code, LoRA adapters, configs, inference script, experiment logs", "Code, adapter LoRA, config, script suy luận, log thí nghiệm")],
  ];
  const cw = (CW - 2 * 0.35) / 3, y = 1.5, h = 2.0;
  items.forEach(([t, d], i) => {
    const x = M + i * (cw + 0.35);
    card(s, x, y, cw, h, i === 0 ? C.accent5 : C.background2);
    txt(s, t, { x: x + 0.25, y: y + 0.15, w: cw - 0.5, h: 0.45, fontSize: 16, bold: true, color: C.text2 });
    txt(s, d, { x: x + 0.25, y: y + 0.65, w: cw - 0.5, h: h - 0.75, fontSize: 13 });
  });
  const y2 = 3.75, cw2 = (CW - 0.4) / 2, h2 = 1.75;
  card(s, M, y2, cw2, h2);
  txt(s, L("Practical value", "Giá trị thực tiễn"), { x: M + 0.25, y: y2 + 0.1, w: cw2 - 0.5, h: 0.4, fontSize: 14, bold: true, color: C.text2 });
  bullets(s, [L("Read signs, labels and prices in Vietnamese photos", "Đọc biển hiệu, nhãn hàng, giá cả trong ảnh tiếng Việt"), L("Assistive technology for visually impaired users", "Công nghệ hỗ trợ người khiếm thị"), L("Menus, forms and documents in Vietnamese", "Thực đơn, biểu mẫu, tài liệu tiếng Việt")],
    { x: M + 0.25, y: y2 + 0.5, w: cw2 - 0.5, h: h2 - 0.55, fontSize: 12.5, paraSpaceAfter: 2 });
  card(s, M + cw2 + 0.4, y2, cw2, h2, C.accent4);
  txt(s, L("Scientific value", "Ý nghĩa khoa học"), { x: M + cw2 + 0.65, y: y2 + 0.1, w: cw2 - 0.5, h: 0.4, fontSize: 14, bold: true, color: C.text2 });
  bullets(s, [L("Systematic PEFT study for a Vietnamese VLM (G3)", "Nghiên cứu PEFT có hệ thống cho VLM tiếng Việt (G3)"), L("Vietnamese-specific handling: tone marks, output language, OCR order (V1–V3)", "Xử lý đặc thù tiếng Việt: dấu, ngôn ngữ đầu ra, thứ tự OCR (V1–V3)"), L("Hallucination measurement for Vietnamese VQA (G4); reproducible protocol", "Đo hallucination cho VQA tiếng Việt (G4); quy trình tái lập được")],
    { x: M + cw2 + 0.65, y: y2 + 0.5, w: cw2 - 0.5, h: h2 - 0.55, fontSize: 12.5, paraSpaceAfter: 2 });
  const team = [["SE182670 · Phạm Lê Huy Hoàng", L("Leader · PEFT / training", "Trưởng nhóm · PEFT / training")], ["SE182458 · Văn Quang Trường", L("Evaluation / deployment", "Đánh giá / triển khai")],
                ["SE193642 · Trần Quốc Huân", L("Dataset / OCR", "Dữ liệu / OCR")], ["SE183249 · Hồ Đắc Toàn", L("VLM / baseline", "VLM / baseline")]];
  const tw = (CW - 3 * 0.25) / 4, y3 = 5.75;
  team.forEach(([n, r], i) => {
    const x = M + i * (tw + 0.25);
    card(s, x, y3, tw, 1.1);
    txt(s, n, { x: x + 0.2, y: y3 + 0.12, w: tw - 0.4, h: 0.45, fontSize: 12.5, bold: true, color: C.text2 });
    txt(s, r, { x: x + 0.2, y: y3 + 0.58, w: tw - 0.4, h: 0.4, fontSize: 12 });
  });
}

// fill the feedback table (A0) now that slide numbers are known
{
  const n = (k) => String(A[k]);
  const rows = [
    [TH("#"), TH(L("Reviewer comment (24/09)", "Góp ý của phản biện (24/09)")), TH(L("What changed", "Nhóm đã sửa")), TH(L("Slide", "Slide"))],
    ["1", L("Problem unclear; input/output unclear; how Vietnamese differs from English", "Bài toán chưa rõ; chưa rõ input/output; tiếng Việt khác tiếng Anh ở đâu"), L("One-sentence task; input/output table; four Vietnamese problems V1–V4", "Bài toán một câu; bảng input/output; bốn vấn đề tiếng Việt V1–V4"), `${n("problem")}–${n("viet")}`],
    ["2", L("Punctuation, spaces, connectives are only preprocessing", "Dấu câu, khoảng trắng, từ nối chỉ là tiền xử lý"), L("Preprocessing separated from the research problem; diacritics-removal evidence cited", "Tách tiền xử lý khỏi vấn đề nghiên cứu; dẫn thí nghiệm bỏ dấu"), n("viet")],
    ["3", L("4-bit/16-bit and VRAM hypotheses do not address Vietnamese", "Giả thuyết 4-bit/16-bit, VRAM chưa giải quyết vấn đề tiếng Việt"), L("Reduced to H1–H4, each tied to V1–V4; memory moved to measurements", "Rút còn H1–H4, mỗi cái gắn V1–V4; bộ nhớ chuyển sang số đo"), n("hyp")],
    ["4", L("No proposed method; no pipeline figure", "Chưa có phương pháp đề xuất; chưa có hình pipeline"), L("One method fixed; 8-step pipeline; prompt template", "Chốt một phương pháp; pipeline 8 bước; mẫu prompt"), n("method")],
    ["5", L("Explain QLoRA; why both LoRA and QLoRA", "Giải thích QLoRA; vì sao đưa cả LoRA và QLoRA"), L("QLoRA explained; QLoRA only, because of the hardware", "Giải thích QLoRA; chỉ dùng QLoRA vì ràng buộc phần cứng"), n("qlora")],
    ["6", L("Comparing models of different generation / size", "So mô hình khác thế hệ, khác kích thước"), L("Model-role table; conclusions only on the same backbone", "Bảng vai trò mô hình; chỉ kết luận trên cùng backbone"), n("qlora")],
    ["7", L("Baselines = prior work; need 3 baselines and Vietnamese SOTA", "Baseline là công trình trước; cần 3 baseline và SOTA tiếng Việt"), L("Prior-work tables for ViTextVQA and ViVQA, with caveats", "Bảng công trình trước cho ViTextVQA và ViVQA, kèm lưu ý"), n("prior")],
    ["8", L("Consider RAG; count tokens", "Khuyến nghị RAG; cần tính số token"), L("Token budget per sample; external RAG optional and conditional", "Ngân sách token mỗi mẫu; RAG ngoài là tuỳ chọn có điều kiện"), n("tokens")],
    ["9", L("Preprocess before splitting", "Tiền xử lý trước khi chia tập"), L("Preprocess → fixed split; tune on dev, report on test", "Tiền xử lý → split cố định; chỉnh trên dev, báo cáo trên test"), `${n("data")}, ${n("method")}`],
    ["10", L("Plan not detailed", "Kế hoạch chưa chi tiết"), L("Week-by-week table with verifiable outputs", "Bảng công việc theo tuần với đầu ra kiểm chứng được"), n("plan")],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c, options: { fontSize: 11.5, bold: j === 0 || j === 3 } })));
  table(fbSlide, rows, { x: M, y: 1.45, w: CW, colW: [0.45, 5.0, 5.48, 1.2], rowH: 0.47 });
}

// ================= PART B — REVIEW 2 =================
// Slide numbers inside Part B are computed from OFF (number of slides before Part B), so Part A can grow without breaking references.
const OFF = pres.slides.length - 1;            // original R2 slide k  ->  k + OFF
const B = (k) => String(k + OFF);
const partBStart = pres.slides.length + 1;
// 2. Summary
pres.addSection({ title: L("B · Summary", "B · Tóm tắt") });
{
  const s = content(L("B · Summary", "B · Tóm tắt"), L("REVIEW 2 · SUMMARY", "REVIEW 2 · TÓM TẮT"), L("Four results of this milestone", "Bốn kết quả chính của mốc này"),
    "Bốn con số cần nhớ. Một: pipeline đánh giá của nhóm tái lập đúng số công bố của Qwen2.5-VL-3B trên TextVQA (79,6 so với 79,3), nên các số tiếng Việt phía sau đáng tin. Hai: QLoRA chỉ train 0,2% tham số nhưng tăng EM rất mạnh trên cả hai bộ dữ liệu, có kiểm định thống kê. Ba: trên toàn bộ test ViTextVQA, B2 vượt SOTA của paper ViTextBLIP-2. Bốn: phương pháp đề xuất, QLoRA nhận biết OCR, tăng thêm 2,5 điểm EM so với cùng QLoRA trên cùng dữ liệu, ổn định qua 3 seed và trên full test.");
  const cw = (CW - 3 * 0.35) / 4, y = 1.75, h = 2.9;
  const items = [
    ["79.6", L("Pipeline check — TextVQA val, Qwen2.5-VL-3B; the official report gives 79.3", "Kiểm chứng pipeline — TextVQA val, Qwen2.5-VL-3B; báo cáo chính thức ghi 79,3"), C.accent1],
    ["+21.1", L("EM from QLoRA over zero-shot, full ViTextVQA test (+51.7 on ViVQA); p ≈ 0", "EM nhờ QLoRA so với zero-shot, full test ViTextVQA (+51,7 ở ViVQA); p ≈ 0"), C.text2],
    ["50.3", L("EM on the full ViTextVQA test vs 25.5 for the published SOTA ViTextBLIP-2", "EM trên full test ViTextVQA so với 25,5 của SOTA công bố ViTextBLIP-2"), C.text2],
    ["+2.5", L("EM from OCR-aware QLoRA vs the same QLoRA without OCR (full test, p = 10⁻¹⁴; stable over 3 seeds)", "EM nhờ QLoRA nhận biết OCR so với cùng QLoRA không OCR (full test, p = 10⁻¹⁴; ổn định qua 3 seed)"), C.accent2],
  ];
  items.forEach(([big, label, col], i) => {
    const x = M + i * (cw + 0.35);
    card(s, x, y, cw, h, i === 3 ? C.accent5 : C.background2);
    stat(s, x + 0.3, y + 0.45, cw - 0.6, FMT(big), label, col, 36);
  });
  card(s, M, 5.0, CW, 1.05, C.accent4);
  txt(s, L("Efficient by design: QLoRA trains 7.37 M parameters (0.196%), a 29.5 MB adapter, on a single NVIDIA L4 GPU — no full fine-tuning.",
           "Hiệu quả theo thiết kế: QLoRA chỉ train 7,37 triệu tham số (0,196%), adapter 29,5 MB, trên một GPU NVIDIA L4 — không full fine-tuning."),
      { x: M + 0.35, y: 5.0, w: CW - 0.7, h: 1.05, fontSize: 16, valign: "middle", color: C.text2 });
}

// 3. Feedback
pres.addSection({ title: L("B · Feedback", "B · Góp ý") });
{
  const s = content(L("B · Feedback", "B · Góp ý"), L("REVIEW 2 · PART 01 · RESPONSE TO FEEDBACK", "REVIEW 2 · PHẦN 01 · XỬ LÝ GÓP Ý"), L("Advisor feedback (07/10) and what we did", "Góp ý của GVHD (07/10) và việc nhóm đã làm"),
    "Sau buổi họp ngày 07/10, thầy Trí yêu cầu năm việc trước khi bàn cải tiến. Bảng này cho thấy mỗi yêu cầu được xử lý ở slide nào trong Phần B. Mười góp ý của phản biện Review 1 đã được sửa ở Phần A, xem slide tiếp thu góp ý ở đầu Phần A.");
  const rows = [
    [TH(L("Feedback", "Góp ý")), TH(L("Action", "Việc đã làm")), TH(L("Slide", "Slide"))],
    [L("No baseline architecture diagram", "Chưa có sơ đồ kiến trúc baseline"), L("Component-level diagram of Qwen2.5-VL-3B with LoRA placement (Fig. 7A)", "Sơ đồ từng thành phần của Qwen2.5-VL-3B và vị trí gắn LoRA (Hình 7A)"), B(5)],
    [L("Explain the dataset and its noise", "Giải thích dataset và nhiễu"), L("ViVQA build pipeline + noise points N1–N4 (Fig. 6A); train–test leakage check; 400-sample manual audit", "Quy trình tạo ViVQA + điểm nhiễu N1–N4 (Hình 6A); kiểm tra rò rỉ train–test; duyệt tay 400 mẫu"), `${B(4)}, ${B(14)}`],
    [L("Re-run the baseline with a stated evaluation method", "Chạy lại baseline với phương pháp đánh giá rõ ràng"), L("Frozen protocol (Fig. 7B); pipeline validated on a published number (R0); zero-shot B1 on both datasets", "Giao thức cố định (Hình 7B); kiểm chứng pipeline bằng số đã công bố (R0); B1 zero-shot trên 2 dataset"), `${B(6)}–${B(8)}`],
    [L("Know the SOTA", "Nắm SOTA"), L("ViTextBLIP-2 compared on the full test; Table 7 LMM row checked with Qwen2-VL-7B (R1)", "So với ViTextBLIP-2 trên full test; kiểm dòng LMM ở Table 7 bằng Qwen2-VL-7B (R1)"), `${B(9)}, ${B(12)}–${B(13)}`],
    [L("Run experiments before discussing improvements", "Chạy thí nghiệm rồi mới bàn cải tiến"), L("QLoRA (B2) + three controls; then the proposed OCR-aware QLoRA (B3) with ablations", "QLoRA (B2) + ba đối chứng; sau đó phương pháp đề xuất QLoRA nhận biết OCR (B3) kèm ablation"), `${B(10)}–${B(18)}`],
  ].map((r, i) => i === 0 ? r : r.map((c) => ({ text: c, options: { fontSize: 14 } })));
  table(s, rows, { x: M, y: 1.65, w: CW, colW: [3.6, 7.3, 1.23], rowH: 0.62 });
}

// 4. Dataset & noise
pres.addSection({ title: L("B · Data & baseline", "B · Dữ liệu & baseline") });
{
  const s = content(L("B · Data & baseline", "B · Dữ liệu & baseline"), L("REVIEW 2 · PART 02 · DATASET UNDERSTANDING", "REVIEW 2 · PHẦN 02 · HIỂU DỮ LIỆU"), L("ViVQA is machine-translated — noise is built in", "ViVQA được dịch máy — nhiễu nằm sẵn trong dữ liệu"),
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
  const s = content(L("B · Data & baseline", "B · Dữ liệu & baseline"), L("REVIEW 2 · PART 03 · BASELINE ARCHITECTURE", "REVIEW 2 · PHẦN 03 · KIẾN TRÚC BASELINE"), "Qwen2.5-VL-3B-Instruct",
    "Đây là sơ đồ baseline mà thầy yêu cầu. Ảnh đi qua vision encoder 32 khối, được gộp còn 256 đến 512 token, rồi ghép với câu hỏi đưa vào decoder 36 lớp. Toàn bộ trọng số gốc đóng băng và lượng tử 4-bit. Phần duy nhất được train là LoRA ở 4 phép chiếu q, k, v, o của self-attention — ô màu cam. Khung bên phải tóm tắt đầu vào và đầu ra của từng khối: ảnh được resize tối đa 401.408 điểm ảnh rồi cắt thành patch 14 nhân 14; patch merger gộp 2 nhân 2 patch thành 1 token 2048 chiều, ra 256 đến 512 token ảnh; prompt P0 ghép token ảnh với câu hỏi theo chat template; decoder sinh đáp án tham lam, tối đa 32 token. Lưu ý: baseline KHÔNG dùng OCR — OCR là phần cải tiến B3 ở phần 06.");
  const f = fig(s, "a7", M, 1.45, 8.3, 5.45);
  const x = M + f.w + 0.4, w = CW - f.w - 0.4;
  card(s, x, 1.45, w, 1.85, C.accent5);
  txt(s, L("Trained (QLoRA)", "Được train (QLoRA)"), { x: x + 0.25, y: 1.55, w: w - 0.5, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  bullets(s, [L("LoRA r = 16, α = 32 on q, k, v, o", "LoRA r = 16, α = 32 trên q, k, v, o"),
              L("7,372,800 params = 0.196% · 29.5 MB", "7.372.800 tham số = 0,196% · 29,5 MB"),
              L("Everything else frozen, LLM in 4-bit NF4", "Còn lại đóng băng, LLM 4-bit NF4")], { x: x + 0.25, y: 1.95, w: w - 0.5, h: 1.3, fontSize: 12, paraSpaceAfter: 2 });
  card(s, x, 3.45, w, 3.4);
  txt(s, L("Data flow (in → out)", "Luồng dữ liệu (vào → ra)"), { x: x + 0.25, y: 3.55, w: w - 0.5, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  const flow = [
    [L("1 · Image → ViT: ", "1 · Ảnh → ViT: "), L("≤ 401,408 px → 14×14 patches → 32 blocks", "≤ 401.408 px → patch 14×14 → 32 khối")],
    [L("2 · Patch merger: ", "2 · Patch merger: "), L("2×2 patches → 1 token (2048-d) → 256–512 visual tokens", "2×2 patch → 1 token (2048 chiều) → 256–512 token ảnh")],
    [L("3 · Prompt P0: ", "3 · Prompt P0: "), L("chat template + visual tokens + Vietnamese question", "chat template + token ảnh + câu hỏi tiếng Việt")],
    [L("4 · Decoder (+LoRA): ", "4 · Decoder (+LoRA): "), L("greedy, ≤ 32 new tokens → normalize → EM / ANLS", "sinh tham lam, ≤ 32 token → chuẩn hoá → EM / ANLS")],
  ];
  const runs = [];
  flow.forEach(([a, b], i) => {
    runs.push({ text: a, options: { bold: true, color: HEX.ink } });
    runs.push({ text: b, options: { breakLine: true } });
    if (i < flow.length - 1) runs.push({ text: " ", options: { fontSize: 3, breakLine: true } });
  });
  txt(s, runs, { x: x + 0.25, y: 3.98, w: w - 0.5, h: 2.3, fontSize: 11.5 });
  txt(s, L("No OCR in the baseline — OCR is the B3 improvement (Part 06).", "Baseline không dùng OCR — OCR là cải tiến B3 (Phần 06)."),
      { x: x + 0.25, y: 6.3, w: w - 0.5, h: 0.45, fontSize: 11, italic: true, color: HEX.teal });
}

// 6. Evaluation protocol
{
  const s = content(L("B · Data & baseline", "B · Dữ liệu & baseline"), L("REVIEW 2 · PART 03 · EVALUATION PROTOCOL", "REVIEW 2 · PHẦN 03 · GIAO THỨC ĐÁNH GIÁ"), L("Every run follows one frozen protocol", "Mọi run theo một giao thức cố định"),
    "Mọi run dùng chung một giao thức: split và seed cố định, cùng prompt, cấu hình khoá trong file config, cùng hàm chuẩn hoá và metric, và log đầy đủ commit, GPU, thời gian. So sánh hai run luôn dùng kiểm định ghép cặp McNemar cho EM và bootstrap cho ANLS. Luật kết luận được chốt trước khi chạy để tránh chọn kết quả có lợi. Dòng chân trang giải thích các từ viết tắt.");
  const f = fig(s, "b7", M, 1.45, CW, 2.4);
  const y = 1.45 + f.h + 0.3, cw = (CW - 2 * 0.4) / 3, h = 6.3 - y;
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
  footnote(s, L("EM = Exact Match · VQA-Acc = VQA Accuracy (min(#matching gold / 3, 1); = EM with one gold answer) · ANLS = Average Normalized Levenshtein Similarity (threshold 0.5) · "
              + "F1 = token-level F1 · bf16 = bfloat16 · NF4 = 4-bit NormalFloat (QLoRA) · CI95 = 95% confidence interval · McNemar = paired test on questions where exactly one run is correct",
                "EM = Exact Match (khớp tuyệt đối) · VQA-Acc = VQA Accuracy (min(số gold khớp / 3, 1); = EM khi chỉ có 1 gold) · ANLS = Average Normalized Levenshtein Similarity (ngưỡng 0,5) · "
              + "F1 = F1 theo token · bf16 = bfloat16 · NF4 = 4-bit NormalFloat (QLoRA) · CI95 = khoảng tin cậy 95% · McNemar = kiểm định ghép cặp trên các câu chỉ một run đúng"), 6.38, 0.6);
}

// 7. Pipeline validation
pres.addSection({ title: L("B · Results", "B · Kết quả") });
{
  const s = content(L("B · Results", "B · Kết quả"), L("REVIEW 2 · PART 04 · PIPELINE VALIDATION (R0)", "REVIEW 2 · PHẦN 04 · KIỂM CHỨNG PIPELINE (R0)"), L("Our pipeline reproduces the published number", "Pipeline của nhóm tái lập được số đã công bố"),
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
              L("Keeps 94.5% of the accuracy — reported as an efficiency measurement, not a hypothesis (Part A)", "Giữ 94,5% độ chính xác — báo cáo như số đo hiệu quả, không phải giả thuyết (Phần A)")], { x: x + 0.3, y: 4.75, w: w - 0.6, h: 1.9, fontSize: 15 });
}

// 8. Zero-shot B1
{
  const s = content(L("B · Results", "B · Kết quả"), L("REVIEW 2 · PART 04 · ZERO-SHOT BASELINE (B1)", "REVIEW 2 · PHẦN 04 · BASELINE ZERO-SHOT (B1)"), L("Zero-shot fails mostly on answer format", "Zero-shot sai chủ yếu ở định dạng câu trả lời"),
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
  const s = content(L("B · Results", "B · Kết quả"), L("REVIEW 2 · PART 04 · BACKBONE ABLATION (R1)", "REVIEW 2 · PHẦN 04 · ABLATION BACKBONE (R1)"), L("A larger model is not better here", "Model lớn hơn không tốt hơn ở bài này"),
    "Table 7 của paper ViTextVQA có dòng QwenVL-7b đạt EM 28 trên 100 câu. Nhóm chạy Qwen2-VL-7B trong đúng điều kiện của baseline. Kết quả 7B kém 3B 12,6 điểm EM, p rất nhỏ, chậm hơn và tốn VRAM hơn. Nguyên nhân chính là 7B trả lời dài, trung bình 8,2 từ. Table 7 chỉ có 100 câu nên sai số khoảng ±9 điểm, không đủ để chọn giữa 3B và 7B. Vì vậy nhóm giữ Qwen2.5-VL-3B. Nguồn của từng model và paper ghi ở chân trang, kèm thư mục run trong repo.");
  const fmt = FMT;
  const rows = [
    [TH(L("Run (ViTextVQA, 2,000 q.)", "Run (ViTextVQA, 2.000 câu)")), TH("EM"), TH("F1"), TH(L("words", "số từ")), TH(L("s / q", "s / câu")), TH("VRAM GB")],
    [L("B1 · Qwen2.5-VL-3B [1] · 4-bit · 512 tok.", "B1 · Qwen2.5-VL-3B [1] · 4-bit · 512 token"), "28.8", "53.8", "5.5", "0.86", "≈ 2.5"],
    [L("R1b · Qwen2-VL-7B [2] · 4-bit · 512 tok.", "R1b · Qwen2-VL-7B [2] · 4-bit · 512 token"), "16.2", "41.3", "8.0", "1.02", "6.2"],
    [L("R1 · Qwen2-VL-7B [2] · bf16 · full res.", "R1 · Qwen2-VL-7B [2] · bf16 · đủ độ phân giải"), "16.7", "43.5", "8.2", "2.85", "20.3"],
    [L("ViTextVQA paper [3] Table 7 · “QwenVL-7b” (100 q.)", "Paper ViTextVQA [3] Table 7 · “QwenVL-7b” (100 câu)"), "28.0", "53.8", "–", "–", "–"],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: fmt(c), options: { fontSize: 14, align: j ? "center" : "left", bold: i === 1 } })));
  table(s, rows, { x: M, y: 1.55, w: CW, colW: [5.03, 1.2, 1.2, 1.4, 1.4, 1.9], rowH: 0.52 });
  const y = 4.4, cw = (CW - 2 * 0.4) / 3, h = 1.85;
  [[L("7B < 3B under identical conditions", "7B < 3B trong cùng điều kiện"), L("−12.6 EM, McNemar p ≈ 10⁻³³; higher resolution does not help (R1 = R1b, p = 0.45)", "−12,6 EM, McNemar p ≈ 10⁻³³; tăng độ phân giải không giúp (R1 = R1b, p = 0,45)")],
   [L("Why: instruction following", "Vì sao: tuân thủ chỉ dẫn"), L("7B writes 8.2 words per answer vs gold 3.7 — EM fails even when the text is read correctly", "7B trả lời 8,2 từ so với gold 3,7 — EM trượt dù đọc đúng chữ")],
   [L("Table 7 cannot decide", "Table 7 không đủ để quyết định"), L("n = 100 → ±9 EM; B1 is already consistent with it. Decision: keep Qwen2.5-VL-3B", "n = 100 → ±9 EM; B1 đã nhất quán với nó. Quyết định: giữ Qwen2.5-VL-3B")]]
    .forEach(([t, b], i) => {
      const x = M + i * (cw + 0.4);
      card(s, x, y, cw, h, i === 2 ? C.accent4 : C.background2);
      txt(s, t, { x: x + 0.25, y: y + 0.15, w: cw - 0.5, h: 0.45, fontSize: 16, bold: true, color: C.text2 });
      txt(s, b, { x: x + 0.25, y: y + 0.65, w: cw - 0.5, h: h - 0.8, fontSize: 14 });
    });
  footnote(s, L("[1] Bai et al., Qwen2.5-VL Technical Report, arXiv:2502.13923, 2025 · [2] Wang et al., Qwen2-VL, arXiv:2409.12191, 2024 · "
              + "[3] Nguyen et al., ViTextVQA, arXiv:2404.10652, 2024 (ext. in Expert Systems with Applications, 2025). "
              + "Our runs: experiments/W07_B1_zeroshot_…_test2000, W07_R1_…, W07_R1b_… (same 2,000 questions, seed 42).",
                "[1] Bai và cs., Qwen2.5-VL Technical Report, arXiv:2502.13923, 2025 · [2] Wang và cs., Qwen2-VL, arXiv:2409.12191, 2024 · "
              + "[3] Nguyen và cs., ViTextVQA, arXiv:2404.10652, 2024 (bản mở rộng: Expert Systems with Applications, 2025). "
              + "Run của nhóm: experiments/W07_B1_zeroshot_…_test2000, W07_R1_…, W07_R1b_… (cùng 2.000 câu, seed 42)."), 6.35, 0.62);
}

// 10. QLoRA B2
pres.addSection({ title: L("B · Improvement", "B · Cải tiến") });
{
  const s = content(L("B · Improvement", "B · Cải tiến"), L("REVIEW 2 · PART 05 · QLoRA (B2)", "REVIEW 2 · PHẦN 05 · QLoRA (B2)"), L("QLoRA, one epoch: large and significant gains", "QLoRA một epoch: tăng lớn và có ý nghĩa thống kê"),
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
  const s = content(L("B · Improvement", "B · Cải tiến"), L("REVIEW 2 · PART 05 · CONTROLS", "REVIEW 2 · PHẦN 05 · ĐỐI CHỨNG"), L("The gain is real, not leakage and not just format", "Mức tăng là thật, không do rò rỉ, không chỉ do định dạng"),
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
  const s = content(L("B · Improvement", "B · Cải tiến"), L("REVIEW 2 · PART 05 · COMPARISON WITH SOTA", "REVIEW 2 · PHẦN 05 · SO VỚI SOTA"), L("Full ViTextVQA test: above the published SOTA", "Full test ViTextVQA: cao hơn SOTA đã công bố"),
    "Trên toàn bộ 10.028 câu test, B2 đạt EM 50,3 và F1 71,2, cao hơn ViTextBLIP-2 là SOTA của paper, với cận dưới khoảng tin cậy vẫn cao hơn số công bố. B1 zero-shot chỉ cao hơn về EM, còn F1 thì ngang. Chạy lại trên 2.000 câu chung cho kết quả trùng 100%. Giới hạn: paper không công bố cách chuẩn hoá đáp án.");
  const fmt = FMT;
  const rows = [
    [TH(L("Model (full test, 10,028 q.)", "Model (full test, 10.028 câu)")), TH(L("EM [CI95]", "EM [CI95]")), TH(L("F1 [CI95]", "F1 [CI95]")), TH(L("vs SOTA (rule: CI95 lower bound > published)", "so với SOTA (luật: cận dưới CI95 > số công bố)"))],
    [L("ViTextBLIP-2 · Nguyen et al. [3], Table 6 (trained)", "ViTextBLIP-2 · Nguyen và cs. [3], Table 6 (có train)"), "25.48", "53.95", "–"],
    [L("B1 · Qwen2.5-VL-3B zero-shot", "B1 · Qwen2.5-VL-3B zero-shot"), "29.12 [28.23, 29.99]", "53.89 [53.11, 54.66]", L("higher on EM; F1 indistinguishable", "cao hơn về EM; F1 ngang")],
    [L("B2 · + QLoRA, 1 epoch", "B2 · + QLoRA, 1 epoch"), "50.26 [49.27, 51.23]", "71.16 [70.42, 71.89]", L("higher on EM and F1", "cao hơn cả EM và F1")],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: fmt(c).replace(/\[(\d+,\d+), (\d)/, "[$1; $2"), options: { fontSize: 14, align: j && j < 3 ? "center" : "left", bold: i === 3, color: i === 3 ? HEX.ink : HEX.text, fill: i === 3 ? { color: HEX.tealt } : undefined } })));
  table(s, rows, { x: M, y: 1.55, w: CW, colW: [4.0, 2.55, 2.55, 3.03], rowH: 0.62 });
  const y = 4.25, cw = (CW - 2 * 0.4) / 3, h = 1.85;
  [[L("Reproducible", "Tái lập được"), L("Full-test predictions match the earlier subset runs on all 2,000 shared questions (100%).", "Dự đoán full test trùng 100% với các run tập con trên 2.000 câu chung.")],
   [L("Fair in kind", "Công bằng về loại"), L("Both B2 and ViTextBLIP-2 are trained on the ViTextVQA train split.", "Cả B2 và ViTextBLIP-2 đều được train trên tập train ViTextVQA.")],
   [L("Limitation", "Giới hạn"), L("The paper does not publish its answer normalization; the comparison assumes equivalent scoring.", "Paper không công bố cách chuẩn hoá đáp án; phép so giả định cách chấm tương đương.")]]
    .forEach(([t, b], i) => {
      const x = M + i * (cw + 0.4);
      card(s, x, y, cw, h, i === 2 ? C.accent5 : C.background2);
      txt(s, t, { x: x + 0.25, y: y + 0.15, w: cw - 0.5, h: 0.45, fontSize: 16, bold: true, color: C.text2 });
      txt(s, b, { x: x + 0.25, y: y + 0.65, w: cw - 0.5, h: h - 0.8, fontSize: 14 });
    });
  footnote(s, L("[3] Nguyen et al., ViTextVQA, arXiv:2404.10652, 2024. CI95 = paired bootstrap, 10,000 resamples, seed 42. "
              + "Our numbers: experiments/W07_B1_zeroshot_…_testfull, W07_B2_eval_…_testfull, W07_fulltest_summary_L4_seed42.",
                "[3] Nguyen và cs., ViTextVQA, arXiv:2404.10652, 2024. CI95 = bootstrap ghép cặp, 10.000 lần lấy mẫu, seed 42. "
              + "Số của nhóm: experiments/W07_B1_zeroshot_…_testfull, W07_B2_eval_…_testfull, W07_fulltest_summary_L4_seed42."), 6.3, 0.5);
}

// 12b. How the SOTA works vs. our approach (added after internal review 18/10)
{
  const s = content(L("B · Improvement", "B · Cải tiến"), L("REVIEW 2 · PART 05 · SOTA APPROACH VS OURS", "REVIEW 2 · PHẦN 05 · CÁCH LÀM CỦA SOTA SO VỚI NHÓM"),
    L("How ViTextBLIP-2 works — and how we differ", "ViTextBLIP-2 làm thế nào — và nhóm khác ở đâu"),
    "Đây là quy trình của SOTA. ViTextBLIP-2 giữ đóng băng một Vision Transformer, một hệ OCR là SwinTextSpotter dùng nguyên không fine-tune, và mô hình ngôn ngữ tiếng Việt ViT5; phần duy nhất được train là Q-Former khởi tạo từ BLIP-2, có nhiệm vụ trộn đặc trưng ảnh, câu hỏi và chữ OCR. Paper cũng chỉ ra sắp xếp chữ OCR từ trái-trên xuống phải-dưới giúp hầu hết các model. Nhóm khác ở ba điểm. Một: dùng một VLM đã được pretrain sẵn khả năng đọc chữ, Qwen2.5-VL-3B, lượng tử 4-bit. Hai: chỉ gắn LoRA, 0,2% tham số. Ba: B2 không dùng OCR ngoài — model tự đọc chữ từ ảnh; OCR được đưa vào ở B3, phần 06, dưới dạng một dòng văn bản trong prompt. Về đánh giá, paper báo EM và F1 nhưng không công bố cách chuẩn hoá; nhóm báo thêm ANLS và công khai hàm chuẩn hoá. Một lưu ý khi đọc paper: con số 25,48 / 53,95 nằm ở cột sắp xếp theo độ tin cậy OCR của Table 10, còn bảng ablation lại báo 24,83 / 52,51 cho model đầy đủ — paper không giải thích chênh lệch này.");
  const hd = [TH(L("Aspect", "Khía cạnh")), TH(L("ViTextBLIP-2 (SOTA, paper [3])", "ViTextBLIP-2 (SOTA, paper [3])")), TH(L("Ours: B2 = Qwen2.5-VL-3B + QLoRA", "Nhóm: B2 = Qwen2.5-VL-3B + QLoRA"))];
  const body = [
    [L("Vision", "Thị giác"), L("Frozen pre-trained ViT", "ViT pretrained, đóng băng"), L("Qwen2.5-VL ViT (32 blocks), frozen", "ViT của Qwen2.5-VL (32 khối), đóng băng")],
    [L("Scene text (OCR)", "Chữ trong ảnh (OCR)"), L("SwinTextSpotter [4], not fine-tuned, Vietnamese charset; OCR tokens fused as features", "SwinTextSpotter [4], không fine-tune, bộ ký tự tiếng Việt; token OCR trộn như đặc trưng"),
                                                L("No external OCR — the VLM reads text from pixels (OCR added in B3, Part 06)", "Không dùng OCR ngoài — VLM tự đọc chữ từ ảnh (OCR thêm ở B3, Phần 06)")],
    [L("Language model", "Mô hình ngôn ngữ"), L("ViT5 [5] (Vietnamese T5), frozen", "ViT5 [5] (T5 tiếng Việt), đóng băng"), L("Qwen2.5-3B decoder, 36 layers, 4-bit NF4", "Decoder Qwen2.5-3B, 36 lớp, 4-bit NF4")],
    [L("What is trained", "Phần được train"), L("Q-Former (initialized from BLIP-2) fusing image, question and OCR", "Q-Former (khởi tạo từ BLIP-2) trộn ảnh, câu hỏi và OCR"), L("LoRA on q, k, v, o: 7.37 M params (0.196%)", "LoRA trên q, k, v, o: 7,37 triệu tham số (0,196%)")],
    [L("Data handling", "Xử lý dữ liệu"), L("OCR tokens ordered top-left → bottom-right (helps most models)", "Sắp chữ OCR trái-trên → phải-dưới (giúp hầu hết model)"), L("Official split; Vietnamese normalization (NFC, lowercase, tone marks kept)", "Split chính thức; chuẩn hoá tiếng Việt (NFC, chữ thường, giữ dấu thanh)")],
    [L("Training data", "Dữ liệu train"), L("Full ViTextVQA train", "Toàn bộ train ViTextVQA"), L("Full ViTextVQA train, 1 epoch (3.03 h on one L4)", "Toàn bộ train ViTextVQA, 1 epoch (3,03 h trên 1 L4)")],
    [L("Metrics", "Metric"), L("EM, F1; answer normalization not published", "EM, F1; không công bố cách chuẩn hoá"), L("EM, ANLS (+ F1 to compare); normalization published in code", "EM, ANLS (+ F1 để so); hàm chuẩn hoá công khai trong code")],
    [L("Full test (EM / F1)", "Full test (EM / F1)"), "25.48 / 53.95", "50.26 / 71.16"],
  ];
  const rows = [hd].concat(body.map((r, i) => r.map((c, j) => ({ text: FMT(c), options: { fontSize: 12, bold: j === 0 || i === body.length - 1,
      color: j === 0 ? HEX.ink : HEX.text, fill: i === body.length - 1 ? { color: HEX.tealt } : undefined } }))));
  table(s, rows, { x: M, y: 1.45, w: CW, colW: [2.2, 5.0, 4.93], rowH: 0.5 });
  footnote(s, L("[3] Nguyen et al., ViTextVQA, arXiv:2404.10652, 2024 (ext. ESWA 2025) · [4] Huang et al., SwinTextSpotter, CVPR 2022 · [5] Phan et al., ViT5, NAACL SRW 2022. "
              + "Note: the paper's 25.48 / 53.95 is the “OCR confidence” column of its Table 10; its ablation reports 24.83 / 52.51 for the full model.",
                "[3] Nguyen và cs., ViTextVQA, arXiv:2404.10652, 2024 (bản mở rộng ESWA 2025) · [4] Huang và cs., SwinTextSpotter, CVPR 2022 · [5] Phan và cs., ViT5, NAACL SRW 2022. "
              + "Lưu ý: số 25,48 / 53,95 của paper nằm ở cột “OCR confidence” trong Table 10; bảng ablation báo 24,83 / 52,51 cho model đầy đủ."), 6.3, 0.62);
}

// 13. Error analysis (moved before PART 06: it motivates the OCR-aware method)
{
  const s = content(L("B · Improvement", "B · Cải tiến"), L("REVIEW 2 · PART 05 · ERROR & HALLUCINATION ANALYSIS", "REVIEW 2 · PHẦN 05 · PHÂN TÍCH LỖI & HALLUCINATION"), L("Where B2 still fails", "B2 còn sai ở đâu"),
    "Phân loại tự động 4.988 câu B2 còn sai trên full test ViTextVQA. 9% chỉ sai dấu thanh, ví dụ thủy và thúy — đây là thách thức riêng của tiếng Việt. 17% chỉ lệch một hai ký tự, thường là đọc chữ hoặc số. Khoảng 27% sai về độ dài: thừa hoặc thiếu chữ so với gold. 47% còn lại sai về nội dung hoặc vùng ảnh. Nhóm đã duyệt tay 200 lỗi của B2 có xem ảnh, 4 người cùng nhãn gợi ý của AI, trưởng nhóm phân xử các chỗ lệch. Kết quả: 54% câu bị EM chấm sai thực ra đúng ý — EM quá khắt khe; 12,5% do gold nhiễu; ở ViTextVQA lỗi thật chủ yếu là đọc sai chữ 28% và đọc nhầm vùng chữ 23%; hallucination thật chỉ 7%. Chính các lỗi đọc chữ này dẫn tới phương pháp đề xuất ở phần sau. Độ đồng thuận giữa hai người gán nhãn độc lập kappa 0,62; bỏ 42 dòng có AI hỗ trợ thì kết luận không đổi. Nguồn từng con số ghi ở chân trang: lệnh tái tạo phân loại tự động và file kết quả duyệt tay trong repo.");
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
  table(s, rows, { x: M, y: 4.62, w: 7.0, colW: [1.8, 2.45, 2.75], rowH: 0.44 });
  const x = M + 7.4, w = CW - 7.4;
  card(s, x, 1.5, w, 4.95);
  txt(s, L("Manual audit: 200 B2 errors (adjudicated)", "Duyệt tay 200 lỗi B2 (đã phân xử)"), { x: x + 0.3, y: 1.65, w: w - 0.6, h: 0.75, fontSize: 16, bold: true, color: C.text2 });
  bullets(s, [
    L("54% are correct in meaning: EM is too strict (ViVQA 62%, ViTextVQA 46%)", "54% thực ra đúng ý: EM quá khắt khe (ViVQA 62%, ViTextVQA 46%)"),
    L("Noisy gold: 12.5% (ViVQA 19%)", "Gold nhiễu: 12,5% (ViVQA 19%)"),
    L("ViTextVQA real errors: OCR 28%, wrong text region 23%, tone marks 8%", "Lỗi thật ở ViTextVQA: đọc sai chữ 28%, nhầm vùng chữ 23%, dấu thanh 8%"),
    L("Hallucination (not in the image): 7% of errors (95% CI 4–11%)", "Hallucination (không có trong ảnh): 7% số lỗi (CI95 4–11%)"),
    L("4 annotators + AI pre-labels; κ = 0.62 (2 independent annotators); same conclusions without the 42 AI-assisted rows", "4 người + nhãn gợi ý của AI; κ = 0,62 (2 người độc lập); bỏ 42 dòng có AI hỗ trợ, kết luận không đổi"),
  ], { x: x + 0.3, y: 2.4, w: w - 0.6, h: 3.95, fontSize: 14 });
  footnote(s, L("Sources — buckets: python -m src.eval.answer_format --buckets on experiments/W07_B2_eval_qlora_r16_ep1_vitextvqa_testfull_… (4,988 = 10,028 − 5,040 correct). "
              + "Manual audit: docs/W07/audit/W07_error_audit_final.json (labels in docs/W07/audit/labels/, computed by compute_error_audit.py).",
                "Nguồn — phân loại tự động: python -m src.eval.answer_format --buckets trên experiments/W07_B2_eval_qlora_r16_ep1_vitextvqa_testfull_… (4.988 = 10.028 − 5.040 câu đúng). "
              + "Duyệt tay: docs/W07/audit/W07_error_audit_final.json (nhãn ở docs/W07/audit/labels/, tính bằng compute_error_audit.py)."), 6.4, 0.55);
}

// ================= PART 06 · OCR-aware QLoRA (B3) — added W09 =================
// Numbers: experiments/W08_ocr_* (OCR choice, inference-only ablation), W09_ocr_factorial_2x2_*, W09_B3_seed_stability_*,
// W09_fulltest_B3_vs_B2sub_*, W09_vivqa_epoch_select_* (see docs/W08/README.md, docs/W09/README.md).
pres.addSection({ title: L("B · OCR-aware QLoRA", "B · QLoRA nhận biết OCR") });

// 14. Method
{
  const s = content(L("B · OCR-aware QLoRA", "B · QLoRA nhận biết OCR"), L("REVIEW 2 · PART 06 · PROPOSED METHOD (B3)", "REVIEW 2 · PHẦN 06 · PHƯƠNG PHÁP ĐỀ XUẤT (B3)"),
    L("OCR-aware QLoRA: train the model to use OCR text", "QLoRA nhận biết OCR: dạy model dùng văn bản OCR"),
    "Từ phân tích lỗi, nhóm đề xuất B3: QLoRA nhận biết OCR. Ảnh được chạy OCR trước — PaddleOCR dò vùng chữ, VietOCR đọc chữ tiếng Việt — rồi đúng một dòng 'Văn bản đọc được trong ảnh' được chèn trước câu hỏi. Điểm mấu chốt: dòng OCR có mặt cả lúc train lẫn lúc test, nên model học khi nào tin OCR, khi nào tin ảnh. Backend OCR được chọn trên dev300 bằng luật chốt trước: PaddleOCR bản Latin làm rơi dấu tiếng Việt, ví dụ 'Kiểu Thẩm' thành 'Kiu Thm'; VietOCR giữ được đáp án trong văn bản OCR ở 37,7% câu so với 27,0%.");
  // pipeline row (native shapes)
  const y = 1.55, h = 1.3, gap = 0.32;
  const boxes = [
    [L("Image", "Ảnh"), L("scene text", "chữ trong ảnh"), C.background2, C.text2],
    [L("PaddleOCR", "PaddleOCR"), L("detects text lines", "dò dòng chữ"), C.background2, C.text2],
    [L("VietOCR", "VietOCR"), L("reads Vietnamese, keeps tone marks", "đọc tiếng Việt, giữ dấu"), C.background2, C.text2],
    [L("Prompt", "Prompt"), L("1 OCR line + question", "1 dòng OCR + câu hỏi"), C.accent4, C.text2],
    [L("Qwen2.5-VL + LoRA", "Qwen2.5-VL + LoRA"), L("LoRA trained with the OCR line", "LoRA train cùng dòng OCR"), C.accent5, C.text2],
  ];
  const bw = (CW - gap * (boxes.length - 1)) / boxes.length;
  boxes.forEach(([t, sub, fill, col], i) => {
    const x = M + i * (bw + gap);
    card(s, x, y, bw, h, fill);
    txt(s, t, { x: x + 0.1, y: y + 0.12, w: bw - 0.2, h: 0.42, fontSize: 15, bold: true, color: col, align: "center" });
    txt(s, sub, { x: x + 0.15, y: y + 0.62, w: bw - 0.3, h: 0.6, fontSize: 12, color: C.text1, align: "center" });
    if (i < boxes.length - 1)
      s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bw + 0.05, y: y + h / 2 - 0.11, w: gap - 0.1, h: 0.22, fill: { color: HEX.muted }, line: { color: HEX.muted, width: 0 }, objectName: oid("arrow") });
  });
  // prompt example
  card(s, M, 3.1, CW, 0.75, C.background2);
  txt(s, [
    { text: L("Prompt line:  ", "Dòng prompt:  "), options: { bold: true, color: HEX.ink } },
    { text: "Văn bản đọc được trong ảnh: NHÀ THUỐC TÂM AN | 0919 467 868 …   Câu hỏi: tên nhà thuốc là gì?", options: { fontFace: "Courier New", fontSize: 13 } },
  ], { x: M + 0.3, y: 3.1, w: CW - 0.6, h: 0.75, fontSize: 14, valign: "middle" });
  // three cards
  const cy = 4.15, ch = 2.7, cw = (CW - 2 * 0.4) / 3;
  [[L("Why OCR", "Vì sao OCR"), [L("Audit of real B2 errors on ViTextVQA: OCR misreads 28%, wrong text region 23%, tone marks 8%", "Lỗi thật của B2 ở ViTextVQA: đọc sai chữ 28%, nhầm vùng chữ 23%, sai dấu 8%"),
                                   L("Vietnamese scene text is a stated research gap (G2)", "Chữ tiếng Việt trong ảnh là gap nghiên cứu G2")], C.background2],
   [L("Which OCR (rule fixed on dev300)", "Chọn OCR (luật chốt, dev300)"), [L("PaddleOCR Latin drops tone marks: “Kiểu Thẩm” → “Kiu Thm”", "PaddleOCR bản Latin làm rơi dấu: “Kiểu Thẩm” → “Kiu Thm”"),
                                   L("Gold answer inside OCR text: VietOCR 37.7% vs 27.0% (p = 0.0008) → VietOCR", "Đáp án nằm trong văn bản OCR: VietOCR 37,7% so với 27,0% (p = 0,0008) → VietOCR")], C.background2],
   [L("What is trained", "Train gì"), [L("Same QLoRA as B2 (r = 16, 0.196% params); only the prompt gains the OCR line", "QLoRA y hệt B2 (r = 16, 0,196% tham số); chỉ thêm dòng OCR vào prompt"),
                                   L("Controlled test: 8,000-question train subset, B3 vs B2-sub (same subset, no OCR)", "Phép thử có kiểm soát: tập con 8.000 câu, B3 so với B2-sub (cùng tập, không OCR)")], C.accent4]]
    .forEach(([t, items, fill], i) => {
      const x = M + i * (cw + 0.4);
      card(s, x, cy, cw, ch, fill);
      txt(s, t, { x: x + 0.25, y: cy + 0.15, w: cw - 0.5, h: 0.45, fontSize: 16, bold: true, color: C.text2 });
      bullets(s, items, { x: x + 0.25, y: cy + 0.65, w: cw - 0.5, h: ch - 0.8, fontSize: 14 });
    });
}

// 15. 2x2 factorial
{
  const s = content(L("B · OCR-aware QLoRA", "B · QLoRA nhận biết OCR"), L("REVIEW 2 · PART 06 · ABLATION: OCR AT TRAIN × OCR AT TEST", "REVIEW 2 · PHẦN 06 · ABLATION: OCR LÚC TRAIN × OCR LÚC TEST"),
    L("OCR helps only when the model is trained with it", "OCR chỉ giúp khi model được train cùng OCR"),
    "Đây là ablation bắt buộc có và không có OCR, làm thành bảng hai nhân hai trên cùng tập con và cùng 2.000 câu test. Chỉ chèn OCR lúc test cho model chưa train cùng OCR: không tăng, kể cả với zero-shot B1 và B2 ở tuần 8. Train cùng OCR mà lúc test bỏ OCR: cũng không tăng. Chỉ khi có cả hai thì EM tăng 4,35 điểm, p cỡ 10 mũ trừ 8. Tương tác là 4,9 điểm, khoảng tin cậy 3,2 đến 6,6. Vậy đóng góp không phải là thêm OCR vào prompt, mà là dạy model cách dùng OCR.");
  // 2x2 grid
  const gx = M + 1.95, gy = 1.95, cw = 2.45, chh = 1.55, g = 0.15;
  txt(s, L("Test without OCR", "Test không OCR"), { x: gx, y: gy - 0.45, w: cw, h: 0.4, fontSize: 14, bold: true, color: C.text2, align: "center" });
  txt(s, L("Test with OCR", "Test có OCR"), { x: gx + cw + g, y: gy - 0.45, w: cw, h: 0.4, fontSize: 14, bold: true, color: C.text2, align: "center" });
  txt(s, L("Train without OCR", "Train không OCR"), { x: M, y: gy, w: 1.85, h: chh, fontSize: 14, bold: true, color: C.text2, valign: "middle" });
  txt(s, L("Train with OCR", "Train có OCR"), { x: M, y: gy + chh + g, w: 1.85, h: chh, fontSize: 14, bold: true, color: C.text2, valign: "middle" });
  const cells = [[0, 0, "45.55", L("B2-sub", "B2-sub"), C.background2], [0, 1, "44.90", L("B2-sub + OCR at test", "B2-sub + OCR lúc test"), C.background2],
                 [1, 0, "45.00", L("B3, OCR removed at test", "B3, bỏ OCR lúc test"), C.background2], [1, 1, "49.25", L("B3 (proposed)", "B3 (đề xuất)"), C.accent5]];
  cells.forEach(([r, c, v, lab, fill]) => {
    const x = gx + c * (cw + g), y = gy + r * (chh + g);
    card(s, x, y, cw, chh, fill);
    txt(s, FMT(v), { x, y: y + 0.18, w: cw, h: 0.75, fontFace: "Cambria", fontSize: 34, bold: true, color: r && c ? C.accent2 : C.text2, align: "center" });
    txt(s, lab, { x: x + 0.1, y: y + 0.95, w: cw - 0.2, h: 0.45, fontSize: 13, color: C.text1, align: "center" });
  });
  txt(s, L("EM (%), ViTextVQA test subset (2,000 q.), seed 42", "EM (%), ViTextVQA tập con test (2.000 câu), seed 42"),
      { x: gx, y: gy + 2 * chh + g + 0.12, w: 2 * cw + g, h: 0.35, fontSize: 12, color: C.accent3, align: "center" });
  // effects
  const x = M + 7.45, w = CW - 7.45;
  card(s, x, 1.5, w, 2.3);
  txt(s, L("Paired tests (McNemar, EM)", "Kiểm định ghép cặp (McNemar, EM)"), { x: x + 0.25, y: 1.62, w: w - 0.5, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  bullets(s, [
    L("OCR at test only: −0.65 (p = 0.49)", "Chỉ OCR lúc test: −0,65 (p = 0,49)"),
    L("OCR at train only: −0.55 (p = 0.35)", "Chỉ OCR lúc train: −0,55 (p = 0,35)"),
    L("Both at train and test: +4.35 (p = 2 × 10⁻⁸)", "Cả train lẫn test: +4,35 (p = 2 × 10⁻⁸)"),
    L("Removing OCR from B3: −4.25 (p = 4 × 10⁻⁹)", "Bỏ OCR khỏi B3: −4,25 (p = 4 × 10⁻⁹)"),
  ], { x: x + 0.25, y: 2.05, w: w - 0.5, h: 1.7, fontSize: 14 });
  card(s, x, 4.0, w, 1.4, C.accent4);
  txt(s, [
    { text: L("Interaction +4.9 EM ", "Tương tác +4,9 EM "), options: { bold: true, color: HEX.ink } },
    { text: L("(CI95 +3.2 to +6.6) → OCR is needed at both train and test", "(CI95 +3,2 đến +6,6) → cần OCR ở cả train lẫn test") },
  ], { x: x + 0.25, y: 4.0, w: w - 0.5, h: 1.4, fontSize: 15, valign: "middle" });
  card(s, M, 5.85, CW, 1.0, C.accent5);
  txt(s, [
    { text: L("Takeaway  ", "Kết luận  "), options: { bold: true, color: HEX.ink } },
    { text: L("the contribution is not “add OCR to the prompt” — it is teaching the model, with QLoRA, when to trust OCR and when to trust the image.",
              "đóng góp không phải “thêm OCR vào prompt” — mà là dùng QLoRA dạy model khi nào tin OCR, khi nào tin ảnh.") },
  ], { x: M + 0.3, y: 5.85, w: CW - 0.6, h: 1.0, fontSize: 15, valign: "middle" });
}

// 16. Seeds + full test
{
  const s = content(L("B · OCR-aware QLoRA", "B · QLoRA nhận biết OCR"), L("REVIEW 2 · PART 06 · ROBUSTNESS", "REVIEW 2 · PHẦN 06 · ĐỘ VỮNG"),
    L("The gain holds across seeds and on the full test", "Mức tăng giữ vững qua các seed và trên full test"),
    "Hai câu hỏi hội đồng hay đặt: một seed có phải may mắn không, và tập con có đại diện không. Ba seed huấn luyện 42, 43, 44, cùng tập con: cả ba seed B3 đều thắng B2-sub có ý nghĩa, trung bình cộng 3,57 điểm EM, độ lệch chuẩn 0,76. Trên toàn bộ 10.028 câu test, B3 thắng B2-sub 2,54 điểm, p cỡ 10 mũ trừ 14, và cao hơn ViTextBLIP-2 ở cả EM lẫn F1. Nhóm cũng đính chính: B3 dùng 23% dữ liệu vẫn thấp hơn B2 train toàn bộ 1,16 điểm; OCR bù được khoảng 69% phần mất do cắt dữ liệu. Về ANLS, mức tăng là 1,55 điểm, dưới ngưỡng 3 điểm của H2.");
  s.addChart(pres.charts.BAR, [
    { name: L("B2-sub (no OCR)", "B2-sub (không OCR)"), labels: ["seed 42", "seed 43", "seed 44"], values: [45.55, 44.75, 46.05] },
    { name: L("B3 (OCR-aware)", "B3 (nhận biết OCR)"), labels: ["seed 42", "seed 43", "seed 44"], values: [49.25, 49.0, 48.8] },
  ], chartBase({ x: M, y: 1.5, w: 5.6, h: 4.0, barDir: "col", barGrouping: "clustered", chartColors: [HEX.muted, HEX.amber], showLegend: true, legendPos: "b",
                 valAxisMinVal: 40, valAxisMaxVal: 52, showTitle: true, title: L("EM (%), test subset 2,000 q., 3 training seeds", "EM (%), tập con 2.000 câu, 3 seed huấn luyện"), dataLabelFormatCode: "0.00", objectName: oid("chart") }));
  card(s, M, 5.7, 5.6, 1.15, C.accent4);
  txt(s, [
    { text: L("+3.57 ± 0.76 EM", "+3,57 ± 0,76 EM"), options: { bold: true, color: HEX.ink } },
    { text: L(" · all 3 seeds p ≤ 1.4 × 10⁻⁴ → stable (pre-registered rule)", " · cả 3 seed p ≤ 1,4 × 10⁻⁴ → ổn định (luật chốt trước)") },
  ], { x: M + 0.25, y: 5.7, w: 5.1, h: 1.15, fontSize: 14, valign: "middle" });
  const x = M + 6.0, w = CW - 6.0;
  const rows = [
    [TH(L("Full test (10,028 q.)", "Full test (10.028 câu)")), TH(L("Train data", "Dữ liệu train")), TH("EM"), TH("F1")],
    ["ViTextBLIP-2 (paper)", L("full", "toàn bộ"), "25.48", "53.95"],
    ["B2-sub", "23%", "46.56", "67.83"],
    [L("B3 (OCR-aware)", "B3 (nhận biết OCR)"), "23%", "49.10", "70.03"],
    [L("B2 (QLoRA, no OCR)", "B2 (QLoRA, không OCR)"), L("100%", "100%"), "50.26", "71.16"],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: FMT(c), options: { fontSize: 14, align: j ? "center" : "left", bold: i === 3,
      color: i === 3 ? HEX.ink : HEX.text, fill: i === 3 ? { color: HEX.ambert } : undefined } })));
  table(s, rows, { x, y: 1.5, w, colW: [2.55, 1.45, 1.1, 1.03], rowH: 0.5 });
  card(s, x, 4.25, w, 2.6);
  bullets(s, [
    L("B3 vs B2-sub: +2.54 EM (672 vs 417 questions only one model gets right, p = 10⁻¹⁴)", "B3 so với B2-sub: +2,54 EM (672 so với 417 câu chỉ một model đúng, p = 10⁻¹⁴)"),
    L("Above ViTextBLIP-2 on EM and F1 (CI95 lower bounds 48.1 / 69.3)", "Cao hơn ViTextBLIP-2 cả EM và F1 (cận dưới CI95 48,1 / 69,3)"),
    L("Correction: still 1.16 EM below full-data B2 (p = 0.003); OCR recovers ≈ 69% of the data-cut gap with 25% of the training time", "Đính chính: vẫn thấp hơn B2 dữ liệu đầy đủ 1,16 EM (p = 0,003); OCR bù ≈ 69% phần mất do cắt dữ liệu với 25% thời gian train"),
  ], { x: x + 0.25, y: 4.4, w: w - 0.5, h: 2.35, fontSize: 14 });
}

// 17. ViVQA epoch selection
{
  const s = content(L("B · OCR-aware QLoRA", "B · QLoRA nhận biết OCR"), L("REVIEW 2 · PART 06 · ViVQA: TRAINING LENGTH", "REVIEW 2 · PHẦN 06 · ViVQA: ĐỘ DÀI HUẤN LUYỆN"),
    L("ViVQA: a second epoch helps — chosen without touching test", "ViVQA: epoch thứ hai có ích — chọn mà không chạm test"),
    "Với ViVQA, câu hỏi là train một hay hai epoch. Val của ViVQA được tách theo câu hỏi nên 43% câu val dùng chung ảnh với train; train lâu hơn dễ nhớ ảnh và làm val đẹp giả. Vì vậy nhóm chỉ chọn epoch trên 688 câu val có ảnh chưa từng xuất hiện trong train. Lịch hai epoch sau epoch 2 hơn B2 một epoch 3,9 điểm, có ý nghĩa, nên được chọn. Chỉ khi đó mới chấm test, một lần: EM tăng từ 73,8 lên 76,6, p cỡ 10 mũ trừ 9. Kiểm tra thêm sau khi có kết quả: trên phần test sạch, ảnh và cặp hỏi đáp chưa từng thấy, mức tăng vẫn là 3,2 điểm — nên đây không phải do model nhớ ảnh. Chi phí: gấp đôi thời gian train, 6,1 giờ.");
  s.addChart(pres.charts.BAR, [
    { name: "EM", labels: [L("C1 · B2, 1-epoch schedule", "C1 · B2, lịch 1 epoch"), L("C2 · 2-epoch schedule, after epoch 1", "C2 · lịch 2 epoch, sau epoch 1"), L("C3 · 2-epoch schedule, after epoch 2", "C3 · lịch 2 epoch, sau epoch 2")], values: [73.11, 71.51, 77.03] },
  ], chartBase({ x: M, y: 1.5, w: 7.0, h: 3.6, barDir: "bar", chartColors: [HEX.muted, HEX.muted, HEX.amber], showLegend: false, valAxisMinVal: 60, valAxisMaxVal: 80,
                 valAxisHidden: true, valGridLine: { style: "none" }, catAxisOrientation: "maxMin", showTitle: true,
                 title: L("EM (%) on image-disjoint val (688 q.)", "EM (%) trên val sạch, ảnh không trùng train (688 câu)"), dataLabelFormatCode: "0.0", objectName: oid("chart") }));
  card(s, M, 5.3, 7.0, 1.55, C.accent4);
  txt(s, L("Rule fixed before the run: highest EM on image-disjoint val; keep C1 unless significantly better → C3 (+3.9 EM, p = 1.4 × 10⁻⁴)",
           "Luật chốt trước: EM cao nhất trên val sạch; giữ C1 trừ khi hơn có ý nghĩa → C3 (+3,9 EM, p = 1,4 × 10⁻⁴)"),
      { x: M + 0.25, y: 5.3, w: 6.5, h: 1.55, fontSize: 14, valign: "middle", color: C.text2 });
  const x = M + 7.4, w = CW - 7.4;
  card(s, x, 1.5, w, 2.6, C.accent5);
  stat(s, x + 0.3, 1.65, w - 0.6, FMT("73.8 → 76.6"), L("ViVQA test EM (3,001 q.), touched once: +2.9, p = 4.6 × 10⁻⁹", "EM test ViVQA (3.001 câu), chạm một lần: +2,9, p = 4,6 × 10⁻⁹"), C.text2);
  txt(s, L("Not image memorisation (post-hoc): on the clean test subset — unseen images and unseen QA, n = 1,219 — +3.2 EM (69.1 → 72.3, p = 1.5 × 10⁻⁴)",
           "Không phải nhớ ảnh (kiểm sau): trên phần test sạch — ảnh và cặp hỏi–đáp chưa thấy, n = 1.219 — +3,2 EM (69,1 → 72,3, p = 1,5 × 10⁻⁴)"),
      { x: x + 0.3, y: 3.2, w: w - 0.6, h: 0.85, fontSize: 13, italic: true, color: C.text1 });
  card(s, x, 4.3, w, 2.55);
  txt(s, L("Why image-disjoint val", "Vì sao dùng val sạch"), { x: x + 0.3, y: 4.42, w: w - 0.6, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  bullets(s, [
    L("512 of 1,200 val questions share an image with train", "512 / 1.200 câu val dùng chung ảnh với train"),
    L("Longer training memorises images → plain val would favour it", "Train lâu hơn nhớ ảnh → val thường thiên vị"),
    L("Cost: 6.1 h instead of 3.0 h on one L4", "Chi phí: 6,1 giờ thay vì 3,0 giờ trên một L4"),
  ], { x: x + 0.3, y: 4.85, w: w - 0.6, h: 1.95, fontSize: 14 });
}

// 18. Hypotheses scoreboard
pres.addSection({ title: L("B · Analysis & plan", "B · Phân tích & kế hoạch") });
{
  const s = content(L("B · Analysis & plan", "B · Phân tích & kế hoạch"), L("REVIEW 2 · PART 07 · HYPOTHESES", "REVIEW 2 · PHẦN 07 · GIẢ THUYẾT"), L("What the evidence says so far", "Bằng chứng đến thời điểm này"),
    "Đối chiếu với bốn giả thuyết H1 đến H4 ở Phần A. H1 được ủng hộ mạnh: QLoRA tăng 51,7 điểm EM trên ViVQA và 21,1 trên ViTextVQA. H2 chỉ ủng hộ một phần: OCR cộng hưởng mạnh với QLoRA, nhưng ANLS chỉ tăng 1,55 điểm, dưới ngưỡng 3 điểm đã đặt — nhóm báo đúng như vậy. H3 chưa được ủng hộ với dạng few-shot đã thử. H4: duyệt tay 200 lỗi của B2 cho thấy chỉ khoảng 7% là hallucination thật; nhưng H4 so với zero-shot nên cần đo B1 ở tuần 11 — vẫn để Đang chờ. Hai dòng cuối không phải giả thuyết: hiệu quả 4-bit là số đo, giữ 94,5% độ chính xác với VRAM giảm 77%; ablation cấu hình LoRA làm ở tuần 10.");
  const OK = L("Supported", "Ủng hộ"), PART = L("Partly", "Một phần"), NO = L("Not supported", "Chưa ủng hộ"), WAIT = L("Pending", "Đang chờ"), MEAS = L("Measured", "Đã đo"), PLAN = L("Planned", "Kế hoạch");
  const col = { [OK]: HEX.teal, [PART]: HEX.amber, [NO]: HEX.red, [WAIT]: HEX.muted, [MEAS]: HEX.ink, [PLAN]: HEX.muted };
  const rows = [
    [TH(L("Hypothesis (Part A, revised numbering)", "Giả thuyết (Phần A, đánh số mới)")), TH(L("Evidence", "Bằng chứng")), TH(L("Status", "Trạng thái"))],
    [L("H1  QLoRA ≥ +5 EM over zero-shot", "H1  QLoRA ≥ +5 EM so với zero-shot"), L("+51.7 (ViVQA), +21.1 (ViTextVQA full), p ≈ 0", "+51,7 (ViVQA), +21,1 (ViTextVQA đầy đủ), p ≈ 0"), OK],
    [L("H2  OCR text ≥ +3 ANLS, best with QLoRA", "H2  Chữ OCR ≥ +3 ANLS, tốt nhất khi kết hợp QLoRA"), L("Synergy: interaction +4.9 EM; but ANLS +1.55 ± 0.40 (< +3)", "Cộng hưởng: tương tác +4,9 EM; nhưng ANLS +1,55 ± 0,40 (< +3)"), PART],
    [L("H3  Vietnamese format prompt ≥ +3 EM", "H3  Prompt định dạng tiếng Việt ≥ +3 EM"), L("Few-shot variant: −2.1 / −1.2 EM", "Dạng few-shot: −2,1 / −1,2 EM"), NO],
    [L("H4  QLoRA does not increase hallucination", "H4  QLoRA không làm tăng hallucination"), L("Audit: 7% of B2 errors are hallucinations (CI 4–11%); B1 comparison in W11", "Duyệt tay: 7% lỗi của B2 là hallucination (CI 4–11%); so với B1 ở W11"), WAIT],
    [L("Efficiency (measurement, not a hypothesis)", "Hiệu quả (số đo, không phải giả thuyết)"), L("4-bit keeps 94.5% accuracy; VRAM −77% (R0 vs R0b)", "4-bit giữ 94,5% độ chính xác; VRAM −77% (R0 so với R0b)"), MEAS],
    [L("Ablations A1, A2  LoRA rank, target modules", "Ablation A1, A2  rank, module gắn LoRA"), L("Planned W10", "Kế hoạch W10"), PLAN],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c, options: { fontSize: 14, bold: j === 2, color: j === 2 ? col[c] : HEX.text } })));
  table(s, rows, { x: M, y: 1.5, w: CW, colW: [5.2, 4.93, 2.0], rowH: 0.54 });
  card(s, M, 5.6, CW, 1.25, C.accent4);
  txt(s, L("RQ1: zero-shot EM 22.1 (ViVQA), 29.1 (ViTextVQA).  RQ2: QLoRA +51.7 / +21.1 EM.  RQ3: OCR-aware QLoRA +2.5 EM over the same QLoRA (full test).",
           "RQ1: zero-shot EM 22,1 (ViVQA), 29,1 (ViTextVQA).  RQ2: QLoRA +51,7 / +21,1 EM.  RQ3: QLoRA nhận biết OCR +2,5 EM so với cùng QLoRA (full test)."),
      { x: M + 0.35, y: 5.6, w: CW - 0.7, h: 1.25, fontSize: 15, valign: "middle", color: C.text2 });
}

// 19. Next steps
{
  const s = content(L("B · Analysis & plan", "B · Phân tích & kế hoạch"), L("REVIEW 2 · PART 07 · NEXT STEPS", "REVIEW 2 · PHẦN 07 · BƯỚC TIẾP THEO"), L("Plan to Review 3 (week 13)", "Kế hoạch đến Review 3 (tuần 13)"),
    "Từ tuần 10 đến tuần 13: train QLoRA nhận biết OCR trên toàn bộ dữ liệu ViTextVQA để xem có vượt B2 không, ablation rank và module gắn LoRA, đo hallucination có hệ thống kể cả B1 và B3 để kết luận H4, adapter dùng chung là tuỳ chọn, rồi dựng prototype web: tải ảnh, chạy OCR, hỏi bằng tiếng Việt — cho Review 3.");
  const steps = [
    ["W10", L("OCR-aware QLoRA on the full ViTextVQA train set; LoRA rank and target-module ablations (ablations A1, A2)", "QLoRA nhận biết OCR trên toàn bộ train ViTextVQA; ablation rank và module gắn LoRA (ablation A1, A2)")],
    ["W11", L("Systematic hallucination evaluation incl. B1 and B3 (H4); shared adapter for both datasets (optional)", "Đo hallucination có hệ thống, kể cả B1 và B3 (H4); adapter dùng chung cho 2 dataset (tuỳ chọn)")],
    ["W12", L("Web prototype: upload image → OCR → answer in Vietnamese", "Prototype web: tải ảnh → OCR → trả lời tiếng Việt")],
    ["W13", L("Review 3: this deck + Part C (final results, ablations, error analysis, live demo)", "Review 3: bộ slide này + Phần C (kết quả cuối, ablation, phân tích lỗi, demo trực tiếp)")],
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

// 20. Thank you
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: L("B · Analysis & plan", "B · Phân tích & kế hoạch") });
  s.addText(L("Thank you", "Cảm ơn hội đồng"), { placeholder: "title" });
  s.addText([
    { text: L("We welcome your feedback.", "Nhóm rất mong nhận được góp ý."), options: { fontSize: 20, breakLine: true } },
    { text: " ", options: { fontSize: 12, breakLine: true } },
    { text: L("Every number in this talk comes from a logged, reproducible run (fixed seed, config and evaluation protocol).", "Mọi con số trong bài đều có log và tái lập được (seed, config và giao thức đánh giá cố định)."), options: { fontSize: 15, color: "A8C4E0" } },
  ], { placeholder: "body" });
  s.addNotes("Cảm ơn hội đồng đã lắng nghe. Nhóm sẵn sàng trả lời câu hỏi.");
}

// ---------- fill the Contents slide (needs final slide numbers) ----------
{
  const s = contentsSlide, last = pres.slides.length;
  const parts = [
    [L("Part A · Review 1 (revised)", "Phần A · Review 1 (đã sửa)"), `${A.fb}–${partBStart - 1}`, C.accent4,
     [L("Response to the 10 reviewer comments (24/09)", "Tiếp thu 10 góp ý phản biện (24/09)"), L("Problem, input/output, Vietnamese problems V1–V4", "Bài toán, input/output, vấn đề tiếng Việt V1–V4"),
      L("Gaps, prior work, RQ1–RQ4, hypotheses H1–H4", "Khoảng trống, công trình trước, RQ1–RQ4, giả thuyết H1–H4"), L("Data, proposed method, QLoRA, token budget", "Dữ liệu, phương pháp đề xuất, QLoRA, ngân sách token"),
      L("Plan, risks, outputs, team", "Kế hoạch, rủi ro, sản phẩm, nhóm")]],
    [L("Part B · Review 2", "Phần B · Review 2"), `${partBStart}–${last}`, C.accent5,
     [L("Summary and response to advisor feedback (07/10)", "Tóm tắt và xử lý góp ý GVHD (07/10)"), L("Dataset noise, baseline architecture, evaluation protocol", "Nhiễu dữ liệu, kiến trúc baseline, giao thức đánh giá"),
      L("R0 pipeline check, zero-shot B1, backbone ablation", "Kiểm chứng R0, zero-shot B1, ablation backbone"), L("QLoRA B2, controls, SOTA, error analysis", "QLoRA B2, đối chứng, SOTA, phân tích lỗi"),
      L("Proposed OCR-aware QLoRA B3, hypotheses H1–H4, next steps", "Đề xuất QLoRA nhận biết OCR B3, giả thuyết H1–H4, bước tiếp theo")]],
  ];
  const cw = (CW - 0.4) / 2, y = 1.6, h = 4.0;
  parts.forEach(([t, range, fill, items], i) => {
    const x = M + i * (cw + 0.4);
    card(s, x, y, cw, h, fill);
    txt(s, t, { x: x + 0.3, y: y + 0.2, w: cw - 2.0, h: 0.55, fontFace: "Cambria", fontSize: 22, bold: true, color: C.text2 });
    txt(s, L("slides ", "slide ") + range, { x: x + cw - 1.9, y: y + 0.25, w: 1.6, h: 0.5, fontSize: 14, bold: true, color: C.accent1, align: "right" });
    bullets(s, items, { x: x + 0.3, y: y + 1.0, w: cw - 0.6, h: h - 1.2, fontSize: 17, paraSpaceAfter: 8 });
  });
  txt(s, L("Hypotheses use one numbering (H1–H4) across all parts. Parts C (Review 3) and D (final defense) will be appended; earlier parts are revised after each review.",
           "Giả thuyết dùng một cách đánh số (H1–H4) cho mọi phần. Phần C (Review 3) và D (bảo vệ cuối) sẽ được nối thêm; các phần trước được sửa sau mỗi lần review."),
      { x: M, y: 6.3, w: CW, h: 0.55, fontSize: 13, color: C.accent3 });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
