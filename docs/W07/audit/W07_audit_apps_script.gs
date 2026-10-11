/**
 * W07 error audit — tạo 4 Google Sheet gán nhãn, gắn link ảnh, chia sẻ cho từng người, theo dõi tiến độ.
 *
 * Điều kiện: trên Google Drive đã có
 *   My Drive/ViVQA-VLM/audit/W07/images/{vivqa,vitextvqa}/   (ảnh, do make_audit_sheet.py tạo)
 *   My Drive/ViVQA-VLM/audit/W07/phan_cong/W07_audit_team_data.json   (Cursor chép vào)
 *
 * Cách dùng (script.google.com → New project → dán file này):
 *   1) Điền TEAM bên dưới (tên + Gmail của 3 bạn; dòng của Hoàng để trống email).
 *   2) Chọn hàm setupAudit → Run → cấp quyền (Advanced → Go to ... (unsafe) vì script của chính bạn).
 *   3) Mở file "W07_links" trong thư mục phan_cong → gửi link cho từng người.
 *   Xem tiến độ: chạy capNhatTienDo → mở file "W07_theo_doi".
 * Chạy lại setupAudit an toàn: file đã có sẽ KHÔNG bị tạo lại hay ghi đè (chỉ bổ sung quyền chia sẻ).
 */
const TEAM = [
  { key: 'nguoi1_Hoang', ten: 'Hoàng', email: '' },          // chủ sở hữu, không cần chia sẻ
  { key: 'nguoi2', ten: 'TEN_BAN_2', email: 'EMAIL_BAN_2@gmail.com' },
  { key: 'nguoi3', ten: 'TEN_BAN_3', email: 'EMAIL_BAN_3@gmail.com' },
  { key: 'nguoi4', ten: 'TEN_BAN_4', email: 'EMAIL_BAN_4@gmail.com' },
];
const W07_PATH = ['ViVQA-VLM', 'audit', 'W07'];
const DATA_FILE = 'W07_audit_team_data.json';
const YELLOW = '#FFF2CC', BLUE = '#DDEBF7', NAVY = '#13294B';
const WIDTHS = [50, 85, 85, 170, 330, 220, 220, 250, 105, 105, 240, 110];

function folderByPath_(parts) {
  let f = DriveApp.getRootFolder();
  parts.forEach(function (p) {
    const it = f.getFoldersByName(p);
    if (!it.hasNext()) throw new Error('Không thấy thư mục "' + p + '" trong "' + f.getName() + '"');
    f = it.next();
  });
  return f;
}

function fileInFolder_(folder, name) {
  const it = folder.getFilesByName(name);
  return it.hasNext() ? it.next() : null;
}

function imageMap_(imagesFolder) {
  const map = {};
  const sub = imagesFolder.getFolders();
  while (sub.hasNext()) {
    const f = sub.next();
    const files = f.getFiles();
    while (files.hasNext()) { const x = files.next(); map[f.getName() + '/' + x.getName()] = x.getUrl(); }
  }
  return map;
}

function q_(s) { return '"' + String(s).replace(/"/g, '""') + '"'; }

function buildSheet_(ss, data, rows, imgs) {
  const g = ss.getSheets()[0];
  g.setName('huong_dan');
  g.getRange(1, 1).setValue('Hướng dẫn gán nhãn lỗi B2 — W07 error audit').setFontWeight('bold').setFontSize(14);
  g.getRange(3, 1, data.guide.length, 2).setValues(data.guide).setWrap(true).setVerticalAlignment('top');
  g.setColumnWidth(1, 260); g.setColumnWidth(2, 820);

  const ws = ss.insertSheet('error_audit');
  const cols = data.columns, n = rows.length;
  ws.getRange(1, 1, 1, cols.length).setValues([cols]).setFontWeight('bold').setFontColor('#FFFFFF').setBackground(NAVY);
  const values = rows.map(function (r) { return cols.map(function (c) { return r[c] == null ? '' : r[c]; }); });
  ws.getRange(2, 2, n, cols.length - 1).setNumberFormat('@');   // giữ nguyên chuỗi (số 0 đầu, ngày tháng…)
  ws.getRange(2, 1, n, cols.length).setValues(values).setWrap(true).setVerticalAlignment('top');
  let missing = 0;
  const links = rows.map(function (r) {
    const url = imgs[r.dataset + '/' + r.anh];
    if (!url) { missing++; return [r.anh]; }
    return ['=HYPERLINK(' + q_(url) + ',' + q_(r.anh) + ')'];
  });
  ws.getRange(2, 4, n, 1).setValues(links);
  ws.getRange(2, 8, n, 5).setBackground(YELLOW);
  rows.forEach(function (r, i) { if (r.phan === 'hieu_chinh') ws.getRange(i + 2, 1, 1, 7).setBackground(BLUE); });
  ws.getRange(2, 8, n, 1).setDataValidation(SpreadsheetApp.newDataValidation().requireValueInList(data.labels, true).setAllowInvalid(false).build());
  ws.getRange(2, 9, n, 2).setDataValidation(SpreadsheetApp.newDataValidation().requireValueInList(['Có', 'Không'], true).setAllowInvalid(false).build());
  WIDTHS.forEach(function (w, j) { ws.setColumnWidth(j + 1, w); });
  ws.setFrozenRows(1); ws.setFrozenColumns(4);
  ws.getRange(1, 1, n + 1, 7).protect().setDescription('Cột dữ liệu gốc — không sửa').setWarningOnly(true);
  ss.setActiveSheet(ws);
  return missing;
}

function setupAudit() {
  const w07 = folderByPath_(W07_PATH);
  const pc = folderByPath_(W07_PATH.concat(['phan_cong']));
  const images = folderByPath_(W07_PATH.concat(['images']));
  const df = fileInFolder_(pc, DATA_FILE);
  if (!df) throw new Error('Thiếu ' + DATA_FILE + ' trong phan_cong (Cursor chưa chép hoặc Drive chưa đồng bộ xong).');
  const data = JSON.parse(df.getBlob().getDataAsString('UTF-8'));
  const imgs = imageMap_(images);
  const out = [['người', 'tên', 'email', 'số dòng', 'link', 'ảnh thiếu']];
  TEAM.forEach(function (m) {
    const rows = data.people[m.key];
    if (!rows) throw new Error('Không có dữ liệu cho ' + m.key);
    const name = 'W07_audit_' + m.key;
    let f = fileInFolder_(pc, name), missing = '(đã có, giữ nguyên)';
    if (!f) {
      const ss = SpreadsheetApp.create(name);
      missing = buildSheet_(ss, data, rows, imgs);
      f = DriveApp.getFileById(ss.getId());
      f.moveTo(pc);
    }
    if (m.email && m.email.indexOf('@') > 0 && m.email.indexOf('EMAIL_') !== 0) {
      f.addEditor(m.email);
      images.addViewer(m.email);
    }
    out.push([m.key, m.ten, m.email, rows.length, f.getUrl(), missing]);
  });
  let lf = fileInFolder_(pc, 'W07_links');
  const ls = lf ? SpreadsheetApp.open(lf) : SpreadsheetApp.create('W07_links');
  if (!lf) DriveApp.getFileById(ls.getId()).moveTo(pc);
  const sh = ls.getSheets()[0];
  sh.clear(); sh.getRange(1, 1, out.length, out[0].length).setValues(out); sh.autoResizeColumns(1, 6);
  out.forEach(function (r) { Logger.log(r.join(' | ')); });
  Logger.log('Xong. Mở file W07_links trong ' + pc.getName() + '. (Thư mục W07 gốc KHÔNG được chia sẻ.)');
  Logger.log('W07 folder: ' + w07.getUrl());
}

function capNhatTienDo() {
  const pc = folderByPath_(W07_PATH.concat(['phan_cong']));
  let tf = fileInFolder_(pc, 'W07_theo_doi');
  const ss = tf ? SpreadsheetApp.open(tf) : SpreadsheetApp.create('W07_theo_doi');
  if (!tf) DriveApp.getFileById(ss.getId()).moveTo(pc);
  const out = [['file', 'tổng dòng', 'đã điền nguoi_duyet', 'hiệu chỉnh xong (/20)', '% xong', 'sửa lần cuối']];
  TEAM.forEach(function (m) {
    const f = fileInFolder_(pc, 'W07_audit_' + m.key);
    if (!f) return;
    const v = SpreadsheetApp.open(f).getSheetByName('error_audit').getDataRange().getValues().slice(1);
    const filled = function (r) { return String(r[11]).trim() !== ''; };
    const done = v.filter(filled).length;
    const cal = v.filter(function (r) { return r[1] === 'hieu_chinh' && filled(r); }).length;
    out.push([f.getName(), v.length, done, cal, Math.round(100 * done / v.length) + '%', f.getLastUpdated()]);
  });
  const sh = ss.getSheets()[0];
  sh.clear(); sh.getRange(1, 1, out.length, out[0].length).setValues(out);
  Logger.log('Xem W07_theo_doi: ' + ss.getUrl());
}
