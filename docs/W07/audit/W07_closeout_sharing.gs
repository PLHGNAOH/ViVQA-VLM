/**
 * W07 closeout — kiểm tra và THU HỒI mọi quyền chia sẻ trên Google Drive của chủ tài khoản.
 *
 * Cách dùng (script.google.com → mở lại dự án W07_audit_setup → bấm + cạnh "Tệp" → Tập lệnh → dán file này):
 *   1) Chạy kiemTraChiaSe   → CHỈ ĐỌC. Tạo/ghi file "W07_bao_cao_chia_se" ở thư mục gốc My Drive:
 *                              liệt kê mọi thư mục/file đang chia sẻ cho người khác hoặc mở bằng link.
 *   2) Xem báo cáo. Nếu đồng ý thu hồi hết → chạy thuHoiQuyen (xoá mọi người khác + tắt chia sẻ link).
 *   3) Chạy lại kiemTraChiaSe → báo cáo phải còn 0 dòng.
 * Phạm vi quét: MỌI thư mục trong My Drive (đệ quy) + file trong thư mục ≤ 500 file
 *   + mọi file Google (Sheets/Docs/Colab…) do bạn sở hữu + mọi file mở bằng link.
 *   Thư mục > 500 file (vd. ảnh dataset) chỉ kiểm quyền của THƯ MỤC (file con thừa hưởng quyền thư mục).
 * Giới hạn: Apps Script chạy tối đa 6 phút; nếu báo "chưa quét hết" thì chạy lại (đã quét sẽ quét lại, an toàn).
 */
const MAX_FILES_PER_FOLDER = 500;
const TIME_BUDGET_MS = 5 * 60 * 1000;
const REPORT_NAME = 'W07_bao_cao_chia_se';

function me_() { return Session.getEffectiveUser().getEmail().toLowerCase(); }

function others_(item, owner) {
  let emails = [];
  try { emails = item.getEditors().concat(item.getViewers()).map(function (u) { return u.getEmail().toLowerCase(); }); } catch (e) { return ['(không đọc được quyền)']; }
  return emails.filter(function (e, i) { return e && e !== owner && emails.indexOf(e) === i; });
}

function access_(item) {
  try { return String(item.getSharingAccess()); } catch (e) { return 'UNKNOWN'; }
}

function inspect_(item, kind, path, owner, out, seen) {
  const id = item.getId();
  if (seen[id]) return; seen[id] = true;
  let ownerEmail = '';
  try { ownerEmail = item.getOwner() ? item.getOwner().getEmail().toLowerCase() : ''; } catch (e) {}
  if (ownerEmail && ownerEmail !== owner) return;                 // không phải của mình → bỏ qua
  const acc = access_(item), ppl = others_(item, owner);
  if (acc !== 'PRIVATE' || ppl.length) out.push({ item: item, kind: kind, path: path, access: acc, people: ppl });
}

function scan_() {
  const t0 = Date.now(), owner = me_(), out = [], seen = {};
  let complete = true;
  const walk = function (folder, path) {
    if (Date.now() - t0 > TIME_BUDGET_MS) { complete = false; return; }
    inspect_(folder, 'thư mục', path, owner, out, seen);
    const files = folder.getFiles(); let n = 0;
    while (files.hasNext() && n < MAX_FILES_PER_FOLDER) { const f = files.next(); n++; inspect_(f, 'file', path + '/' + f.getName(), owner, out, seen); }
    const subs = folder.getFolders();
    while (subs.hasNext()) { const s = subs.next(); walk(s, path + '/' + s.getName()); }
  };
  walk(DriveApp.getRootFolder(), 'My Drive');
  ['mimeType contains \'vnd.google\' and \'me\' in owners',
   'visibility = \'anyoneWithLink\'', 'visibility = \'anyoneCanFind\'',
   'visibility = \'domainWithLink\'', 'visibility = \'domainCanFind\''].forEach(function (q) {
    try {
      const it = DriveApp.searchFiles(q);
      while (it.hasNext()) { const f = it.next(); inspect_(f, 'file', '(tìm kiếm) ' + f.getName(), owner, out, seen); }
    } catch (e) { Logger.log('Bỏ qua truy vấn: ' + q + ' — ' + e); }
  });
  return { rows: out, complete: complete, owner: owner };
}

function writeReport_(res, title) {
  const it = DriveApp.getRootFolder().getFilesByName(REPORT_NAME);
  const ss = it.hasNext() ? SpreadsheetApp.open(it.next()) : SpreadsheetApp.create(REPORT_NAME);
  const sh = ss.getSheets()[0]; sh.clear();
  const data = [[title + (res.complete ? '' : '  — CHƯA QUÉT HẾT (hết giờ), chạy lại'), '', '', '', ''],
                ['loại', 'đường dẫn', 'chia sẻ link', 'người khác có quyền', 'url']];
  res.rows.forEach(function (r) { data.push([r.kind, r.path, r.access, r.people.join(', '), r.item.getUrl()]); });
  sh.getRange(1, 1, data.length, 5).setValues(data);
  sh.getRange(2, 1, 1, 5).setFontWeight('bold');
  Logger.log(title + ': ' + res.rows.length + ' mục còn chia sẻ' + (res.complete ? '' : ' (CHƯA QUÉT HẾT)') + '. Báo cáo: ' + ss.getUrl());
}

function kiemTraChiaSe() {
  const res = scan_();
  writeReport_(res, 'Kiểm tra (chỉ đọc) ' + new Date().toLocaleString('vi-VN'));
}

function thuHoiQuyen() {
  const res = scan_();
  let fixed = 0, failed = [];
  res.rows.forEach(function (r) {
    try {
      r.people.forEach(function (e) {
        try { r.item.removeEditor(e); } catch (x) {}
        try { r.item.removeViewer(e); } catch (x) {}
      });
      if (r.access !== 'PRIVATE') r.item.setSharing(DriveApp.Access.PRIVATE, DriveApp.Permission.NONE);
      fixed++;
    } catch (e) { failed.push(r.path + ': ' + e); }
  });
  Logger.log('Đã thu hồi: ' + fixed + ' mục. Lỗi: ' + failed.length);
  failed.forEach(function (x) { Logger.log('  ' + x); });
  writeReport_(scan_(), 'Sau khi thu hồi ' + new Date().toLocaleString('vi-VN'));
}
