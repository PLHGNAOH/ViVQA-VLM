// Thu hồi nhanh v2: khoá THƯ MỤC trước (file con tự hết quyền), bỏ qua nhanh file đã sạch, tự dừng trước 6 phút.
function thuHoiNhanh() {
  const t0 = Date.now(), LIMIT = 5 * 60 * 1000;
  const emails = TEAM.map(function (m) { return m.email.toLowerCase(); }).filter(function (e) { return e.indexOf('@') > 0; });
  const qs = [];
  emails.forEach(function (e) { qs.push("'" + e + "' in writers", "'" + e + "' in readers"); });
  qs.push("visibility = 'anyoneWithLink'", "visibility = 'anyoneCanFind'", "visibility = 'domainWithLink'", "visibility = 'domainCanFind'");
  const done = {}; let n = 0, skipped = 0, timeout = false;
  const fix = function (item) {
    if (done[item.getId()]) return; done[item.getId()] = true;
    const ppl = item.getEditors().concat(item.getViewers()).map(function (u) { return u.getEmail().toLowerCase(); });
    const hit = emails.filter(function (e) { return ppl.indexOf(e) >= 0; });
    const pub = String(item.getSharingAccess()) !== 'PRIVATE';
    if (!hit.length && !pub) { skipped++; return; }
    hit.forEach(function (e) { try { item.removeEditor(e); } catch (x) {} try { item.removeViewer(e); } catch (x) {} });
    if (pub) { try { item.setSharing(DriveApp.Access.PRIVATE, DriveApp.Permission.NONE); } catch (x) {} }
    n++; Logger.log('Đã khoá: ' + item.getName());
  };
  ['searchFolders', 'searchFiles'].forEach(function (fn) {          // THƯ MỤC trước, file sau
    qs.forEach(function (q) {
      if (timeout) return;
      try {
        const it = DriveApp[fn](q + " and 'me' in owners");
        while (it.hasNext()) { if (Date.now() - t0 > LIMIT) { timeout = true; return; } fix(it.next()); }
      } catch (e) { Logger.log('Bỏ qua: ' + fn + ' | ' + q); }
    });
  });
  Logger.log((timeout ? 'TẠM DỪNG (hết giờ) — bấm Chạy lại. ' : 'XONG. ') + 'Đã khoá ' + n + ' mục, bỏ qua ' + skipped + ' mục đã sạch.');
}