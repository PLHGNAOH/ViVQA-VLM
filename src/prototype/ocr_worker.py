"""
ocr_worker.py — OCR chạy ở TIẾN TRÌNH RIÊNG cho prototype (W13 bước 7).

Vấn đề (W08, notebook 10/19): `paddlepaddle-gpu` mang thư viện CUDA/NCCL riêng. Nạp Paddle GPU chung một tiến trình
với torch + Qwen2.5-VL dễ hỏng thư viện của nhau. Cache OCR của mọi run eval được dựng ở phiên RIÊNG (không có
transformers), với Paddle GPU 3.3.1 + VietOCR — đó là môi trường đã chứng minh chạy được.

Cách làm: giao diện + Qwen chạy ở tiến trình chính; OCR chạy trong một tiến trình con giữ engine nạp sẵn
(PaddleDetVietOCREngine của src/data/ocr.py — CÙNG engine đã dựng cache), nói chuyện qua stdin/stdout bằng JSON
từng dòng. Tiến trình chính không import paddle.

Giao thức (một dòng JSON mỗi thông điệp; dòng trả lời có tiền tố PROTO_PREFIX để tách khỏi log lẫn vào stdout):
    con  -> cha : @@OCR@@ {"ready": true, "info": {...engine.info}, "backend": ...}
    cha  -> con : {"id": 1, "image": "/path/a.jpg"}
    con  -> cha : @@OCR@@ {"id": 1, "ok": true, "result": {"w","h","lines"}, "sec": 0.42}
                  @@OCR@@ {"id": 1, "ok": false, "error": "..."}
    cha  -> con : {"quit": true}
Mọi thứ Paddle/VietOCR in ra (kể cả từ C) bị chuyển sang stderr -> file log, không lẫn vào kênh giao thức.

Chạy tay (kiểm engine):   python -m src.prototype.ocr_worker --backend paddledet_vietocr --device gpu
Kiểm thử offline:         python -m src.prototype.ocr_worker --selftest     (engine mock, tiến trình con thật)
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

PROTO_PREFIX = "@@OCR@@ "
WORKER_VERSION = "ocr-worker-v1"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Các mục engine quyết định kết quả OCR — phải trùng cache đã dùng lúc eval (như luật notebook 19)
ENGINE_MUST_MATCH = ("paddleocr", "det_model", "rec_model", "vietocr_config", "lang")
REF_MANIFEST = os.path.join(REPO_ROOT, "experiments", "W11_ocr_cache_vivqa", "ocr_cache_manifest.json")


# =============================================================================
# 1) PHÍA TIẾN TRÌNH CON
# =============================================================================
def _worker_main(backend: str, device: str, noise: bool = False) -> int:
    # Kênh giao thức = bản sao fd 1 gốc; sau đó fd 1 trỏ sang stderr -> print/log của thư viện không phá giao thức
    proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def send(obj: Dict[str, Any]) -> None:
        proto.write(PROTO_PREFIX + json.dumps(obj, ensure_ascii=False) + "\n")
        proto.flush()

    from src.data.ocr import get_ocr_engine
    try:
        if noise:
            print("NOISE: thư viện in linh tinh ra stdout lúc nạp model")
            os.write(1, b"NOISE from C-level write\n")
        engine = get_ocr_engine(backend, device=device)
    except Exception as e:                                   # nạp engine hỏng -> báo cha rồi thoát
        send({"ready": False, "error": f"{type(e).__name__}: {e}"})
        return 1
    send({"ready": True, "backend": backend, "device": device, "info": engine.info,
          "worker": WORKER_VERSION, "pid": os.getpid()})
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            send({"id": None, "ok": False, "error": "dòng không phải JSON"})
            continue
        if msg.get("quit"):
            break
        rid, path = msg.get("id"), msg.get("image")
        t0 = time.time()
        try:
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"không thấy ảnh {path}")
            if noise:
                print(f"NOISE: đang đọc {os.path.basename(path)}")
            res = engine.read(path)
            send({"id": rid, "ok": True, "result": res, "sec": round(time.time() - t0, 4)})
        except Exception as e:
            send({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}", "sec": round(time.time() - t0, 4)})
    return 0


# =============================================================================
# 2) PHÍA TIẾN TRÌNH CHA
# =============================================================================
class OCRWorkerError(RuntimeError):
    pass


class OCRWorkerClient:
    """Khởi động tiến trình OCR con, gửi ảnh, nhận dòng OCR. An toàn khi gọi từ nhiều luồng (khoá tuần tự)."""

    def __init__(self, backend: str = "paddledet_vietocr", device: str = "gpu", python: Optional[str] = None,
                 log_path: Optional[str] = None, start_timeout: float = 900.0, read_timeout: float = 180.0,
                 noise: bool = False):
        self.backend, self.device = backend, device
        self.read_timeout = read_timeout
        self.log_path = log_path or os.path.join(tempfile.gettempdir(), "vivqa_ocr_worker.log")
        self._log = open(self.log_path, "a", encoding="utf-8")
        cmd = [python or sys.executable, "-u", "-m", "src.prototype.ocr_worker", "--serve",
               "--backend", backend, "--device", device] + (["--noise"] if noise else [])
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO_ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        self._proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=self._log, text=True, encoding="utf-8", bufsize=1)
        self._q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        self._lock = threading.Lock()
        self._next_id = 0
        hello = self._get(start_timeout, "khởi động OCR")
        if not hello.get("ready"):
            tail = self.log_tail()
            self.close()
            raise OCRWorkerError(f"OCR không khởi động được: {hello.get('error')}\n{tail}")
        self.info: Dict[str, Any] = hello.get("info") or {}
        self.pid = hello.get("pid")

    def _pump(self) -> None:
        for line in self._proc.stdout:                       # chỉ nhận dòng giao thức; dòng khác (nếu có) bỏ qua
            if line.startswith(PROTO_PREFIX):
                try:
                    self._q.put(json.loads(line[len(PROTO_PREFIX):]))
                except json.JSONDecodeError:
                    pass
        self._q.put(None)                                    # stdout đóng = tiến trình con đã chết

    def _get(self, timeout: float, what: str) -> Dict[str, Any]:
        try:
            msg = self._q.get(timeout=timeout)
        except queue.Empty:
            raise OCRWorkerError(f"quá {timeout:.0f} s chờ {what}\n{self.log_tail()}")
        if msg is None:
            raise OCRWorkerError(f"tiến trình OCR đã dừng (mã {self._proc.poll()}) khi {what}\n{self.log_tail()}")
        return msg

    def log_tail(self, n_chars: int = 2000) -> str:
        try:
            if not self._log.closed:
                self._log.flush()
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                return "--- log OCR (cuối) ---\n" + f.read()[-n_chars:]
        except OSError:
            return ""

    def read(self, image_path: str) -> Dict[str, Any]:
        """{"w","h","lines":[{"text","conf","box"}], "sec"}; lỗi đọc ảnh -> OCRWorkerError."""
        with self._lock:
            if self._proc.poll() is not None:
                raise OCRWorkerError(f"tiến trình OCR đã dừng (mã {self._proc.returncode})\n{self.log_tail()}")
            self._next_id += 1
            rid = self._next_id
            self._proc.stdin.write(json.dumps({"id": rid, "image": os.path.abspath(image_path)}, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            msg = self._get(self.read_timeout, f"OCR ảnh {os.path.basename(image_path)}")
            if msg.get("id") != rid:
                raise OCRWorkerError(f"lệch id giao thức: gửi {rid}, nhận {msg.get('id')}")
            if not msg.get("ok"):
                raise OCRWorkerError(msg.get("error", "lỗi không rõ"))
            return {**msg["result"], "sec": msg.get("sec")}

    def lines(self, image_path: str) -> List[Dict[str, Any]]:
        """Dạng `ocr_fn` mà src.prototype.core.Backend cần."""
        return self.read(image_path)["lines"]

    def close(self) -> None:
        try:
            if self._proc.poll() is None:
                self._proc.stdin.write(json.dumps({"quit": True}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=20)
        except Exception:
            pass
        if self._proc.poll() is None:
            self._proc.kill()
        try:
            self._log.close()
        except Exception:
            pass

    def __enter__(self) -> "OCRWorkerClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def check_engine_matches_cache(info: Dict[str, Any], manifest_path: str = REF_MANIFEST) -> Dict[str, Any]:
    """So engine của tiến trình OCR với engine đã dựng cache (biên bản W11). Trả {"ok", "diff", "device_note"}."""
    with open(manifest_path, encoding="utf-8") as f:
        ref = json.load(f)["engine_info"]
    diff = {k: {"live": info.get(k), "cache": ref.get(k)} for k in ENGINE_MUST_MATCH if info.get(k) != ref.get(k)}
    note = None
    if info.get("device") != ref.get("device") or info.get("paddlepaddle") != ref.get("paddlepaddle"):
        note = (f"khác môi trường (chỉ ghi nhận): device {info.get('device')} vs {ref.get('device')}, "
                f"paddlepaddle {info.get('paddlepaddle')} vs {ref.get('paddlepaddle')}")
    return {"ok": not diff, "diff": diff, "device_note": note}


# =============================================================================
# 3) KIỂM THỬ OFFLINE (engine mock, tiến trình con THẬT)
# =============================================================================
def _selftest() -> None:
    from PIL import Image
    tmp = tempfile.mkdtemp(prefix="ocrw_")
    img = os.path.join(tmp, "a.jpg")
    Image.new("RGB", (64, 32), "white").save(img)
    log = os.path.join(tmp, "w.log")
    with OCRWorkerClient(backend="mock", device="cpu", log_path=log, start_timeout=120, read_timeout=60,
                         noise=True) as c:
        assert c.info.get("engine") == "mock", c.info
        r1 = c.read(img)
        assert r1["lines"][0]["text"] == "MOCK a.jpg" and len(r1["lines"]) == 2, r1
        assert c.lines(img) == r1["lines"]
        try:
            c.read(os.path.join(tmp, "khong_co.jpg"))
            raise AssertionError("ảnh thiếu phải báo lỗi")
        except OCRWorkerError as e:
            assert "không thấy ảnh" in str(e)
        assert c.read(img)["lines"] == r1["lines"]          # vẫn chạy tiếp sau một ảnh lỗi
        # nhiều luồng gọi cùng lúc -> khoá tuần tự, không lệch id
        out: List[Any] = []
        ths = [threading.Thread(target=lambda: out.append(c.read(img)["lines"][0]["text"])) for _ in range(6)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        assert out == ["MOCK a.jpg"] * 6, out
        proc = c._proc
    with open(log, encoding="utf-8") as f:
        assert "NOISE" in f.read()                           # log thư viện đi vào file, không phá giao thức
    assert proc.poll() is not None, "tiến trình OCR phải tắt sau close()"
    # engine không nạp được -> lỗi rõ ràng, không treo
    try:
        OCRWorkerClient(backend="khong_ton_tai", device="cpu", log_path=log, start_timeout=60)
        raise AssertionError("backend sai phải báo lỗi")
    except OCRWorkerError as e:
        assert "không khởi động được" in str(e)
    # so engine với cache (biên bản W11 trong repo)
    ref = json.load(open(REF_MANIFEST, encoding="utf-8"))["engine_info"]
    assert check_engine_matches_cache(dict(ref))["ok"]
    assert not check_engine_matches_cache({**ref, "det_model": "PP-OCRv5_server_det"})["ok"]
    assert check_engine_matches_cache({**ref, "device": "cpu"})["device_note"]
    print("✅ selftest ocr_worker OK (giao thức, log tách kênh, ảnh lỗi, đa luồng, tắt tiến trình, engine = cache)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", action="store_true", help="chạy vai tiến trình con (do OCRWorkerClient gọi)")
    ap.add_argument("--backend", default="paddledet_vietocr")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--noise", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if a.serve:
        sys.exit(_worker_main(a.backend, a.device, a.noise))
    with OCRWorkerClient(backend=a.backend, device=a.device) as c:     # chạy tay: in engine + kiểm với cache
        print(json.dumps(c.info, ensure_ascii=False, indent=1))
        print(check_engine_matches_cache(c.info))


if __name__ == "__main__":
    main()
