"""红星光 CSL S1 后端：FastAPI + WebSocket 静态手势识别。

上行 {type:"frame", data:"data:image/jpeg;base64,...", ts}
下行 {type:"static_word", word, conf, ts}
YOLO 推理跑在独立后台线程，避免阻塞 WebSocket 事件循环。
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
import cv2
import numpy as np

from inference.static_yolo import StaticYOLO

ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = ROOT / "frontend/index.html"
CERT_DIR = ROOT / "certs"
FRAME_TIMEOUT_S = 2.0  # 防抖状态超过该时长无帧则重置

# 平板经局域网访问需 secure context 才能用摄像头（getUserMedia），
# 唯一可用路径是 HTTPS。证书就绪即走 TLS；本机调试可设 NO_HTTPS=1 或 --no-https。
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

app = FastAPI(title="红星光 CSL S1 YOLO 旁路")

_model: Optional[StaticYOLO] = None

# 后台推理线程与 WS 事件循环之间的通道
_in_q: queue.Queue = queue.Queue(maxsize=8)
_out_q: queue.Queue = queue.Queue(maxsize=16)


def _build_model() -> StaticYOLO:
    return StaticYOLO()


def _infer_loop() -> None:
    """后台线程：从输入队列取帧 → YOLO 推理 → 结果写回输出队列。"""
    while True:
        frame = _in_q.get()
        if frame is None:
            break
        ts = frame.get("ts")
        try:
            word, conf = _model.infer(frame["bgr"])
        except Exception:
            word, conf = None, None
        if word is not None and conf is not None:
            try:
                _out_q.put_nowait(
                    {"type": "static_word", "word": word, "conf": conf, "ts": ts}
                )
            except queue.Full:
                # 输出积压时丢弃最旧结果，保证下游 WS 消费是最新的
                try:
                    _out_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    _out_q.put_nowait(
                        {"type": "static_word", "word": word, "conf": conf, "ts": ts}
                    )
                except queue.Full:
                    pass
        _in_q.task_done()


@app.on_event("startup")
async def _startup() -> None:
    global _model
    _model = _build_model()
    _worker = threading.Thread(target=_infer_loop, name="yolo-infer", daemon=True)
    _worker.start()
    print("[startup] YOLO static model loaded")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok" if _model is not None else "loading",
            "model_loaded": _model is not None,
            "classes": len(_model.names) if _model else 0,
        }
    )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.websocket("/ws/recog")
async def ws_recog(ws: WebSocket) -> None:
    await ws.accept()
    if _model is None:
        await ws.send_json({"type": "error", "msg": "model not loaded"})
        await ws.close()
        return

    last_frame_ts: Optional[float] = None

    async def _pump() -> None:
        """把推理结果从后台线程 queue 推给浏览器。

        queue.get 是阻塞调用，须丢进线程执行器，避免冻结 WS 事件循环。
        """
        while True:
            try:
                msg = await asyncio.to_thread(_out_q.get, timeout=FRAME_TIMEOUT_S)
            except queue.Empty:
                # 超时无结果：清理防抖状态，避免旧手势在恢复后立即重发
                _model.reset()
                continue
            try:
                await ws.send_json(msg)
            except Exception:
                return

    pump_task = asyncio.ensure_future(_pump())
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if data.get("type") != "frame":
                continue
            b64 = data.get("data", "")
            if not b64.startswith("data:image/"):
                continue
            try:
                jpg = base64.b64decode(b64.split(",", 1)[1])
            except (IndexError, ValueError):
                continue
            arr = np.frombuffer(jpg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            ts = data.get("ts", int(time.time() * 1000))
            try:
                _in_q.put_nowait({"bgr": img, "ts": ts})
            except queue.Full:
                # 输入积压超过 8 帧：旧帧作废、新帧取代
                try:
                    _in_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    _in_q.put_nowait({"bgr": img, "ts": ts})
                except queue.Full:
                    pass
            last_frame_ts = ts
    except WebSocketDisconnect:
        pass
    finally:
        pump_task.cancel()
        _model.reset()


def _certs_ready() -> tuple[str, str] | None:
    cert, key = CERT_DIR / "cert.pem", CERT_DIR / "key.pem"
    if cert.is_file() and key.is_file():
        return str(cert), str(key)
    return None


def main() -> None:
    import uvicorn

    no_https = "--no-https" in sys.argv or os.environ.get("NO_HTTPS") == "1"
    kwargs = {}
    if not no_https:
        ssl = _certs_ready()
        if ssl:
            kwargs.update(ssl_certfile=ssl[0], ssl_keyfile=ssl[1])
            print(f"[server] HTTPS enabled: {ssl[0]}")
        else:
            print(
                "[server] certs/cert.pem + key.pem 缺失，退化为明文 HTTP（仅限本机调试）"
            )
    else:
        print("[server] --no-https：明文 HTTP（仅限本机调试）")

    uvicorn.run("backend.main:app", host=DEFAULT_HOST, port=DEFAULT_PORT, **kwargs)


if __name__ == "__main__":
    main()
