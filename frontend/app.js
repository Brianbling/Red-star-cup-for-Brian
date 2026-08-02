'use strict';

// WS 地址跟随页面协议：页面是 https（平板经局域网访问）→ wss；否则 ws
const WS_URL = (location.protocol === 'https:'
  ? 'wss://' + location.host
  : 'ws://' + location.host) + '/ws/recog';
const FPS = 10;                 // 上行帧率
const FRAME_MS = 1000 / FPS;    // 采样周期
const JPEG_QUALITY = 0.6;       // 与后端契约一致
const W = 640, H = 480;         // 480p 目标尺寸
const CARD_MAX = 5;             // 结果卡片上限
const FRAME_TIMEOUT_MS = 400;   // 超过该时长无新帧 → 判定连接卡住

// ---- DOM ----
const $ = id => document.getElementById(id);
const video      = $('video');
const btnStart   = $('btn-start');
const btnStop    = $('btn-stop');
const camState   = $('cam-state');
const wsDot      = $('ws-dot');
const wsLabel    = $('ws-label');
const lampConn   = $('lamp-conn');
const lampRecog  = $('lamp-recog');
const lampConf   = $('lamp-conf');
const wordList   = $('word-list');
const footWs     = $('foot-ws');
const footSent   = $('foot-sent');
const footRecv   = $('foot-recv');
const footDrop   = $('foot-drop');

// ---- 状态 ----
let ws = null;
let stream = null;
let lastFrameTs = 0;
let sentCount = 0;
let recvCount = 0;
let dropCount = 0;
let lastWord = '';         // 防抖：同词不重复入卡
let lastWordTs = 0;

// ---- 摄像头 ----
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    video.classList.remove('placeholder');
    btnStart.disabled = true;
    btnStop.disabled = false;
    camState.textContent = '运行中 · 10fps · 480p JPEG 上行';
  } catch (err) {
    camState.textContent = '摄像头启动失败：' + (err && err.message ? err.message : err);
    alert('无法访问摄像头，请检查浏览器权限设置。');
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
    video.srcObject = null;
  }
  btnStart.disabled = false;
  btnStop.disabled = true;
  camState.textContent = '未启动 · 10fps · 480p JPEG';
  setRecogState('空闲');
}

// ---- 取帧上行（防抖：上一次发送未落盘则不叠发，计丢帧） ----
const frameCanvas = document.createElement('canvas');
frameCanvas.width = W;
frameCanvas.height = H;
const frameCtx = frameCanvas.getContext('2d');

function sendFrame() {
  if (!stream || !ws || ws.readyState !== WebSocket.OPEN) return;
  if (video.readyState < 2) return; // 无可用帧

  const now = Date.now();
  if (now - lastFrameTs < FRAME_MS) return; // 节流到 10fps
  lastFrameTs = now;

  // 发送缓冲区积压超过约 4 帧时，旧帧作废、新帧取代 → 计丢帧
  // （后端无逐帧 ack，唯一可靠的丢帧信号是浏览器 send 缓冲积压）
  if (ws.bufferedAmount > 4 * (W * H * 0.5)) {
    dropCount++;
  }

  frameCtx.drawImage(video, 0, 0, W, H);
  const dataUrl = frameCanvas.toDataURL('image/jpeg', JPEG_QUALITY);

  try {
    ws.send(JSON.stringify({ type: 'frame', data: dataUrl, ts: now }));
    sentCount++;
  } catch (err) {
    // 发送失败通常意味着连接已断，交给 onclose 统一处理
  }
  renderStats();
}

// ---- WebSocket ----
function connect() {
  // 建连期状态灯置"连接中"
  wsDot.className = 'dot';
  wsLabel.textContent = '连接中';
  lampConn.textContent = '连接中';
  lampConn.className = 'lamp-value connecting';
  footWs.textContent = '连接中';

  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    wsDot.className = 'dot ok';
    wsLabel.textContent = '已连接';
    lampConn.textContent = '已连接';
    lampConn.className = 'lamp-value recog';
    footWs.textContent = '已连接';
    // 连接恢复后重启采样
    lastFrameTs = 0;
  };

  ws.onmessage = (ev) => {
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch (err) {
      return;
    }
    recvCount++;
    if (msg.type === 'static_word') {
      // 后端已做同类别连续帧防抖；前端再按 2.5s 去重展示，避免同一手势反复入卡
      const now = Date.now();
      if (msg.word && (msg.word !== lastWord || now - lastWordTs > 2500)) {
        lastWord = msg.word;
        lastWordTs = now;
        addCard(msg.word, msg.conf, msg.ts);
      }
      setRecogState('识别中');
      lampConf.textContent = formatConf(msg.conf);
    }
    renderStats();
  };

  ws.onclose = () => {
    wsDot.className = 'dot bad';
    wsLabel.textContent = '未连接';
    lampConn.textContent = '未连接';
    lampConn.className = 'lamp-value idle';
    footWs.textContent = '未连接';
    setRecogState('空闲');
    // 断线重连
    setTimeout(connect, 2000);
  };

  ws.onerror = () => {
    wsDot.className = 'dot bad';
    wsLabel.textContent = '连接错误';
    footWs.textContent = '错误';
  };
}

// ---- 结果渲染 ----
function addCard(word, conf, ts) {
  // 移除占位提示
  const empty = wordList.querySelector('.empty-hint');
  if (empty) empty.remove();

  const li = document.createElement('li');
  li.className = 'word-card';
  const time = ts ? new Date(ts).toLocaleTimeString('zh-CN', { hour12: false }) : '--';
  li.innerHTML =
    '<span class="word">' + escapeHtml(word) + '</span>' +
    '<span class="meta"><span class="conf">' + formatConf(conf) +
    '</span><br><span class="time">' + time + '</span></span>';
  wordList.prepend(li);

  // 只保留最近 CARD_MAX 条
  while (wordList.children.length > CARD_MAX) {
    wordList.removeChild(wordList.lastChild);
  }
}

function setRecogState(text) {
  lampRecog.textContent = text;
  lampRecog.className = 'lamp-value ' + (text === '识别中' ? 'recog' : 'idle');
}

function formatConf(conf) {
  const c = Number(conf);
  if (isNaN(c)) return '--';
  return (c * 100).toFixed(0) + '%';
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function renderStats() {
  footSent.textContent = sentCount;
  footRecv.textContent = recvCount;
  footDrop.textContent = dropCount;
}

// ---- 驱动 ----
btnStart.addEventListener('click', startCamera);
btnStop.addEventListener('click', stopCamera);

// 全局采样循环：有流且 WS 打开时定时取帧
setInterval(sendFrame, FRAME_MS);
connect();
