"""
finial4.py
===================================================
finial3 精简版：**砍掉"推理后视频保存"功能**，其余全部保留。

相对 finial3 的唯一差别：
    ✗ 不再把带 AI 痕迹的视频写到 luzhi_annotated/（这个功能被砍）
    ✓ 抓拍 jpg 仍然是"带检测框"的那一帧（取证/上云的图照样有标注）

保留的功能：
    1. **必须联网才会启动**：开机后阻塞探活外网，不通就死等 + 定期重连 WiFi
    2. 摄像头硬件编码录像，每段 2 分钟存 MP4 到 luzhi/
    3. 后台 AI 逐段读 MP4：YOLOP 压线 + YOLOv8 抛物 + 应急车道占用 + HyperLPR3 车牌
    4. 触发违规：存带框抓拍 jpg → 异步传 OSS → 异步推事件给 L610(→MQTT)
    5. 驾驶疲劳提醒；SIGTERM/SIGINT 优雅停机

依赖：
    pip3 install pyserial pynmea2 hyperlpr3 oss2

------------------------------------------------------------
开机自启（替换 finial3）：

    # 0) ASCII 软链
    sudo ln -sf /home/sunrise/finial4.py /home/sunrise/finialnow.py

    # 1) 改 ExecStart 指向 finial4（用上面那个软链）
    sudo nano /etc/systemd/system/yingjianrecord.service
    #   ExecStart=/usr/bin/python3 -u /home/sunrise/finialnow.py
    #   建议同时加： Environment=PYTHONUNBUFFERED=1

    # 2) 重载并重启
    sudo systemctl daemon-reload
    sudo systemctl restart yingjianrecord.service
    sudo journalctl -u yingjianrecord -f

注意：本脚本与 l610_rdk_sim2 - 副本2.py 是两个独立服务，都要开机自启。
"""

import os
import re
import glob
import cv2
import json
import time
import queue
import signal
import socket
import argparse
import threading
import subprocess
import numpy as np
import oss2
from datetime import datetime, timedelta
from hobot_dnn import pyeasy_dnn as dnn

# === Horizon Hardware API ===
try:
    from hobot_vio import libsrcampy
except ImportError:
    print("[FATAL ERROR] hobot_vio not found. Must run on RDK X5.")
    exit(1)

# === GPS (optional) ===
try:
    import serial
    import pynmea2
    GPS_AVAILABLE = True
except ImportError:
    print("[WARN] pyserial / pynmea2 not installed. GPS disabled.")
    GPS_AVAILABLE = False

# === HyperLPR3 (optional, robust) ===
try:
    import hyperlpr3 as lpr3
    HYPERLPR_AVAILABLE = True
except Exception as e:
    print(f"[WARN] hyperlpr3 unavailable ({type(e).__name__}: {e}).")
    print("       License plate disabled (fallback to NOPLATE).")
    print("       修复：rm -rf ~/.hyperlpr3  然后再 import 一次。")
    HYPERLPR_AVAILABLE = False
    lpr3 = None

# ================= Configuration =================
# --- WiFi ---
WIFI_CONNECT = True
# 多个候选热点：从上到下依次尝试，连上任意一个就够了。
# 以后要加热点，就往这个列表里再加一行 ("热点名", "密码")。
WIFI_NETWORKS = [
    ("vivo X100 Pro", "houzisi666"),
    ("队友热点名_改成实际的", "队友密码_改成实际的"),
]
WIFI_DEVICE = "wlan0"
WIFI_TIMEOUT = 30
WIFI_SETTLE_SEC = 3
WIFI_RETRY = 3

# --- 必须联网才启动 ---
REQUIRE_INTERNET = True
INTERNET_CHECK_HOSTS = [
    ("oss-cn-shanghai.aliyuncs.com", 443),
    ("aliyun.com", 443),
    ("www.baidu.com", 443),
]
INTERNET_CHECK_TIMEOUT = 3.0
ONLINE_RETRY_INTERVAL = 5
ONLINE_LOG_INTERVAL = 30
WIFI_REKICK_INTERVAL = 60

VIDEO_DIR = "/home/sunrise/car_video_file/luzhi"
SNAPSHOT_DIR = "/home/sunrise/violation_snapshots/luzhi"

# --- Recorder ---
CAMERA_ID = 0
SEGMENT_TIME = 120
WANT_WIDTH = 1280
WANT_HEIGHT = 720
WANT_FPS = 90

# 每多少帧推理一次（不存视频后，这只影响检测密度与 AI 负载）。默认 3。
AI_FRAME_STRIDE = 3

# --- Models ---
YOLOP_MODEL_PATH = "/home/sunrise/models/yolop_320.bin"
YOLOV8_MODEL_PATH = "/home/sunrise/models/model.bin"

# --- AI thresholds ---
YOLOP_CONF_THRES = 0.50
YOLOP_NMS_THRES = 0.45
YOLOP_CROSSING_DENSITY = 0.25
YOLOV8_CONF_THRES = 0.2    # 实测抛物最高分约0.29、噪声约0.06，取0.2兼顾召回；偏低，注意观察误报
YOLOV8_NMS_THRES = 0.45

# --- Emergency lane occupancy（应急车道占用）---
EMERGENCY_LANE_ENABLE = True
EMERGENCY_LANE_SIDE = "right"           # "right" or "left"
EMERGENCY_LANE_X_MIN_RATIO = 0.72
EMERGENCY_LANE_X_MAX_RATIO = 0.98
EMERGENCY_LANE_Y_MIN_RATIO = 0.55
EMERGENCY_LANE_MIN_BOX_H_RATIO = 0.18
EMERGENCY_LANE_CONFIRM_SECONDS = 2.0

# --- Driver rest reminder（疲劳提醒）---
REST_REMINDER_ENABLE = True
REST_REMINDER_AFTER_SECONDS = 4 * 60 * 60
REST_REMINDER_REPEAT_SECONDS = 4 * 60 * 60

# --- License plate ---
PLATE_MIN_CONF = 0.50

# --- Low-light enhancement ---
ENHANCE_MODE = 'auto'
DARK_BRIGHTNESS_THRESHOLD = 60

# --- GPS ---
GPS_PORT = "/dev/ttyUSB0"
GPS_BAUD = 9600
LOCAL_TZ_OFFSET = timedelta(hours=8)

# --- Time validity ---
MIN_VALID_YEAR = 2024
TIME_WAIT_TIMEOUT = 120

# --- Aliyun OSS ---
OSS_ACCESS_KEY_ID = 'LTAI5t6MuwmAotkkAQePHBik'
OSS_ACCESS_KEY_SECRET = 'Tft7C9KczuuBovrrbEqlwPrFYsZvB8'
OSS_ENDPOINT = 'oss-cn-shanghai.aliyuncs.com'
OSS_BUCKET_NAME = 'rdk-car-video'

# --- L610 事件推送（图片走 OSS，事件 JSON 走这里给 L610 → MQTT）---
L610_PUSH_ENABLED = True
L610_EVENT_HOST = "127.0.0.1"
L610_EVENT_PORT = 9610
L610_EVENT_TIMEOUT = 2.0

EVENT_NAME_MAP = {
    "Line_Crossing": "车辆压线行驶",
    "Throwing_Garbage": "车窗抛物",
    "Emergency_Lane_Occupied": "占用应急车道",
}

# 优雅停机：收到 SIGTERM/SIGINT 时置位，各线程据此收尾
stop_event = threading.Event()

video_queue = queue.Queue()

try:
    auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
    bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET_NAME)
    print("[OSS] Aliyun OSS initialized.")
except Exception as e:
    print(f"[OSS] Init failed: {e}")
    bucket = None
# ========================================================


# ================= WiFi =================
def _is_root():
    try:
        return os.geteuid() == 0
    except AttributeError:
        return False


def _nmcli_cmd(args, need_sudo=True):
    if need_sudo and not _is_root():
        return ['sudo', 'nmcli'] + args
    return ['nmcli'] + args


def _run(cmd, timeout=10, capture=True):
    try:
        kwargs = dict(timeout=timeout)
        if capture:
            kwargs.update(capture_output=True, text=True)
        return subprocess.run(cmd, **kwargs)
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return None


def connect_wifi(device=WIFI_DEVICE, timeout=WIFI_TIMEOUT, retries=WIFI_RETRY):
    """
    依次尝试 WIFI_NETWORKS 里的每个候选热点，连上任意一个即返回 True。
    已经联网时直接跳过，避免把好端端的连接断掉重连。
    """
    ok, _ = _check_internet()
    if ok:
        print("[WiFi] 已联网，跳过重连。")
        return True

    names = [s for s, _ in WIFI_NETWORKS]
    print(f"[WiFi] 候选热点（连上任一即可）: {names}  uid={os.geteuid() if hasattr(os, 'geteuid') else 'na'}")

    for attempt in range(1, retries + 1):
        r = _run(_nmcli_cmd(['device', 'status'], need_sudo=False), timeout=5)
        if r is None:
            print(f"[WiFi] nmcli not available (attempt {attempt}/{retries}), wait 5s and retry.")
            time.sleep(5)
            continue

        for ssid, password in WIFI_NETWORKS:
            print(f"[WiFi] 尝试 '{ssid}' (第 {attempt}/{retries} 轮) ...")
            _run(_nmcli_cmd(['device', 'disconnect', device]), timeout=10)
            r = _run(
                _nmcli_cmd(['device', 'wifi', 'connect', ssid, 'password', password]),
                timeout=timeout,
            )
            if r is not None and r.returncode == 0:
                print(f"[WiFi] OK，已连上 '{ssid}'. {(r.stdout or '').strip()}")
                time.sleep(WIFI_SETTLE_SEC)
                ip = _run(["hostname", "-I"], timeout=3)
                if ip and ip.returncode == 0:
                    print(f"[WiFi] IP: {(ip.stdout or '').strip()}")
                return True
            else:
                err = (r.stderr or r.stdout or "").strip() if r is not None else "timeout"
                print(f"[WiFi] '{ssid}' 连接失败: {err}")

        time.sleep(5)

    print(f"[WiFi] {retries} 轮都没连上任何候选热点（本轮）。")
    return False


# ================= Internet readiness =================
def _check_internet():
    for host, port in INTERNET_CHECK_HOSTS:
        try:
            with socket.create_connection((host, port), timeout=INTERNET_CHECK_TIMEOUT):
                return True, (host, port)
        except (OSError, socket.timeout):
            continue
    return False, None


def wait_until_online():
    """阻塞直到外网可达；期间定期重连 WiFi。收到停机信号则直接退出进程。"""
    print("=" * 56)
    print("[Net] REQUIRE_INTERNET=True：联网前脚本绝不往下走")
    print(f"[Net] 探活目标: {', '.join(f'{h}:{p}' for h, p in INTERNET_CHECK_HOSTS)}")
    print("=" * 56)

    connect_wifi()

    attempt = 0
    last_log_t = time.monotonic()
    last_wifi_kick_t = time.monotonic()

    while not stop_event.is_set():
        attempt += 1
        ok, who = _check_internet()
        if ok:
            host, port = who
            print(f"[Net] ✓ ONLINE — TCP {host}:{port} 可达（探活第 {attempt} 次）")
            return

        now = time.monotonic()
        if now - last_wifi_kick_t > WIFI_REKICK_INTERVAL:
            print("[Net] 还连不上外网，重新触发 WiFi 连接 ...")
            connect_wifi(retries=1)
            last_wifi_kick_t = time.monotonic()
        if now - last_log_t > ONLINE_LOG_INTERVAL:
            print(f"[Net] 仍在等联网（已探活 {attempt} 次）... 还会一直等下去。")
            last_log_t = time.monotonic()

        stop_event.wait(ONLINE_RETRY_INTERVAL)

    print("[Net] 收到停机信号，联网等待中止，退出。")
    raise SystemExit(0)


# ================= GPS Reader =================
class GPSState:
    def __init__(self):
        self._lock = threading.Lock()
        self._lat = None
        self._lon = None
        self._utc_dt = None
        self._utc_dt_recv_mono = None
        self._has_fix = False

    def update_pos_time(self, utc_dt, lat, lon):
        with self._lock:
            self._utc_dt = utc_dt
            self._utc_dt_recv_mono = time.monotonic()
            self._lat = lat
            self._lon = lon
            self._has_fix = True

    def update_time_only(self, utc_dt):
        with self._lock:
            self._utc_dt = utc_dt
            self._utc_dt_recv_mono = time.monotonic()

    def clear_fix(self):
        with self._lock:
            self._has_fix = False

    def get_local_time(self):
        with self._lock:
            if self._utc_dt is None or self._utc_dt_recv_mono is None:
                return None
            elapsed = time.monotonic() - self._utc_dt_recv_mono
            return self._utc_dt + LOCAL_TZ_OFFSET + timedelta(seconds=elapsed)

    def get_position(self):
        with self._lock:
            return self._lat, self._lon, self._has_fix


gps_state = GPSState()


def gps_reader_thread():
    if not GPS_AVAILABLE:
        print("[GPS] Disabled (libs missing).")
        return
    print(f"[GPS] Reader on {GPS_PORT} @ {GPS_BAUD}")
    while not stop_event.is_set():
        ser = None
        try:
            ser = serial.Serial(GPS_PORT, GPS_BAUD, timeout=1)
            print("[GPS] Port opened.")
        except Exception as e:
            print(f"[GPS] Open failed: {e}. Retry in 5s.")
            stop_event.wait(5)
            continue

        try:
            while not stop_event.is_set():
                raw = ser.readline()
                if not raw:
                    continue
                try:
                    line = raw.decode('ascii', errors='ignore').strip()
                except Exception:
                    continue
                if not line.startswith('$'):
                    continue
                try:
                    msg = pynmea2.parse(line)
                except pynmea2.ParseError:
                    continue
                if getattr(msg, 'sentence_type', None) == 'RMC':
                    if getattr(msg, 'datestamp', None) and getattr(msg, 'timestamp', None):
                        utc_dt = datetime.combine(msg.datestamp, msg.timestamp)
                        if msg.status == 'A':
                            try:
                                lat = float(msg.latitude) if msg.latitude else None
                                lon = float(msg.longitude) if msg.longitude else None
                                if lat is not None and lon is not None and (lat != 0 or lon != 0):
                                    gps_state.update_pos_time(utc_dt, lat, lon)
                                else:
                                    gps_state.update_time_only(utc_dt)
                            except (ValueError, TypeError):
                                gps_state.update_time_only(utc_dt)
                        else:
                            gps_state.update_time_only(utc_dt)
                            gps_state.clear_fix()
        except Exception as e:
            print(f"[GPS] Read error: {e}")
        finally:
            try:
                if ser:
                    ser.close()
            except Exception:
                pass
        stop_event.wait(2)


# ================= Time helpers =================
def get_current_time():
    gps_time = gps_state.get_local_time()
    if gps_time is not None and gps_time.year >= MIN_VALID_YEAR:
        return gps_time, 'GPS'
    sys_time = datetime.now()
    if sys_time.year >= MIN_VALID_YEAR:
        return sys_time, 'SYS'
    return sys_time, 'INVALID'


def wait_for_valid_time(timeout=TIME_WAIT_TIMEOUT):
    print(f"[Time] Waiting up to {timeout}s for valid time source...")
    start = time.monotonic()
    last_log = 0.0
    while time.monotonic() - start < timeout:
        if stop_event.is_set():
            return
        t, src = get_current_time()
        if src != 'INVALID':
            print(f"[Time] OK. Source={src}, now={t.strftime('%Y-%m-%d %H:%M:%S')}")
            return
        if time.monotonic() - last_log > 10:
            print(f"[Time] Still waiting... system year={t.year}")
            last_log = time.monotonic()
        time.sleep(1)
    print(f"[Time] WARNING: No valid time after {timeout}s. Filenames may be wrong.")


def rest_reminder_thread():
    if not REST_REMINDER_ENABLE:
        return
    start_mono = time.monotonic()
    next_remind = start_mono + REST_REMINDER_AFTER_SECONDS
    while not stop_event.is_set():
        now = time.monotonic()
        if now >= next_remind:
            hours = (now - start_mono) / 3600.0
            print("=" * 56)
            print(f"[REST] Device has been running for {hours:.1f} hours.")
            print("[REST] Please remind the driver to stop and rest.")
            print("=" * 56)
            next_remind = now + REST_REMINDER_REPEAT_SECONDS
        stop_event.wait(5)


def fmt_time_minute_for_name():
    t, _ = get_current_time()
    return t.strftime("%Y%m%d_%H%M")


def fmt_position_for_name():
    lat, lon, has_fix = gps_state.get_position()
    if not has_fix or lat is None or lon is None:
        return "NOGPS"
    return f"{lat:.6f}_{lon:.6f}"


def reserve_unique_path(directory, base_name, ext):
    p = os.path.join(directory, f"{base_name}{ext}")
    if not os.path.exists(p):
        return p
    for i in range(2, 100):
        p = os.path.join(directory, f"{base_name}_{i}{ext}")
        if not os.path.exists(p):
            return p
    return os.path.join(directory, f"{base_name}_{int(time.time())}{ext}")


# ================= License plate =================
_FILENAME_SAFE_RE = re.compile(r'[^\w一-鿿]')


def sanitize_plate_for_filename(text):
    if not text:
        return ""
    return _FILENAME_SAFE_RE.sub('', text)


class PlateRecognizer:
    def __init__(self):
        self._catcher = None
        if not HYPERLPR_AVAILABLE:
            return
        try:
            print("[Plate] Initializing HyperLPR3...")
            self._catcher = lpr3.LicensePlateCatcher()
            print("[Plate] HyperLPR3 ready.")
        except Exception as e:
            print(f"[Plate] HyperLPR3 init failed: {e}. Fallback to NOPLATE.")
            self._catcher = None

    def recognize(self, bgr_image):
        if self._catcher is None or bgr_image is None:
            return None
        try:
            results = self._catcher(bgr_image)
        except Exception as e:
            print(f"  [Plate] Inference error: {e}")
            return None
        if not results:
            return None
        best = max(results, key=lambda r: r[1] if len(r) > 1 else 0.0)
        text = best[0] if len(best) > 0 else ''
        conf = best[1] if len(best) > 1 else 0.0
        if conf < PLATE_MIN_CONF:
            return None
        cleaned = sanitize_plate_for_filename(text)
        return cleaned if cleaned else None


# ================= setup =================
def setup_directories():
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    print(f"[*] Videos (raw)  : {VIDEO_DIR}")
    print(f"[*] Snapshots     : {SNAPSHOT_DIR}")


# ================= Aliyun upload =================
def build_oss_object_key(file_name):
    """OSS 上的对象路径；L610 推到云的 image_key 用同一个值，云端可据此取图。"""
    t, _ = get_current_time()
    date_prefix = t.strftime('%Y-%m-%d')
    return f"violation_logs/{date_prefix}/{file_name}"


def upload_to_aliyun_async(local_file_path, file_name):
    if bucket is None:
        return

    def _upload():
        if not os.path.exists(local_file_path):
            print(f"  [Network] Error: File not found locally: {local_file_path}")
            return
        object_name = build_oss_object_key(file_name)
        try:
            result = bucket.put_object_from_file(object_name, local_file_path)
            if result.status == 200:
                print(f"  [Network] Uploaded: {file_name}")
            else:
                print(f"  [Network] Upload failed status={result.status}")
        except Exception as e:
            print(f"  [Network] Exception: {e}")

    threading.Thread(target=_upload, daemon=True).start()


# ================= L610 event push =================
def event_tag_to_cn(event_tag):
    parts = event_tag.split("_and_")
    return " + ".join(EVENT_NAME_MAP.get(p, p) for p in parts)


def build_event_info(event_tag, plate_str, snap_name):
    """组装推给 L610 上报服务的事件包（纯文本元数据，不含图片）。"""
    lat, lon, has_fix = gps_state.get_position()
    t, _ = get_current_time()
    return {
        "plate": plate_str,
        "event": event_tag,
        "event_name": event_tag_to_cn(event_tag),
        "captured_at": t.strftime("%Y-%m-%d %H:%M:%S"),
        "latitude": lat if has_fix else None,
        "longitude": lon if has_fix else None,
        "location": f"{lat:.6f},{lon:.6f}" if has_fix else "NOGPS",
        "snapshot_file": snap_name,
        "image_key": build_oss_object_key(snap_name),
    }


def notify_l610_async(info):
    """
    非阻塞地把事件 JSON 推给本机 L610 上报服务（- 副本2.py，tcp 9610）。
    协议：一条事件 = 一行 JSON，\\n 结尾。失败只打 log，不影响主链路。
    """
    if not L610_PUSH_ENABLED:
        return
    payload = (json.dumps(info, ensure_ascii=False) + "\n").encode("utf-8")

    def _send():
        try:
            with socket.create_connection(
                (L610_EVENT_HOST, L610_EVENT_PORT),
                timeout=L610_EVENT_TIMEOUT,
            ) as s:
                s.sendall(payload)
            print(f"  [L610] Pushed: plate={info['plate']} event={info['event']}")
        except (ConnectionRefusedError, socket.timeout) as e:
            print(f"  [L610] 推送失败（l610 服务未启动？）: {e}")
        except Exception as e:
            print(f"  [L610] 推送异常: {e}")

    threading.Thread(target=_send, daemon=True).start()


# ================= 低光增强 =================
_GAMMA = 0.7
_GAMMA_LUT = np.array([((i / 255.0) ** (1.0 / _GAMMA)) * 255 for i in range(256)]).astype(np.uint8)


def enhance_low_light(frame_bgr):
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return cv2.LUT(enhanced, _GAMMA_LUT)


def mean_brightness(frame_bgr):
    return float(np.mean(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)))


def maybe_enhance(frame_bgr):
    if ENHANCE_MODE == 'always':
        return enhance_low_light(frame_bgr), True, None
    if ENHANCE_MODE == 'auto':
        bri = mean_brightness(frame_bgr)
        if bri < DARK_BRIGHTNESS_THRESHOLD:
            return enhance_low_light(frame_bgr), True, bri
        return frame_bgr, False, bri
    return frame_bgr, False, None


# ================= YOLOP =================
def bgr2nv12(bgr_img):
    height, width = bgr_img.shape[:2]
    if height % 2 != 0 or width % 2 != 0:
        bgr_img = cv2.resize(bgr_img, (width + width % 2, height + height % 2))
        height, width = bgr_img.shape[:2]
    yuv_i420 = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420)
    y_plane = yuv_i420[:height, :]
    u_plane = yuv_i420[height:height + height // 4, :].reshape(-1)
    v_plane = yuv_i420[height + height // 4:, :].reshape(-1)
    uv_plane = np.empty((height // 2 * width), dtype=np.uint8)
    uv_plane[0::2] = u_plane
    uv_plane[1::2] = v_plane
    uv_plane = uv_plane.reshape(height // 2, width)
    return np.vstack((y_plane, uv_plane))


def get_emergency_lane_roi(orig_w, orig_h):
    """返回应急车道判定 ROI (x1,y1,x2,y2)；按 SIDE 放在画面左 / 右侧。"""
    if not EMERGENCY_LANE_ENABLE:
        return None

    if EMERGENCY_LANE_SIDE == "left":
        x1 = int(orig_w * (1.0 - EMERGENCY_LANE_X_MAX_RATIO))
        x2 = int(orig_w * (1.0 - EMERGENCY_LANE_X_MIN_RATIO))
    else:
        x1 = int(orig_w * EMERGENCY_LANE_X_MIN_RATIO)
        x2 = int(orig_w * EMERGENCY_LANE_X_MAX_RATIO)

    y1 = int(orig_h * EMERGENCY_LANE_Y_MIN_RATIO)
    y2 = orig_h
    x1, x2 = max(0, x1), min(orig_w, x2)
    y1, y2 = max(0, y1), min(orig_h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def postprocess_yolop(frame_bgr, outputs):
    det_out = outputs[0].buffer.squeeze()
    lane_line_out = outputs[2].buffer.squeeze()

    orig_h, orig_w = frame_bgr.shape[:2]
    lane_line_mask = np.argmax(lane_line_out, axis=0).astype(np.uint8)
    lane_line_resized = cv2.resize(lane_line_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    boxes, confidences = [], []
    for det in det_out:
        score = det[4] * det[5]
        if score > YOLOP_CONF_THRES:
            cx, cy, w, h = det[0:4]
            cx_orig, cy_orig = cx * orig_w / 320, cy * orig_h / 320
            w_orig, h_orig = w * orig_w / 320, h * orig_h / 320
            x1, y1 = int(cx_orig - w_orig / 2), int(cy_orig - h_orig / 2)
            boxes.append([x1, y1, int(w_orig), int(h_orig)])
            confidences.append(float(score))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, YOLOP_CONF_THRES, YOLOP_NMS_THRES)

    is_violating = False
    is_emergency_lane = False
    overlay = frame_bgr.copy()
    overlay[lane_line_resized == 1] = [0, 0, 255]
    result_img = cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0)

    emergency_roi = get_emergency_lane_roi(orig_w, orig_h)
    if emergency_roi is not None:
        rx1, ry1, rx2, ry2 = emergency_roi
        cv2.rectangle(result_img, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
        cv2.putText(result_img, "EMG-LANE ROI", (rx1, max(25, ry1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    min_box_h = orig_h * EMERGENCY_LANE_MIN_BOX_H_RATIO

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            x1, y1, x2, y2 = max(0, x), max(0, y), min(orig_w, x + w), min(orig_h, y + h)
            if w * h < 1500 or y2 < orig_h * 0.45:
                continue

            # ---- 压线检测 ----
            trigger_h = max(2, int(h * 0.15))
            trigger_y1 = max(0, y2 - trigger_h)
            trigger_w = max(2, int(w * 0.50))
            trigger_x1 = min(orig_w, x1 + int(w * 0.25))
            trigger_x2 = min(orig_w, trigger_x1 + trigger_w)

            trigger_roi_mask = lane_line_resized[trigger_y1:y2, trigger_x1:trigger_x2]
            overlap_pixels = np.sum(trigger_roi_mask)
            trigger_area = trigger_w * trigger_h
            overlap_density = overlap_pixels / trigger_area if trigger_area > 0 else 0

            box_color = (255, 0, 0)
            label = f"Car: {confidences[i]:.2f}"

            if overlap_density > YOLOP_CROSSING_DENSITY:
                is_violating = True
                box_color = (0, 0, 255)
                label = f"CROSSING! Den:{overlap_density:.2f}"
                cv2.rectangle(result_img, (trigger_x1, trigger_y1), (trigger_x2, y2), (0, 255, 255), -1)

            # ---- 应急车道占用检测（车尾着地点 in ROI + 车够近）----
            if emergency_roi is not None:
                bottom_cx = (x1 + x2) // 2
                bottom_cy = y2
                rx1, ry1, rx2, ry2 = emergency_roi
                in_roi = (rx1 <= bottom_cx <= rx2) and (ry1 <= bottom_cy <= ry2)
                near_enough = (y2 - y1) >= min_box_h
                if in_roi and near_enough:
                    is_emergency_lane = True
                    box_color = (255, 255, 0)
                    label = f"EMG-LANE h={y2 - y1}"

            cv2.rectangle(result_img, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    return result_img, is_violating, is_emergency_lane


# ================= YOLOv8 =================
class YOLOv8ThrowDetector:
    def __init__(self, model_path, conf_thresh, nms_thresh):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        print(f"[AI] Loading YOLOv8 Model: {model_path}")
        self.models = dnn.load(model_path)
        self.model = self.models[0]
        input_ptr = self.model.inputs[0].properties
        self.input_shape = input_ptr.shape
        self.tensor_type = input_ptr.tensor_type
        if self.input_shape[1] == 3:
            self.layout, self.model_h, self.model_w = "NCHW", self.input_shape[2], self.input_shape[3]
        else:
            self.layout, self.model_h, self.model_w = "NHWC", self.input_shape[1], self.input_shape[2]

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, dw, dh

    def preprocess(self, frame_bgr):
        img_padded, self.ratio, self.dw, self.dh = self.letterbox(frame_bgr, (self.model_w, self.model_h))
        if "NV12" in str(self.tensor_type):
            yuv = cv2.cvtColor(img_padded, cv2.COLOR_BGR2YUV_I420)
            h, w = img_padded.shape[:2]
            y = yuv[:h, :]
            u, v = yuv[h: h + h // 4, :].reshape((-1)), yuv[h + h // 4:, :].reshape((-1))
            uv = np.zeros((h // 2, w), dtype=np.uint8)
            uv[:, 0::2], uv[:, 1::2] = u.reshape((h // 2, w // 2)), v.reshape((h // 2, w // 2))
            return np.concatenate([y.reshape(-1), uv.reshape(-1)])
        rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        if self.layout == "NCHW":
            return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))
        return np.ascontiguousarray(rgb)

    def postprocess(self, outputs, orig_w, orig_h):
        raw_buffer = np.array(outputs[0].buffer).copy()
        raw_shape = outputs[0].properties.shape
        if len(raw_shape) == 4:
            raw = raw_buffer.reshape(raw_shape[1], raw_shape[2])
        elif len(raw_shape) == 3:
            raw = raw_buffer.squeeze(0)
        else:
            raw = np.squeeze(raw_buffer)
        if raw.shape[0] == 5 and raw.shape[1] > 5:
            raw = raw.T
        if len(raw.shape) != 2 or raw.shape[1] < 5:
            return [], []
        mask = raw[:, 4] > self.conf_thresh
        valid_data = raw[mask]
        boxes, confidences = [], []
        for row in valid_data:
            cx, cy, w, h, score = row[:5]
            orig_cx, orig_cy = (cx - self.dw) / self.ratio, (cy - self.dh) / self.ratio
            orig_w, orig_h = w / self.ratio, h / self.ratio
            boxes.append([int(orig_cx - orig_w / 2), int(orig_cy - orig_h / 2), int(orig_w), int(orig_h)])
            confidences.append(float(score))
        return boxes, confidences


# ================= AI Thread =================
def ai_processing_thread():
    print("[AI] Initializing YOLOP model...")
    yolop_models = dnn.load(YOLOP_MODEL_PATH)
    yolov8_detector = YOLOv8ThrowDetector(YOLOV8_MODEL_PATH, YOLOV8_CONF_THRES, YOLOV8_NMS_THRES)
    plate_recognizer = PlateRecognizer()
    print(f"[AI] Models loaded. Enhance={ENHANCE_MODE} stride={AI_FRAME_STRIDE} (no annotated video)")

    while True:
        video_path = video_queue.get()
        if video_path is None:
            break

        print(f"\n[AI] Analysis: {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)

        # 把"确认秒数"换算成"确认帧数"（按源 fps / stride）
        src_fps = cap.get(cv2.CAP_PROP_FPS) or float(WANT_FPS)
        if src_fps <= 1e-3:
            src_fps = float(WANT_FPS)
        eff_fps = max(1.0, src_fps / float(AI_FRAME_STRIDE))
        emergency_confirm_frames = max(1, int(EMERGENCY_LANE_CONFIRM_SECONDS * eff_fps))

        frame_count = 0
        last_snapshot_time = 0
        emergency_lane_hits = 0
        enhanced_count = 0
        processed_count = 0

        try:
            while cap.isOpened():
                if stop_event.is_set():
                    print("[AI] stop signal, abort current segment.")
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % AI_FRAME_STRIDE == 0:
                    processed_count += 1

                    frame, was_enhanced, _ = maybe_enhance(frame)
                    if was_enhanced:
                        enhanced_count += 1

                    orig_h, orig_w = frame.shape[:2]

                    nv12_yolop = bgr2nv12(cv2.resize(frame, (320, 320)))
                    yolop_outs = yolop_models[0].forward(nv12_yolop)
                    processed_frame, is_crossing, emergency_lane_raw = postprocess_yolop(frame, yolop_outs)

                    if emergency_lane_raw:
                        emergency_lane_hits += 1
                    else:
                        emergency_lane_hits = 0
                    is_emergency_lane = emergency_lane_hits >= emergency_confirm_frames

                    tensor_yolov8 = yolov8_detector.preprocess(frame)
                    yolov8_outs = yolov8_detector.model.forward([tensor_yolov8])
                    boxes_v8, confs_v8 = yolov8_detector.postprocess(yolov8_outs, orig_w, orig_h)
                    indices_v8 = cv2.dnn.NMSBoxes(boxes_v8, confs_v8, yolov8_detector.conf_thresh, yolov8_detector.nms_thresh)

                    is_throwing = False
                    if len(indices_v8) > 0:
                        for i in indices_v8.flatten():
                            x, y, w, h = boxes_v8[i]
                            cv2.rectangle(processed_frame, (max(0, x), max(0, y)),
                                          (min(orig_w, x + w), min(orig_h, y + h)), (0, 255, 255), 3)
                            cv2.putText(processed_frame, f"THROW {confs_v8[i]:.2f}", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            is_throwing = True

                    if (is_crossing or is_throwing or is_emergency_lane) and (time.time() - last_snapshot_time > 1.5):
                        event_parts = []
                        if is_crossing:
                            event_parts.append("Line_Crossing")
                        if is_throwing:
                            event_parts.append("Throwing_Garbage")
                        if is_emergency_lane:
                            event_parts.append("Emergency_Lane_Occupied")
                        event_tag = "_and_".join(event_parts)

                        cv2.putText(processed_frame, f"ALERT: {event_tag}", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                        plate_text = plate_recognizer.recognize(frame)
                        plate_str = plate_text if plate_text else "NOPLATE"

                        time_str = fmt_time_minute_for_name()
                        pos_str = fmt_position_for_name()
                        base = f"{time_str}_{plate_str}_{pos_str}_{event_tag}"
                        snap_path = reserve_unique_path(SNAPSHOT_DIR, base, ".jpg")
                        snap_name = os.path.basename(snap_path)

                        # 存"带框"的抓拍图（取证 + 上云用的就是它）
                        cv2.imwrite(snap_path, processed_frame)
                        print(f"  [!!!] Violation captured: {snap_name}")

                        # 1) 图片直传 OSS
                        upload_to_aliyun_async(snap_path, snap_name)
                        # 2) 事件元数据推给 L610 → MQTT
                        notify_l610_async(build_event_info(event_tag, plate_str, snap_name))

                        last_snapshot_time = time.time()
                        if is_emergency_lane:
                            emergency_lane_hits = 0

                frame_count += 1
        finally:
            cap.release()

        print(f"[AI] Finished {os.path.basename(video_path)}  "
              f"(enhanced {enhanced_count}/{processed_count})")


# ================= Recorder Thread =================
def recorder_thread():
    print(f"[Recorder] Init hardware encoder...")
    enc = libsrcampy.Encoder()
    ret = enc.encode(0, 1, WANT_WIDTH, WANT_HEIGHT)
    if ret != 0:
        print("[Recorder] FATAL: hardware encoder init failed.")
        os._exit(1)

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WANT_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WANT_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, WANT_FPS)

    if not cap.isOpened():
        print(f"[Recorder] FATAL: cannot open USB camera {CAMERA_ID}.")
        os._exit(1)

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[Recorder] Camera actual FPS = {actual_fps}, starting {SEGMENT_TIME}s segments.")

    try:
        while not stop_event.is_set():
            time_str = fmt_time_minute_for_name()
            mp4_path = reserve_unique_path(VIDEO_DIR, time_str, ".mp4")
            mp4_base = os.path.splitext(os.path.basename(mp4_path))[0]
            h264_path = os.path.join(VIDEO_DIR, f"temp_{mp4_base}.h264")

            print(f"\n[Recorder] Recording: {os.path.basename(mp4_path)}")
            start_time = time.time()

            with open(h264_path, 'wb') as f:
                while (time.time() - start_time) < SEGMENT_TIME:
                    if stop_event.is_set():
                        break
                    ret_frame, frame = cap.read()
                    if not ret_frame:
                        break
                    nv12_data = bgr2nv12(frame)
                    enc.encode_file(nv12_data.tobytes())
                    encoded_bytes = enc.get_img()
                    if encoded_bytes is not None:
                        if isinstance(encoded_bytes, np.ndarray):
                            f.write(encoded_bytes.tobytes())
                        else:
                            f.write(encoded_bytes)

            print(f"[Recorder] Muxing to MP4 format...")
            ffmpeg_cmd = ['ffmpeg', '-y', '-framerate', str(WANT_FPS), '-i', h264_path, '-c', 'copy', mp4_path]
            subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if os.path.exists(h264_path):
                os.remove(h264_path)

            print(f"[Recorder] Saved {os.path.basename(mp4_path)}, passing to AI.")
            video_queue.put(mp4_path)

    except KeyboardInterrupt:
        print("\n[Recorder] Stop requested.")
    finally:
        cap.release()
        try:
            enc.close()
        except Exception:
            pass
        video_queue.put(None)


# ================= Main =================
def _install_signal_handlers():
    def _handler(signum, frame):
        print(f"\n[Main] 收到信号 {signum}，请求优雅停机 ...")
        stop_event.set()
    try:
        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)
    except (ValueError, OSError):
        pass


def main():
    global ENHANCE_MODE, DARK_BRIGHTNESS_THRESHOLD, WIFI_CONNECT
    global AI_FRAME_STRIDE, REQUIRE_INTERNET, L610_PUSH_ENABLED
    global EMERGENCY_LANE_ENABLE, EMERGENCY_LANE_SIDE, EMERGENCY_LANE_X_MIN_RATIO
    global EMERGENCY_LANE_X_MAX_RATIO, EMERGENCY_LANE_Y_MIN_RATIO
    global EMERGENCY_LANE_MIN_BOX_H_RATIO, EMERGENCY_LANE_CONFIRM_SECONDS
    global REST_REMINDER_ENABLE, REST_REMINDER_AFTER_SECONDS, REST_REMINDER_REPEAT_SECONDS

    parser = argparse.ArgumentParser(description="finial4 - 录制 + 越线/抛物/应急车道检测 + L610上云 + 必须联网（不存推理视频）")
    parser.add_argument("--no-wifi", action="store_true",
                        help="（仅调试用）跳过 WiFi 自动连接 + 跳过联网阻塞")
    parser.add_argument("--allow-offline", action="store_true",
                        help="（仅调试用）连 WiFi 但不要求外网可达，立即往下走")
    parser.add_argument("--no-l610", action="store_true",
                        help="不把事件 JSON 推给本机 L610 服务")
    parser.add_argument("--enhance", choices=['none', 'always', 'auto'], default=None,
                        help=f"低光增强模式（默认 {ENHANCE_MODE}）")
    parser.add_argument("--dark-threshold", type=int, default=None,
                        help=f"auto 模式暗帧亮度阈值（默认 {DARK_BRIGHTNESS_THRESHOLD}）")
    parser.add_argument("--stride", type=int, default=None,
                        help=f"AI 推理帧步长（默认 {AI_FRAME_STRIDE}，越大越省 CPU、检测越稀）")
    parser.add_argument("--no-emergency-lane", action="store_true",
                        help="关闭应急车道占用检测")
    parser.add_argument("--emergency-side", choices=["left", "right"], default=None,
                        help=f"应急车道在画面哪侧（默认 {EMERGENCY_LANE_SIDE}）")
    parser.add_argument("--emergency-x-min", type=float, default=None)
    parser.add_argument("--emergency-x-max", type=float, default=None)
    parser.add_argument("--emergency-y-min", type=float, default=None)
    parser.add_argument("--emergency-min-box-h", type=float, default=None,
                        help=f"应急车道车框最小高度比例（默认 {EMERGENCY_LANE_MIN_BOX_H_RATIO}）")
    parser.add_argument("--emergency-confirm", type=float, default=None,
                        help=f"确认占用所需的视频秒数（默认 {EMERGENCY_LANE_CONFIRM_SECONDS}）")
    parser.add_argument("--no-rest-reminder", action="store_true",
                        help="关闭疲劳提醒")
    parser.add_argument("--rest-reminder-hours", type=float, default=None,
                        help="疲劳提醒间隔小时（默认 4）")
    args = parser.parse_args()

    if args.no_wifi:
        WIFI_CONNECT = False
        REQUIRE_INTERNET = False
    if args.allow_offline:
        REQUIRE_INTERNET = False
    if args.no_l610:
        L610_PUSH_ENABLED = False
    if args.enhance is not None:
        ENHANCE_MODE = args.enhance
    if args.dark_threshold is not None:
        DARK_BRIGHTNESS_THRESHOLD = args.dark_threshold
    if args.stride is not None and args.stride >= 1:
        AI_FRAME_STRIDE = args.stride
    if args.no_emergency_lane:
        EMERGENCY_LANE_ENABLE = False
    if args.emergency_side is not None:
        EMERGENCY_LANE_SIDE = args.emergency_side
    if args.emergency_x_min is not None:
        EMERGENCY_LANE_X_MIN_RATIO = min(1.0, max(0.0, args.emergency_x_min))
    if args.emergency_x_max is not None:
        EMERGENCY_LANE_X_MAX_RATIO = min(1.0, max(0.0, args.emergency_x_max))
    if args.emergency_y_min is not None:
        EMERGENCY_LANE_Y_MIN_RATIO = min(1.0, max(0.0, args.emergency_y_min))
    if args.emergency_min_box_h is not None:
        EMERGENCY_LANE_MIN_BOX_H_RATIO = min(1.0, max(0.0, args.emergency_min_box_h))
    if args.emergency_confirm is not None and args.emergency_confirm >= 0:
        EMERGENCY_LANE_CONFIRM_SECONDS = args.emergency_confirm
    if args.no_rest_reminder:
        REST_REMINDER_ENABLE = False
    if args.rest_reminder_hours is not None and args.rest_reminder_hours > 0:
        REST_REMINDER_AFTER_SECONDS = int(args.rest_reminder_hours * 3600)
        REST_REMINDER_REPEAT_SECONDS = REST_REMINDER_AFTER_SECONDS

    _install_signal_handlers()

    print(f"[Main] running as {'root' if _is_root() else 'user'} (uid={os.geteuid()})")
    print(f"[Main] WIFI_CONNECT={WIFI_CONNECT} REQUIRE_INTERNET={REQUIRE_INTERNET} "
          f"L610_PUSH={L610_PUSH_ENABLED}")
    print(f"[Main] stride={AI_FRAME_STRIDE}  annotated_video=OFF（已砍）")
    print(f"[Main] emergency_lane={EMERGENCY_LANE_ENABLE} side={EMERGENCY_LANE_SIDE} "
          f"x={EMERGENCY_LANE_X_MIN_RATIO:.2f}-{EMERGENCY_LANE_X_MAX_RATIO:.2f} "
          f"y_min={EMERGENCY_LANE_Y_MIN_RATIO:.2f} min_box_h={EMERGENCY_LANE_MIN_BOX_H_RATIO:.2f} "
          f"confirm={EMERGENCY_LANE_CONFIRM_SECONDS:.1f}s")
    print(f"[Main] rest_reminder={REST_REMINDER_ENABLE} interval={REST_REMINDER_AFTER_SECONDS / 3600.0:.1f}h")

    # 1) 联网保障：死等到联网（默认）/ 调试跳过
    if WIFI_CONNECT and REQUIRE_INTERNET:
        wait_until_online()
    elif WIFI_CONNECT and not REQUIRE_INTERNET:
        connect_wifi()
        print("[Net] --allow-offline 已开，不强制要求外网可达。")
    else:
        print("[Net] --no-wifi 已开，整段联网逻辑跳过（仅调试用！）")

    setup_directories()

    # 2) GPS 后台
    gps_worker = threading.Thread(target=gps_reader_thread, daemon=True)
    gps_worker.start()

    # 3) 等一个有效时间源
    wait_for_valid_time()

    # 4) 疲劳提醒
    rest_worker = threading.Thread(target=rest_reminder_thread, daemon=True)
    rest_worker.start()

    # 5) AI + 录像
    ai_worker = threading.Thread(target=ai_processing_thread)
    ai_worker.start()

    recorder_thread()
    ai_worker.join()
    print("Program exited safely.")


if __name__ == "__main__":
    main()
