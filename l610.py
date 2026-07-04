"""
l610_rdk_sim2 - 副本2.py
========================================
RDK X5 ↔ L610 (Cat.1) 串口 → MQTT 上行脚本（v2）

v2 与 v1（- 副本.py）的差别：
    v1: 监视 finial1.py 的抓拍目录，靠"文件名"传递事件 → 状态难持久化
    v2: 起一个本机 TCP 服务，由 finial2.py 主动把事件 JSON 推过来
        每条事件只送一次，自然没有重复 / 漏报问题；
        图片不经过 L610（finial2 直传 OSS），L610 只负责文本元数据上行。

数据流：
    finial2.py 检测到违规
        ├─ 抓拍 .jpg → 直传 Aliyun OSS
        └─ 事件 JSON → TCP 127.0.0.1:L610_EVENT_PORT  (本脚本)
    本脚本
        └─ AT+MQTTPUB → 云端 MQTT Broker

接线（RDK X5 40Pin 默认 UART → L610 模组）：
    Pin 8  (UART_TXD, BCM 14)  →  L610 RXD
    Pin 10 (UART_RXD, BCM 15)  →  L610 TXD
    Pin 6 / Pin 14 (GND)       →  L610 GND
    L610 VCC 用独立 4V 电源（不要从 RDK 的 3.3V/5V 取）。
"""

import serial
import time
import json
import socket
import threading
import queue
from datetime import datetime


# ========== 需要你修改的配置 ==========
# RDK X5 上 40Pin 引出的 TTL UART 一般是 /dev/ttyS1，
# 不同固件可能是 /dev/ttyS3，请先 `ls -l /dev/ttyS*` 确认。
# 注意：finial2.py 的 GPS 用的是 /dev/ttyUSB0，别和它冲突。
SERIAL_PORT = "/dev/ttyS1"
BAUDRATE = 115200

# 接收 finial2.py 推送的本机端口
L610_EVENT_HOST = "127.0.0.1"
L610_EVENT_PORT = 9610

# 云端 MQTT Broker
ECS_IP = "101.37.187.166"
MQTT_PORT = 1883

DEVICE_ID = "rdk_x5_l610_001"

MQTT_USER = "L610"
MQTT_PASS = "123"

EVENT_TOPIC = "car/event"
CMD_TOPIC = f"car/cmd/{DEVICE_ID}"
STATUS_TOPIC = f"car/status/{DEVICE_ID}"

# 心跳间隔（秒）
STATUS_INTERVAL = 60
# =====================================


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class L610Controller:
    def __init__(self, port, baudrate=115200):
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=0.5,
        )
        # finial2 推过来的事件先入队，主循环串行下发到 L610（避免多线程同时写串口）
        self.event_queue = queue.Queue()

    def close(self):
        self.ser.close()

    def read_available(self):
        """读取串口当前已有的数据"""
        data = self.ser.read_all()
        if not data:
            return ""
        text = data.decode("utf-8", errors="ignore")
        if text.strip():
            print(text, end="")
        return text

    def send_at(self, cmd, wait=1.0):
        """发送一条 AT 指令，并等待一会儿读取返回"""
        print(f"\n>> {cmd}")
        self.ser.write((cmd + "\r\n").encode("utf-8"))
        time.sleep(wait)
        return self.read_available()

    def json_to_at_payload(self, data: dict):
        """
        把 JSON 转成 L610 AT+MQTTPUB 可用的字符串。
        关键点：把 JSON 里的双引号 " 替换成 \\22。
        """
        text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        return text.replace('"', "\\22")

    def init_module(self):
        """初始化 L610"""
        self.send_at("AT")
        self.send_at("ATE0")
        self.send_at("AT+CPIN?")
        self.send_at("AT+CSQ")
        self.send_at("AT+CEREG?")
        self.send_at("AT+CGATT?")
        self.send_at("AT+MIPCALL?")

    def connect_mqtt(self):
        """连接 ECS 上的 MQTT Broker"""
        self.send_at("AT+MQTTCLOSE=1", wait=1)
        self.send_at(f'AT+MQTTUSER=1,"{MQTT_USER}","{MQTT_PASS}"', wait=1)
        self.send_at(
            f'AT+MQTTOPEN=1,"{ECS_IP}",{MQTT_PORT},0,60',
            wait=6,
        )

    def subscribe_cmd(self):
        """订阅云端下发命令 Topic"""
        self.send_at(f'AT+MQTTSUB=1,"{CMD_TOPIC}",0', wait=2)

    def publish_json(self, topic, data):
        """发布 JSON 数据到指定 Topic"""
        payload = self.json_to_at_payload(data)
        cmd = f'AT+MQTTPUB=1,"{topic}",0,0,"{payload}"'
        self.send_at(cmd, wait=2)

    def publish_status(self):
        """发布设备在线状态"""
        data = {
            "device_id": DEVICE_ID,
            "module": "Fibocom L610",
            "network": "LTE Cat.1",
            "status": "online",
            "time": now_str(),
        }
        self.publish_json(STATUS_TOPIC, data)

    def publish_vehicle_event(self, info):
        """
        把 finial2 推过来的事件包透传到 MQTT。
        info 期望含：plate, event, event_name, captured_at,
                     latitude, longitude, location, image_key
        本端只在最外层套一层 device_id + 上行时间。
        """
        data = {
            "device_id": DEVICE_ID,
            "plate": info.get("plate", "NOPLATE"),
            "event": info.get("event"),
            "event_name": info.get("event_name"),
            "time": now_str(),
            "captured_at": info.get("captured_at"),
            "location": info.get("location"),
            "latitude": info.get("latitude"),
            "longitude": info.get("longitude"),
            "image_key": info.get("image_key"),
        }
        print(
            f"\n>>> 上报违规事件: plate={data['plate']} "
            f"event={data['event']} image_key={data['image_key']}"
        )
        self.publish_json(EVENT_TOPIC, data)

    # ---------- TCP server: 接 finial2.py 的推送 ----------
    def start_event_server(self):
        """启动后台 TCP 服务，接 finial2 推送的事件 JSON。"""
        t = threading.Thread(target=self._event_server_loop, daemon=True)
        t.start()
        return t

    def _event_server_loop(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind((L610_EVENT_HOST, L610_EVENT_PORT))
        except OSError as e:
            print(f"[Event] bind {L610_EVENT_HOST}:{L610_EVENT_PORT} 失败: {e}")
            return
        srv.listen(8)
        print(f"[Event] TCP 监听 {L610_EVENT_HOST}:{L610_EVENT_PORT}（等 finial2 推送）")
        while True:
            try:
                conn, _ = srv.accept()
            except OSError as e:
                print(f"[Event] accept error: {e}")
                continue
            threading.Thread(
                target=self._handle_event_conn,
                args=(conn,),
                daemon=True,
            ).start()

    def _handle_event_conn(self, conn):
        """协议：每个事件一行 JSON，以 \\n 结尾。"""
        with conn:
            conn.settimeout(5.0)
            buf = b""
            try:
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if not line.strip():
                            continue
                        try:
                            info = json.loads(line.decode("utf-8", errors="ignore"))
                        except json.JSONDecodeError as e:
                            print(f"[Event] JSON 解析失败: {e}")
                            continue
                        self.event_queue.put(info)
                        print(
                            f"[Event] 收到 finial2 推送: "
                            f"plate={info.get('plate')} event={info.get('event')}"
                        )
            except socket.timeout:
                pass
            except Exception as e:
                print(f"[Event] 连接异常: {e}")

    # ---------- 主循环 ----------
    def run_loop(self):
        """
        1. 监听云端下发命令（cmd=status 时立即上报设备状态）
        2. 定期心跳上报设备在线
        3. 把事件队列里的内容串行下发到 L610 → MQTT
        """
        print("\n" + "=" * 50)
        print("监听云端命令 + 转发 finial2 事件")
        print(f"订阅 Topic:  {CMD_TOPIC}")
        print(f"上报 Topic:  {EVENT_TOPIC}")
        print(f"事件入口:    tcp://{L610_EVENT_HOST}:{L610_EVENT_PORT}")
        print('云端命令示例: {"cmd":"status"}')
        print("按 Ctrl+C 退出")
        print("=" * 50 + "\n")

        last_status_time = time.time()

        while True:
            text = self.read_available()

            # 响应云端 status 命令
            if "status" in text:
                print("\n收到云端 status 命令，立即上报设备状态。")
                self.publish_status()
                last_status_time = time.time()

            # 定期心跳
            if time.time() - last_status_time > STATUS_INTERVAL:
                self.publish_status()
                last_status_time = time.time()

            # 把 finial2 推过来的事件依次转发
            while True:
                try:
                    info = self.event_queue.get_nowait()
                except queue.Empty:
                    break
                self.publish_vehicle_event(info)
                time.sleep(0.5)  # 两条 MQTT 之间稍微留间隔

            time.sleep(0.2)


def main():
    l610 = L610Controller(SERIAL_PORT, BAUDRATE)

    try:
        print("=" * 50)
        print("RDK X5 ↔ L610 Cat.1 → MQTT (v2: TCP 接 finial2 推送)")
        print("=" * 50)

        print("\n[1/4] 初始化 L610...")
        l610.init_module()

        print("\n[2/4] 连接 MQTT 服务器...")
        l610.connect_mqtt()

        print("\n[3/4] 订阅云端命令 Topic...")
        l610.subscribe_cmd()

        print("\n[4/4] 上报设备在线状态...")
        l610.publish_status()

        # 启 TCP 服务，开始接 finial2 推送
        l610.start_event_server()

        # 进入持续监听 + 转发循环
        l610.run_loop()

    except KeyboardInterrupt:
        print("\n用户退出。")
    finally:
        l610.close()


if __name__ == "__main__":
    main()
