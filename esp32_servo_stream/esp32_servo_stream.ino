/*
esp32_servo_stream.ino — PCA9685 8舵机TCP流驱动 (ESP32-C3-DevKitM-1)

替代树莓派上的 servo_stream.py。协议完全相同：
  connect :8123 → "ready 8\n"
  发送 "90 90 90 90 90 90 90 90\n" → 驱动舵机（8个角度，空格分隔）
  发送 "q\n" → 释放舵机 + "stopped\n"

通道顺序 = q1lite.xml 执行器顺序：
  ch0=FR_hip ch1=FR_knee ch2=RR_hip ch3=RR_knee
  ch4=FL_hip ch5=FL_knee ch6=RL_hip ch7=RL_knee

脉宽映射（MG90D）：500-2500us 对应 0-180°
最新命令优先：一次读入全部积压数据，只执行最新有效行（防网络卡顿后的补课回放）

════════════════════════════════════════════════════════════════

接线：
  PCA9685 SDA → ESP32-C3 GPIO4  (J3 pin 11，非strap引脚，安全)
  PCA9685 SCL → ESP32-C3 GPIO5  (J3 pin 10，非strap引脚，安全)
  PCA9685 VCC → ESP32-C3 3.3V   (逻辑供电，PCA9685 VDD 2.3-5.5V)
  PCA9685 GND → ESP32-C3 GND    (必须与外部5V电源共地)
  PCA9685 V+  → 外部5V电源       (舵机动力，绝不用ESP32的3.3V/5V)

烧录（Arduino IDE）：
  1. 开发板管理器 → 安装 "esp32 by Espressif Systems"
     (URL: https://espressif.github.io/arduino-esp32/package_esp32_index.json)
  2. 开发板 → ESP32C3 Dev Module (USB CDC On Boot: Disabled)
  3. 库管理器 → 安装 "Adafruit PWM Servo Driver Library" (自动带 Adafruit BusIO)
  4. 修改下方 WIFI_SSID / WIFI_PASS
  5. 上传 → 串口监视器 115200 → 读 IP 地址

烧录（PlatformIO）：
  cd esp32_servo_stream && pio run -t upload && pio device monitor

Mac 端使用：
  python test_servo_groups.py --transport tcp --host <esp32_ip>
  python q1lite_servo_bridge.py --transport tcp --host <esp32_ip> --drive 0.4
  mjpython fly_embodied_q1lite.py --servo --servo-transport tcp --servo-host <esp32_ip>
*/

#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ══ 用户配置（修改这里） ═══════════════════════════════════════════
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";
const uint16_t TCP_PORT = 8123;

// I2C 引脚（GPIO4/GPIO5 = J3 pin 11/10，非strap引脚）
// 默认 I2C 是 GPIO8(SDA)/GPIO9(SCL)——是strap引脚，故覆盖之
#define I2C_SDA 4
#define I2C_SCL 5

// ══ 协议常量（与 servo_stream.py 一致） ════════════════════════════
const uint8_t  N_SERVOS = 8;        // 通道 0-7
const uint16_t US_MIN = 500;        // 微秒，对应 0°（MG90D）
const uint16_t US_MAX = 2500;       // 微秒，对应 180°
// ══════════════════════════════════════════════════════════════════

// ── 行解析（固定缓冲，无堆分配） ──────────────────────────────────
// 注意：struct 必须在所有使用它的函数之前定义，否则 Arduino 预处理器
// 自动生成的函数原型会插在 struct 之前导致 "has not been declared" 错误。
char lineBuf[160];
size_t lineLen = 0;

struct LineResult {
  float angles[8];
  bool  haveAngles;
  bool  stop;
};

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();  // 0x40, &Wire
WiFiServer server(TCP_PORT);

// ── 舵机驱动 ──────────────────────────────────────────────────────
// tick = us * 4096 / 20000  (50Hz, 12-bit)
//   500us → 102, 1500us → 307, 2500us → 512

void setServoAngle(uint8_t ch, float angleDeg) {
  if (angleDeg < 0.0f)   angleDeg = 0.0f;   // clamp（ServoKit 越界会 raise）
  if (angleDeg > 180.0f) angleDeg = 180.0f;
  uint32_t us = US_MIN + (uint32_t)(angleDeg * (US_MAX - US_MIN) / 180.0f + 0.5f);
  uint16_t tick = (uint16_t)(us * 4096UL / 20000UL);
  pwm.setPWM(ch, 0, tick);                   // on=0：脉冲从相位0开始
}

void applyAngles(const float a[8]) {
  for (uint8_t i = 0; i < N_SERVOS; i++) setServoAngle(i, a[i]);
}

void releaseAll() {
  // FULL OFF（LEDn_OFF bit 12）→ 舵机无力矩
  // 注意：不是 setPWM(ch, 0, 0)——那不是文档定义的 full-off
  for (uint8_t ch = 0; ch < 16; ch++) pwm.setPWM(ch, 0, 4096);
}

// ── PCA9685 初始化 ────────────────────────────────────────────────

bool pca9685Init() {
  // 必须在 pwm.begin() 之前调用：
  // Adafruit_I2CDevice::begin() 内部调用无参 Wire.begin()，
  // 而 ESP32 上对已启动的总线再调用 begin() 是 no-op（保留我们的引脚）
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);                     // PCA9685 支持 Fm+

  // 先做 I2C 地址探测（begin() 在 2.4.x 返回 void，无法判断成败）
  Wire.beginTransmission(0x40);
  if (Wire.endTransmission() != 0) {
    Serial.println("PCA9685 init FAILED (check wiring, addr 0x40)");
    return false;
  }

  pwm.begin();                               // begin() 默认设 1000Hz！
  pwm.setOscillatorFrequency(25000000);      // 默认；与 Pi 端 ServoKit 假设一致
  pwm.setPWMFreq(50);                        // 必须显式设 50Hz（舵机标准）
  delay(10);                                 // 等待 prescale 生效

  // 启动保险：PCA9685 复位不清除 PWM 输出，先全部关断
  releaseAll();
  return true;
}

// ── 行解析（固定缓冲，无堆分配） ──────────────────────────────────

void parseLine(char* cmd, LineResult* result) {
  while (*cmd == ' ' || *cmd == '\t') cmd++;  // 前导空白
  if (*cmd == '\0') return;                   // 空行
  if (strcmp(cmd, "q") == 0) { result->stop = true; return; }

  float v[N_SERVOS];
  int n = 0;
  char* tok = strtok(cmd, " \t");
  while (tok && n < N_SERVOS) {
    v[n++] = strtof(tok, nullptr);
    tok = strtok(nullptr, " \t");
  }
  // 恰好 8 个 token 才有效（多余或不足都跳过）
  if (n == N_SERVOS && !tok) {
    memcpy(result->angles, v, sizeof v);
    result->haveAngles = true;
  }
}

// ── 主逻辑 ────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);                       // DevKitM-1 板载 USB-UART 桥
  Serial.println();
  Serial.println("=== esp32_servo_stream starting ===");

  // WiFi 连接
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  // 关闭 WiFi 省电（modem sleep）：默认省电模式下射频每 ~100-300ms 才醒一次，
  // TCP 数据在路由器侧积攒成批，醒来时一次性涌入 → latest-wins 丢中间帧
  // → 步态停顿-爆发式不连贯。关掉后延迟降到 ~1-3ms，50Hz 流平滑到达。
  WiFi.setSleep(false);
  Serial.print("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(200);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("servo bridge IP: ");
  Serial.println(WiFi.localIP());
  Serial.print("TCP port: ");
  Serial.println(TCP_PORT);

  // PCA9685
  if (!pca9685Init()) {
    Serial.println("HALT: PCA9685 not found. Check wiring, then reset.");
    while (true) delay(1000);                 // 停在这里，串口可见
  }

  // TCP 服务器
  server.begin();
  server.setNoDelay(true);                    // 关 Nagle（50Hz 小包低延迟）

  Serial.println("Ready. Waiting for connection...");
}

void loop() {
  // WiFi 断线重连（断线期间舵机保持最后姿态）
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi lost, reconnecting...");
    WiFi.disconnect();
    WiFi.reconnect();
    delay(2000);
    return;
  }

  WiFiClient client = server.accept();        // 3.x API（available() 已弃用）
  if (!client) return;

  Serial.println("client connected");
  client.println("ready 8");                  // 协议握手
  lineLen = 0;

  while (client.connected() || client.available()) {
    LineResult result = {};                   // 每批次重置
    result.haveAngles = false;
    result.stop = false;

    // ── latest-wins drain：读光全部积压，只保留最新有效命令 ──
    while (client.available()) {
      char c = (char)client.read();
      if (c == '\n') {
        lineBuf[lineLen] = '\0';
        parseLine(lineBuf, &result);
        lineLen = 0;
        if (result.stop) break;               // 'q' 立即停止读取
      } else if (c != '\r') {                 // 容忍 CRLF
        if (lineLen < sizeof(lineBuf) - 1) lineBuf[lineLen++] = c;
        else lineLen = 0;                     // 超长行丢弃（垃圾防护）
      }
    }

    // 应用最新命令（若 'q' 与 park 行被网络合并，先到位再释放）
    if (result.haveAngles) applyAngles(result.angles);
    if (result.stop) break;

    if (!client.available()) delay(1);        // 空闲时让出 CPU 给 WiFi 任务
  }

  // 断线或 'q'：先短暂停留（让合并的 park 指令到位），再释放
  delay(500);
  releaseAll();
  client.println("stopped");
  client.stop();
  Serial.println("client disconnected");
}
