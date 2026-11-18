// Eddy line protocol firmware for a two-wheel robot (Arduino / ESP32 / Pico).
//
// NOT YET TESTED ON A REAL BOARD. This is a starting point: check it with the wheels off the
// ground, and check that pulling the USB cable stops the motors within WATCHDOG_S.
//
// It reports no odometry and no range sensor, so Eddy drives by time. Add wheel encoders or a
// distance sensor and report them in describe() and state() for better driving.
//
// Protocol: see setup/microcontroller.md.

// ---- Set these for your robot -------------------------------------------------------------
const int LEFT_PWM = 5, LEFT_DIR = 4;        // left motor driver pins
const int RIGHT_PWM = 6, RIGHT_DIR = 7;      // right motor driver pins
const float WHEEL_SPACING_M = 0.15;          // distance between the two wheels
const float MAX_WHEEL_SPEED = 0.40;          // wheel speed (m/s) at full PWM
const float MAX_LINEAR = 0.30;               // what Eddy may ask for, m/s
const float MAX_ANGULAR = 2.0;               // what Eddy may ask for, rad/s
const float WATCHDOG_S = 0.5;                // stop this long after the last drive command
const long BAUD = 115200;
// --------------------------------------------------------------------------------------------

unsigned long lastDrive = 0;
bool driving = false;
String line;

void setMotor(int pwmPin, int dirPin, float speed) {
  // speed in m/s at the wheel; sign gives the direction.
  float duty = speed / MAX_WHEEL_SPEED;
  if (duty > 1) duty = 1;
  if (duty < -1) duty = -1;
  digitalWrite(dirPin, duty >= 0 ? HIGH : LOW);
  analogWrite(pwmPin, (int)(fabs(duty) * 255));
}

void drive(float v, float w) {
  // Differential drive: turn the body speeds into wheel speeds.
  float left = v - w * WHEEL_SPACING_M / 2;
  float right = v + w * WHEEL_SPACING_M / 2;
  setMotor(LEFT_PWM, LEFT_DIR, left);
  setMotor(RIGHT_PWM, RIGHT_DIR, right);
}

void stopAll() {
  drive(0, 0);
  driving = false;
}

void describe() {
  Serial.print("{\"eddy\":1,\"name\":\"rover\",\"drive\":{\"max_linear\":");
  Serial.print(MAX_LINEAR, 3);
  Serial.print(",\"max_angular\":");
  Serial.print(MAX_ANGULAR, 3);
  Serial.print("},\"joints\":[],\"gripper\":false,\"sensors\":[],\"watchdog\":");
  Serial.print(WATCHDOG_S, 2);
  Serial.println("}");
}

void state() {
  // No odometry and no range sensor: an empty state. Add "x","y","th" and "range" here.
  Serial.println("{}");
}

void handle(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;
  if (cmd == "EDDY?") { describe(); return; }
  if (cmd == "?") { state(); return; }
  if (cmd == "S") { stopAll(); Serial.println("OK"); return; }
  if (cmd.startsWith("V ")) {
    int space = cmd.indexOf(' ', 2);
    if (space < 0) { Serial.println("ERR V needs two numbers"); return; }
    float v = cmd.substring(2, space).toFloat();
    float w = cmd.substring(space + 1).toFloat();
    v = constrain(v, -MAX_LINEAR, MAX_LINEAR);
    w = constrain(w, -MAX_ANGULAR, MAX_ANGULAR);
    drive(v, w);
    lastDrive = millis();
    driving = (v != 0 || w != 0);
    Serial.println("OK");
    return;
  }
  if (cmd.startsWith("J")) { Serial.println("ERR unsupported this robot has no joints"); return; }
  if (cmd.startsWith("G")) { Serial.println("ERR unsupported this robot has no gripper"); return; }
  Serial.print("ERR unknown command ");
  Serial.println(cmd);
}

void setup() {
  pinMode(LEFT_PWM, OUTPUT); pinMode(LEFT_DIR, OUTPUT);
  pinMode(RIGHT_PWM, OUTPUT); pinMode(RIGHT_DIR, OUTPUT);
  stopAll();
  Serial.begin(BAUD);
}

void loop() {
  // Watchdog: if Eddy (or the cable) goes quiet, stop.
  if (driving && millis() - lastDrive > (unsigned long)(WATCHDOG_S * 1000)) stopAll();

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') { handle(line); line = ""; }
    else if (c != '\r' && line.length() < 120) line += c;
  }
}
