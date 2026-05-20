#define DEBUG_MODULE "VYAW"

#include "visual_yaw_uart.h"

#include <stddef.h>
#include <stdint.h>

#include "FreeRTOS.h"
#include "queue.h"
#include "task.h"

#include "config.h"
#include "debug.h"
#include "log.h"
#include "param.h"
#include "static_mem.h"
#include "system.h"
#include "uart1.h"

#define VISUAL_YAW_UART_BAUDRATE 115200
#define VISUAL_YAW_RX_TASK_NAME "VYAW_RX"
#define VISUAL_YAW_TX_TASK_NAME "VYAW_TX"
#define VISUAL_YAW_TASK_STACKSIZE configMINIMAL_STACK_SIZE
#define VISUAL_YAW_RX_TASK_PRI 2
#define VISUAL_YAW_TX_TASK_PRI 1
#define VISUAL_YAW_TX_QUEUE_LENGTH 32

#define FRAME_START_MASK 0x80
#define FRAME_TYPE_MASK 0x7F
#define FRAME_TYPE_VISUAL_YAW 0x10
#define FRAME_TYPE_TARGET_CAPTURE 0x11

#define VISUAL_YAW_RAW_BYTES 8
#define VISUAL_YAW_DATA_BYTES 10
#define VISUAL_YAW_CRC_PAYLOAD_BYTES 7
#define TARGET_CAPTURE_RAW_BYTES 6
#define TARGET_CAPTURE_DATA_BYTES 7
#define TARGET_CAPTURE_FRAME_BYTES (1 + TARGET_CAPTURE_DATA_BYTES)
#define TARGET_CAPTURE_CRC_PAYLOAD_BYTES 5

#define TARGET_CAPTURE_COMMAND_CURRENT 0x01
#define TARGET_CAPTURE_COMMAND_RECORD_START 0x02
#define TARGET_CAPTURE_COMMAND_RECORD_STOP_SAVE 0x03
#define TARGET_CAPTURE_COMMAND_RECORD_ABORT 0x04
#define TARGET_CAPTURE_FLAG_REQUIRE_ACK 0x01
#define VISUAL_YAW_KNOWN_FLAGS (VISUAL_YAW_FLAG_TARGET_VALID | \
                                VISUAL_YAW_FLAG_PREDICTION_VALID | \
                                VISUAL_YAW_FLAG_TARGET_CAPTURE_ACK)
#define YAW_Q_SCALE 10000.0f

static xQueueHandle txQueue;
STATIC_MEM_QUEUE_ALLOC(txQueue, VISUAL_YAW_TX_QUEUE_LENGTH, sizeof(uint8_t));

static void visualYawRxTask(void *arg);
static void visualYawTxTask(void *arg);
STATIC_MEM_TASK_ALLOC(visualYawRxTask, VISUAL_YAW_TASK_STACKSIZE);
STATIC_MEM_TASK_ALLOC(visualYawTxTask, VISUAL_YAW_TASK_STACKSIZE);

static bool isInit = false;

static float latestYawRad = 0.0f;
static uint8_t latestSeq = 0;
static uint8_t latestTargetSeq = 0;
static uint8_t latestFlags = 0;
static uint8_t latestOpenmvAgeMs = 0;
static TickType_t latestFrameTick = 0;
static uint8_t haveFrame = 0;
static uint8_t frameFresh = 0;

static uint8_t targetCommandSeq = 0;
static uint8_t captureParam = 0;
static uint8_t recordParam = 0;

static uint32_t framesOk = 0;
static uint32_t framesBadCrc = 0;
static uint32_t framesBadFlags = 0;
static uint32_t framesUnknown = 0;
static uint32_t frameRestarts = 0;
static uint32_t targetRequests = 0;
static uint32_t targetRequestDrops = 0;

static uint16_t crc16_ccitt(const uint8_t *data, size_t n)
{
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < n; i++) {
    crc ^= (uint16_t)data[i] << 8;
    for (int j = 0; j < 8; j++) {
      crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
  }
  return crc;
}

static void pack7(const uint8_t *in, size_t rawLen, uint8_t *out)
{
  uint32_t acc = 0;
  int nbits = 0;
  size_t w = 0;
  for (size_t i = 0; i < rawLen; i++) {
    acc = (acc << 8) | in[i];
    nbits += 8;
    while (nbits >= 7) {
      nbits -= 7;
      out[w++] = (uint8_t)((acc >> nbits) & 0x7F);
    }
  }
  if (nbits > 0) {
    out[w++] = (uint8_t)((acc << (7 - nbits)) & 0x7F);
  }
}

static void unpack7(const uint8_t *in, size_t packedLen, uint8_t *out, size_t rawLen)
{
  uint32_t acc = 0;
  int nbits = 0;
  size_t outIdx = 0;
  for (size_t i = 0; i < packedLen; i++) {
    acc = (acc << 7) | (in[i] & 0x7F);
    nbits += 7;
    if (nbits >= 8 && outIdx < rawLen) {
      nbits -= 8;
      out[outIdx++] = (uint8_t)((acc >> nbits) & 0xFF);
    }
  }
}

static void putU16Be(uint8_t *dst, uint16_t v)
{
  dst[0] = (uint8_t)((v >> 8) & 0xFF);
  dst[1] = (uint8_t)(v & 0xFF);
}

static bool sendAllOrDrop(const uint8_t *data, size_t n)
{
  if (!isInit || data == NULL) {
    return false;
  }
  if (uxQueueSpacesAvailable(txQueue) < n) {
    return false;
  }
  for (size_t i = 0; i < n; i++) {
    if (xQueueSend(txQueue, &data[i], 0) != pdTRUE) {
      return false;
    }
  }
  return true;
}

static void applyVisualYawFrame(uint8_t startByte, const uint8_t raw[VISUAL_YAW_RAW_BYTES])
{
  uint16_t rxCrc = ((uint16_t)raw[6] << 8) | raw[7];
  uint8_t crcPayload[VISUAL_YAW_CRC_PAYLOAD_BYTES];
  crcPayload[0] = startByte;
  for (int i = 0; i < 6; i++) {
    crcPayload[i + 1] = raw[i];
  }

  uint16_t exCrc = crc16_ccitt(crcPayload, VISUAL_YAW_CRC_PAYLOAD_BYTES);
  if (rxCrc != exCrc) {
    framesBadCrc++;
    return;
  }

  uint8_t flags = raw[2];
  if ((flags & ~VISUAL_YAW_KNOWN_FLAGS) != 0) {
    framesBadFlags++;
    return;
  }

  int16_t yawQ = (int16_t)(((uint16_t)raw[4] << 8) | raw[5]);
  taskENTER_CRITICAL();
  latestSeq = raw[0];
  latestTargetSeq = raw[1];
  latestFlags = flags;
  latestOpenmvAgeMs = raw[3];
  latestYawRad = (float)yawQ / YAW_Q_SCALE;
  latestFrameTick = xTaskGetTickCount();
  haveFrame = 1;
  taskEXIT_CRITICAL();

  framesOk++;
}

static void startFrameOrDrop(uint8_t b, int *dataIdx, uint8_t *startByte)
{
  uint8_t frameType = b & FRAME_TYPE_MASK;
  if (frameType == FRAME_TYPE_VISUAL_YAW) {
    *startByte = b;
    *dataIdx = 0;
  } else {
    *dataIdx = -1;
    framesUnknown++;
  }
}

static void visualYawRxTask(void *arg)
{
  systemWaitStart();

  uint8_t dataBuf[VISUAL_YAW_DATA_BYTES];
  uint8_t raw[VISUAL_YAW_RAW_BYTES];
  uint8_t startByte = 0;
  int dataIdx = -1;

  while (1) {
    char c;
    uart1Getchar(&c);
    uint8_t b = (uint8_t)c;

    if (b & FRAME_START_MASK) {
      if (dataIdx >= 0) {
        frameRestarts++;
      }
      startFrameOrDrop(b, &dataIdx, &startByte);
      continue;
    }

    if (dataIdx < 0) {
      continue;
    }

    dataBuf[dataIdx++] = b;
    if (dataIdx == VISUAL_YAW_DATA_BYTES) {
      unpack7(dataBuf, VISUAL_YAW_DATA_BYTES, raw, VISUAL_YAW_RAW_BYTES);
      applyVisualYawFrame(startByte, raw);
      dataIdx = -1;
    }
  }
}

static void visualYawTxTask(void *arg)
{
  systemWaitStart();

  while (1) {
    uint8_t byte;
    if (xQueueReceive(txQueue, &byte, portMAX_DELAY) == pdTRUE) {
      uart1SendData(1, &byte);
    }
  }
}

void visualYawUartInit(void)
{
  if (isInit) {
    return;
  }

  uart1Init(VISUAL_YAW_UART_BAUDRATE);
  txQueue = STATIC_MEM_QUEUE_CREATE(txQueue);

  STATIC_MEM_TASK_CREATE(visualYawRxTask, visualYawRxTask,
                         VISUAL_YAW_RX_TASK_NAME, NULL,
                         VISUAL_YAW_RX_TASK_PRI);
  STATIC_MEM_TASK_CREATE(visualYawTxTask, visualYawTxTask,
                         VISUAL_YAW_TX_TASK_NAME, NULL,
                         VISUAL_YAW_TX_TASK_PRI);

  isInit = true;
}

bool visualYawGetLatest(float *yawRad, uint8_t *targetSeq, uint8_t *flags, uint32_t *ageMs)
{
  float yaw;
  uint8_t target;
  uint8_t frameFlags;
  uint8_t openmvAge;
  TickType_t frameTick;
  uint8_t has;

  taskENTER_CRITICAL();
  yaw = latestYawRad;
  target = latestTargetSeq;
  frameFlags = latestFlags;
  openmvAge = latestOpenmvAgeMs;
  frameTick = latestFrameTick;
  has = haveFrame;
  taskEXIT_CRITICAL();

  uint32_t age = openmvAge;
  if (has) {
    age += T2M(xTaskGetTickCount() - frameTick);
  }

  if (yawRad != NULL) {
    *yawRad = yaw;
  }
  if (targetSeq != NULL) {
    *targetSeq = target;
  }
  if (flags != NULL) {
    *flags = frameFlags;
  }
  if (ageMs != NULL) {
    *ageMs = age;
  }

  frameFresh = (has && age < 255) ? 1 : 0;
  return has != 0;
}

bool visualYawIsFresh(uint32_t timeoutMs)
{
  uint32_t ageMs = 0;
  uint8_t flags = 0;
  bool has = visualYawGetLatest(NULL, NULL, &flags, &ageMs);
  return has &&
         ageMs <= timeoutMs &&
         ((flags & (VISUAL_YAW_FLAG_TARGET_VALID | VISUAL_YAW_FLAG_PREDICTION_VALID)) ==
          (VISUAL_YAW_FLAG_TARGET_VALID | VISUAL_YAW_FLAG_PREDICTION_VALID));
}

static void visualYawSendCommand(uint8_t command, uint8_t reason, uint8_t flags)
{
  uint8_t raw[TARGET_CAPTURE_RAW_BYTES];
  uint8_t crcPayload[TARGET_CAPTURE_CRC_PAYLOAD_BYTES];
  uint8_t frame[TARGET_CAPTURE_FRAME_BYTES];
  uint8_t start = FRAME_START_MASK | FRAME_TYPE_TARGET_CAPTURE;

  raw[0] = targetCommandSeq++;
  raw[1] = command;
  raw[2] = reason;
  raw[3] = flags;

  crcPayload[0] = start;
  for (int i = 0; i < 4; i++) {
    crcPayload[i + 1] = raw[i];
  }
  putU16Be(&raw[4], crc16_ccitt(crcPayload, TARGET_CAPTURE_CRC_PAYLOAD_BYTES));

  frame[0] = start;
  pack7(raw, TARGET_CAPTURE_RAW_BYTES, &frame[1]);

  if (sendAllOrDrop(frame, TARGET_CAPTURE_FRAME_BYTES)) {
    targetRequests++;
  } else {
    targetRequestDrops++;
  }
}

void visualYawRequestTargetCapture(uint8_t reason)
{
  visualYawSendCommand(TARGET_CAPTURE_COMMAND_CURRENT, reason, TARGET_CAPTURE_FLAG_REQUIRE_ACK);
}

void visualYawStartRecording(uint8_t reason)
{
  visualYawSendCommand(TARGET_CAPTURE_COMMAND_RECORD_START, reason, 0);
}

void visualYawStopAndSaveRecording(uint8_t reason)
{
  visualYawSendCommand(TARGET_CAPTURE_COMMAND_RECORD_STOP_SAVE, reason, 0);
}

void visualYawAbortRecording(uint8_t reason)
{
  visualYawSendCommand(TARGET_CAPTURE_COMMAND_RECORD_ABORT, reason, 0);
}

static void captureParamChanged(void)
{
  if (captureParam != 0) {
    visualYawRequestTargetCapture(VISUAL_YAW_TARGET_REASON_PARAM_REQUEST);
    captureParam = 0;
  }
}

static void recordParamChanged(void)
{
  if (recordParam == 1) {
    visualYawStartRecording(VISUAL_YAW_TARGET_REASON_PARAM_REQUEST);
  } else if (recordParam == 2) {
    visualYawStopAndSaveRecording(VISUAL_YAW_TARGET_REASON_PARAM_REQUEST);
  } else if (recordParam == 3) {
    visualYawAbortRecording(VISUAL_YAW_TARGET_REASON_PARAM_REQUEST);
  }
  recordParam = 0;
}

PARAM_GROUP_START(vyaw)
PARAM_ADD_WITH_CALLBACK(PARAM_UINT8, capture, &captureParam, captureParamChanged)
PARAM_ADD_WITH_CALLBACK(PARAM_UINT8, record, &recordParam, recordParamChanged)
PARAM_GROUP_STOP(vyaw)

LOG_GROUP_START(vyaw)
LOG_ADD(LOG_UINT8, has, &haveFrame)
LOG_ADD(LOG_UINT8, fresh, &frameFresh)
LOG_ADD(LOG_UINT8, seq, &latestSeq)
LOG_ADD(LOG_UINT8, target, &latestTargetSeq)
LOG_ADD(LOG_UINT8, flags, &latestFlags)
LOG_ADD(LOG_UINT8, openmvAge, &latestOpenmvAgeMs)
LOG_ADD(LOG_FLOAT, yaw, &latestYawRad)
LOG_ADD(LOG_UINT32, ok, &framesOk)
LOG_ADD(LOG_UINT32, crc, &framesBadCrc)
LOG_ADD(LOG_UINT32, badFlags, &framesBadFlags)
LOG_ADD(LOG_UINT32, unknown, &framesUnknown)
LOG_ADD(LOG_UINT32, restart, &frameRestarts)
LOG_ADD(LOG_UINT32, txReq, &targetRequests)
LOG_ADD(LOG_UINT32, txDrop, &targetRequestDrops)
LOG_GROUP_STOP(vyaw)
