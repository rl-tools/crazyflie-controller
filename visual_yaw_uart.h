#pragma once

#include <stdbool.h>
#include <stdint.h>

#define VISUAL_YAW_TARGET_REASON_CONTROLLER_ACTIVATED 1
#define VISUAL_YAW_TARGET_REASON_PARAM_REQUEST 2
#define VISUAL_YAW_TARGET_REASON_CONTROLLER_DEACTIVATED 3

#define VISUAL_YAW_FLAG_TARGET_VALID 0x01
#define VISUAL_YAW_FLAG_PREDICTION_VALID 0x02
#define VISUAL_YAW_FLAG_TARGET_CAPTURE_ACK 0x04

void visualYawUartInit(void);
bool visualYawGetLatest(float *yawRad, uint8_t *targetSeq, uint8_t *flags, uint32_t *ageMs);
bool visualYawIsFresh(uint32_t timeoutMs);
void visualYawRequestTargetCapture(uint8_t reason);
void visualYawStartRecording(uint8_t reason);
void visualYawStopAndSaveRecording(uint8_t reason);
void visualYawAbortRecording(uint8_t reason);
