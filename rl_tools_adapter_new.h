#ifndef RL_TOOLS_ADAPTER_NEW_H
#define RL_TOOLS_ADAPTER_NEW_H
#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif


typedef struct {
    float position[3];
    float orientation[4]; // Quaternion: w, x, y, z
    float linear_velocity[3];
    float angular_velocity[3];
    float previous_action[4];
} RLtoolsObservation;
typedef struct {
    float action[4];
} RLtoolsAction;

#define RL_TOOLS_STATUS_TIMING_JITTER_LOW_THRESHOLD 0.8
#define RL_TOOLS_STATUS_TIMING_JITTER_HIGH_THRESHOLD 1.2
#define RL_TOOLS_STATUS_TIMING_BIAS_LOW_THRESHOLD 0.95
#define RL_TOOLS_STATUS_TIMING_BIAS_HIGH_THRESHOLD 1.05

#define RL_TOOLS_STATUS_OK                           (0ULL)
#define RL_TOOLS_STATUS_BIT_SOURCE_CONTROL           (1ULL << 1)
#define RL_TOOLS_STATUS_BIT_SOURCE_CONTROL_ORIGINAL  (1ULL << 2)
#define RL_TOOLS_STATUS_BITS_ISSUE_SHIFT             (8ULL)
#define RL_TOOLS_STATUS_BITS_ISSUE                   (0xfffffffULL << RL_TOOLS_STATUS_BITS_ISSUE_SHIFT)
#define RL_TOOLS_STATUS_BIT_TIMESTAMP_INVALID        (0x01ULL << (RL_TOOLS_STATUS_BITS_ISSUE_SHIFT))
#define RL_TOOLS_STATUS_BIT_TIMING_ISSUE             (0x02ULL << (RL_TOOLS_STATUS_BITS_ISSUE_SHIFT))
#define RL_TOOLS_STATUS_BIT_TIMING_JITTER            (1ULL << (RL_TOOLS_STATUS_BITS_ISSUE_SHIFT + 8 + 0))
#define RL_TOOLS_STATUS_BIT_TIMING_BIAS              (1ULL << (RL_TOOLS_STATUS_BITS_ISSUE_SHIFT + 8 + 1))
#define RL_TOOLS_STATUS_BITS_MAGNITUDE_SHIFT         (RL_TOOLS_STATUS_BITS_ISSUE_SHIFT + 8 + 1 + 16)
#define RL_TOOLS_STATUS_BITS_MAGNITUDE               (0xffffULL << RL_TOOLS_STATUS_BITS_MAGNITUDE_SHIFT)

typedef uint64_t RLtoolsStatus;

#ifdef __cplusplus
extern "C" {
#endif
    void rl_tools_init();
    void rl_tools_reset();
    float rl_tools_test(RLtoolsAction* action);
    // note: DON'T pass an uint32 timestamp here, which might wrap around after ~1h
    bool rl_tools_healthy(RLtoolsStatus status);
    RLtoolsStatus rl_tools_control(uint64_t microseconds, RLtoolsObservation* observation, RLtoolsAction* action);
    const char* rl_tools_get_checkpoint_name();
    char* rl_tools_get_status_message(RLtoolsStatus status);
    float rl_tools_timing_get_bias(bool original);
#ifdef __cplusplus
}
#endif



#endif