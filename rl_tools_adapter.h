#ifdef __cplusplus
extern "C"
#endif
void rl_tools_init();
#ifdef __cplusplus
extern "C"
#endif
float rl_tools_test(float*);
#ifdef __cplusplus
extern "C"
#endif
bool rl_tools_get_action_history(float* output, int length);
#ifdef __cplusplus
extern "C"
#endif
void rl_tools_reset();
#ifdef __cplusplus
extern "C"
#endif
void rl_tools_control(float* state, float* actions);
#ifdef __cplusplus
extern "C"
#endif
char* rl_tools_get_checkpoint_name();


