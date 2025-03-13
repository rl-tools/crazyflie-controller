#ifdef __cplusplus
extern "C"{
#endif
    void rl_tools_init();
    void rl_tools_reset();
    float rl_tools_test(float*);
    void rl_tools_control(float* state, float* actions);
    char* rl_tools_get_checkpoint_name();
#ifdef __cplusplus
}
#endif

