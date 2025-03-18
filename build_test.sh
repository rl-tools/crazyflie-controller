g++ -I external/rl_tools/include/ rl_tools_adapter_new.cpp rl_tools_adapter_test.cpp \
    -fsanitize=address,undefined -fstack-protector-strong && ./a.out
