# project(crazyflie_controller_test)


# add_executable(crazyflie_controller_test rl_tools_adapter_test.cpp rl_tools_adapter_new.cpp)
# target_include_directories(crazyflie_controller_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/rl_tools/include)
# target_compile_definitions(crazyflie_controller_test PRIVATE RL_TOOLS_ENABLE_INFORMATIVE_STATUS_MESSAGES)

project(crazyflie_controller_test)

# Create the executable
add_executable(crazyflie_controller_test 
    rl_tools_adapter_test.cpp 
    rl_tools_adapter_new.cpp
)

# Include directories
target_include_directories(crazyflie_controller_test 
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/rl_tools/include
)

# Base compile definitions
target_compile_definitions(crazyflie_controller_test 
    PRIVATE RL_TOOLS_ENABLE_INFORMATIVE_STATUS_MESSAGES
)

# Enable maximum warnings and treat them as errors
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(crazyflie_controller_test PRIVATE
        -Wextra            # Extra warnings
        -Werror            # Treat warnings as errors
        -Wshadow           # Warn on variable shadowing
        -Woverflow         # Warn on overflow
        -Wcast-align       # Warn on misaligned casts
        -Wformat=2         # Enhanced format string checking
        -Wnull-dereference # Warn on null dereferences
        -Warray-bounds     # Warn on array bounds violations
        -Wno-unused-local-typedefs
    )
elseif(MSVC)
    target_compile_options(crazyflie_controller_test PRIVATE
        /W4               # Warning level 4
        /WX               # Treat warnings as errors
        /sdl              # Enable additional security checks
    )
endif()

# Enable sanitizers (only effective in Debug builds typically)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(crazyflie_controller_test PRIVATE
        -fsanitize=address          # AddressSanitizer
        -fsanitize=undefined        # UndefinedBehaviorSanitizer
        -fsanitize=leak             # LeakSanitizer
        -fno-omit-frame-pointer     # Better stack traces
    )
    target_link_options(crazyflie_controller_test PRIVATE
        -fsanitize=address
        -fsanitize=undefined
        -fsanitize=leak
    )
endif()

# Stack protection and other security features
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(crazyflie_controller_test PRIVATE
        -fstack-protector-strong      # Strong stack-smashing protection
        # -fstack-check               # Stack overflow checking
        # -mbranch-protection=standard # Branch protection (on supported arch)
        -D_FORTIFY_SOURCE=2         # Buffer overflow protection
        -fcf-protection             # Control-flow protection
    )
endif()

# Enable debug symbols and disable optimizations for better debugging
if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(crazyflie_controller_test PRIVATE
        -g3                 # Maximum debug info
        -O0                 # No optimizations
    )
endif()

# Set C++ standard with additional safety
set_target_properties(crazyflie_controller_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF      # Disable compiler extensions
)