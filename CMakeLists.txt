cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_FLAGS "--allow-unsupported-compiler -Xptxas=-v --expt-relaxed-constexpr" CACHE STRING "" FORCE)
project(ARC LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "89") # RTX 4050

add_compile_definitions(UNICODE _UNICODE NOMINMAX WIN32)

# MSVC flags (CXX only â€” exclude CUDA). /analyze is Release-only (slow in dev).
# Off by choice, not by breakage: it is the only thing that checks a to_text
# format string against its arguments, and the fmtlib false positives that
# forced it off left with fmtlib.
# add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/analyze;/analyze:stacksize$<SEMICOLON>65536;/analyze:external->")

if(MSVC)
    string(REGEX REPLACE "/Ob[0-9]" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/MP>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")  # source and exec charset agree
    add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2;/Oi;/Ot;/GS-;/GL;/fp:fast;/Qpar;/favor:blend>")
    add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:/O2;/Ob2;/Oi;/Ot;/Zi>")
endif()

# CUDA: emit device line-number info in RelWithDebInfo (the profiling preset) so
# Nsight Compute can map hardware counters back to cuda_solver.cu source lines
# (paired with `ncu --import-source yes` in profile_ncu.bat). -lineinfo keeps the
# kernel FULLY optimized â€” it only adds a line table â€” unlike -G, which disables
# optimization and would make the profiled kernel unrepresentative. Release stays
# lean with no line table.
add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-lineinfo>")

find_package(Vulkan REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter)

# Shaders
set(SHADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders")
set(SHADERS_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/out")

add_custom_command(
    OUTPUT "${SHADERS_OUT_DIR}/vert.spv"
    COMMAND Vulkan::glslc -fshader-stage=vertex "${SHADERS_DIR}/vert.glsl" -o "${SHADERS_OUT_DIR}/vert.spv"
    DEPENDS "${SHADERS_DIR}/vert.glsl"
    COMMENT "Compiling vertex shader..."
)

add_custom_command(
    OUTPUT "${SHADERS_OUT_DIR}/frag.spv"
    COMMAND Vulkan::glslc -fshader-stage=fragment "${SHADERS_DIR}/frag.glsl" -o "${SHADERS_OUT_DIR}/frag.spv"
    DEPENDS "${SHADERS_DIR}/frag.glsl"
    COMMENT "Compiling fragment shader..."
)

# Resources
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/app.rc" "${CMAKE_CURRENT_BINARY_DIR}/icon.ico"
    COMMAND Python::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/scripts/make_resources.py" "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/make_resources.py" "${CMAKE_CURRENT_SOURCE_DIR}/assets/arc_logo.svg"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Generating resources..."
)

# Executable
set(SOURCES
    "src/main.cpp"
    "src/cuda_solver.cu"
    "${SHADERS_OUT_DIR}/vert.spv"
    "${SHADERS_OUT_DIR}/frag.spv"
    "${CMAKE_CURRENT_BINARY_DIR}/app.rc"
)
add_executable(ARC WIN32 ${SOURCES})

# Backup
add_custom_command(
    TARGET ARC POST_BUILD
    COMMAND Python::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/scripts/backup.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running backup..."
)

# Include Directories
target_include_directories(ARC PRIVATE include)
target_include_directories(ARC PRIVATE ${Vulkan_INCLUDE_DIRS})

# Linker options. WIN32 target => /subsystem:windows (no console), but the entry
# point is plain main(), so point the CRT at mainCRTStartup instead of WinMain.
target_link_options(ARC PRIVATE "/ENTRY:mainCRTStartup")
target_link_options(ARC PRIVATE "$<$<CONFIG:Release>:/LTCG;/OPT:REF;/OPT:ICF>")
target_link_options(ARC PRIVATE "$<$<CONFIG:RelWithDebInfo>:/DEBUG;/OPT:REF;/OPT:ICF>")

# Libraries
target_link_libraries(ARC PRIVATE Vulkan::Vulkan)
target_link_libraries(ARC PRIVATE user32 gdi32 shell32 advapi32 winmm ws2_32)

# Text helper test. Its own target so it never touches the app, and it runs
# itself at build time, so a broken helper stops the build instead of waiting to
# be remembered. Silent when it passes.
add_executable(text_format_test "tests/text_format_test.cpp")
target_include_directories(text_format_test PRIVATE include)
add_custom_command(
    TARGET text_format_test POST_BUILD
    COMMAND text_format_test
    COMMENT "Checking text helpers..."
)
