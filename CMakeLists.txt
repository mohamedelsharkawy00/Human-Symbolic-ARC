cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_FLAGS "--allow-unsupported-compiler -Xptxas=-v --expt-relaxed-constexpr -Wreorder -Wmissing-launch-bounds -Wdefault-stream-launch -Wext-lambda-captures-this -Xcompiler /W4" CACHE STRING "" FORCE)
# Tried --Ofast-compile=min (nvcc 13.2) on 2026-09-09: it cut a fresh Release
# build to well under half, almost all of it out of the CUDA compile, where both
# cicc and ptxas came down by roughly four times. It does not go in the line
# above as it stands: fast-compile switches device code to per-function
# compilation, and ptxas then rejects the whole-program enable_smem_spilling
# pragma in cuda_solver.cu ("Pragma 'enable_smem_spilling' is not allowed for
# per-function compilation modes"). That measurement therefore came from a build
# with the pragma commented out, so it also gives up the shared-memory spilling
# the occupancy cap relies on. Solving looked the same speed by eye, unmeasured.
# If it is wanted, it belongs in a separate dev preset with the pragma behind an
# #if, never in Release.
project(ARC LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "89") # RTX 4050

# WIN32_LEAN_AND_MEAN keeps windows.h from pulling in the old winsock.h, so
# winsock2.h (32_command_server.hpp) no longer has to come before every windows.h.
add_compile_definitions(UNICODE _UNICODE NOMINMAX WIN32 WIN32_LEAN_AND_MEAN)

# MSVC flags (CXX only â€” exclude CUDA). /analyze is Release-only (slow in dev).
# Off by choice, not by breakage: it is the only thing that checks a to_text
# format string against its arguments, and the fmtlib false positives that
# forced it off left with fmtlib. Run on 2026-09-05 over all 112 call sites and
# came back clean, proven by planting a bad one and watching C6273 fire. On while
# its remaining findings are being worked through.
add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/analyze;/analyze:stacksize$<SEMICOLON>65536;/analyze:external->")

if(MSVC)
    string(REGEX REPLACE "/Ob[0-9]" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/W4>")
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
if(ARC_BUILD_TIMING_DIR)
    # Keep the profiling build's shaders fresh and isolated too.
    set(SHADERS_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
endif()

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

# THE SOLVER'S HEADERS: include/core, and nothing else on its include path, so
# a core file that reached for anything in include/gui would not build.
add_library(arc_core INTERFACE)
target_include_directories(arc_core INTERFACE include/core)
# Off since 2026-09-09: nothing calls it yet and building it costs roughly a
# third of every clean build. The reasoning sits above the kernel in cuda_solver.cu.
option(ARC_ENABLE_CUDA_PIECES "Build the currently unused CUDA pieces extraction API" OFF)
target_compile_definitions(arc_core INTERFACE ARC_ENABLE_CUDA_PIECES=$<BOOL:${ARC_ENABLE_CUDA_PIECES}>)

# THE SOLVER'S GPU HALF: the CUDA kernel. Only the program that searches links it.
add_library(arc_gpu STATIC "src/cuda_solver.cu")
target_link_libraries(arc_gpu PUBLIC arc_core)

# The solver on its own, no window: every search, the GUI's included, runs in
# one of these, and it is what Kaggle runs. Built here too, so what Kaggle
# compiles is compiled on every build and can be run on a desk.
add_executable(arc_headless "src/kaggle_main.cpp")
target_link_libraries(arc_headless PRIVATE arc_gpu)
# It reads its settings with getenv, the call Kaggle's compiler has; MSVC would
# rather it used its own _dupenv_s.
target_compile_definitions(arc_headless PRIVATE _CRT_SECURE_NO_WARNINGS)

# ONE PROGRAM, ONE PUZZLE, NO SEARCH: hands a written program to a puzzle and
# shows what it drew beside what was wanted, pair by pair. It answers only
# "can the words we already have say this?", which is the question a new op
# should have to survive before it is added. Run it from the repo root:
#   arc_spell <task-id> "<program>"      one program
#   arc_spell <task-id> --file <path>    one a line, each reported in turn
add_executable(arc_spell "scripts/spell.cpp")
target_link_libraries(arc_spell PRIVATE arc_gpu)
target_compile_definitions(arc_spell PRIVATE _CRT_SECURE_NO_WARNINGS)

# WHAT THE READING SOLVES WITHOUT SEARCHING, counted over a whole dataset, and
# every puzzle's inference readout, in one run into out/<label>. The guard on
# say_the_reading: a sentence it spells that draws the wrong thing is reported
# here and nowhere else. scripts\said.bat builds it and runs it, and
# scripts\readouts.py compares two runs. Built with ARC_NO_GPU (the Linux
# cloud), it writes the readouts alone.
add_executable(arc_said "scripts/said.cpp")
target_link_libraries(arc_said PRIVATE arc_gpu)
target_compile_definitions(arc_said PRIVATE _CRT_SECURE_NO_WARNINGS)

# The GUI: a window onto the search, which runs in an arc_headless child. It
# links the solver's headers and not its GPU half, so it cannot create a CUDA
# context: GUI code that reached for the GPU would fail to link.
set(SOURCES
    "src/gui_main.cpp"
    "${SHADERS_OUT_DIR}/vert.spv"
    "${SHADERS_OUT_DIR}/frag.spv"
    "${CMAKE_CURRENT_BINARY_DIR}/app.rc"
)
add_executable(ARC WIN32 ${SOURCES})
target_link_libraries(ARC PRIVATE arc_core)

# Backup
set(ARC_BACKUP_COMMAND Python::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/scripts/backup.py")
if(ARC_BUILD_TIMING_DIR)
    set(ARC_BACKUP_COMMAND Python::Interpreter
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/profile_build.py" --time-command
        "${ARC_BUILD_TIMING_DIR}" backup --
        "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/backup.py")
endif()
add_custom_command(
    TARGET ARC POST_BUILD
    COMMAND ${ARC_BACKUP_COMMAND}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running backup..."
)

# Include Directories. include/core arrives with arc_core.
# It starts arc_headless.exe, so building it builds that too.
add_dependencies(ARC arc_headless)
target_include_directories(ARC PRIVATE include/gui)
target_include_directories(ARC PRIVATE ${Vulkan_INCLUDE_DIRS})

# Linker options. WIN32 target => /subsystem:windows (no console), but the entry
# point is plain main(), so point the CRT at mainCRTStartup instead of WinMain.
target_link_options(ARC PRIVATE "/ENTRY:mainCRTStartup")
target_link_options(ARC PRIVATE "$<$<CONFIG:Release>:/LTCG;/OPT:REF;/OPT:ICF>")
target_link_options(ARC PRIVATE "$<$<CONFIG:RelWithDebInfo>:/DEBUG;/OPT:REF;/OPT:ICF>")
target_link_options(arc_headless PRIVATE "$<$<CONFIG:Release>:/LTCG;/OPT:REF;/OPT:ICF>")
target_link_options(arc_headless PRIVATE "$<$<CONFIG:RelWithDebInfo>:/DEBUG;/OPT:REF;/OPT:ICF>")

# Libraries
target_link_libraries(ARC PRIVATE Vulkan::Vulkan)
target_link_libraries(ARC PRIVATE user32 gdi32 shell32 advapi32 winmm ws2_32)
