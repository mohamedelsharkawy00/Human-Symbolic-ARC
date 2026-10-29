cmake_minimum_required(VERSION 3.18)
project(ARC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# WIN32_LEAN_AND_MEAN keeps windows.h from pulling in the old winsock.h, so
# winsock2.h (36_command_server.hpp) no longer has to come before every windows.h.
add_compile_definitions(UNICODE _UNICODE NOMINMAX WIN32 WIN32_LEAN_AND_MEAN)

# MSVC flags. /analyze is Release-only (slow in dev).
# Off by choice, not by breakage: it is the only thing that checks a to_text
# format string against its arguments, and the fmtlib false positives that
# forced it off left with fmtlib. Run on 2026-09-05 over all 112 call sites and
# came back clean, proven by planting a bad one and watching C6273 fire. Commented
# out 2026-10-08: it cost about 25 s of each C++ file's 32 s.
# add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/analyze;/analyze:stacksize$<SEMICOLON>65536;/analyze:external->")

if(MSVC)
    string(REGEX REPLACE "/Ob[0-9]" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/W4>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/MP>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")  # source and exec charset agree
    # No /GL (and so no /LTCG) since 2026-10-08: a scored run ran as fast without it,
    # and the code generation it moves into the link made each link 6-26 s.
    add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2;/Oi;/Ot;/GS-;/fp:fast;/Qpar;/favor:blend>")
    add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:/O2;/Ob2;/Oi;/Ot;/Zi>")
endif()

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

# The solver on its own, no window: every reading the GUI shows runs in one of
# these, and it is what Kaggle runs. Built here too, so what Kaggle compiles is
# compiled on every build and can be run on a desk. It alone has the ops
# (include/ops), what runs a program, and the tools that run programs
# (scripts: said, spell) on its include path.
add_executable(arc_headless "src/kaggle_main.cpp")
target_link_libraries(arc_headless PRIVATE arc_core)
target_include_directories(arc_headless PRIVATE include/ops scripts)
# It reads its settings with getenv, the call Kaggle's compiler has; MSVC would
# rather it used its own _dupenv_s.
target_compile_definitions(arc_headless PRIVATE _CRT_SECURE_NO_WARNINGS)

# The GUI: a window onto the reading, which runs in an arc_headless child. It
# has the solver's headers and not its ops, so it cannot run a program: GUI
# code that tried would fail to link.
# Its code compiles apart, in arc_gui, because Ninja holds back every compile
# of a target until the targets it depends on are built: inside ARC, gui_main.cpp
# waited on arc_headless's whole build. Now only ARC's link waits.
add_library(arc_gui OBJECT "src/gui_main.cpp")
target_link_libraries(arc_gui PRIVATE arc_core)
target_include_directories(arc_gui PRIVATE include/gui ${Vulkan_INCLUDE_DIRS})
set(SOURCES
    "${SHADERS_OUT_DIR}/vert.spv"
    "${SHADERS_OUT_DIR}/frag.spv"
    "${CMAKE_CURRENT_BINARY_DIR}/app.rc"
)
add_executable(ARC WIN32 ${SOURCES})
target_link_libraries(ARC PRIVATE arc_gui)

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

# It starts arc_headless.exe, so building it builds that too.
add_dependencies(ARC arc_headless)

# Linker options. WIN32 target => /subsystem:windows (no console), but the entry
# point is plain main(), so point the CRT at mainCRTStartup instead of WinMain.
target_link_options(ARC PRIVATE "/ENTRY:mainCRTStartup")
target_link_options(ARC PRIVATE "$<$<CONFIG:Release>:/OPT:REF;/OPT:ICF>")
target_link_options(ARC PRIVATE "$<$<CONFIG:RelWithDebInfo>:/DEBUG;/OPT:REF;/OPT:ICF>")
target_link_options(arc_headless PRIVATE "$<$<CONFIG:Release>:/OPT:REF;/OPT:ICF>")
target_link_options(arc_headless PRIVATE "$<$<CONFIG:RelWithDebInfo>:/DEBUG;/OPT:REF;/OPT:ICF>")

# Libraries
target_link_libraries(ARC PRIVATE Vulkan::Vulkan)
target_link_libraries(ARC PRIVATE user32 gdi32 shell32 advapi32 winmm ws2_32)
