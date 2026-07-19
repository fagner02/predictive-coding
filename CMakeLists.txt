cmake_minimum_required(VERSION 3.30)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libs" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_WAYLAND ON CACHE BOOL "GLFW use wayland" FORCE)
set(GLFW_BUILD_X11 OFF CACHE BOOL "GLFW use x11" FORCE)
set(USE_WAYLAND ON CACHE BOOL "" FORCE)
set(USE_X11 OFF CACHE BOOL "" FORCE)

project(PredictiveCode)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

add_executable(predictive_code
    main.cpp
    inputs/input.hpp
    inputs/simple_input.cpp
    inputs/xor_input.cpp
    inputs/majority_input.cpp
    inputs/palindrome_input.cpp
    inputs/grammarinput.cpp
    neuron.cpp
)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG        6.0
)
FetchContent_MakeAvailable(raylib)

FetchContent_Declare(
    eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen
    GIT_TAG 5.0.1
)
FetchContent_MakeAvailable(eigen3)

FetchContent_Declare(
    rlimgui
    GIT_REPOSITORY https://github.com/raylib-extras/rlImGui.git
    GIT_TAG        Raylib_6_0
)
FetchContent_MakeAvailable(rlimgui)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.92.7
)
FetchContent_Declare(
    implot
    GIT_REPOSITORY https://github.com/epezent/implot.git
    GIT_TAG v1.0
)

FetchContent_MakeAvailable(imgui)
add_library(imgui STATIC)
target_sources(imgui PRIVATE
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
)

target_include_directories(imgui PUBLIC
${imgui_SOURCE_DIR}
${imgui_SOURCE_DIR}/backends
${rlimgui_SOURCE_DIR}
)

target_link_libraries(imgui PUBLIC OpenGL::GL)

FetchContent_MakeAvailable(implot)

add_library(implot INTERFACE)
target_sources(implot INTERFACE
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp
)

target_include_directories(implot INTERFACE ${implot_SOURCE_DIR})
target_link_libraries(implot INTERFACE imgui)

target_sources(predictive_code PRIVATE
    ${rlimgui_SOURCE_DIR}/rlImGui.cpp)

target_link_libraries(predictive_code PRIVATE
raylib
OpenGL::GL
GLEW::GLEW
implot
imgui
)

target_include_directories(predictive_code PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${eigen3_SOURCE_DIR}
)
