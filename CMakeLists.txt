cmake_minimum_required(VERSION 3.30)

set(SFML_STATIC_LIBRARIES OFF CACHE BOOL "" FORCE)
set(SFML_BUILD_AUDIO OFF CACHE BOOL "" FORCE)
set(SFML_BUILD_NETWORK OFF CACHE BOOL "" FORCE)
set(SFML_STATIC_LIBRARIES OFF CACHE BOOL "Use shared SFML" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libs" FORCE)

project(PredictiveCode)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include_directories(${CMAKE_SOURCE_DIR}/include
 ${CMAKE_SOURCE_DIR}
 ${CMAKE_SOURCE_DIR}/src)

include(FetchContent)

add_executable(predictive_code
    main.cpp 
)

find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

FetchContent_Declare(
    ImGui-SFML
    GIT_REPOSITORY https://github.com/SFML/imgui-sfml
    GIT_TAG        v2.6.1
)
FetchContent_Declare(
    ImGui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.0
)

FetchContent_MakeAvailable(ImGui)
set(IMGUI_DIR ${imgui_SOURCE_DIR} CACHE PATH "ImGui source directory" FORCE)

FetchContent_MakeAvailable(ImGui-SFML)
target_link_libraries(predictive_code PRIVATE
OpenGL::GL
glfw
GLEW::GLEW
Eigen3::Eigen
ImGui-SFML::ImGui-SFML
)

target_include_directories(predictive_code PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
)