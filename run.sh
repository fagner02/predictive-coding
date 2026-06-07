g++ -I/usr/include -I./include -I. -L/lib/x86_64-linux-gnu \
main.cpp \
shaders/shader.cpp \
vertex_array.cpp \
shapes/point.cpp \
shapes/plane.cpp \
shapes/cube.cpp \
include/imgui/imgui.cpp \
include/imgui/imgui_draw.cpp \
include/imgui/imgui_tables.cpp \
include/imgui/imgui_widgets.cpp \
include/imgui/imgui_impl_glfw.cpp \
include/imgui/imgui_impl_opengl3.cpp \
-o main -lGLEW -lGL -lglfw -lsfml-graphics -lsfml-window -lsfml-system