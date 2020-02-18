cmake_minimum_required(VERSION 3.10)
project (projeto_basico)
add_executable(hello hello.cpp func.cpp) 
target_compile_options(hello PUBLIC -O3)
