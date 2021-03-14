cmake_minimum_required(VERSION 3.4...3.18)
project(ctcdecoder)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(pybind11)
pybind11_add_module(ctcdecoder ctcdecoder/prefix_beam_search.cpp)
