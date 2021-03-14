cmake_minimum_required(VERSION 3.15)
project(CTC_decoder)

set(CMAKE_CXX_STANDARD 14)

find_package(pybind11 REQUIRED)
pybind11_add_module(ctcdecoder ctcdecoder/prefix_beam_search.cpp)

add_executable(CTC_decoder test.cpp)
