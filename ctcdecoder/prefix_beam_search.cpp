//
// Created by Kaiyu Xie on 2021/3/14.
//

#include <vector>
#include <pybind11/pybind11.h>

namespace py = pybind11;


std::pair<std::vector<int>, float> decode() {
    return make_pair(std::vector<int>(), 0);
}

PYBIND11_MODULE(test, m) {
    m.doc() = "CTC Decoder";
    m.def("decode", &decode);
}
