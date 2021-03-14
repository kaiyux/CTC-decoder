#include <iostream>
#include "ctcdecoder/prefix_beam_search.cpp"

int main() {
    auto res = decode();
    auto labels = res.first;
    auto score = res.second;

    std::cout << "Labels:" << std::endl;
    for (auto l:labels) {
        std::cout << l << " ";
    }
    std::cout << std::endl;

    std::cout << "Score:" << std::endl;
    std::cout << score << std::endl;

    return 0;
}
