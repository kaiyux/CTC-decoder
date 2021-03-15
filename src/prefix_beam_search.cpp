//
// Created by Kaiyu Xie on 2021/3/14.
//

#include <unordered_map>
#include <cmath>
#include <vector>
#include <cfloat>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"

namespace py = pybind11;


class uint32_vector_hasher {
public:
    std::size_t operator()(std::vector<int> const &vec) const {
        std::size_t seed = vec.size();
        for (auto &i : vec) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

float logsumexp(const std::vector<float> &nums) {
    bool all_neg_inf = true;
    float max_num = -FLT_MAX;
    const double EPS = 0.0000001;
    for (auto n:nums) {
        if (abs(n - (-FLT_MAX)) > EPS) {
            all_neg_inf = false;
        }
        max_num = std::max(max_num, n);
    }
    if (all_neg_inf) {
        return -FLT_MAX;
    }
    float tmp_sum = 0;
    for (auto n:nums) {
        tmp_sum += exp(n - max_num);
    }
    auto lsp = log(tmp_sum);
    return max_num + lsp;
}

std::pair<std::vector<int>, float> decode(py::array_t<float> probs, int beam_size, int blank) {
    auto probs_buf = probs.request();

    const auto T = probs_buf.shape[0];
    const auto S = probs_buf.shape[1];
    auto *probs_ptr = (float *) probs_buf.ptr;

    std::vector<std::pair<std::vector<int>, std::pair<float, float>>> beam;
    auto first_beam = std::make_pair(std::vector<int>(), std::make_pair(0.0, -FLT_MAX));
    beam.emplace_back(first_beam);
    for (int t = 0; t < T; ++t) {
        // Loop over time

        std::unordered_map<std::vector<int>, std::pair<float, float>, uint32_vector_hasher> next_beam;

        for (int s = 0; s < S; ++s) {
            // Loop over vocab
            auto p = probs_ptr[t * S + s];

            for (const auto &b:beam) {
                // Loop over beam
                auto prefix = b.first;
                auto p_b = b.second.first;
                auto p_nb = b.second.second;

                if (s == blank) {
                    float n_p_b = -FLT_MAX;
                    float n_p_nb = -FLT_MAX;
                    if (next_beam.find(prefix) != next_beam.end()) {
                        n_p_b = next_beam[prefix].first;
                        n_p_nb = next_beam[prefix].second;
                    }
                    n_p_b = logsumexp(std::vector<float>{n_p_b, p_b + p, p_nb + p});
                    next_beam[prefix] = std::make_pair(n_p_b, n_p_nb);
                    continue;
                }

                int end_t = -1;
                if (!prefix.empty()) {
                    end_t = prefix.back();
                }
                auto n_prefix = prefix;
                n_prefix.push_back(s);
                float n_p_b = -FLT_MAX;
                float n_p_nb = -FLT_MAX;
                if (next_beam.find(n_prefix) != next_beam.end()) {
                    n_p_b = next_beam[n_prefix].first;
                    n_p_nb = next_beam[n_prefix].second;
                }
                if (s != end_t) {
                    n_p_nb = logsumexp(std::vector<float>{n_p_nb, p_b + p, p_nb + p});
                } else {
                    n_p_nb = logsumexp(std::vector<float>{n_p_nb, p_b + p});
                }
                next_beam[n_prefix] = std::make_pair(n_p_b, n_p_nb);
                if (s == end_t) {
                    n_p_b = -FLT_MAX;
                    n_p_nb = -FLT_MAX;
                    if (next_beam.find(prefix) != next_beam.end()) {
                        n_p_b = next_beam[prefix].first;
                        n_p_nb = next_beam[prefix].second;
                    }
                    n_p_nb = logsumexp(std::vector<float>{n_p_nb, p_nb + p});
                    next_beam[prefix] = std::make_pair(n_p_b, n_p_nb);
                }
            }
        }
        beam = std::vector<std::pair<std::vector<int>, std::pair<float, float>>>(next_beam.begin(),
                                                                                 next_beam.end());
        sort(beam.begin(), beam.end(), [](std::pair<std::vector<int>, std::pair<float, float>> &a,
                                          std::pair<std::vector<int>, std::pair<float, float>> &b) {
            return logsumexp(std::vector<float>{a.second.first, a.second.second}) >
                   logsumexp(std::vector<float>{b.second.first, b.second.second});
        });
        if (beam_size <= beam.size()) {
            beam = std::vector<std::pair<std::vector<int>, std::pair<float, float>>>(beam.begin(),
                                                                                     beam.begin() + beam_size);
        }
    }

    auto best = beam[0];
    return std::make_pair(best.first, -logsumexp(std::vector<float>{best.second.first, best.second.second}));
}

PYBIND11_MODULE(ctcdecoder, m) {
    m.doc() = "CTC Decode";
    m.def("decode", &decode);
}
