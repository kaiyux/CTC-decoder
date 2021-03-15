import sys
import numpy as np
from time import time
from src.prefix_beam_search import decode

sys.path.append('./cmake-build-release')
import ctcdecoder

if __name__ == '__main__':
    np.random.seed(3)

    time_len = 100
    output_dim = 100

    probs = np.random.rand(time_len, output_dim)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    # print('probs: ', probs)
    probs = np.log(probs)

    print('----------------- python -----------------')
    tik = time()
    labels, score = decode(probs, 10, 0)
    tok = time()
    print('Labels: ', labels)
    print("Score {:.3f}".format(score))
    print('Time: ', tok - tik)

    print('----------------- cpp -----------------')
    tik = time()
    labels, score = ctcdecoder.decode(probs, 10, 0)
    tok = time()
    print('Labels: ', labels)
    print("Score {:.3f}".format(score))
    print('Time: ', tok - tik)
