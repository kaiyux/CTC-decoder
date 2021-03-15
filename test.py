import sys
import numpy as np
from time import time
import random
from src.prefix_beam_search import decode

sys.path.append('./cmake-build-release')
import ctcdecoder

if __name__ == '__main__':
    np.random.seed(3)

    python_time = 0
    cpp_time = 0
    epochs = 100

    for i in range(epochs):
        time_len = random.randint(10, 200)
        output_dim = random.randint(10, 200)

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
        python_time += tok - tik

        print('----------------- cpp -----------------')
        tik = time()
        labels, score = ctcdecoder.decode(probs, 10, 0)
        tok = time()
        print('Labels: ', labels)
        print("Score {:.3f}".format(score))
        print('Time: ', tok - tik)
        cpp_time += tok - tik

    print(f'Time cost: Python/cpp - {python_time / cpp_time}')
