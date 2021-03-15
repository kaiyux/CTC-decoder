# CTC-Decoder
## A cpp reimplementation
This repo is a cpp reimplementation for `Awni Hannun`'s version of CTC decoder. It runs 4.5x faster in my MacBook Pro & No extra dependency is needed.
## Usage
Firstly
```bash
mkdir cmake-build-release
cd cmake-build-release

/path/to/cmake -DCMAKE_BUILD_TYPE=Release -G "CodeBlocks - Unix Makefiles" /path/to/CTC-decoder

/path/to/cmake --build /path/to/CTC-decoder/cmake-build-release --target ctcdecoder -- -j 4
```
Then scripts in Python
```Python
import sys
sys.path.append('/path/to/cmake-build-release')
import ctcdecoder

probs = np.random.rand(time_len, output_dim)
probs = probs / np.sum(probs, axis=1, keepdims=True)
probs = np.log(probs)
labels, score = ctcdecoder.decode(probs, 10, 0)
```
