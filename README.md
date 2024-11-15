## Installation

1. Create a conda environment and install the required dependencies:
```bash
mamba env create -f environment.yaml
```

2. Activate the environment:
```bash
conda activate find
```

3. Install bitsandbytes package from source to enable quantization:
```bash
git clone https://github.com/TimDettmers/bitsandbytes.git && cd bitsandbytes/
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=cuda -S .
make
pip install .
```