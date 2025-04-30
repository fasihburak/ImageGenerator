# ai-prompt-to-image-generator


## Stack

Python: 3.12.3

Flask: 3.0.3

## How to Run?

```bash
python -m venv zenv
source zenv/bin/activate # Mac
```

-   Install basic requirements

```bash
pip install -r requirements.txt

# OR INITIAL INSTALLATION

pip install --upgrade pip
pip install --upgrade setuptools

pip install transformers diffusers flask pillow
```

-   Install PyTorch ([CUDA](https://pytorch.org/get-started/locally/))

```bash
# CUDA 12.1 (Windows)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.1 (Windows)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# MAC
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# OR (Others)
pip install torch torchvision
```

### To Run

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser!

