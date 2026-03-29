#!/bin/bash

sudo apt install -y ffmpeg

python3.12 -m venv venv
source venv/bin/activate
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install .

echo "traiNNer-redux dependencies installed successfully!"
