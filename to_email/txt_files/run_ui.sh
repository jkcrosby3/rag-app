#!/bin/bash

echo "Installing requirements..."
pip install -r requirements.txt
echo "Requirements installed"

echo "Running document UI..."
python -m src.web.document_ui
