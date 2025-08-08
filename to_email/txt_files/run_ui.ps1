Write-Host "Installing requirements..."
pip install -r requirements.txt
Write-Host "Requirements installed"

Write-Host "Running document UI..."
python -m src.web.document_ui
