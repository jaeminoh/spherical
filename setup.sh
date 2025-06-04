# requirements
uv sync 
source .venv/bin/activate
pre-commit install
uv pip install torch --index-url https://download.pytorch.org/whl/cpu