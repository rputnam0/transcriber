.PHONY: setup lint test run docker-build docker-run docker-run-watch

HF_CACHE ?= $(HOME)/.cache/huggingface

setup:
	uv python install 3.11
	uv sync
	pre-commit install

lint:
	uv run ruff . && uv run black --check .

test:
	uv run pytest -q

# Local GPU run (WhisperX backend by default)
run:
	uv run transcribe $(INPUT) --model $(MODEL) --output-dir outputs --backend whisperx \
	--min-speakers $(MIN) --max-speakers $(MAX)

# Container build and run (GPU): adjust CUDA/torch channel via ARGs if needed
docker-build:
	docker build -t transcriber:gpu .

docker-run:
	docker run --rm --gpus all \
	  -v $(PWD):/app \
	  -v $(HF_CACHE):/root/.cache/huggingface \
	  -e HF_HOME=/root/.cache/huggingface \
	  -e HF_TOKEN=$$HF_TOKEN \
	  -w /app \
	  transcriber:gpu \
	  "uv run transcribe $(INPUT) --model $(MODEL) --output-dir /app/outputs --backend whisperx --min-speakers $(MIN) --max-speakers $(MAX)"

docker-run-watch:
	docker run --rm --gpus all \
	  -v $(PWD):/app \
	  -v $(HF_CACHE):/root/.cache/huggingface \
	  -e HF_HOME=/root/.cache/huggingface \
	  -e HF_TOKEN=$$HF_TOKEN \
	  -w /app \
	  transcriber:gpu \
	  "uv run transcribe audio/ --watch"
