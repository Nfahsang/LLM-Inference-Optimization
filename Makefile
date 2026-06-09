.DEFAULT_GOAL := help

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL        ?= meta-llama/Llama-3.1-8B-Instruct
DRAFT_MODEL  ?= meta-llama/Llama-3.2-1B-Instruct
MODEL_QWEN   ?= Qwen/Qwen2.5-7B-Instruct
DTYPE        ?= float16
NUM_PROMPTS  ?= 1000
REQUEST_RATE ?= 10
DATASET      ?= ./data/ShareGPT_V3_unfiltered_cleaned_split.json
RESULTS_DIR  ?= ./results
VLLM_PORT    ?= 8000
SGLANG_PORT  ?= 30000
DATE         := $(shell date +%Y%m%d)

# ── Help ──────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@printf "\nTargets:\n"
	@printf "  setup           Install vLLM, SGLang, and eval dependencies\n"
	@printf "  data            Download ShareGPT benchmark dataset\n"
	@printf "  bench           FP16 baseline on vLLM (alias for bench-fp16)\n"
	@printf "  bench-fp16      vLLM FP16 baseline benchmark\n"
	@printf "  bench-sglang    SGLang FP16 baseline benchmark\n"
	@printf "  bench-quant     Full quantization sweep (INT8, INT4-AWQ, INT4-GPTQ)\n"
	@printf "  bench-specdec   Speculative decoding benchmark (Phase 2)\n"
	@printf "  eval-mt-bench   Run MT-Bench quality evaluation\n"
	@printf "\nVariables (override with VAR=value):\n"
	@printf "  MODEL           $(MODEL)\n"
	@printf "  NUM_PROMPTS     $(NUM_PROMPTS)\n"
	@printf "  REQUEST_RATE    $(REQUEST_RATE)\n"
	@printf "  DTYPE           $(DTYPE)\n"
	@printf "\n"

# ── Setup ─────────────────────────────────────────────────────────────────────
.PHONY: setup
setup:
	pip install --upgrade pip
	pip install vllm
	pip install sglang[all]
	pip install fschat lm-eval datasets huggingface_hub
	@echo "Verify GPU: nvidia-smi"
	@echo "Verify vLLM: python -c \"import vllm; print(vllm.__version__)\""

# ── Data ──────────────────────────────────────────────────────────────────────
.PHONY: data
data:
	mkdir -p data
	python -c "\
	from huggingface_hub import hf_hub_download; \
	hf_hub_download(repo_id='anon8231489123/ShareGPT_Vicuna_unfiltered', \
	                filename='ShareGPT_V3_unfiltered_cleaned_split.json', \
	                repo_type='dataset', local_dir='./data')"
	@echo "Dataset ready at $(DATASET)"

# ── Phase 1: Baseline benchmarks ──────────────────────────────────────────────
.PHONY: bench bench-fp16
bench: bench-fp16

bench-fp16:
	$(eval OUT := $(RESULTS_DIR)/$(DATE)_vllm_fp16)
	mkdir -p $(OUT)
	@echo "Starting vLLM server (FP16)..."
	python -m vllm.entrypoints.openai.api_server \
		--model $(MODEL) \
		--dtype float16 \
		--port $(VLLM_PORT) \
		--max-model-len 4096 &
	@echo "Waiting for server..."
	@until curl -s http://localhost:$(VLLM_PORT)/health > /dev/null; do sleep 2; done
	python benchmarks/benchmark_serving.py \
		--backend openai-chat \
		--base-url http://localhost:$(VLLM_PORT) \
		--model $(MODEL) \
		--dataset-name sharegpt \
		--dataset-path $(DATASET) \
		--num-prompts $(NUM_PROMPTS) \
		--request-rate $(REQUEST_RATE) \
		--save-result \
		--result-dir $(OUT)
	@pkill -f "vllm.entrypoints" || true
	@echo "Results saved to $(OUT)"

.PHONY: bench-sglang
bench-sglang:
	$(eval OUT := $(RESULTS_DIR)/$(DATE)_sglang_fp16)
	mkdir -p $(OUT)
	@echo "Starting SGLang server (FP16)..."
	python -m sglang.launch_server \
		--model-path $(MODEL) \
		--dtype float16 \
		--port $(SGLANG_PORT) &
	@until curl -s http://localhost:$(SGLANG_PORT)/health > /dev/null; do sleep 2; done
	python -m sglang.bench_serving \
		--backend sglang \
		--host 127.0.0.1 \
		--port $(SGLANG_PORT) \
		--model $(MODEL) \
		--dataset-name sharegpt \
		--dataset-path $(DATASET) \
		--num-prompts $(NUM_PROMPTS) \
		--request-rate $(REQUEST_RATE) \
		--result-filename $(OUT)/result.json
	@pkill -f "sglang.launch_server" || true
	@echo "Results saved to $(OUT)"

.PHONY: bench-quant
bench-quant:
	@echo "Running quantization sweep..."
	$(MAKE) bench-fp16
	$(MAKE) _bench-vllm-single QUANT_MODEL=neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 TAG=fp8
	$(MAKE) _bench-vllm-single QUANT_MODEL=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 TAG=int4_awq
	$(MAKE) _bench-vllm-single QUANT_MODEL=ModelCloud/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 TAG=int4_gptq

.PHONY: _bench-vllm-single
_bench-vllm-single:
	$(eval OUT := $(RESULTS_DIR)/$(DATE)_vllm_$(TAG))
	mkdir -p $(OUT)
	python -m vllm.entrypoints.openai.api_server \
		--model $(QUANT_MODEL) \
		--port $(VLLM_PORT) \
		--max-model-len 4096 &
	@until curl -s http://localhost:$(VLLM_PORT)/health > /dev/null; do sleep 2; done
	python benchmarks/benchmark_serving.py \
		--backend openai-chat \
		--base-url http://localhost:$(VLLM_PORT) \
		--model $(QUANT_MODEL) \
		--dataset-name sharegpt \
		--dataset-path $(DATASET) \
		--num-prompts $(NUM_PROMPTS) \
		--request-rate $(REQUEST_RATE) \
		--save-result \
		--result-dir $(OUT)
	@pkill -f "vllm.entrypoints" || true

# ── Phase 2: Speculative decoding ─────────────────────────────────────────────
.PHONY: bench-specdec
bench-specdec:
	$(eval OUT := $(RESULTS_DIR)/$(DATE)_vllm_specdec)
	mkdir -p $(OUT)
	python -m vllm.entrypoints.openai.api_server \
		--model $(MODEL) \
		--dtype float16 \
		--speculative-model $(DRAFT_MODEL) \
		--num-speculative-tokens 5 \
		--port $(VLLM_PORT) &
	@until curl -s http://localhost:$(VLLM_PORT)/health > /dev/null; do sleep 2; done
	python benchmarks/benchmark_serving.py \
		--backend openai-chat \
		--base-url http://localhost:$(VLLM_PORT) \
		--model $(MODEL) \
		--dataset-name sharegpt \
		--dataset-path $(DATASET) \
		--num-prompts $(NUM_PROMPTS) \
		--request-rate $(REQUEST_RATE) \
		--save-result \
		--result-dir $(OUT)
	@pkill -f "vllm.entrypoints" || true

# ── Evaluation ────────────────────────────────────────────────────────────────
.PHONY: eval-mt-bench
eval-mt-bench:
	@echo "MT-Bench requires FastChat. Run from eval/mt_bench/:"
	@echo "  cd eval/mt_bench"
	@echo "  python gen_model_answer.py --model-path $(MODEL) --model-id llama3-8b"
	@echo "  python gen_judgment.py --model-list llama3-8b --judge-model gpt-4"
	@echo "  python show_result.py"
	@echo "See eval/mt_bench/README.md for setup."
