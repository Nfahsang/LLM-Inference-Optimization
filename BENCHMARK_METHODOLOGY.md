# Benchmark Methodology

This document defines the experimental design for all benchmark results in this repo. Every result table links back here. If you can't reproduce a number from this doc plus the exact CLI in the results directory, it's a bug.

---

## Hardware

| Field | Value |
|---|---|
| GPU | NVIDIA A100 80GB SXM4 |
| GPU host | Runpod community cloud |
| CPU | _fill in from `lscpu` on provisioned instance_ |
| RAM | _fill in_ |
| CUDA version | _fill in from `nvcc --version`_ |
| Driver version | _fill in from `nvidia-smi`_ |

---

## Software versions

| Component | Version |
|---|---|
| vLLM | _fill in: `python -c "import vllm; print(vllm.__version__)"`_ |
| SGLang | _fill in_ |
| Python | _fill in_ |
| PyTorch | _fill in_ |
| transformers | _fill in_ |

---

## Models

| Model | Source | Quantization | Notes |
|---|---|---|---|
| Llama 3.1 8B Instruct | meta-llama/Llama-3.1-8B-Instruct | FP16 | Primary model |
| Llama 3.1 8B Instruct | neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 | FP8 | Pre-quantized |
| Llama 3.1 8B Instruct | hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | INT4-AWQ | Pre-quantized |
| Llama 3.1 8B Instruct | ModelCloud/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 | INT4-GPTQ | Pre-quantized |
| Llama 3.2 1B Instruct | meta-llama/Llama-3.2-1B-Instruct | FP16 | Draft model for specdec |
| Qwen 2.5 7B Instruct | Qwen/Qwen2.5-7B-Instruct | FP16 | Secondary model (Week 4+) |

---

## What counts as a "request"

A request is a single prompt-completion pair submitted to the server's `/v1/chat/completions` endpoint. Requests are issued concurrently at a fixed **request rate** (requests/sec, Poisson-distributed arrivals). The server processes them according to its scheduling policy (continuous batching for vLLM; RadixAttention for SGLang).

**Not counted:** server warmup period, failed requests, health-check pings.

---

## Metrics

| Metric | Definition | Unit |
|---|---|---|
| **Throughput** | Total output tokens / total wall-clock time | tokens/sec |
| **TTFT** | Time from request submission to first token received | ms |
| **ITL** | Time between consecutive output tokens, per request | ms |
| **p50 / p95 / p99 ITL** | Percentile of ITL distribution across all tokens across all requests | ms |
| **$/Mtoken** | (GPU $/hr × wall-clock hours) / (total output Mtokens) | $/Mtok |

**Why p99 ITL and not mean:** Mean is dominated by fast tokens; p99 reflects the worst-case user-visible stutter. FAANG infra teams care about tail latency.

---

## Workload distribution

### ShareGPT-style (primary)
- Dataset: `ShareGPT_V3_unfiltered_cleaned_split.json`
- Sampled prompt lengths: _fill in from `scripts/analyze_dataset.py` output_
- Approximate input length distribution: _fill in_
- Approximate output length distribution: _fill in_
- Number of prompts per run: **1000**

### Long-context (Week 4+)
- Dataset: LongBench subset — _fill in specific split_
- Input lengths: 2048–8192 tokens
- Output lengths: 256–512 tokens

---

## Benchmark configurations swept

| Dimension | Values |
|---|---|
| Quantization | FP16, FP8, INT4-AWQ, INT4-GPTQ |
| Batch size (max concurrent) | 8, 32 |
| Sequence length | 512in/256out, 2048in/256out |
| Engine | vLLM, SGLang |
| Request rate | 10 req/s (primary); saturating load |

---

## Exact benchmark commands

All raw CLI invocations are committed alongside their results in `results/<run_dir>/CMD.txt`. To reproduce any result:

```bash
cat results/<run_dir>/CMD.txt | bash
```

---

## Exclusions and known limitations

- Benchmarks are single-GPU only; multi-GPU tensor parallelism is out of scope.
- Results reflect community cloud GPU performance; variance between runs is ~5%. Three runs averaged where noted.
- MT-Bench evaluation requires an OpenAI API key for the GPT-4 judge. Scores are compared relatively (baseline vs. specdec), not as absolute ground truth.
- Model download time and server startup time are excluded from throughput calculations.
