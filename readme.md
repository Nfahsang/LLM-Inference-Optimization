# LLM Inference Optimization

> Rigorous throughput, latency, and cost benchmarks for Llama 3.1 8B and Qwen 2.5 7B across vLLM and SGLang with FP16/INT8/INT4 quantization — plus a speculative decoding implementation — on a single A100 80GB.

🚧 **Results coming end of Week 4.** Follow progress in [LOG.md](LOG.md).

---

## Quick start

```bash
make setup         # install dependencies (run on GPU server)
make bench         # reproduce FP16 baseline
make bench-quant   # run full quantization sweep
make eval-mt-bench # run MT-Bench quality evaluation
```

## Results

_Populated end of Week 4. See [Write-up #1](WRITEUP_1.md) when published._

| Model | Engine | Quant | Batch | Throughput (tok/s) | p99 ITL (ms) | $/Mtoken |
|---|---|---|---|---|---|---|
| Llama 3.1 8B | vLLM | FP16 | 32 | — | — | — |
| Llama 3.1 8B | SGLang | FP16 | 32 | — | — | — |
| Llama 3.1 8B | vLLM | INT4-AWQ | 32 | — | — | — |

## Methodology

See [BENCHMARK_METHODOLOGY.md](BENCHMARK_METHODOLOGY.md) for full experimental design: hardware, metric definitions, workload distribution, and exact benchmark commands.

## Write-ups

- [Write-up #1](WRITEUP_1.md) — Baseline benchmark study: vLLM vs. SGLang, FP16 vs. INT8 vs. INT4 _(coming Week 4)_
- [Write-up #2](WRITEUP_2.md) — Speculative decoding: implementation, tuning, and upstream contribution _(coming Week 11)_

## Upstream PR

_Link added when merged._

## License

[MIT](LICENSE) — Nicholas Fah-Sang · [Fah-Sang.me](https://Fah-Sang.me) · [GitHub @Nfahsang](https://github.com/Nfahsang)
