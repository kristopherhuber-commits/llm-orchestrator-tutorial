# Tutorial Progress Log

## Status log [x] means section has been completed 
- [x] §1 Conceptual map
- [x] §2 WSL2 + Blackwell setup
- [x] §3 Capacity planning
- [x] §4 Quantization formats
- [x] §5 Inference engines
- [ ] §6 Candidate model set
- [ ] §7 Sampling and structured output
- [ ] §8 Speculative decoding and prefix caching
- [ ] §9 Tokenizers and chat templates
- [ ] §10 What makes a coding-agent harness
- [ ] §11 Edit formats
- [ ] §12 Harness-by-harness walkthrough
- [ ] §13 Connecting local models to harnesses
- [ ] §14 Connecting Vertex Gemini
- [ ] §15 Claude Code under Pro subscription
- [ ] §16 MCP demystified
- [ ] §17 Reviewing my existing MCP server
- [ ] §18 MCP across harnesses
- [ ] §19 MCP in benchmarking
- [ ] §20 Benchmark taxonomy and hygiene
- [ ] §21 Aider polyglot benchmark
- [ ] §22 SWE-bench Verified and Lite
- [ ] §23 Terminal-Bench
- [ ] §24 LiveCodeBench, BigCodeBench, MultiPL-E, RepoBench
- [ ] §25 General-agent benchmarks
- [ ] §26 Eval harnesses
- [ ] §27 Orchestrator-comparison experiment design
- [ ] §28 Reproducibility protocol
- [ ] §29 Observability
- [ ] §30 LLM-as-judge
- [ ] §31 Cost and latency modeling
- [ ] §32 Finding and vetting benchmark repos
- [ ] §33 Recommendation deliverable
- [ ] §34 License and commercial-use checklist
- [ ] §35 Maintenance

## Environment state
- **WSL distro / version:** Ubuntu 24.04.4 LTS / WSL 2.7.3.0
- **CUDA version (inside WSL):** 12.5 (default), 12.8 (for llama.cpp builds)
- **Python version / venv location:** 3.12.3 / `<repo>/.venv`
- **llama.cpp commit / build flags:** 769cc93a43b51bf6013986180c73ee60cf24cede / CUDA_ARCH=120, FORCE_CUBLAS=OFF, CUDAToolkit_ROOT=/usr/local/cuda-12.8
- **Ollama version:** 0.24.0
- **vLLM version / install method:** 0.21.0 / pip in `.venv`, PyTorch 2.11.0+cu130

## Models installed
| Model | Size | Quant | Engine | Context | Notes |
|-------|------|-------|--------|---------|-------|
| Qwen2.5-Coder-7B-Instruct | 7B | Q4_K_M GGUF | llama.cpp/Ollama | 32K | 4.7 GB file on disk |
| Qwen2.5-Coder-7B-Instruct | 7B | AWQ INT4 | vLLM | 16K/32K | 5.0 GB files in local directory |

## Benchmark results
| Date | Benchmark | Model | Engine | Harness | Edit format | Sampler | Score / Throughput | Wall-clock | Cost | Notes |
|------|-----------|-------|--------|---------|-------------|---------|-------|------------|------|-------|
| 2026-05-18 | llama-bench | Qwen2.5-Coder-7B-Instruct | llama.cpp | n/a | n/a | t=0 | pp512: 15158.14 t/s, tg128: 242.24 t/s | n/a | $0 | Full GPU offload (-ngl 99), FlashAttention enabled |
| 2026-05-19 | vllm bench | qwen2.5-coder-7b-awq | vLLM | n/a | n/a | default | Total: 1311.40 t/s, Gen: 262.28 t/s | 31.23s | $0 | AWQ quantization, PagedAttention enabled, Max concurrency 8, Throttled arrival (4 RPS) |
| 2026-05-19 | vllm bench | qwen2.5-coder-7b-awq | vLLM | n/a | n/a | default | Total: 5460.18 t/s, Gen: 1092.04 t/s | 7.50s | $0 | AWQ, Chunked Prefill Enabled, Max concurrency 32, Max sequences 32, Throttled arrival (32 RPS) |

## Decisions made / lessons learned
- OODA loop maps directly onto agentic loop
- vLLM FlashInfer sampler fails capability check on sm_120 without CUDA 12.9; workaround: `VLLM_USE_FLASHINFER_SAMPLER=0`
- vLLM requires explicit split of `--model` string alias and `--tokenizer` local path flags when running synthetic file benchmarks against local weights.
- Client bench tools target legacy `/v1/completions` for random syntax strings; passing `/v1/chat/completions` throws a `400 Bad Request` unless structural array payloads are passed.
- Chunked prefill decouples large prompt ingestion spikes inside WSL virtual memory tables, expanding parallel gen throughput up to 1092.04 tok/s.