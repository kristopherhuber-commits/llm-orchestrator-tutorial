# Tutorial Progress Log

## Status log [x] means setion has been completed 
- [x] §1 Conceptual map
- [x] §2 WSL2 + Blackwell setup
- [x] §3 Capacity planning
- [x] §4 Quantization formats
- [ ] §5 Inference engines
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
- **Python version / venv location:** 3.12.3 / `<repo>/.venv` (moved from ~/.venvs/vllm in §5; recreated from frozen requirements at `configs/vllm-requirements.frozen.txt`)
- **llama.cpp commit / build flags:** 769cc93a43b51bf6013986180c73ee60cf24cede / CUDA_ARCH=120, FORCE_CUBLAS=OFF, CUDAToolkit_ROOT=/usr/local/cuda-12.8
- **Ollama version:** 0.24.0
- **vLLM version / install method:** 0.21.0 / pip in ~/.venvs/vllm, PyTorch 2.11.0+cu130

## Models installed
| Model | Size | Quant | Engine | Context | Notes |
|-------|------|-------|--------|---------|-------|
|       |      |       |        |         |       |

## Harnesses installed
| Harness | Version | Endpoint configured | Notes |
|---------|---------|---------------------|-------|
|         |         |                     |       |

## Benchmark results
| Date | Benchmark | Model | Engine | Harness | Edit format | Sampler | Score | Wall-clock | Cost | Notes |
|------|-----------|-------|--------|---------|-------------|---------|-------|------------|------|-------|
|      |           |       |        |         |             |         |       |            |      |       |

## Open questions / parked items
- [ ]

## Decisions made / lessons learned
- OODA loop maps directly onto agentic loop; orient+decide collapse into single LLM forward pass
- ADK (Google Agent Development Kit) is a harness-construction framework, not a harness; not needed for this tutorial
- TUI = terminal UI; OpenCode uses Bubble Tea-based TUI with vim keybindings
- User preference: vim as editor, vim keybindings in bash shell
- CUDA 13.1 reported by nvidia-smi is driver max, not installed toolkit (had 12.5)
- llama.cpp must build against CUDA 12.8 explicitly; CUDA 13.x crashes MMQ kernel on sm_120
- vLLM FlashInfer sampler fails capability check on sm_120 without CUDA 12.9; workaround: VLLM_USE_FLASHINFER_SAMPLER=0
- WSL 2.7.0+ required for loopback TCP to work reliably with mirrored networking mode
- vLLM startup alias: VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve <model> ...
- Measured: 1.5B Q4_K_M = 2830 MiB net VRAM (840 MB weights + KV pre-alloc + runtime)
- Idle VRAM baseline on this system: ~1929 MiB
- Formula M_weights = N_params × b_param validated against hardware

## Files / configs to remember
- `~/llama.cpp/build/bin/llama-cli` — llama.cpp CLI binary
- `~/llama.cpp/build/bin/llama-server` — llama.cpp OpenAI-compatible server
- `/etc/systemd/system/ollama.service.d/boot-delay.conf` — 45s boot delay for Ollama on Blackwell
- `<repo>/.venv` — vLLM Python venv (relocated in §5; activate with `source .venv/bin/activate`)
- `<repo>/configs/vllm-requirements.frozen.txt` — pinned vLLM 0.21.0 + torch 2.11.0+cu130 + flashinfer 0.6.8 install manifest (Blackwell-verified)
- `<repo>/models/gguf/` — GGUF model weights for llama.cpp / Ollama (gitignored)
- `<repo>/models/hf/` — HuggingFace-format model weights (safetensors / AWQ / GPTQ) for vLLM (gitignored)