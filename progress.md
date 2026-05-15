# Tutorial Progress Log

## Status summary
- **Last session date:** 2026-05-13
- **Current section:** §2 — WSL2 + Blackwell setup (engines not yet installed)
- **Next section planned:** §2 — install llama.cpp, Ollama, vLLM
- **Blockers:** none

## Completed sections
- [x] §1 Conceptual map
- [ ] §2 WSL2 + Blackwell setup  ← environment verified; engines not yet installed
- [ ] §3 Capacity planning
- [ ] §4 Quantization formats
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
- **WSL distro / version:** Ubuntu 24.04.4 LTS (noble)
- **NVIDIA driver (Windows host):** 591.86
- **CUDA version (inside WSL):** 13.1
- **Docker:** 29.4.0
- **Python version / venv location:** 3.12.3 / not yet created
- **llama.cpp commit / build flags:** not yet installed
- **Ollama version:** not yet installed
- **vLLM version / install method:** not yet installed
- **Other engines:** not yet installed

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

## Files / configs to remember
-