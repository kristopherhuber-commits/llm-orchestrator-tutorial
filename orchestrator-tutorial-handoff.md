# Agentic LLM Evaluation Tutorial — Context & Handoff Document

**Document version:** 1.0
**Created:** 2026-05-09
**Purpose:** Persistent context for a multi-week tutorial on evaluating agentic coding LLMs (local + hosted), benchmarking them, and producing recommendations.

---

## How to use this document

### For the user (you)

This file is your **anchor across conversations**. A 35-section technical tutorial cannot fit in a single Claude conversation — context windows are finite and degrade well before the conversation forces a restart. Use this file as follows:

1. **Keep it in version control** (a private GitHub repo, a local git folder, or even just a OneDrive/Dropbox folder with backup). Treat it as a living document.
2. **At the start of every new Claude conversation**, paste this file (or attach it). Then state which section you want to work on next, and paste the **Progress Log** (template at the end of this document) so the new conversation knows what's already done.
3. **Update the Progress Log after every working session.** Record: section number worked on, what you installed/configured, what worked, what broke, what config files you ended up with, what numbers you got, what questions you parked. This is the single most valuable habit for a multi-week project.
4. **Start a fresh conversation per major part (Parts 0–5), or whenever the current one feels sluggish.** Don't try to run all 35 sections in one chat. A good rule of thumb: if a conversation has more than ~20 substantial back-and-forths or you've pasted long logs/code multiple times, start a new one.
5. **Save artifacts locally.** Any config file, script, Modelfile, Docker compose, or benchmark result Claude generates — save it to a local repo. Don't rely on conversation scrollback.

### For the next LLM picking this up

If you are an LLM reading this because the user pasted it into a new conversation: this document is the authoritative context for a tutorial project the user is working through with a prior Claude conversation. The user's hardware, goals, constraints, prior decisions, and the agreed-upon tutorial structure are all below. The user will tell you which section they want help with next. Respect the prior decisions documented here unless the user explicitly revisits them. Do not regenerate the outline — work within it.

---

## 1. User profile

- **Education:** Graduate-level physics and math.
- **Programming:** Python; uses VS Code.
- **Operating environment:** Windows 11 host, all tutorial work performed inside WSL2.
- **Stated communication preferences:** Treat the user as a domain expert in the subject they're asking about. Be detailed and precise. Be helpful but not obsequious. Provide mathematical equations or proofs when germane. Do not guess — ask for clarification when uncertain.

## 2. Hardware

- **GPU:** NVIDIA RTX 5090 (Blackwell, sm_120, 32 GB GDDR7).
- **CPU:** AMD Ryzen 9 9950X.
- **System RAM:** 96 GB.
- **Motherboard:** Gigabyte X870E AORUS Master (AM5).
- **OS:** Windows 11 with WSL2 for all tutorial work.

**Implications for the tutorial:**
- The "interesting" model band is 14B–70B, not the 7–8B tier most home-lab tutorials default to.
- Comfortable territory: Qwen 2.5-Coder 32B at Q5/Q6 fully on GPU; Mixtral 8x7B; Qwen3 30B-A3B MoE; DeepSeek-V2-Lite. 70B-class at Q4_K_M with partial CPU offload is feasible.
- **Blackwell toolchain caveat:** As of the cutoff, full sm_120 performance requires CUDA 12.8+, recent llama.cpp builds, vLLM with PyTorch 2.5+/2.6 nightly, and recent Ollama. Older binaries silently fall back to slow paths or fail to load. The tutorial includes a dedicated setup-and-verification section.

## 3. Goals

The user wants to be able to:

1. **Download and run agentic coding harnesses** in WSL: Claude Code, Gemini CLI, Aider, OpenCode, and similar.
2. **Run and benchmark local LLMs** served via Ollama, llama.cpp, vLLM, or SGLang.
3. **Understand the inference-engine vs model-family distinction** (the "Ollama vs Llama" framing is a category error — see §5 below).
4. **Recommend (model + harness + edit-format + sampler) tuples** to coworkers based on task profile and constraints, backed by reproducible benchmarks.
5. **Understand and apply current coding-agent benchmarks** (Aider polyglot, SWE-bench Verified/Lite, Terminal-Bench, LiveCodeBench, etc.) including their hygiene issues (contamination, reproducibility, pass@k semantics, judge bias).
6. **Find, vet, download, and run community benchmark repos** locally and reproduce or audit their results.
7. **Understand MCP** (Model Context Protocol) properly, given the user has already built a Python MCP server for signal-processing/filtering using Claude Code but doesn't yet have a clear conceptual model of MCP beyond "it's just a function."

## 4. Cloud / API constraints

- **Google Vertex AI:** API key available. Programmatic access to Gemini models is fine. **Use this for all hosted-LLM benchmarking.**
- **Anthropic:** Pro subscription on both Google AI and Anthropic. Claude Code CLI works via subscription auth. **No Anthropic API key.** Therefore Claude Code can be evaluated qualitatively or on small task sets (manual scoring), but not at SWE-bench-scale apples-to-apples without an API key. The tutorial structures the eval section around this asymmetry honestly.
- **No air-gap requirement** stated; coworkers' deployment constraints are unspecified, so the recommendation framework will surface license/commercial-use considerations explicitly (see §35).

## 5. Conceptual decisions already made

These were debated and resolved in the prior conversation. A future LLM should not relitigate them unless the user explicitly asks.

### 5.1 "Ollama vs Llama" is a category error
- **Llama** = a *model family* (Meta's Llama 3, 3.1, 3.2, 3.3, 4).
- **Ollama** = an *inference runtime* that wraps `llama.cpp` and adds a registry, server, and CLI.
- The two real axes:
  - **Model family:** Llama, Qwen, Mistral/Mixtral, Gemma, DeepSeek, GLM, Phi, Granite.
  - **Inference engine:** llama.cpp (and wrappers Ollama, LM Studio, GPT4All), vLLM, SGLang, TGI, ExLlamaV2, MLX (Apple), TensorRT-LLM (NVIDIA prod).
- Tradeoff axes: throughput, latency, memory efficiency (paged attention, prefix caching), quantization formats supported, OpenAI-compatible API surface, concurrency.

### 5.2 "Harness" has two meanings — tutorial disambiguates them
- **Agent harness:** the agentic scaffolding (Aider, OpenCode, Claude Code, Continue, Cline, Goose).
- **Eval harness:** the framework that runs the model against benchmark tasks (`inspect_ai`, `lm-evaluation-harness`, OpenAI Evals, SWE-bench's harness).

### 5.3 MCP framing
The user's intuition that "an MCP server is just a function the LLM can call" is essentially correct at the protocol level. MCP exists as a separate concept from per-vendor function calling because it standardizes:
1. **Transport and lifecycle** — JSON-RPC over stdio or HTTP/SSE, with capability negotiation. The same server works across Claude Code, Gemini CLI, Cursor, Continue, OpenCode, etc., without reimplementation.
2. **Three primitives, not just one** — Tools (callable functions), Resources (readable items: files, DB rows), and Prompts (templates).
3. **Sampling** — A server can ask the host to run an LLM call on its behalf, inverting the usual control flow.

The tutorial covers the protocol layer explicitly, then revisits the user's existing filtering MCP server in that light.

### 5.4 Tool-call reliability is the bottleneck for local agents
A model that scores well on MMLU can still be unusable as an agent if it hallucinates tool names or breaks JSON mid-call. This shows up especially at low-bit quants. Constrained decoding (GBNF grammars, Outlines, XGrammar, vLLM `guided_json`) is a first-class topic, not a footnote.

### 5.5 KV cache dominates VRAM in agentic loops
Agent loops accumulate tool outputs fast — coding agents can pass 32K tokens within a few file reads. KV-cache memory ≈ 2 × layers × hidden_dim × context × bytes-per-element × batch. On a 32 GB card with a 32B model, KV cache often consumes more VRAM than weights. Capacity planning math is in Section 3 of the tutorial.

### 5.6 Benchmark hygiene is non-negotiable
- **Contamination:** many open models have seen public benchmarks in training. SWE-bench Verified exists partly to mitigate this.
- **Reproducibility:** temperature, seed, sampler settings, exact model checkpoint hash, exact harness version. Numbers from blog posts often don't reproduce.
- **Pass@k semantics:** some leaderboards report pass@1 with t=0; others sample.
- **Sandboxing:** SWE-bench-style benchmarks require Docker per task — disk and compute budgeting matters.
- **GPU non-determinism:** even at temperature 0, runs aren't bit-identical. Report median + IQR across N runs.

### 5.7 Aider polyglot is the highest-ROI single benchmark for the user's goals
It's purpose-built for comparing model+harness combos for code editing, supports edit-format ablations (whole-file, search-replace, unified diff), and is cheap to run. Gets its own tutorial section.

### 5.8 Cost / latency realism
Agent loops issue 50–500 LLM calls per task. Local 70B at Q4 on a 5090 is roughly tens of tok/s; frontier hosted is hundreds. End-to-end task wallclock on local can be 10–100× hosted. Recommendations must be on a Pareto frontier (quality × cost × latency), not a single ranking.

### 5.9 Claude Code asymmetry
Without an Anthropic API key, Claude Code is evaluated *qualitatively* on a small task set. Apples-to-apples large-scale runs use Vertex Gemini + local models.

---

## 6. Tutorial structure — 35 sections across 6 parts

The user agreed to **staged section-by-section delivery**, not a single upfront 200-page document. Reasons: tooling moves fast (especially Blackwell support); each section is best generated close to when the user will execute it; and it forces a "do, then proceed" cadence.

### Part 0 — Foundations and environment

1. **Conceptual map of the local-LLM stack.** Model family vs inference engine vs agent harness vs eval harness. Where Ollama, llama.cpp, vLLM, SGLang, ExLlamaV2, TGI sit. Where Aider, Claude Code, Gemini CLI, OpenCode, Continue, Cline, Goose sit. Why "Ollama vs Llama" is a category error.

2. **WSL2 + Blackwell (5090) setup.** Windows 11 NVIDIA driver requirements, WSL2 CUDA toolkit (12.8+) install, verifying `nvidia-smi` inside WSL, Docker Desktop with WSL2 backend, disk quota and `.wslconfig` tuning for 96 GB RAM, building/installing recent llama.cpp, vLLM, and Ollama with sm_120 support.

3. **Hardware-aware capacity planning.** VRAM math: weights memory = params × bytes-per-param; KV cache memory = 2 × layers × hidden × context × bytes × batch. Worked examples for 14B, 32B, 70B at various quants and contexts on 32 GB VRAM. When and how to offload to CPU RAM.

### Part 1 — Local model serving

4. **Quantization formats in practice.** GGUF (Q4_K_M, Q5_K_M, Q6_K, Q8_0, IQ-quants, K-quants), AWQ, GPTQ, EXL2, FP8. Quality vs size vs speed tradeoffs. Why agentic / tool-use workloads degrade faster than chat at low bits.

5. **Inference engines deep-dive.**
   - llama.cpp directly (CLI + `llama-server`), build flags, `--n-gpu-layers`, `--cache-reuse`.
   - Ollama as a llama.cpp wrapper: model registry, Modelfile, OpenAI-compatible endpoint, what it adds and what it hides.
   - vLLM: paged attention, continuous batching, prefix caching, `guided_json`, when it beats llama.cpp on a 5090.
   - SGLang briefly: RadixAttention, when to prefer it.
   - Practical decision tree: "use X if Y."

6. **Serving the candidate model set for coding.** Coverage of Qwen2.5-Coder 7B/14B/32B, Qwen3 family, DeepSeek-Coder-V2-Lite, Codestral, GLM-4 32B, Llama 3.1/3.3, Gemma 2/3. License notes for each. Which fit the 5090 and at what quant/context.

7. **Sampling and structured output.** Temperature, top-p, min-p, DRY/XTC. Constrained decoding via GBNF, Outlines, XGrammar, vLLM `guided_json`. How to make a mid-tier local model reliably emit tool calls.

8. **Speculative decoding and prefix caching.** Setting up a draft+target pair on llama.cpp and on vLLM. Measuring tokens/sec with and without. Prefix caching configuration for agent loops.

9. **Tokenizers, chat templates, and tool-call templates.** Inspecting Jinja templates inside GGUF, fixing broken templates, model-specific tool-call special tokens (Llama 3.1 `<|python_tag|>`, Qwen `<tool_call>`, Hermes). Debugging "the model knows the tool but never calls it" failures.

### Part 2 — Agent harnesses

10. **What makes a coding-agent harness.** The component list: planner / loop controller, tool set (read/edit/run/search/test), edit format (whole-file, search-replace, unified diff, AST patch), context management strategy (repo map, RAG, file-window), retry/repair policy, evaluator (test-runner integration). How these components interact.

11. **Edit formats and why they matter.** Aider's whole/diff/udiff comparison results. Why a model that fails on udiff often succeeds on search-replace. How to choose edit format per model.

12. **Harness-by-harness walkthrough.** Install, configure, run a canned task, and inspect the system prompts and tool schemas of: Aider, Claude Code, Gemini CLI, OpenCode, Continue (CLI mode), Cline (where it overlaps), Goose. Note which support local OpenAI-compatible endpoints out of the box and which need workarounds.

13. **Connecting local models to harnesses.** Pointing each harness at a local OpenAI-compatible endpoint (llama.cpp / Ollama / vLLM). Pitfalls: tool-call format mismatches, context length truncation, missing `tools` field support.

14. **Connecting Vertex Gemini to harnesses.** Auth via service account or `gcloud`, model naming on Vertex vs AI Studio, cost and rate-limit considerations, Aider/OpenCode Vertex configuration.

15. **Claude Code under a Pro subscription.** What you can and cannot benchmark without an API key. Practical strategy: small qualitative task set, manual scoring, vs. the apples-to-apples Vertex+local matrix.

### Part 3 — MCP

16. **MCP demystified.** Protocol: JSON-RPC over stdio/SSE/HTTP. Three primitives: tools, resources, prompts. Plus sampling (server-initiated LLM calls). Why this exists separately from per-vendor function calling.

17. **Reviewing your existing Python MCP server.** Mapping the user's filtering server onto the protocol's tool primitive. Adding resource and prompt primitives if they fit. Logging and error semantics.

18. **MCP across harnesses.** Configuring the same server in Claude Code, Gemini CLI, OpenCode, Continue. What breaks across harnesses (auth, transport, schema dialects).

19. **MCP in benchmarking.** Whether to expose evaluation harnesses *as* MCP servers (sometimes useful), and how MCP-tool-availability changes a benchmark's meaning.

### Part 4 — Benchmarks for coding agents

20. **Benchmark taxonomy and hygiene.** Pass@k semantics, contamination, reproducibility, sandboxing requirements, prompt-format sensitivity, judge-model bias, leaderboard cherry-picking. The "report variance, not just mean" rule.

21. **Aider polyglot benchmark.** What it tests, how to run it, edit-format ablations, cost estimation. Why this is the highest-ROI single benchmark for the user's goals.

22. **SWE-bench Verified and Lite.** Task structure, Docker per-instance sandbox, harness setup (`SWE-bench` repo, `sb-cli`), partial-run strategies, disk and compute budget. What "resolved" actually means.

23. **Terminal-Bench.** Shell-environment agentic tasks; how it complements SWE-bench.

24. **LiveCodeBench, BigCodeBench, MultiPL-E, RepoBench.** Where each is informative and where it's saturated or noisy. Why HumanEval/MBPP no longer carry signal.

25. **General-agent benchmarks worth knowing.** GAIA, τ-bench, OSWorld — short overview for literature literacy.

26. **Eval harnesses (the other meaning of harness).** `inspect_ai` (preferred for agentic evals), `lm-evaluation-harness`, OpenAI Evals. How to write a custom eval in `inspect_ai` for a task you care about.

### Part 5 — Running, comparing, and recommending

27. **Designing your own orchestrator-comparison experiment.** Defining "orchestrator" = (model, inference engine, harness, edit format, sampler, tools). The full factorial is too large; how to pick a sensible fractional design. Picking primary and secondary metrics (resolved rate, cost per resolved, wallclock, tokens, tool-call validity).

28. **Reproducibility protocol.** Pinning model checkpoint hashes, harness commits, container digests, sampler params, seeds. Acknowledging GPU non-determinism. Reporting median + IQR across N runs.

29. **Observability.** Adding Langfuse / Phoenix / OpenLLMetry tracing to agent runs. Extracting per-step token counts, tool-call sequences, retries, failure modes. Turning failed traces into diagnoses.

30. **LLM-as-judge — when and how.** Using a stronger model (e.g., Gemini 2.5 Pro) to grade open-ended outputs. Bias toward verbose answers, self-preference bias, prompt-injection-via-output. Mitigations: rubric prompts, pairwise comparison, multiple judges.

31. **Cost and latency modeling.** Per-task cost on Vertex; per-task wallclock and energy on local. Building a Pareto frontier (quality vs cost vs latency) instead of a single ranking. How to present this to coworkers.

32. **Finding and downloading community benchmark repos.** GitHub discovery patterns (Papers-with-Code, leaderboard repos, awesome-lists). Vetting a benchmark repo: license, last-commit recency, reproducibility scripts, Docker images, dataset provenance, contamination disclosures. Forking, pinning, running, and contributing fixes back.

33. **Producing a recommendation deliverable for coworkers.** Decision matrix template: task profile (greenfield vs refactor vs debugging vs polyglot) × constraint profile (air-gapped vs hosted-OK, latency-sensitive vs throughput-OK, cost ceiling) → recommended (model, harness, edit format) tuple. With evidence links to benchmark runs.

34. **License and commercial-use checklist.** Per-model summary (Apache 2.0, Llama community license, Gemma terms, DeepSeek terms, Qwen Apache 2.0, etc.) and what coworkers can actually deploy.

35. **Maintenance: keeping the matrix fresh.** New model cadence, automating re-benchmarks on new releases, when to retire benchmarks that have been gamed or contaminated.

**Load-bearing sections for the stated goals:** 4, 10, 11, 16, 21, 22, 27, 32. The rest is scaffolding.

---

## 7. Working agreements

1. **Staged delivery.** One section (or a small group of related sections) per Claude conversation. Don't try to fit everything in one chat.
2. **Each section is do-then-proceed.** Complete the practical work for a section before requesting the next. This is how technical tutorials should be consumed.
3. **The user maintains a local progress log** (template below). Pasted into new conversations along with this document.
4. **Decisions in §5 are settled** unless the user explicitly revisits them. A future LLM should not re-derive Ollama-vs-Llama from scratch every time.
5. **Tutorial sections may be reordered** if circumstances warrant — e.g., if benchmark hygiene questions come up while installing a harness, jump to Section 20 briefly. The numbering is a default order, not a hard sequence.
6. **No guessing on commands or library APIs.** If a future LLM is uncertain about, e.g., the current vLLM flag for prefix caching on Blackwell, it should say so and either web-search or ask.

---

## 8. Progress log template

Copy this into a separate file (`progress.md` in your tutorial repo) and update after every working session. Paste it into new Claude conversations alongside this handoff document.

```markdown
# Tutorial Progress Log

## Status summary
- **Last session date:** YYYY-MM-DD
- **Current section:** §X — [name]
- **Next section planned:** §Y — [name]
- **Blockers:** [none / brief description]

## Completed sections
- [ ] §1 Conceptual map
- [ ] §2 WSL2 + Blackwell setup
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
- **WSL distro / version:**
- **NVIDIA driver (Windows host):**
- **CUDA version (inside WSL):**
- **Docker (inside WSL or Docker Desktop WSL backend):**
- **Python version / venv location:**
- **llama.cpp commit / build flags:**
- **Ollama version:**
- **vLLM version / install method:**
- **Other engines (SGLang, ExLlamaV2, etc.):**

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
-

## Files / configs to remember
- `path/to/config.yaml` —
- `path/to/Modelfile` —
- `path/to/inspect_eval.py` —
```

---

## 9. Style notes for any LLM continuing this tutorial

- The user has a graduate-level physics/math background. Equations and mathematical reasoning are welcome and often expected. Don't dumb things down.
- Be direct. Avoid filler, restating the question, and excessive hedging.
- When the user asks a yes/no question that admits a real answer, give the answer first, then the reasoning.
- If the user's question contains a misconception, correct it explicitly. The user has stated they prefer this.
- When uncertain about current tooling versions, library APIs, or fast-moving facts, web-search rather than guess. The user's prior conversation explicitly relied on this.
- Honor the user's preference: detailed, precise, helpful but not obsequious.

---

*End of handoff document. The next conversation should begin with: "I've read the handoff. Ready to work on Section X." or similar.*
