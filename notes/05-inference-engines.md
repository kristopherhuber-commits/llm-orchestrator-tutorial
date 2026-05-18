# §5 — Inference Engines Deep-Dive

**Goal:** Understand what an inference engine *is*, learn the three engines you have installed (llama.cpp, Ollama, vLLM) at a level deep enough to debug them, and end with measured throughput/memory numbers for the same model across all three.

**Running example model:** Qwen2.5-Coder 7B
- llama.cpp / Ollama: GGUF Q4_K_M (~4.7 GB)
- vLLM: AWQ INT4 (~5.0 GB)

**Hardware budget for this section:** ~30 GB free VRAM, ~890 GB free disk. Generous.

---

## 0. What an inference engine actually does

An inference engine is the program that takes:

1. A serialized model (weights + architecture metadata + tokenizer)
2. A request (prompt tokens, sampling parameters, optional tools/grammar)

and produces output tokens by running the forward pass repeatedly. That sounds trivial. It isn't. A modern engine is doing all of this:

| Concern | What the engine handles |
|---|---|
| **Memory layout** | Where weights live (VRAM vs RAM), tensor parallelism, expert sharding |
| **KV cache management** | Allocation strategy, eviction, reuse across requests |
| **Batching** | Combining requests into a single forward pass without head-of-line blocking |
| **Sampling** | Top-k/top-p/min-p/repetition penalties applied efficiently on-GPU |
| **Constrained decoding** | Mask logits to enforce a grammar (JSON schema, regex, GBNF) |
| **Speculative decoding** | Run a small draft model, verify with the target — §8 territory |
| **Transport** | OpenAI-compatible HTTP, native protobuf, gRPC, etc. |

The three "axes of differentiation" between engines:

1. **Throughput** — tokens/sec across N concurrent requests
2. **Latency** — time-to-first-token (TTFT) and inter-token latency (ITL) for a single request
3. **Memory efficiency** — how much VRAM is wasted to fragmentation, padding, or pessimistic allocation

llama.cpp optimizes for (2) and (3) at batch=1. vLLM optimizes for (1) at high concurrency. Ollama is llama.cpp with UX bolted on. SGLang is vLLM with smarter prefix sharing.

This is the framing the whole section hangs on.

---

## 1. llama.cpp

### 1.1 Architecture

llama.cpp is a C++ library (`libllama`) plus a family of CLI binaries:

| Binary | Purpose |
|---|---|
| `llama-cli` | Interactive chat / one-shot completion. Good for sanity-checking. |
| `llama-server` | OpenAI-compatible HTTP server. This is what harnesses connect to. |
| `llama-bench` | Microbenchmark: prompt-processing tok/s (prefill) and text-generation tok/s (decode). |
| `llama-quantize` | Convert one GGUF quant to another. |
| `llama-perplexity` | Measure perplexity on wiki/etc. — quant quality check. |

Single-process, single-GPU by default. CUDA backend is the relevant one for you (sm_120). It uses a custom kernel set: **MMQ** (mat-mul quantized) for quantized weights, with CUBLAS as fallback for FP16/BF16 paths.

### 1.2 The flags that actually matter

```
--model PATH                # path to .gguf file
--n-gpu-layers N            # offload first N layers to GPU; -1 = all
--ctx-size N                # max context window the server will allocate KV for
--batch-size N              # logical batch (default 2048) — affects prompt-processing batching
--ubatch-size N             # physical micro-batch sent to GPU (default 512)
--parallel N                # number of concurrent slots in llama-server
--cont-batching             # continuous batching across slots (default on in recent builds)
--cache-reuse N             # reuse KV cache between requests when prefix matches (chars)
--flash-attn on|off|auto    # FlashAttention kernel (faster, lower memory). Recent builds require a value.
--threads N                 # CPU threads (matters only for offloaded layers)
--port 8080                 # HTTP port for llama-server
--host 0.0.0.0              # bind interface (use 127.0.0.1 for local-only)
```

**On your 5090** for a 7B Q4_K_M model:

- `--n-gpu-layers -1` (full GPU offload — 7B at Q4 is ~4.7 GB, fits trivially)
- `--ctx-size 32768` or `--ctx-size 65536` (Qwen2.5-Coder supports up to 128K; pick what you'll actually use — KV cache scales linearly with this)
- `--flash-attn` (always on for sm_120 — your build supports it)
- `--parallel 4` (for benchmarking multi-request later)
- `--cache-reuse 256` (prefix-cache window in tokens — critical for agent loops)

### 1.3 The KV cache math, applied

Qwen2.5-Coder 7B has 28 layers, hidden_dim 3584, num_kv_heads 4, head_dim 128. KV cache per token at FP16:

```
KV bytes/token = 2 (K and V)
              × num_kv_heads × head_dim
              × num_layers
              × bytes_per_element
            = 2 × 4 × 128 × 28 × 2
            = 57,344 bytes ≈ 56 KiB / token
```

At 32K context: 56 KiB × 32,768 ≈ **1.75 GiB** per slot. With `--parallel 4`, llama.cpp pre-allocates 4 × 1.75 = **7 GiB** of KV cache alongside the 4.7 GiB of weights. Total ≈ 12 GiB. Fits comfortably; this is why your 5090 is the right card for this work.

(GQA — grouped-query attention — is doing the heavy lifting here. num_kv_heads=4 vs num_attention_heads=28 means 7× less KV memory than a dense-attention model of the same hidden_dim. Qwen2.5-Coder, DeepSeek, Llama 3+, and Mistral all use GQA. Older models without GQA blow up KV cache 4–8×.)

### 1.4 Hands-on

**Step 1 — Pull the model:**

```bash
source /home/huber/git/llm-orchestrator-tutorial/.venv/bin/activate    # the venv from §2 has `hf` already
mkdir -p /home/huber/git/llm-orchestrator-tutorial/models/gguf
cd /home/huber/git/llm-orchestrator-tutorial/models/gguf
hf download \
    Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
    qwen2.5-coder-7b-instruct-q4_k_m.gguf \
    --local-dir .
```

Notes:
- In `huggingface_hub` 1.15+, the CLI is `hf` — `huggingface-cli` prints a deprecation warning and exits.
- No HF token needed for Qwen GGUFs (public repo).
- The venv is only needed for the download; llama.cpp itself has no Python dependency.
- The repo's `.venv/bin/activate` auto-exports `HF_HOME=<repo>/.cache/huggingface` and `VLLM_USE_FLASHINFER_SAMPLER=0`. Sourcing the venv handles both.

**Step 2 — Microbenchmark with `llama-bench`:**

```bash
~/llama.cpp/build/bin/llama-bench \
    -m /home/huber/git/llm-orchestrator-tutorial/models/gguf/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
    -ngl 99 \
    -p 512 -n 128 \
    -fa 1
```

What this measures:
- `pp512` row: prompt-processing throughput at 512-token prompts (this is prefill — compute-bound)
- `tg128` row: text-generation throughput at 128-token decode (this is decode — memory-bandwidth-bound)
- Reports tok/s with confidence intervals.

**Record both numbers in `progress.md` under "Benchmark results".** These are your llama.cpp baselines.

**Step 3 — Run `llama-server`:**

```bash
~/llama.cpp/build/bin/llama-server \
    -m /home/huber/git/llm-orchestrator-tutorial/models/gguf/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
    -ngl 99 \
    --ctx-size 32768 \
    --parallel 4 \
    --cont-batching \
    --cache-reuse 256 \
    --flash-attn on \
    --host 127.0.0.1 --port 8080 \
    --alias qwen2.5-coder-7b
```

Test it with curl:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen2.5-coder-7b",
        "messages": [{"role": "user", "content": "Write a Python function that returns the nth Fibonacci number."}],
        "temperature": 0
    }' | jq -r '.choices[0].message.content'
```

The endpoint is OpenAI-compatible — harnesses (Aider, OpenCode, Continue) point at `http://127.0.0.1:8080/v1` and treat it as if it were OpenAI.

**Watch VRAM during this:** in a second terminal, `watch -n 1 nvidia-smi`. Note the steady-state VRAM with KV pre-allocated and the bump when you send a request.

---

## 2. Ollama

### 2.1 What it wraps and what it adds

Ollama is `llama.cpp` packaged as a daemon + registry + CLI. It runs as a systemd service (`ollama serve`), maintains a local model store at `~/.ollama/models`, and exposes both a native API (`/api/generate`) and an OpenAI-compatible one (`/v1/chat/completions`) on port 11434.

What Ollama *adds* over raw llama.cpp:

| Feature | Detail |
|---|---|
| **Registry** | `ollama pull qwen2.5-coder:7b` — no manual HF URL juggling |
| **Modelfile** | Lightweight `Dockerfile`-style spec to combine a base model with a system prompt, params, template overrides |
| **Model lifecycle** | Loads model on first request, evicts after `OLLAMA_KEEP_ALIVE` (default 5 min) |
| **GPU memory negotiation** | Auto-decides `n-gpu-layers` based on free VRAM — good enough usually, hides the knob |
| **Hot-swap** | Switches between models in a single endpoint by `model` field in the request |

What Ollama *hides* (and where this hurts):

| Hidden thing | When it bites |
|---|---|
| llama.cpp build flags | You can't toggle FlashAttention, change `--cache-reuse`, or tune `--ubatch-size`. Ollama picks defaults. |
| Exact quant variant | `qwen2.5-coder:7b` is a specific Q4_K_M — but the resolution is opaque. `ollama show <model> --modelfile` to dig. |
| Per-request concurrency | Controlled by `OLLAMA_NUM_PARALLEL` env var (default 1 in older versions, 4 in recent). Not exposed per-request. |
| KV cache type | FP16 by default. To get Q8 KV cache (smaller, slight quality cost), set `OLLAMA_KV_CACHE_TYPE=q8_0`. |

**The mental model:** Ollama is llama.cpp with sane defaults and friction removed, at the cost of tuning control. For exploratory work it's faster than llama.cpp. For squeezing every tok/s out of your 5090, it isn't.

### 2.2 Modelfile anatomy

```
FROM qwen2.5-coder:7b
PARAMETER num_ctx 32768
PARAMETER temperature 0
PARAMETER top_p 0.95
SYSTEM "You are a careful, terse coding assistant."
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
```

Save as `~/git/llm-orchestrator-tutorial/configs/ollama-qwen-coder.Modelfile`, then `ollama create qwen-coder-32k -f <path>`.

Most users never touch Modelfiles. You will, when you need to (a) raise context window, (b) fix a broken chat template (§9), or (c) standardize system prompts across harnesses.

### 2.3 Hands-on

**Step 1 — Pull the same model via Ollama:**

```bash
ollama pull qwen2.5-coder:7b
ollama list
ollama show qwen2.5-coder:7b
```

The third command shows you the quant (should be Q4_K_M), context length, template, and parameter count. Compare to the GGUF you pulled in §1.

**Step 2 — Time generation against Ollama's OpenAI endpoint:**

```bash
# Warm up (loads model into VRAM)
curl -s http://127.0.0.1:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "qwen2.5-coder:7b", "messages":[{"role":"user","content":"ping"}], "max_tokens": 8}' > /dev/null

# Timed run: 128 decode tokens
time curl -s http://127.0.0.1:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen2.5-coder:7b",
      "messages": [{"role":"user","content":"Write a Python function that returns the nth Fibonacci number with memoization. Just the code, no explanation."}],
      "temperature": 0,
      "max_tokens": 128
    }' | jq -r '.choices[0].message.content'
```

Note the wall-clock time. tok/s ≈ 128 / (wall_clock − TTFT). Cruder than `llama-bench` but it's what you'll measure end-to-end in harnesses.

**Step 3 — Check what Ollama actually loaded:**

```bash
ollama ps                                  # shows loaded models + GPU/CPU split + size
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

This is the moment of truth: if `ollama ps` shows `100% GPU`, Ollama auto-decided full offload. If it shows `xx% / yy% GPU/CPU`, it's spilling — usually because something else is holding VRAM. Your `--n-gpu-layers` equivalent.

**Stop the Ollama-loaded model before moving on to vLLM** (frees VRAM):

```bash
curl -s http://127.0.0.1:11434/api/generate -d '{"model":"qwen2.5-coder:7b","keep_alive":0}'
```

---

## 3. vLLM

### 3.1 Why vLLM exists at all

llama.cpp at batch=1 is excellent. llama.cpp at batch=N is okay. **vLLM at batch=N is a different category of system.** The reason is two ideas:

### 3.1.1 PagedAttention

Naïve KV-cache allocation reserves the full `max_seq_len × hidden` block per request. If your `--ctx-size` is 32K but the average request is 2K, you waste 15× the necessary memory. Multiply by `--parallel N` and the waste compounds.

PagedAttention treats the KV cache like an OS page table:
- VRAM is divided into fixed-size **blocks** (default 16 tokens × hidden dim per block).
- Each request holds a list of block IDs (the "page table").
- Blocks are allocated on demand as the sequence grows.
- When a request finishes, blocks return to a free pool.

Effect: ~96% memory efficiency vs ~30–60% for naïve allocation. This is the **why** behind vLLM's much higher concurrent throughput at the same VRAM footprint.

### 3.1.2 Continuous batching

Static batching waits for the slowest request in a batch before forming the next batch — head-of-line blocking. Continuous batching reschedules at every decode step: as soon as a request finishes, a queued one slots in. Combined with PagedAttention's free-list, this means you don't pay for unused KV slots.

llama.cpp has continuous batching too (`--cont-batching`), but it's grafted on; vLLM was designed around it.

### 3.2 Other things vLLM gives you

| Feature | What it does | When you'll care |
|---|---|---|
| **Prefix caching** | Hash-tree of common prompt prefixes; cache hit skips prefill | Agent loops with repeated system prompts — huge win |
| **`guided_json`** | Constrained decoding via XGrammar (was Outlines) — enforces JSON schema | §7 — getting reliable tool calls out of mid-tier models |
| **Speculative decoding** | Draft+target setup natively | §8 |
| **AWQ/GPTQ/FP8** | Native quant support, often faster than GGUF at the same bit count | This section |
| **Multi-LoRA** | Serve many fine-tunes on one base model | Future, not for you yet |

### 3.3 Critical caveats on your 5090

From your `progress.md` — you already learned these:

- `VLLM_USE_FLASHINFER_SAMPLER=0` is **required** on sm_120 with CUDA 12.8 — FlashInfer's sampler capability check fails. Newer CUDA (12.9+) reportedly fixes it; you're on 12.8/13.0 PyTorch.
- vLLM 0.21.0 with PyTorch 2.11.0+cu130 is what you installed.

Sane launch command for your card:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
/home/huber/git/llm-orchestrator-tutorial/.venv/bin/vllm serve \
    Qwen/Qwen2.5-Coder-7B-Instruct-AWQ \
    --port 8000 \
    --host 127.0.0.1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --quantization awq_marlin \
    --dtype float16
```

Flag-by-flag:

- `--max-model-len 32768` — same context window as llama.cpp for fair comparison
- `--gpu-memory-utilization 0.85` — vLLM pre-allocates this fraction of free VRAM as a KV pool. Default 0.9; lower if you want to leave headroom for another process.
- `--enable-prefix-caching` — turn on; agent workloads benefit massively
- `--quantization awq_marlin` — Marlin is the optimized INT4 kernel set; faster than the reference AWQ path on Ampere+
- `--dtype float16` — explicit; AWQ activations stay in FP16

### 3.4 Hands-on

**Step 1 — Pull the AWQ weights:**

```bash
mkdir -p /home/huber/git/llm-orchestrator-tutorial/models/hf
huggingface-cli download \
    Qwen/Qwen2.5-Coder-7B-Instruct-AWQ \
    --local-dir /home/huber/git/llm-orchestrator-tutorial/models/hf/Qwen2.5-Coder-7B-Instruct-AWQ
```

(About 5 GB, ~5 min download depending on your connection.)

**Step 2 — Serve:**

```bash
source /home/huber/git/llm-orchestrator-tutorial/.venv/bin/activate

VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve \
    /home/huber/git/llm-orchestrator-tutorial/models/hf/Qwen2.5-Coder-7B-Instruct-AWQ \
    --port 8000 --host 127.0.0.1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --quantization awq_marlin \
    --dtype float16 \
    --served-model-name qwen2.5-coder-7b-awq
```

Wait for `Application startup complete.` and `INFO: Uvicorn running on http://127.0.0.1:8000`.

**Step 3 — Single-request smoke test (compare to llama-server):**

```bash
time curl -s http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen2.5-coder-7b-awq",
      "messages":[{"role":"user","content":"Write a Python function that returns the nth Fibonacci number with memoization. Just the code."}],
      "temperature": 0,
      "max_tokens": 128
    }' | jq -r '.choices[0].message.content'
```

**Step 4 — Multi-request throughput (where vLLM shines):**

vLLM ships a benchmark client. Use it:

```bash
# In a third terminal, with vLLM still serving
/home/huber/git/llm-orchestrator-tutorial/.venv/bin/vllm bench serve \
    --model qwen2.5-coder-7b-awq \
    --host 127.0.0.1 --port 8000 \
    --dataset-name random \
    --num-prompts 64 \
    --random-input-len 512 \
    --random-output-len 128 \
    --max-concurrency 8
```

This will report:
- Request throughput (req/s)
- Output token throughput (tok/s, aggregate)
- TTFT mean / P50 / P99
- ITL mean / P50 / P99

**The interesting comparison:** record output-token throughput at `--max-concurrency 1` and at `--max-concurrency 8`. The ratio is approximately how much you gain from continuous batching + PagedAttention vs single-request serving. On a 5090 with a 7B, expect a 4–6× multiplier at concurrency 8.

**Step 5 — Try `guided_json` (preview for §7):**

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen2.5-coder-7b-awq",
      "messages":[{"role":"user","content":"List three programming languages."}],
      "max_tokens": 200,
      "guided_json": {
        "type":"object",
        "properties":{
          "languages":{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":3}
        },
        "required":["languages"]
      }
    }' | jq -r '.choices[0].message.content'
```

The response is *guaranteed* to be valid JSON matching the schema — logits for tokens that would violate the grammar are masked to −∞ at each decode step. This is structurally how local mid-tier models become reliable tool-callers. §7 goes deeper.

**Step 6 — Shut down vLLM:**

`Ctrl-C` in the serve terminal. Confirm VRAM frees: `nvidia-smi`.

---

## 4. SGLang (brief)

SGLang is the third major OSS engine, from the LMSYS group. Architecturally similar to vLLM (continuous batching, paged KV) with one extra trick:

### RadixAttention

Generalizes prefix caching. Where vLLM's prefix cache is a hash-keyed lookup of complete prompt prefixes, SGLang maintains a **radix tree** of all token sequences it has ever cached. Branching prompts (e.g., one system prompt, many user variations; tree-search agent loops; parallel completions) hit the cache at every shared prefix and *every shared branch from there*.

**When SGLang beats vLLM:**
- Agentic workflows with tree-of-thoughts / branching exploration
- Many concurrent users sharing a system prompt with diverging short tails
- RAG with a stable retrieved-context prefix

**When you wouldn't bother:**
- Single-user agent loops (vLLM's flat prefix cache is enough)
- One-shot completions
- Anything where your engine setup time matters more than steady-state throughput

**Maturity status (as of mid-2025):** SGLang Blackwell support exists but lags vLLM. Not worth installing for this tutorial. Revisit if/when an experiment specifically needs RadixAttention.

You should be able to *recognize* SGLang in a benchmark paper or harness config and know "this is vLLM-class plus tree-aware prefix caching." That's the depth this section requires.

---

## 5. Decision tree — which engine for which job

```
Is this a one-off "I want to chat with a model"?
└─ YES → Ollama. Done. Don't overthink it.

Is this an agent harness pointing at a single local model for solo dev work?
├─ You want zero config friction → Ollama
└─ You want to control KV cache reuse, FlashAttention, exact quant → llama.cpp (llama-server)

Are you benchmarking many requests concurrently (serving multiple harness instances,
running an eval harness with parallel tasks)?
└─ vLLM. Not even close.

Do you need guaranteed-valid JSON / regex-constrained output?
├─ vLLM has guided_json natively (XGrammar)
├─ llama.cpp has GBNF grammars (--grammar-file)
└─ Ollama exposes neither cleanly — fall back to llama.cpp or vLLM

Do you need branching prefix reuse (tree search, parallel sampling from one prefix)?
└─ SGLang — but only worth it if this is the dominant pattern of your workload

Do you need a model that only ships in GGUF (a community quant of a niche fine-tune)?
└─ llama.cpp / Ollama. vLLM's GGUF support exists but isn't its strong suit.

Do you need a model that only ships in safetensors with no GGUF (a brand-new release)?
└─ vLLM (AWQ/GPTQ if quantized; FP16 otherwise).
```

**For this tutorial's downstream sections:**

- §6–§9 (sampling, structured output, spec decoding, templates): use vLLM as the primary playground; it gives you the cleanest knobs.
- §10–§15 (harnesses): each harness mostly just needs an OpenAI endpoint; pick whichever engine is already serving the model you want.
- §20–§26 (benchmarks): vLLM. Concurrent eval execution wants its throughput.

---

## 6. What to record in `progress.md`

After running everything in this section, populate these tables.

**Models installed:**

```
| Qwen2.5-Coder-7B-Instruct | 7B | Q4_K_M GGUF | llama.cpp/Ollama | 32K | ~4.7 GB on disk |
| Qwen2.5-Coder-7B-Instruct | 7B | AWQ INT4    | vLLM             | 32K | ~5.0 GB on disk |
```

**Benchmark results** (one row per engine):

```
| <date> | tok/s pp512+tg128 | Qwen2.5-Coder-7B Q4_K_M | llama.cpp    | n/a | n/a | t=0 | pp=___, tg=___ | n/a | $0 | from llama-bench |
| <date> | tok/s decode      | Qwen2.5-Coder-7B Q4_K_M | Ollama       | n/a | n/a | t=0 | tg=___         | n/a | $0 | 128-tok curl     |
| <date> | tok/s decode @c=1 | Qwen2.5-Coder-7B AWQ    | vLLM         | n/a | n/a | t=0 | tg=___         | n/a | $0 | vllm bench       |
| <date> | tok/s decode @c=8 | Qwen2.5-Coder-7B AWQ    | vLLM         | n/a | n/a | t=0 | tg=___         | n/a | $0 | vllm bench       |
```

**Decisions made / lessons learned** — add whichever of these you actually observed:

- vLLM throughput at concurrency 8 vs 1 = ___× — quantifies continuous batching + PagedAttention gain
- Ollama auto-offload picked ___ GPU layers (vs llama.cpp explicit -ngl 99)
- Prefix cache hit reduced TTFT by ___% on the second identical request (vLLM)

---

## 7. What's next

§6 — Serving the candidate model set for coding. Now that you have one model running on all three engines, §6 expands to the full menu (Qwen2.5-Coder 14B/32B, DeepSeek-Coder-V2-Lite, Codestral, GLM-4 32B) with capacity-planning math from §3 applied per model.
