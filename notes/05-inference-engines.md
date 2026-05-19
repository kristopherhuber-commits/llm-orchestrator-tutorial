```markdown
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

and produces output tokens by running the forward pass repeatedly. A modern engine handles memory layout, KV cache management, continuous batching, sampling kernels, constrained decoding, and transport.

The three "axes of differentiation" between engines:
1. **Throughput** — tokens/sec across N concurrent requests
2. **Latency** — time-to-first-token (TTFT) and inter-token latency (ITL)
3. **Memory efficiency** — how much VRAM is wasted to fragmentation, padding, or pessimistic allocation

`llama.cpp` optimizes for latency and memory efficiency at batch=1. `vLLM` optimizes for throughput at high concurrency. Ollama packages `llama.cpp` for runtime convenience. SGLang is vLLM with smarter prefix sharing.

This is the framing the whole section hangs on.

---

## 1. llama.cpp

### 1.1 Architecture

`llama.cpp` is a native C++ library (`libllama`) plus a family of CLI binaries:

| Binary | Purpose |
|---|---|
| `llama-cli` | Interactive chat / one-shot completion. Good for sanity-checking. |
| `llama-server` | OpenAI-compatible HTTP server. This is what harnesses connect to. |
| `llama-bench` | Microbenchmark: prompt-processing tok/s (prefill) and text-generation tok/s (decode). |
| `llama-quantize` | Convert one GGUF quant to another. |
| `llama-perplexity` | Measure perplexity on wiki/etc. — quant quality check. |

Single-process, single-GPU by default. The CUDA backend targets your Blackwell (`sm_120`) architecture natively, executing specialized mat-mul quantized (MMQ) compute kernels.

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
- `--n-gpu-layers -1` (forces full GPU offload)
- `--ctx-size 32768` (allocates space for 32K context)
- `--flash-attn on` (explicitly enables the FlashAttention mathematical kernel optimization)
- `--parallel 4` (configures concurrent execution slots)
- `--cache-reuse 256` (reuses prefix-cache window to accelerate agent loops)

### 1.3 Hands-on

**Step 1 — Pull the model:**

```bash
source /home/huber/git/llm-orchestrator-tutorial/.venv/bin/activate
mkdir -p /home/huber/git/llm-orchestrator-tutorial/models/gguf
cd /home/huber/git/llm-orchestrator-tutorial/models/gguf
hf download \
    Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
    qwen2.5-coder-7b-instruct-q4_k_m.gguf \
    --local-dir .

```

**Step 2 — Microbenchmark with `llama-bench`:**

```bash
~/llama.cpp/build/bin/llama-bench \
    -m /home/huber/git/llm-orchestrator-tutorial/models/gguf/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
    -ngl 99 \
    -p 512 -n 128 \
    -fa 1

```

* `-ngl 99` forces the engine to offload all layers to the GPU.
* `-p 512` sets the prompt prefill size to establish a compute-bound processing score.
* `-n 128` runs the serial token generation loop 128 times consecutively to get a clean, stable decode benchmark.

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
curl [http://127.0.0.1:8080/v1/chat/completions](http://127.0.0.1:8080/v1/chat/completions) \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen2.5-coder-7b",
        "messages": [{"role": "user", "content": "Write a Python function that returns the nth Fibonacci number."}],
        "temperature": 0
    }' | jq -r '.choices[0].message.content'

```

---

## 2. Ollama

### 2.1 What it wraps and what it adds

Ollama runs as a background service daemon (`ollama serve`) written in Go, wrapping `llama.cpp` source binaries natively via a CGO link layer. It auto-negotiates system VRAM allocation and exposes both a native and an OpenAI-compatible API layer on port `11434`.

Ollama injects structured agent scaffolding (like tool-calling prompt routing templates) out of the box, translating a standardized JSON payload into the exact prompt formatting the underlying LLM expects.

### 2.2 Hands-on

**Step 1 — Pull and Inspect:**

```bash
ollama pull qwen2.5-coder:7b
ollama show qwen2.5-coder:7b --modelfile

```

**Step 2 — Execute Timed Performance Test:**

```bash
time curl -s [http://127.0.0.1:11434/v1/chat/completions](http://127.0.0.1:11434/v1/chat/completions) \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen2.5-coder:7b",
      "messages": [{"role":"user","content":"Write a Python function that returns the nth Fibonacci number with memoization. Just the code, no explanation."}],
      "temperature": 0,
      "max_tokens": 128
    }' | jq

```

*Note: The initial run includes a cold start while Ollama streams the model binary into VRAM. Subsequent runs drop initialization latency.*

**Step 3 — Inspect Runtime State:**

```bash
ollama ps

```

**Step 4 — Unload Model to Free VRAM:**

```bash
curl -s [http://127.0.0.1:11434/api/generate](http://127.0.0.1:11434/api/generate) -d '{"model":"qwen2.5-coder:7b","keep_alive":0}'

```

---

## 3. vLLM

### 3.1 PagedAttention & High-Concurrency

vLLM eliminates VRAM fragmentation waste by dividing the Key-Value (KV) cache into small, dynamically allocated 16-token memory pages, maintaining an operational lookup page table. Combined with continuous batching, it drives efficient multi-tenant concurrency.

### 3.2 Hands-on

**Step 1 — Download AWQ Weights:**

```bash
source /home/huber/git/llm-orchestrator-tutorial/.venv/bin/activate
mkdir -p /home/huber/git/llm-orchestrator-tutorial/models/hf
huggingface-cli download \
    Qwen/Qwen2.5-Coder-7B-Instruct-AWQ \
    --local-dir /home/huber/git/llm-orchestrator-tutorial/models/hf/Qwen2.5-Coder-7B-Instruct-AWQ

```

**Step 2 — Serve with Chunked Prefill and Safe WSL Memory Limits:**

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve \
    /home/huber/git/llm-orchestrator-tutorial/models/hf/Qwen2.5-Coder-7B-Instruct-AWQ \
    --port 8000 \
    --host 127.0.0.1 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.80 \
    --max-num-seqs 32 \
    --enable-chunked-prefill \
    --quantization awq_marlin \
    --dtype float16 \
    --served-model-name qwen2.5-coder-7b-awq

```

*Note: Passing `--enable-chunked-prefill` slices up large prompt ingestion blocks, preventing virtual page allocation deadlocks inside WSL.*

**Step 3 — Run High-Throughput Client Benchmark:**

```bash
source /home/huber/git/llm-orchestrator-tutorial/.venv/bin/activate
vllm bench serve \
    --model qwen2.5-coder-7b-awq \
    --tokenizer /home/huber/git/llm-orchestrator-tutorial/models/hf/Qwen2.5-Coder-7B-Instruct-AWQ \
    --host 127.0.0.1 --port 8000 \
    --endpoint /v1/completions \
    --dataset-name random \
    --num-prompts 64 \
    --random-input-len 512 \
    --random-output-len 128 \
    --request-rate 32 \
    --max-concurrency 32

```

*Note: Explicitly separating the `--model` string identifier from the local directory `--tokenizer` path allows the benchmarking client to calculate sequence distributions without throwing routing errors.*

**Step 4 — Shut Down vLLM:**
Press `Ctrl-C` in the server terminal window. Run `sudo pkill -f vllm` to clear any lingering background processes.

---

## 4. Decision Tree Summary

* Use **Ollama** for simple, local single-user exploratory CLI chats or frictionless default setups.
* Use **llama.cpp** directly when you require granular control over KV cache settings, explicit micro-batch dimensions, or raw grammar constraints.
* Use **vLLM** for team-wide serving APIs, parallel execution tasks, and high-concurrency multi-agent orchestrators.

```

```