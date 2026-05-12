# LLM Orchestrator Tutorial — README

Personal working repo for a multi-week tutorial on evaluating agentic coding LLMs (local + hosted), benchmarking them, and producing recommendations.

This README is **for me**. It's the thing I read after a week away to remember how to pick up where I left off.

---

## TL;DR — how to continue from the last commit

1. **Pull latest** (if I pushed from another machine):
   ```bash
   cd ~/git/llm-orchestrator-tutorial
   git pull
   ```
2. **Check where I left off:** open `progress.md`. The "Current section" line at the top says what's next.
3. **Open a new Claude.ai conversation** in the browser. Don't continue an old one — fresh context per section.
4. **Pick the model:**
   - **Sonnet 4.6** for routine sections (installs, walkthroughs, doc reading).
   - **Opus 4.7** for load-bearing conceptual sections (§4, §10, §11, §16, §21, §22, §27, §32) or when stuck on hard debugging.
5. **Attach two files** to the first message:
   - `orchestrator-tutorial-handoff.md`
   - `progress.md`
6. **First message:** `"I've read the handoff. Ready for Section X."` (replace X with the section from progress.md)
7. **At the end of the session**, ask the conversation: *"Give me the exact progress.md entry to append for this section."* Paste it in, commit, push, close the conversation.

---

## How to start a brand-new section (full workflow)

```bash
# 1. Open the working directory in VS Code with WSL remote
cd ~/git/llm-orchestrator-tutorial
code .

# 2. Pull any remote changes
git pull

# 3. Open progress.md, note the current section number
```

Then in the browser:

- **New Claude.ai chat** → attach `orchestrator-tutorial-handoff.md` + `progress.md` → "Ready for Section X."

After working:

```bash
# 4. Update progress.md with what was done (use entry the conversation gives me)
# 5. Save any configs/scripts under configs/, notes/, etc.
# 6. Commit and push
git add progress.md configs/ notes/ results/
git commit -m "Complete §X — <short name>"
git push
```

---

## Why two Claude products

| Use this... | For... |
|-------------|--------|
| **Claude.ai chat** (browser/desktop) | Conceptual sections, explanations, web research, planning, generating documents. This is where the tutorial gets delivered. |
| **Claude Code** (`claude` in terminal) | Actually running installs, editing config files, debugging live failures on this machine. Use it from inside `~/git/llm-orchestrator-tutorial/`. |

Both can run. Don't try to do everything in one. Use the chat for "teach me," use Claude Code for "do it on my filesystem."

---

## Key rules I keep forgetting

1. **One fresh conversation per section** (or per small group of related sections). Don't try to fit all 35 sections in one chat — context degrades long before the conversation forces a restart.
2. **Always paste both `orchestrator-tutorial-handoff.md` and `progress.md`** at the start of a new conversation. The handoff has my hardware, goals, and decisions; progress.md has where I am.
3. **Update `progress.md` at the end of every session, even if it was a short one.** Future-me has no other way to remember what I configured.
4. **Commit and push after every session.** WSL is fragile; a corrupted distro means weeks of work gone if it's not pushed to GitHub.
5. **The handoff document is version 1.0 and shouldn't drift silently.** If a tutorial section discovers something that contradicts a §5 decision in the handoff, *explicitly* bump the version and update the document in a commit. Don't let it rot.
6. **Save artifacts locally as I go.** Any config file, Modelfile, Docker compose, launch script, or benchmark result — save it under the right subdirectory (see "Layout" below), don't rely on conversation scrollback.
7. **Don't `git add` model weights.** `.gitignore` already excludes `*.gguf`, `*.safetensors`, `*.bin`, `*.pt`. If I add a new format, update `.gitignore` first.
8. **License sanity-check before recommending anything to coworkers.** Different model weights have different commercial-use terms. §34 covers this; don't skip it.
9. **Benchmark hygiene is non-negotiable.** Report median + IQR across N runs, pin model hashes and harness commits, note temperature/seed/sampler. A single number from a single run is not a result.

---

## Repository layout

```
~/git/llm-orchestrator-tutorial/
├── README.md                          # this file
├── orchestrator-tutorial-handoff.md   # context for any new Claude conversation
├── progress.md                        # what I've done, what's next
├── .gitignore
├── configs/                           # Modelfiles, vLLM launch scripts, harness configs
├── benchmarks/                        # cloned benchmark repos (gitignored)
├── results/                           # raw benchmark outputs, traces, judge scores
└── notes/                             # per-section scratch notes
```

Subdirectories don't all exist yet — they get created as the tutorial reaches sections that need them.

---

## Hardware quick-reference

- **GPU:** RTX 5090 (Blackwell, sm_120, 32 GB VRAM)
- **CPU:** Ryzen 9 9950X
- **RAM:** 96 GB
- **OS:** Windows 11 + WSL2 (all work inside WSL)
- **Useful model band:** 14B–70B (not the 7–8B tier most tutorials default to)
- **Blackwell caveat:** Needs CUDA 12.8+, recent llama.cpp / vLLM / Ollama for sm_120 support. Older binaries silently fall back to slow paths.

---

## Cloud/API access

- **Google Vertex AI:** API key available — use this for all programmatic hosted-LLM benchmarking (Gemini family).
- **Anthropic:** Pro subscription only, no API key. Claude Code works via subscription; can't do large-scale apples-to-apples benchmarks without an API key. Plan: small qualitative task set + manual scoring.

---

## Tutorial structure — 35 sections in 6 parts

(Full detail in `orchestrator-tutorial-handoff.md`.)

- **Part 0 — Foundations:** §1–§3 (conceptual map, WSL/Blackwell setup, capacity planning)
- **Part 1 — Local serving:** §4–§9 (quants, engines, models, sampling, spec-decoding, tokenizers/templates)
- **Part 2 — Agent harnesses:** §10–§15 (anatomy, edit formats, walkthrough, local + Vertex + Claude Code)
- **Part 3 — MCP:** §16–§19 (protocol, my filtering server, cross-harness, in benchmarking)
- **Part 4 — Benchmarks:** §20–§26 (hygiene, Aider polyglot, SWE-bench, Terminal-Bench, others, eval harnesses)
- **Part 5 — Comparing & recommending:** §27–§35 (experiment design, reproducibility, observability, judge, cost/latency, finding repos, recommendation doc, licenses, maintenance)

Load-bearing sections (use Opus): **§4, §10, §11, §16, §21, §22, §27, §32.**

---

## If something has gone wrong

- **Lost track of where I am:** open `progress.md`. If that's also unclear, `git log --oneline` shows commit history with section completions.
- **A new Claude conversation seems confused about context:** re-attach `orchestrator-tutorial-handoff.md` explicitly. Don't assume earlier turn loaded it correctly.
- **WSL distro broken / new machine:** `git clone git@github.com:kristopherhuber-commits/llm-orchestrator-tutorial.git ~/git/llm-orchestrator-tutorial` and pick up from `progress.md`.
- **Tool version drift since handoff was written:** if a section's install commands fail because tooling has moved on, ask the conversation to web-search current install instructions. Don't run commands you can't verify are current.
- **Want to revise the tutorial structure or handoff content itself:** that's a different kind of session — do it in a conversation that has the handoff attached, generate a v1.1 handoff, commit it, bump the version line at the top.

---

## Useful commands

```bash
# See where I am
cat progress.md | head -20

# See what I've completed (commit history)
git log --oneline

# Save progress
git add -A && git commit -m "Complete §X — <name>" && git push

# Check disk usage before benchmark runs (SWE-bench needs gigabytes)
df -h ~

# Check VRAM live during a run (in a second terminal pane)
nvidia-smi -l 1
```

---

*Last update to this README: when the repo was first set up. If the workflow changes, update this file.*