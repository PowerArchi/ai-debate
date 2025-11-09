# AI Debate — Multi-Agent LLM Debate System

A small, readable project that demonstrates:

- **A2A (Agent-to-Agent) messaging** — agents exchange structured messages with explicit recipients (`to`).
- **MCP (Model Context Protocol)** — optional external-tool calls via MCP servers; results are logged alongside the transcript when enabled.
- A live **web UI** that streams the debate (Planner at the top, three debaters in the middle row, Judge at the bottom) and persists the last run to **localStorage**.

---

![App Screenshot](https://github.com/PowerArchi/ai-debate/blob/main/screenshots/AI_Debate_topic_multiagent_dominant_architecture.png?raw=true)


## Table of contents

- [Overview](#overview)
- [Key features](#key-features)
- [Architecture](#architecture)
- [Getting started](#getting-started)
  - [Requirements](#requirements)
  - [Install](#install)
  - [Run](#run)
- [Project layout](#project-layout)
- [UI behaviour](#ui-behaviour)
- [Configuration reference](#configuration-reference)
- [License](#license)

---

## Overview

Type a topic, choose the number of rounds, and click **Run**. The app orchestrates a debate flow:

1. **Planner** (optional, full width) — sets up strategy and guardrails
2. **Three model agents** — debate in rounds (OpenAI, Claude, Gemini by default)
3. **Judge** (full width) — delivers the final conclusion

Messages are streamed to the browser via **Server-Sent Events (SSE)**.
The last run is saved to `localStorage` and restored on refresh. Starting a new run clears the saved transcript.

---

## Key features

### A2A messaging

- Each `DebateMessage` includes:
  - `agent` — sender name
  - `content` — natural-language message
  - `type` — `analysis` / `argument` / `rebuttal` / `final_conclusion`
  - `to` — explicit recipients (other agents)
- The orchestrator assigns `to` dynamically per turn, broadcasting to debating agents while excluding the Planner and the Judge.

### Optional MCP integration

- The MCP tool router launches configured MCP servers over stdio, lists their tools, and routes calls (e.g., `router.call("server:tool", **kwargs)`).
- By default, the system runs with a built-in **`local-stub`** server that provides placeholder MCP responses — this keeps the debate flow consistent even when no external tools are connected.
- Agents can invoke tools via a helper that issues structured `{server, tool, args, result}` records in the transcript.
- Tool calls, if any, are logged alongside the debate messages for transparency.

> MCP is enabled by default via `MCP_SERVERS=["local-stub"]`.
> To fully disable tool routing, leave `MCP_SERVERS` empty.

### Clean UI

- Planner card spans the top row; three debaters render side-by-side in the middle row; Judge spans the bottom.
- Live timeline, streaming status indicators, agent name highlighting, and persistent localStorage transcript.

---

## Architecture

- **Flask** — web server providing REST endpoints and the live SSE debate stream
- **LangGraph** — state machine controlling debate phases:
  - Initial planning → multi-round debate → final judgment
- **Agents**
  - `BaseAgent` — generic interface and message handling
  - `ModelAgent` — wraps LLM calls (OpenAI, Anthropic, Gemini)
  - `PlannerAgent` — first-phase setup
  - `JudgeAgent` — produces final verdict once
- **Front-end**
  - Vanilla HTML, CSS, and JavaScript
  - Uses `EventSource` to stream events from Flask
  - Saves last transcript to `localStorage`

---

## Getting started

### Requirements

- Python **3.11**
- At least one LLM API key:
  - `OPENAI_API_KEY` (OpenAI)
  - `ANTHROPIC_API_KEY` (Anthropic)
  - `GOOGLE_API_KEY` (Google Gemini)

### Install
```
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Run
```
export FLASK_APP=app.py   # Windows: set FLASK_APP=app.py
flask run --debug
```

Then open `http://localhost:5000` in your browser.
Enter a topic, select number of rounds, and click **Run Debate**.

---


## Project layout


```
ai-debate/
│
├── src/
│   ├── agents/
│   ├── orchestrator/
│   ├── mcp_tools/
│   ├── config.py
│   └── (other libs)
│
├── app.py
├── templates/
│   └── index.html
│
├── static/
│   └── css/
│       └── style.css
│
├── .env
├── README.md
└── requirements.txt
```

---

## UI behaviour

- Planner appears once at the top; Judge at the bottom.
- Debaters exchange messages over multiple rounds.
- Output is streamed live via SSE.
- The number of rounds is controlled entirely by the front-end input (`rounds-input`)
- The transcript is stored in browser `localStorage` and reloaded on refresh.

---

## Configuration reference

- **OPENAI_API_KEY** — OpenAI API key
- **ANTHROPIC_API_KEY** — Anthropic API key
- **GOOGLE_API_KEY** — Gemini API key
- **AGENTS** *(default: `OpenAI,Claude,Gemini`)* — Comma-separated list of debating agents
- **MCP_SERVERS** *(default: `["local-stub"]`)* — JSON array of MCP servers
- **TOOL_BUDGET_PER_ROUND** *(default: `3`)* — Maximum number of MCP tool calls per agent per round
- **PARALLEL_ANALYSES** *(default: `true`)* — Runs agents’ initial analyses in parallel
- **BARRIERED_ROUNDS** *(default: `true`)* — All agents must finish each round before the next begins

---

## License

MIT License — see `LICENSE` for details.
