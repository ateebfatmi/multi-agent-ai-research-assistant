# Multi-Agent Research Assistant (Agentic AI System)
Minor Project II — Implementation

## Overview
An autonomous multi-agent research assistant built using **LangGraph**, **Google Gemini**, **Tavily Web Search**, and **arXiv API**, complete with an interactive **Chainlit UI**.

### System Architecture
1. **Intent Triage & Router Agent:** Evaluates incoming user queries to differentiate between casual conversation/banter/greetings and genuine research questions. Handles conversational queries directly with wit and personality, while scoping and refining complex research queries.
2. **Planner Agent:** Decomposes open-ended topics into targeted sub-questions across academic (arXiv) and real-world (Web) tracks.
3. **Researcher Agent:** Executes parallel searches across live web and academic literature using ThreadPool concurrency.
4. **Critic / Verification Agent:** Fact-checks evidence, filters noise, detects contradictions, and initiates feedback re-search loops if information is deficient.
5. **Synthesis Agent:** Drafts an academic-style report with inline citations `[1]`, `[2]`, deduplicated references, and automated PDF export.

---

## Quickstart

### 1. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your API keys in `.env`:
- `GEMINI_API_KEY`: [Google AI Studio](https://aistudio.google.com/)
- `TAVILY_API_KEY`: [Tavily AI Search](https://tavily.com/)

### 2. Launch the Application
Run the Chainlit interface:
```bash
source venv/bin/activate
chainlit run app.py -w
```
Open your browser at `http://localhost:8000`.

---

## Running Benchmarks (Baseline Comparison)
To reproduce the experimental evaluation comparing the multi-agent system against a single-prompt LLM:
```bash
source venv/bin/activate
python eval/benchmark.py
```
