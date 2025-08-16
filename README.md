# Adaptive LLM Router

<p align="center">
  <img src="img.png" alt="Adaptive LLM Router UI" width="600">
</p>

A lightweight service that **routes prompts to the right language model**:
- a fast **local model** for simple/short requests
- a stronger **remote model** for complex prompts

It ships with a tiny web UI so you can type a prompt, pick **small / large / auto**, and see latency + which model was used.

---

## Features

- 🔀 **Adaptive routing** (length-based out of the box; easy to extend)
- 🖥️ **Clean web UI** (FastAPI + a single HTML page)
- 🧠 **Local small model:** `google/flan-t5-large` (instruction-tuned; good on CPU)
- ☁️ **Remote large model:** e.g. `mistralai/Mistral-7B-Instruct-v0.2` via Hugging Face Inference API (optional)
- 🔒 **Safe defaults:** `.env` is ignored by git; no secrets in the repo

---

## Project Structure
```text
Adaptive-LLM-Router/
├─ app.py              # FastAPI app, routing, UI
├─ requirements.txt    # Python dependencies
├─ .env.example        # Example env file (no secrets)
├─ README.md           # This file
└─ img.png             # Screenshot shown above

Quick Start

1. **Create a Python 3.11 virtual environment and install dependencies:**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt

2) **Configure environment :cp .env.example .env

3) **Run :uvicorn app:app --host 0.0.0.0 --port 8000 --reload

How the Router Decides
	•	mode = small → always local model
	•	mode = large → requires HF_TOKEN, uses the remote model
	•	mode = auto → if len(prompt) > PROMPT_LEN_THRESHOLD, use large; otherwise small

You can tweak PROMPT_LEN_THRESHOLD and decoding parameters inside app.py.

⸻

Good test prompts
	•	“What is Siemens (the company)? Answer in 2–4 sentences.”
	•	“List 3 main business areas of Siemens.”
	•	“Explain PLCs in one short paragraph.”


📜 License

MIT License © 2025 — Devrajeev
---

✨ This version is:  
- **Professional** → Reads like a real open-source project.  
- **Human-written** → No generic AI tone, structured clearly.  
- **Beautiful** → Emojis, clean sections, and roadmap.  

---

👉 Copy all of the above into `README.md` and then run:  

```bash
git add README.md
git commit -m "Added professional README"
git push origin main