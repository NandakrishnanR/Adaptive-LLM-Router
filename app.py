import os, time, asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# hardcoded local models (no .env, no tokens)
SMALL_MODEL = "distilgpt2"               # completion model (fast, weak)
LARGE_MODEL = "google/flan-t5-large"      # instruction model (good for Q&A)
PROMPT_LEN_THRESHOLD = 160               # auto-switch point
SMALL_CONCURRENCY = 2
LARGE_CONCURRENCY = 1

# lazy init
_small_pipe = None
_large_pipe = None

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
set_seed(42)  # stable results

_small_pipe = None

def _small_generate(prompt: str, max_new_tokens: int) -> str:
    """
    Make distilgpt2 behave: no sampling, beam search, strong anti-repetition,
    and EOS as both eos/pad token.
    """
    global _small_pipe
    if _small_pipe is None:
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        mdl = AutoModelForCausalLM.from_pretrained("distilgpt2")
        _small_pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device=-1)

    # PRIME WITH A PATTERN (distilgpt2 is a continuer, not an answerer)
    primed = (
        "Q: What is Apple?\n"
        "A: Apple is a large technology company known for the iPhone and the Mac.\n\n"
        "Q: What is Google?\n"
        "A: Google is a major internet company best known for Search and Android.\n\n"
        f"Q: {prompt.strip().rstrip('?')}?\n"
        "A:"
    )

    out = _small_pipe(
        primed,
        max_new_tokens=min(max_new_tokens, 48),
        do_sample=False,             # NO SAMPLING → stops the ibleible loop
        num_beams=5,                 # beam search = more sensible text
        early_stopping=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5,
        eos_token_id=_small_pipe.tokenizer.eos_token_id,
        pad_token_id=_small_pipe.tokenizer.eos_token_id,
        return_full_text=False,
        clean_up_tokenization_spaces=True,
    )
    return out[0]["generated_text"].strip()

def _large_generate(prompt: str, max_new_tokens: int) -> str:
    global _large_pipe
    if _large_pipe is None:
        _large_pipe = pipeline("text2text-generation", model=LARGE_MODEL, device=-1)

    tpl = (
        "You are a concise encyclopedia.\n"
        "Answer the question accurately in 2–4 sentences.\n\n"
        f"Question: {prompt}\n"
        "Answer:"
    )

    out = _large_pipe(
        tpl,
        max_new_tokens=min(max_new_tokens, 200),
        do_sample=False,         # turn OFF randomness
        num_beams=4,             # beam search = more reliable
        early_stopping=True
    )
    return out[0]["generated_text"].strip()

app = FastAPI(title="Adaptive LLM Router (Local)")

small_sem = asyncio.Semaphore(SMALL_CONCURRENCY)
large_sem = asyncio.Semaphore(LARGE_CONCURRENCY)
stats = {"small_ms_avg": 0.0, "large_ms_avg": 0.0, "small_n": 0, "large_n": 0}

def _rec(kind: str, ms: float):
    a, n = stats[f"{kind}_ms_avg"], stats[f"{kind}_n"]
    stats[f"{kind}_ms_avg"] = (a*n + ms) / (n+1)
    stats[f"{kind}_n"] = n + 1

class ChatIn(BaseModel):
    prompt: str
    max_new_tokens: int = 120
    mode: str = "auto"    # "small" | "large" | "auto"

class ChatOut(BaseModel):
    model_used: str
    latency_ms: float
    text: str
    routed_reason: str

@app.get("/metrics")
def metrics(): return stats

HTML = """
<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Adaptive LLM Router</title>
<style>
body{margin:0;background:#0f1220;color:#eef;font-family:system-ui,-apple-system,Segoe UI,Roboto}
.wrap{max-width:900px;margin:24px auto;padding:0 16px}
.card{background:#141a33;border:1px solid #2a325b;border-radius:14px;padding:16px;box-shadow:0 8px 30px rgba(0,0,0,.25)}
h1{margin:0 0 8px;font-size:26px}.muted{color:#a8b1d8}
textarea{width:100%;min-height:140px;border-radius:10px;border:1px solid #2a325b;background:#0c1130;color:#eef;padding:12px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px}
input,select{padding:10px;border-radius:10px;border:1px solid #2a325b;background:#0c1130;color:#eef}
button{padding:12px 18px;border-radius:12px;background:#2b3f92;border:0;color:#fff;font-weight:600;cursor:pointer}
.badge{padding:2px 8px;border-radius:999px;background:#12225a;border:1px solid #27407f;color:#cfe0ff;font-size:12px;margin-right:6px}
.out{margin-top:12px;background:#0d1130;border:1px solid #2a325b;border-radius:12px;padding:12px;min-height:64px;white-space:pre-wrap}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:12px}th,td{padding:6px 4px;border-bottom:1px solid #22284a;color:#cbd6ff}
</style>
<div class="wrap">
  <h1>Adaptive LLM Router (Local)</h1>
  <p class="muted">Small: distilgpt2 • Large: flan-t5-base • No tokens. Choose mode.</p>
  <div class="card">
    <textarea id="prompt" placeholder="Ask something..."></textarea>
    <div class="row">
      <label>Max new tokens <input id="tokens" type="number" value="120" min="8" max="512"></label>
      <label>Mode
        <select id="mode">
          <option value="auto" selected>auto</option>
          <option value="small">small (distilgpt2)</option>
          <option value="large">large (flan-t5-base)</option>
        </select>
      </label>
      <button id="go">Generate</button>
    </div>
    <div style="margin-top:8px">
      <span class="badge" id="bm">model: -</span>
      <span class="badge" id="bl">latency: -</span>
      <span class="badge" id="br">reason: -</span>
    </div>
    <div class="out" id="out">Your answer will appear here.</div>
    <table>
      <tr><th>Small avg</th><td id="savg">0 ms</td></tr>
      <tr><th>Large avg</th><td id="lavg">0 ms</td></tr>
      <tr><th>Small calls</th><td id="sn">0</td></tr>
      <tr><th>Large calls</th><td id="ln">0</td></tr>
    </table>
  </div>
</div>
<script>
async function metrics(){const r=await fetch('/metrics');const j=await r.json();
document.getElementById('savg').textContent=(j.small_ms_avg||0).toFixed(1)+' ms';
document.getElementById('lavg').textContent=(j.large_ms_avg||0).toFixed(1)+' ms';
document.getElementById('sn').textContent=j.small_n||0;document.getElementById('ln').textContent=j.large_n||0;}
setInterval(metrics,1500);metrics();
document.getElementById('go').onclick=async()=>{const prompt=document.getElementById('prompt').value.trim();
const tokens=parseInt(document.getElementById('tokens').value||'120',10);
const mode=document.getElementById('mode').value;
document.getElementById('out').textContent='Working…';
const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({"prompt":prompt,"max_new_tokens":tokens,"mode":mode})});
const j=await r.json();
if(!r.ok){document.getElementById('out').textContent='Error: '+JSON.stringify(j);return;}
document.getElementById('out').textContent=j.text||'(empty)';
document.getElementById('bm').textContent='model: '+j.model_used;
document.getElementById('bl').textContent='latency: '+j.latency_ms.toFixed(1)+' ms';
document.getElementById('br').textContent='reason: '+j.routed_reason;metrics();};
</script>
"""

@app.get("/", response_class=HTMLResponse)
def home(_: Request): return HTMLResponse(HTML)

class _In(BaseModel):
    prompt: str
    max_new_tokens: int = 120
    mode: str = "auto"

@app.post("/chat", response_model=ChatOut)
async def chat(q: _In):
    if q.mode == "large":
        async with large_sem:
            t0 = time.perf_counter()
            txt = _large_generate(q.prompt, q.max_new_tokens)
            ms = (time.perf_counter() - t0) * 1000
            _rec("large", ms)
            return ChatOut(model_used=LARGE_MODEL, latency_ms=ms, text=txt, routed_reason="forced_large")

    if q.mode == "small":
        async with small_sem:
            t0 = time.perf_counter()
            txt = _small_generate(q.prompt, q.max_new_tokens)
            ms = (time.perf_counter() - t0) * 1000
            _rec("small", ms)
            return ChatOut(model_used=SMALL_MODEL, latency_ms=ms, text=txt, routed_reason="forced_small")

    if len(q.prompt) >= PROMPT_LEN_THRESHOLD:
        async with large_sem:
            t0 = time.perf_counter()
            txt = _large_generate(q.prompt, q.max_new_tokens)
            ms = (time.perf_counter() - t0) * 1000
            _rec("large", ms)
            return ChatOut(model_used=LARGE_MODEL, latency_ms=ms, text=txt, routed_reason="prompt_long")

    async with small_sem:
        t0 = time.perf_counter()
        txt = _small_generate(q.prompt, q.max_new_tokens)
        ms = (time.perf_counter() - t0) * 1000
        _rec("small", ms)
        return ChatOut(model_used=SMALL_MODEL, latency_ms=ms, text=txt, routed_reason="small_ok")