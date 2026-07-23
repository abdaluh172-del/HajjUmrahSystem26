# -*- coding: utf-8 -*-
"""Robust, shared Gemini API client (v15.6).

Both the AI assistant (assistant.py) and the comment-analysis pipeline
(ai_pipeline.py) call Gemini through this one module, so all the hard parts
live in a single well-tested place:

  * Correct, SUPPORTED model by default (gemini-2.5-flash) — never a retired
    or alias-only id.
  * Two generation variants tried in order: thinking OFF (cheap, correct on
    2.5-class models) then plain (newer 3.x models reject thinkingBudget=0
    with HTTP 400) — so the call is model-agnostic.
  * HTTP 429 (RESOURCE_EXHAUSTED / quota) and 503 handled PROPERLY:
      - honors the server's Retry-After header / RetryInfo.retryDelay,
      - a short bounded backoff+retry (never long enough to hang a request),
      - then falls back to a lighter model that has a SEPARATE, higher
        free-tier quota bucket (gemini-2.5-flash-lite),
      - and finally trips a short module-wide COOLDOWN so the site instantly
        stops calling Gemini (fast graceful fallback everywhere) instead of
        firing hundreds of doomed requests while a video's comments are
        analyzed. The cooldown clears itself automatically.
  * A real timeout on every request.
  * Clear, prefixed logging for Render's log view.
  * Never hangs the UI: on any terminal failure it raises GeminiError with a
    `kind` ("rate_limit" | "auth" | "not_found" | "timeout" | "network" |
    "empty" | "error") so callers can show an accurate, human message or
    fall back to their non-LLM tier.

Nothing here throws on import; it only needs `requests`.
"""
import os
import time
import requests

# --------------------------------------------------------------------- #
# Configuration (all overridable via environment variables)
# --------------------------------------------------------------------- #
DEFAULT_MODEL = "gemini-2.5-flash"          # assistant default (best quality/cost)
LITE_MODEL = "gemini-2.5-flash-lite"        # bulk default: higher free-tier limits
_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

TIMEOUT = int(os.environ.get("GEMINI_TIMEOUT", "30"))          # per HTTP request
RL_RETRIES = int(os.environ.get("GEMINI_RL_RETRIES", "1"))     # extra tries on 429/503
RL_MAX_WAIT = float(os.environ.get("GEMINI_RL_MAX_WAIT", "4")) # never sleep longer (s)
COOLDOWN_SECONDS = int(os.environ.get("GEMINI_COOLDOWN_SECONDS", "60"))
_BACKOFF = [1.0, 2.0, 4.0]

# The four adjustable safety categories are opened up: a pilgrim can ask
# about difficult topics (crowd-crush safety, medical emergencies, grief),
# and comment moderation has to actually SEE the text to classify it. The
# on-topic boundary is enforced by the prompt, not by these filters.
SAFETY_SETTINGS = [
    {"category": c, "threshold": "BLOCK_NONE"} for c in (
        "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]

# Module-wide cooldown: while time.time() < _cooldown_until, every call
# short-circuits to a rate_limit error WITHOUT touching the network.
_cooldown_until = 0.0


class GeminiError(Exception):
    def __init__(self, message, kind="error"):
        super().__init__(message)
        self.kind = kind


# --------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------- #
def api_key() -> str:
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()


def is_configured() -> bool:
    return bool(api_key())


def in_cooldown() -> bool:
    return time.time() < _cooldown_until


def cooldown_remaining() -> int:
    return max(0, int(_cooldown_until - time.time()))


def _set_cooldown(seconds: int):
    global _cooldown_until
    _cooldown_until = max(_cooldown_until, time.time() + seconds)
    print(f"[gemini] quota cooldown engaged for ~{seconds}s "
          f"(site keeps working via fallback tiers)")


def _model_chain(primary: str = None):
    """Ordered, de-duplicated list of models to try. `primary` first, then
    the standard flash + flash-lite pair so a quota hit on one model rolls
    over to the other (they have separate free-tier quota buckets)."""
    env_model = (os.environ.get("GEMINI_MODEL") or "").strip()
    llm_model = (os.environ.get("LLM_MODEL") or "").strip()
    # Only honor LLM_MODEL here if it's actually a Gemini id (it's a shared
    # var that could hold a Claude/GPT name when another provider is active).
    if llm_model and not llm_model.lower().startswith("gemini"):
        llm_model = ""
    head = primary or env_model or llm_model or DEFAULT_MODEL
    chain = [head]
    for m in (DEFAULT_MODEL, LITE_MODEL):
        if m not in chain:
            chain.append(m)
    return chain


def _parse_retry_delay(resp) -> float:
    """Seconds the server asks us to wait, from the Retry-After header or the
    RetryInfo detail in the JSON body. None if not provided."""
    ra = resp.headers.get("Retry-After")
    if ra:
        try:
            return float(ra)
        except (TypeError, ValueError):
            pass
    try:
        for d in (resp.json().get("error", {}) or {}).get("details", []) or []:
            rd = d.get("retryDelay")
            if isinstance(rd, str) and rd.endswith("s"):
                return float(rd[:-1])
    except Exception:
        pass
    return None


def _extract_text(data: dict) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts).strip()


def generate_text(contents, *, system_instruction=None, json_mode=False,
                  max_output_tokens=1000, allow_wait=True, timeout=None,
                  primary_model=None) -> str:
    """Call Gemini and return the reply text.

    contents: Gemini "contents" list, e.g.
        [{"role": "user", "parts": [{"text": "..."}]}]
    system_instruction: optional system prompt string.
    json_mode: request application/json output (used by the classifier).
    allow_wait: if False, never sleep on 429 (bulk analysis fails fast to the
        non-LLM tier instead of stalling); the cooldown still protects the
        rest of the run.
    primary_model: force a specific first model (bulk analysis uses the lite
        model, which has higher free-tier limits, to AVOID 429s up front).

    Raises GeminiError(kind=...) on any terminal failure. Never hangs.
    """
    key = api_key()
    if not key:
        raise GeminiError("GEMINI_API_KEY is not set", kind="auth")
    if in_cooldown():
        raise GeminiError("Gemini is cooling down after a quota limit",
                          kind="rate_limit")

    timeout = timeout or TIMEOUT
    headers = {"x-goog-api-key": key, "content-type": "application/json"}
    variants = [{"thinkingConfig": {"thinkingBudget": 0}}, {}]  # thinking off, then plain
    models = _model_chain(primary_model)

    last_kind, last_detail = "error", "no response"
    rate_limited_models = set()

    for model in models:
        if model in rate_limited_models:
            continue
        url = f"{_API_BASE}/{model}:generateContent"
        for vi, extra in enumerate(variants):
            gen = {"maxOutputTokens": max_output_tokens}
            gen.update(extra)
            if json_mode:
                gen["responseMimeType"] = "application/json"
            body = {"contents": contents, "generationConfig": gen,
                    "safetySettings": SAFETY_SETTINGS}
            if system_instruction:
                body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

            for attempt in range(RL_RETRIES + 1):
                try:
                    r = requests.post(url, headers=headers, json=body, timeout=timeout)
                except requests.Timeout:
                    last_kind, last_detail = "timeout", f"timeout after {timeout}s"
                    print(f"[gemini] TIMEOUT model={model} after {timeout}s")
                    break
                except requests.RequestException as e:
                    last_kind, last_detail = "network", str(e)
                    print(f"[gemini] network error model={model}: {e}")
                    break

                if r.status_code in (429, 503):
                    delay = _parse_retry_delay(r)
                    last_kind = "rate_limit"
                    last_detail = f"HTTP {r.status_code}"
                    print(f"[gemini] {r.status_code} RESOURCE_EXHAUSTED model={model} "
                          f"attempt={attempt + 1}/{RL_RETRIES + 1} "
                          f"retryDelay={delay}: {(r.text or '')[:200]}")
                    if (allow_wait and attempt < RL_RETRIES
                            and (delay is None or delay <= RL_MAX_WAIT)):
                        time.sleep(delay if delay else _BACKOFF[min(attempt, len(_BACKOFF) - 1)])
                        continue
                    rate_limited_models.add(model)
                    break  # move on to the next model (separate quota bucket)

                if not r.ok:
                    txt = (r.text or "")[:300]
                    # 400 caused by thinkingConfig on a 3.x model -> retry
                    # this same model with the plain (no-thinking) variant.
                    if r.status_code == 400 and "think" in txt.lower() and vi == 0:
                        last_detail = f"HTTP 400 (thinking): {txt}"
                        break
                    last_kind = ("auth" if r.status_code in (401, 403)
                                 else "not_found" if r.status_code == 404 else "error")
                    last_detail = f"HTTP {r.status_code}: {txt}"
                    print(f"[gemini] error {r.status_code} model={model}: {txt}")
                    break

                text = _extract_text(r.json())
                if text:
                    return text
                last_kind, last_detail = "empty", "empty candidates / finishReason"
                print(f"[gemini] empty response model={model} variant={vi}")
                break  # try the next variant / model

    if last_kind == "rate_limit":
        _set_cooldown(COOLDOWN_SECONDS)
    raise GeminiError(last_detail, kind=last_kind)


def status() -> dict:
    """Small snapshot for the admin/status panels."""
    return {
        "gemini_configured": is_configured(),
        "gemini_cooldown_active": in_cooldown(),
        "gemini_cooldown_remaining": cooldown_remaining(),
        "gemini_default_model": os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL,
    }
