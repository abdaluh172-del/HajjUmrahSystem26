# -*- coding: utf-8 -*-
""""تعليمات الحج والعمرة" — the specialized Hajj & Umrah AI assistant (v15).

A ChatGPT-style assistant, but locked to ONE domain: Hajj, Umrah, their
rituals, and every pilgrim-facing service at the two Holy Mosques. Anything
else gets a polite redirect instead of an answer.

Design goals (from the product spec):
  * Answer ONLY questions about Hajj/Umrah rituals & Haramain services;
    politely decline anything else and ask for an on-topic question.
  * Never guess: lean on a small curated knowledge base (knowledge_base.py)
    for grounding, cite the category of official source responsible
    (Ministry of Hajj & Umrah / the Grand Mosque & Prophet's Mosque
    presidency / Nusuk platform), and say so plainly when a detail isn't
    confirmed rather than inventing it.
  * Structured, professional answers: a heading, a short intro, ordered
    points/steps, important warnings, shar'i rulings when relevant (with a
    pointer to official Ifta offices for personal rulings), correct
    duas/adhkar, service locations, and a "related info" close.

Tiering mirrors ai_pipeline.py: reuses ANTHROPIC_API_KEY / OPENAI_API_KEY /
GEMINI_API_KEY (same env vars — no extra configuration) for the highest-
quality answers; without a key, falls back to a template built from
knowledge_base.py so the page is never empty/broken, just less
conversational.
"""
import json
import os
import re
import traceback

import requests

import knowledge_base
import llm_client

LLM_TIMEOUT = 30
MAX_HISTORY_MESSAGES = 12  # keep the request small & the assistant focused

# v15.1 fix: "claude-haiku-4-5" (no date suffix) is NOT a valid Anthropic
# model id — every request using it fails with 404, the exception is caught
# by answer()'s try/except, and the assistant silently falls back to the
# bare knowledge-base template forever, even with a perfectly good API key.
# This was the main reason the assistant looked "broken". Overridable via
# the LLM_MODEL env var if the site owner wants a different model/tier.
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
# v15.3: pinned to the stable GA model "gemini-2.5-flash" instead of the
# "gemini-flash-latest" alias. The alias can resolve to an experimental /
# 3.x model that (a) may carry more restrictive rate limits and (b) REJECTS
# generationConfig.thinkingConfig.thinkingBudget=0 with an HTTP 400 — which
# was making this whole tier fall back silently to the knowledge-base
# template ("لم يتم تفعيل نموذج ذكاء اصطناعي بعد") even with a perfectly
# valid GEMINI_API_KEY set. gemini-2.5-flash is fast, cheap, supported, and
# accepts the thinking-off config. _call_gemini() below ALSO retries without
# thinkingConfig, so even a 3.x model set via LLM_MODEL still answers.
# Overridable via the LLM_MODEL env var.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Same reasoning as ai_pipeline.GEMINI_SAFETY_SETTINGS: a pilgrim can
# legitimately ask about difficult topics (crowd-crush safety, medical
# emergencies, grief) without tripping Google's default filters and coming
# back empty — the on-topic/off-topic boundary here is enforced by the
# system prompt + knowledge_base.in_scope(), not by the safety filter.
GEMINI_SAFETY_SETTINGS = [
    {"category": c, "threshold": "BLOCK_NONE"} for c in (
        "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]

LANG_NAMES = {
    "ar": "Arabic", "en": "English", "tr": "Turkish", "ur": "Urdu",
    "hi": "Hindi", "he": "Hebrew",
}


def _system_prompt(lang: str, kb_context: str) -> str:
    lang_name = LANG_NAMES.get(lang, "Arabic")
    prompt = f"""You are "تعليمات الحج والعمرة" (Hajj & Umrah Guidance), a specialized assistant \
inside a Hajj & Umrah pilgrim-feedback platform. You help pilgrims and prospective pilgrims with \
Hajj, Umrah, and Haramain (the two Holy Mosques) services ONLY.

IN-SCOPE topics (answer these): Hajj rituals, Umrah rituals, Ihram, Tawaf, Sa'i, standing at \
Arafah, Muzdalifah, Mina, stoning the Jamarat, the Hady (sacrifice), shaving/trimming, Tawaf \
al-Ifadah, Tawaf al-Wada', the Miqats, Ihram prohibitions, Fidyah, du'as and adhkar, shar'i \
rulings related to Hajj/Umrah, Grand Mosque services, Prophet's Mosque services, Ifta (fatwa) \
offices, guidance offices, lesson/lecture locations, Qur'an circles, restrooms, ablution areas, \
gates, prayer areas, elderly/disability carts, first aid, health centers, lost & found, crowd \
management, transport services, official Hajj/Umrah apps (e.g. Nusuk), and any other service for \
pilgrims within Makkah, Madinah, or the sacred sites (al-Masha'ir al-Muqaddasah).

OUT OF SCOPE (politely decline): anything unrelated to Hajj/Umrah/Haramain services — general \
chit-chat, coding, unrelated travel, politics, other religions' rituals, etc. When a question is \
out of scope, apologize briefly and ask the person to ask something about Hajj or Umrah instead. \
Do NOT answer the off-topic question even partially.

RELIABILITY RULES (critical):
- Never invent rulings, prices, phone numbers, exact locations, or dates. If you are not certain, \
say so plainly and suggest the person confirm with an official source.
- Ground your answers in what is well-established; official sources you may refer to by name are: \
{knowledge_base.OFFICIAL_SOURCES_AR} ({knowledge_base.OFFICIAL_SOURCES_EN}).
- For personal shar'i rulings (e.g. "did I do X correctly", fidyah for a specific situation), give \
the general rule and explicitly recommend confirming with an official Ifta/guidance office rather \
than issuing a personal fatwa yourself.
- Only give du'as/adhkar you are confident are authentic and correctly worded; if unsure of exact \
wording, describe the general content instead of inventing wording.

ANSWER FORMAT (use Markdown):
- A short bold heading line.
- A one-to-two sentence introduction.
- Organized bullet points, and numbered sequential steps when the answer describes a procedure.
- An "⚠️ تنبيه مهم" / "⚠️ Important" callout for any important warning, if relevant.
- Shar'i rulings when relevant, phrased carefully per the reliability rules above.
- Correct du'as/adhkar when relevant.
- Service locations when relevant.
- End with one short line suggesting a related follow-up topic.

LANGUAGE: Respond in {lang_name}. If the person's message is written in a different language, \
respond in the language they used instead.

{("GROUNDING CONTEXT (verified reference material — prefer this over your own memory when it " \
"applies; it may be partial or empty):\n" + kb_context) if kb_context else ""}"""
    return prompt


def _heuristic_in_scope(text: str) -> bool:
    return knowledge_base.in_scope(text)


# v15.7: automatic cross-provider fallback — Gemini is primary; if it hits a
# rate limit / quota or fails for any reason, the assistant transparently
# tries ChatGPT next, then Claude, with none of this visible to the user
# (same chat UI regardless of which one actually answered). When Gemini
# recovers it's tried first again on the very next message — there's no
# "sticky" failover state, each request just walks the list fresh.
PROVIDER_ORDER = ["gemini", "openai", "anthropic"]


def llm_configured() -> str:
    """The provider that would be tried FIRST, under the Gemini -> ChatGPT ->
    Claude order — i.e. the first one with a key configured. See
    PROVIDER_ORDER / answer() for the actual per-request fallback chain."""
    configured = providers_configured()
    for provider in PROVIDER_ORDER:
        if configured.get(provider):
            return provider
    return ""


def providers_configured() -> dict:
    """v15.2: see ai_pipeline.providers_configured() — same idea, reported
    independently here since this module already duplicates llm_configured()
    rather than importing ai_pipeline."""
    return {
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "gemini": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    }


def _call_anthropic(system_prompt: str, messages: list) -> str:
    key = os.environ["ANTHROPIC_API_KEY"]
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={
            "model": os.environ.get("LLM_MODEL", DEFAULT_ANTHROPIC_MODEL),
            "max_tokens": 1000,
            "system": system_prompt,
            "messages": messages,
        },
        timeout=LLM_TIMEOUT,
    )
    if not r.ok:
        # Surface the real reason in the server log (bad model name, bad/expired
        # key, rate limit, etc.) instead of a generic "LLM call failed" with no
        # detail — this is exactly the kind of silent failure that made the
        # assistant look broken with no way to diagnose it from the outside.
        print(f"[assistant] Anthropic API error {r.status_code}: {r.text[:500]}")
    r.raise_for_status()
    data = r.json()
    return "".join(b.get("text", "") for b in data.get("content", [])).strip()


def _call_openai(system_prompt: str, messages: list) -> str:
    key = os.environ["OPENAI_API_KEY"]
    oa_messages = [{"role": "system", "content": system_prompt}] + messages
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": os.environ.get("LLM_MODEL", DEFAULT_OPENAI_MODEL),
            "max_tokens": 1000,
            "messages": oa_messages,
        },
        timeout=LLM_TIMEOUT,
    )
    if not r.ok:
        print(f"[assistant] OpenAI API error {r.status_code}: {r.text[:500]}")
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def _gemini_extract_text(data: dict) -> str:
    """Kept for backward compatibility; the real extraction now lives in
    llm_client. Tolerates an empty/filtered candidate list."""
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts).strip()


def _call_gemini(system_prompt: str, messages: list) -> str:
    """Ask Gemini for the assistant reply via the shared, hardened client
    (llm_client): correct model, thinking-off/plain retry, proper 429/503
    handling with backoff + lighter-model fallback + cooldown, timeout, and
    clear logging. Raises llm_client.GeminiError(kind=...) on failure so
    answer() can show an accurate message / fall back to the KB."""
    # Gemini has no "assistant" role — the model's own turns are "model".
    contents = [{"role": "model" if m["role"] == "assistant" else "user",
                 "parts": [{"text": m["content"]}]} for m in messages]
    return llm_client.generate_text(
        contents,
        system_instruction=system_prompt,
        max_output_tokens=1000,
        allow_wait=True,          # a single chat reply can wait a moment on 429
        timeout=LLM_TIMEOUT,
    )


def _kb_fallback_answer(question: str, lang: str, provider_configured: bool = False,
                        error_kind: str = None) -> dict:
    """Fallback to the knowledge base. Reached either because no LLM key is
    configured, OR because a configured provider's call failed this turn.
    provider_configured distinguishes the two so the "empty knowledge base"
    message doesn't wrongly claim no AI is set up when one actually is.
    error_kind (e.g. "rate_limit") tailors the message to the real cause."""
    if not _heuristic_in_scope(question):
        msg = {
            "ar": "أستطيع الإجابة فقط عن أسئلة متعلقة بالحج والعمرة وخدمات الحرمين الشريفين. "
                  "تفضل بطرح سؤال في هذا النطاق 🙏",
            "en": "I can only answer questions about Hajj, Umrah, and services at the two Holy "
                  "Mosques. Please ask something in that scope 🙏",
        }.get(lang, None) or {
            "ar": "أستطيع الإجابة فقط عن أسئلة متعلقة بالحج والعمرة وخدمات الحرمين الشريفين.",
        }["ar"]
        return {"reply": msg, "engine": "scope-guard", "out_of_scope": True}

    # A clear, specific note prepended when the AI tier was reachable but the
    # request itself failed (rate limit / temporary error). Even when the KB
    # DOES have an answer below, the user is told why it's not the richer AI
    # reply — "رسالة مفهومة بدلاً من عدم الرد".
    ai_note = ""
    if error_kind == "rate_limit":
        ai_note = {
            "ar": "⏳ المساعد مشغول حاليًا بسبب كثرة الطلبات على خدمة الذكاء الاصطناعي "
                  "(تم بلوغ الحد المسموح مؤقتًا). إليك معلومة موثوقة من قاعدة المعرفة، "
                  "ويمكنك إعادة المحاولة بعد دقيقة للحصول على إجابة أكثر تفصيلًا.\n\n",
            "en": "⏳ The assistant is briefly rate-limited (the AI service quota was reached). "
                  "Here's a verified answer from the knowledge base — try again in a minute "
                  "for a more detailed reply.\n\n",
        }.get(lang, "")
    elif provider_configured and error_kind:
        ai_note = {
            "ar": "⚠️ تعذّر الوصول إلى المساعد الذكي مؤقتًا. إليك معلومة من قاعدة المعرفة.\n\n",
            "en": "⚠️ The AI assistant is temporarily unavailable. Here's a knowledge-base answer.\n\n",
        }.get(lang, "")

    hits = knowledge_base.retrieve(question, limit=2)
    if not hits:
        if error_kind == "rate_limit":
            msg = {
                "ar": "المساعد مشغول حاليًا بسبب كثرة الطلبات على خدمة الذكاء الاصطناعي "
                      "(تم بلوغ الحد المسموح مؤقتًا). يُرجى المحاولة بعد دقيقة.",
                "en": "The assistant is busy right now due to high demand on the AI service "
                      "(rate limit reached). Please try again in a minute.",
            }.get(lang) or "المساعد مشغول حاليًا. يُرجى المحاولة بعد دقيقة."
            return {"reply": msg, "engine": "kb-fallback-ratelimited", "out_of_scope": False}
        if provider_configured:
            msg = {
                "ar": "تعذّر الوصول إلى المساعد الذكي مؤقتًا لهذا السؤال. يُرجى المحاولة مرة "
                      f"أخرى بعد قليل، أو مراجعة {knowledge_base.OFFICIAL_SOURCES_AR} للتأكد.",
                "en": "The AI assistant is temporarily unavailable for this question. Please try "
                      f"again shortly, or check {knowledge_base.OFFICIAL_SOURCES_EN}.",
            }.get(lang) or (
                "تعذّر الوصول إلى المساعد الذكي مؤقتًا. يُرجى المحاولة مرة أخرى بعد قليل، أو "
                f"مراجعة {knowledge_base.OFFICIAL_SOURCES_AR}."
            )
        else:
            msg = {
                "ar": "هذا سؤال متعلق بالحج والعمرة، لكن لا تتوفر لديّ حاليًا معلومة موثوقة كافية "
                      "للإجابة عليه بدقة (لم يتم تفعيل نموذج ذكاء اصطناعي بعد). "
                      f"يُرجى مراجعة {knowledge_base.OFFICIAL_SOURCES_AR} للتأكد.",
                "en": "This is a Hajj/Umrah question, but I don't have enough verified information to "
                      "answer it precisely right now (no AI model is configured yet). Please check "
                      f"{knowledge_base.OFFICIAL_SOURCES_EN}.",
            }.get(lang) or (
                "هذا سؤال متعلق بالحج والعمرة، لكن لا تتوفر لديّ حاليًا معلومة موثوقة كافية للإجابة "
                f"عليه بدقة. يُرجى مراجعة {knowledge_base.OFFICIAL_SOURCES_AR}."
            )
        return {"reply": msg, "engine": "kb-fallback-empty", "out_of_scope": False}

    use_ar = lang != "en"
    parts = []
    for e in hits:
        title = e["title_ar"] if use_ar else e["title_en"]
        body = e["body_ar"] if use_ar else e["body_en"]
        parts.append(f"**{title}**\n\n{body}")
    footer = (
        f"\n\n_هذه معلومات عامة من قاعدة معرفة داخلية. للحصول على إجابات أكثر تفصيلًا وسياقًا، "
        f"يمكن لمسؤول الموقع تفعيل الذكاء الاصطناعي التوليدي عبر إضافة مفتاح API. للتأكد من "
        f"التفاصيل الدقيقة راجع {knowledge_base.OFFICIAL_SOURCES_AR}._"
        if use_ar else
        f"\n\n_This is general information from an internal knowledge base. For more detailed, "
        f"conversational answers, the site admin can enable the generative AI tier by adding an "
        f"API key. For precise details, check {knowledge_base.OFFICIAL_SOURCES_EN}._"
    )
    return {"reply": ai_note + "\n\n---\n\n".join(parts) + footer,
            "engine": "kb-fallback", "out_of_scope": False}


_ALL_PROVIDERS_UNAVAILABLE = {
    "ar": "جميع خدمات الذكاء الاصطناعي غير متاحة حالياً، يرجى المحاولة مرة أخرى بعد قليل.",
    "en": "All AI services are currently unavailable, please try again shortly.",
}


def answer(history: list, lang: str = "ar") -> dict:
    """history: list of {"role": "user"|"assistant", "content": str}, oldest
    first, ending with the newest user message. Returns
    {"reply": str, "engine": str, "out_of_scope": bool}. Never raises.

    Provider fallback (v15.7): tries every configured provider in order —
    Gemini first, then ChatGPT, then Claude — completely automatically and
    invisibly to the user (same chat UI either way). The provider that
    actually answered is only ever recorded in `engine` for logs/status, not
    shown to the user. If every configured provider fails this turn, a clear
    message is returned instead of silently going quiet; if none is
    configured at all, falls back to the knowledge-base template as before."""
    history = [h for h in (history or [])
               if h.get("role") in ("user", "assistant") and (h.get("content") or "").strip()]
    if not history or history[-1]["role"] != "user":
        return {"reply": "", "engine": "none", "out_of_scope": False}
    history = history[-MAX_HISTORY_MESSAGES:]
    last_question = history[-1]["content"]

    configured = providers_configured()
    providers = [p for p in PROVIDER_ORDER if configured.get(p)]
    error_kind = None
    attempted = False
    if providers:
        kb_ctx = knowledge_base.context_block(last_question, lang=lang, limit=3)
        system_prompt = _system_prompt(lang, kb_ctx)
        messages = [{"role": h["role"], "content": h["content"]} for h in history]
        for provider in providers:
            attempted = True
            try:
                if provider == "anthropic":
                    text = _call_anthropic(system_prompt, messages)
                elif provider == "openai":
                    text = _call_openai(system_prompt, messages)
                else:
                    text = _call_gemini(system_prompt, messages)
                if text:
                    # Logged for debugging only — never shown to the user,
                    # who sees the exact same chat UI regardless of which
                    # provider answered (per the switching-transparency
                    # requirement).
                    print(f"[assistant] answered via {provider}")
                    return {"reply": text, "engine": f"llm-{provider}", "out_of_scope": False}
            except llm_client.GeminiError as e:
                error_kind = e.kind  # "rate_limit" | "auth" | ... -> tailored message
                print(f"[assistant] {provider} failed ({e.kind}), trying next provider")
                continue
            except Exception as e:
                error_kind = "error"
                print(f"[assistant] {provider} failed, trying next provider: {e}")
                traceback.print_exc()
                continue
        # Every configured provider was tried and none returned an answer.
        if _heuristic_in_scope(last_question):
            msg = _ALL_PROVIDERS_UNAVAILABLE.get(lang) or _ALL_PROVIDERS_UNAVAILABLE["ar"]
            return {"reply": msg, "engine": "all-providers-failed", "out_of_scope": False}
    return _kb_fallback_answer(last_question, lang, provider_configured=attempted,
                               error_kind=error_kind)


def status() -> dict:
    provider = llm_configured()
    default_model = {"anthropic": DEFAULT_ANTHROPIC_MODEL, "openai": DEFAULT_OPENAI_MODEL,
                      "gemini": DEFAULT_GEMINI_MODEL}.get(provider)
    return {"llm_provider": provider,
            "llm_model": os.environ.get("LLM_MODEL") or default_model,
            "llm_providers_configured": providers_configured(),
            "kb_topics": len(knowledge_base.KB),
            **llm_client.status()}
