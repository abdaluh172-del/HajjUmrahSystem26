# -*- coding: utf-8 -*-
"""Apify as an additional official comment source (v15.7).

Apify (https://apify.com) runs third-party "Actors" (scrapers/crawlers) for
platforms like YouTube, Google Maps and Reddit. This module is a thin,
uniform wrapper around Apify's REST API so those actors can feed the SAME
comments pipeline as every other source in external_sources.py / 
google_maps_source.py — same normalized shape, same table, same
moderation/relevance/dedup pipeline. Nothing about Apify is a separate
system: `fetch_all()` below returns the exact `{"external_id", "text",
"author", "created_at", "source", ...}` shape every other fetcher returns,
and app.py's ingest_external_comments() merges it into the same list.

Configuration (site owner controlled, via the admin Settings page — see
app.py's /api/admin/settings/apify — or, as a fallback, plain environment
variables so it also works without touching the database):
    APIFY_API_TOKEN    the account's API token (required to enable Apify)
    APIFY_ENABLED      "0" disables Apify even if a token is set (default
                       "1" = enabled whenever a token is present); the
                       admin Settings page toggle overrides this env var
    APIFY_YOUTUBE_ACTOR       actor id for YouTube comments (default below)
    APIFY_GOOGLE_MAPS_ACTOR   actor id for Google Maps reviews (default below)
    APIFY_REDDIT_ACTOR        actor id for Reddit posts/comments (default below)
    APIFY_SEARCH_QUERY        overrides the default Hajj/Umrah search query
    APIFY_MAX_ITEMS_PER_ACTOR cap per actor run (default 50)

The default actor ids point at well-known, commonly used public Actors on
the Apify Store for each platform. Apify Store listings and their exact
input schemas can change over time or be renamed/replaced by the site
owner's subscription — verify the actor id and its input fields in the
Apify Console before relying on this in production, and override via the
env vars above if a different actor is preferred.

Every function here NEVER raises: a failed/misconfigured actor, an expired
token, or a network error is logged and yields [] for that actor only, so
one broken actor never stops the other sources or the rest of the site
(per the "continue operating using the other sources without interruption
or user-facing errors" requirement).
"""
import os
from datetime import datetime, timezone

import requests

APIFY_BASE = "https://api.apify.com/v2"
TIMEOUT = 60  # actor runs can take a while; run-sync-get-dataset-items blocks until done
DEFAULT_QUERY = "Hajj Umrah experience"
DEFAULT_MAX_ITEMS = 50

# Best-effort defaults — see the module docstring's note on verifying these
# against the site owner's actual Apify Store subscription.
DEFAULT_ACTORS = {
    "youtube": "streamers/youtube-comments-scraper",
    "google_maps": "compass/google-maps-reviews-scraper",
    "reddit": "trudax/reddit-scraper-lite",
}


def _query() -> str:
    return os.environ.get("APIFY_SEARCH_QUERY") or DEFAULT_QUERY


def _max_items() -> int:
    try:
        return max(1, min(int(os.environ.get("APIFY_MAX_ITEMS_PER_ACTOR", DEFAULT_MAX_ITEMS)), 200))
    except (TypeError, ValueError):
        return DEFAULT_MAX_ITEMS


def _actor_id(platform: str) -> str:
    env_key = f"APIFY_{platform.upper()}_ACTOR"
    return (os.environ.get(env_key) or DEFAULT_ACTORS.get(platform) or "").strip()


def env_token() -> str:
    """The token from the environment only (fallback when no admin-panel
    value is set — see app.py's get_setting('apify_api_token'))."""
    return (os.environ.get("APIFY_API_TOKEN") or "").strip()


def env_enabled_default() -> bool:
    return os.environ.get("APIFY_ENABLED", "1") != "0"


def _run_actor_sync(actor_id: str, token: str, run_input: dict) -> list:
    """Runs an actor and blocks until it finishes, returning its dataset
    items directly. Raises on any HTTP/network failure — callers catch it
    per-actor so one broken actor doesn't take down the others."""
    url = f"{APIFY_BASE}/acts/{actor_id.replace('/', '~')}/run-sync-get-dataset-items"
    r = requests.post(url, params={"token": token}, json=run_input, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def _fetch_youtube(token: str) -> list:
    actor_id = _actor_id("youtube")
    if not actor_id:
        return []
    try:
        items = _run_actor_sync(actor_id, token, {
            "searchKeywords": _query(),
            "maxComments": _max_items(),
        })
    except Exception as e:
        print(f"[apify] youtube actor failed: {e}")
        return []
    out = []
    for it in items:
        cid = it.get("commentId") or it.get("id") or it.get("cid")
        text = (it.get("text") or it.get("comment") or "").strip()
        if not cid or not text:
            continue
        out.append({
            "external_id": "apify:youtube:" + str(cid),
            "text": text,
            "author": it.get("author") or it.get("authorName") or "YouTube user",
            "created_at": it.get("publishedAt") or it.get("date")
                          or datetime.now(timezone.utc).isoformat(),
            "source": "apify",
            "language": it.get("language"),
        })
    return out


def _fetch_google_maps(token: str) -> list:
    actor_id = _actor_id("google_maps")
    if not actor_id:
        return []
    try:
        items = _run_actor_sync(actor_id, token, {
            "searchStringsArray": [_query()],
            "maxReviews": _max_items(),
        })
    except Exception as e:
        print(f"[apify] google_maps actor failed: {e}")
        return []
    out = []
    for it in items:
        rid = it.get("reviewId") or it.get("id")
        text = (it.get("text") or it.get("reviewText") or "").strip()
        if not rid or not text:
            continue
        out.append({
            "external_id": "apify:google_maps:" + str(rid),
            "text": text,
            "author": it.get("name") or it.get("reviewerName") or "Google Maps user",
            "created_at": it.get("publishedAtDate") or it.get("date")
                          or datetime.now(timezone.utc).isoformat(),
            "source": "apify",
            "rating": it.get("stars") or it.get("rating"),
            "place_name": it.get("title") or it.get("placeName"),
            "place_url": it.get("url") or it.get("placeUrl"),
        })
    return out


def _fetch_reddit(token: str) -> list:
    actor_id = _actor_id("reddit")
    if not actor_id:
        return []
    try:
        items = _run_actor_sync(actor_id, token, {
            "searches": [_query()],
            "maxItems": _max_items(),
        })
    except Exception as e:
        print(f"[apify] reddit actor failed: {e}")
        return []
    out = []
    for it in items:
        rid = it.get("id")
        text = (it.get("body") or it.get("text") or it.get("title") or "").strip()
        if not rid or not text:
            continue
        out.append({
            "external_id": "apify:reddit:" + str(rid),
            "text": text[:2000],
            "author": "u/" + (it.get("username") or it.get("author") or "reddit"),
            "created_at": it.get("createdAt") or it.get("date")
                          or datetime.now(timezone.utc).isoformat(),
            "source": "apify",
            "title": it.get("title"),
            "community": it.get("communityName") or it.get("subreddit"),
            "permalink": it.get("url") or it.get("permalink"),
        })
    return out


def fetch_all(token: str, enabled: bool = True) -> list:
    """Fetch from every configured Apify actor (flat comment list, same
    normalized shape as external_sources.fetch_all()). Returns [] when
    disabled or no token is set — never raises, and one actor's failure
    never blocks the others (each _fetch_* above already isolates its own
    errors)."""
    token = (token or "").strip()
    if not enabled or not token:
        return []
    return _fetch_youtube(token) + _fetch_google_maps(token) + _fetch_reddit(token)


def configured(token: str = None) -> bool:
    t = token if token is not None else env_token()
    return bool((t or "").strip())
