"""Book Session Store, Parallel Multi-Turn Continuity Manager, and Flipbook Publisher.

This module manages the lifecycle of interactive storybooks and science books in
WonderBook Live:
1. `create_book`: Initializes a new book from a preset (`science_hcl_lab`,
   `fairy_starlight_fox`, `science_space_kitchen`) or custom title, executing Nano Banana
   illustration and Veo motion/voice generation in parallel via `ThreadPoolExecutor`.
2. `run_book_turn`: Orchestrates a complete user voice/question turn:
   - Calls `story_agent.generate_next_page_story` for narrative + cumulative scene state.
   - Loads the previous page's PNG (`prev_image_bytes`) for visual continuity chaining.
   - Runs `generate_page_illustration` (Nano Banana) and `generate_page_motion_and_voice`
     (Veo + Character Voice) **concurrently in parallel threads** to minimize turn latency!
   - Appends the new page and persists all artifacts to `runtime/books/<book_id>/`.
3. `finish_and_publish_book`: Marks the book as `published`, builds a standalone
   shareable HTML flipbook (`export.html`), and syncs all artifacts to Google Cloud
   Storage (`gs://<ARTIFACT_BUCKET>/books/<book_id>/`) with SHA-256 read-back verification.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from config import (
    BOOK_PRESETS,
    BOOKS_DIR,
    DEFAULT_PRESET_ID,
    get_cloud_config,
    get_localized_preset,
    log_function_call,
)
from media_agent import generate_page_illustration, generate_page_motion_and_voice
from story_agent import generate_custom_opening_book, generate_next_page_story


def _generate_book_id() -> str:
    """Create a unique, URL-safe book identifier."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"book_{ts}_{uuid.uuid4().hex[:6]}"


def _sanitize_book_id(book_id: str) -> str | None:
    """Validate that a `book_id` is alphanumeric with underscores/hyphens."""
    cleaned = "".join(ch for ch in (book_id or "") if ch.isalnum() or ch in ("_", "-"))
    if not cleaned or cleaned != book_id:
        return None
    return cleaned


def _sync_bytes_to_gcs(bucket_name: str, object_name: str, data: bytes, content_type: str) -> dict[str, Any]:
    """Upload bytes to Google Cloud Storage and verify read-back SHA-256 digest."""
    log_function_call(
        "_sync_bytes_to_gcs",
        {"bucket_name": bucket_name, "object_name": object_name, "byte_length": len(data)},
    )
    expected_sha256 = hashlib.sha256(data).hexdigest()
    gcs_uri = f"gs://{bucket_name}/{object_name}"
    try:
        from google.cloud import storage  # type: ignore

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        blob.upload_from_string(data, content_type=content_type)
        downloaded = blob.download_as_bytes()
        actual_sha256 = hashlib.sha256(downloaded).hexdigest()
        return {
            "synced": actual_sha256 == expected_sha256,
            "gcs_uri": gcs_uri,
            "sha256": actual_sha256,
            "sha256_verified": actual_sha256 == expected_sha256,
            "bytes": len(downloaded),
        }
    except Exception as exc:
        return {
            "synced": False,
            "gcs_uri": gcs_uri,
            "sha256": expected_sha256,
            "sha256_verified": False,
            "error": str(exc),
        }


def _persist_page_media(
    book_id: str,
    page_num: int,
    image_png: bytes,
    motion_gif: bytes,
    voice_wav: bytes,
) -> dict[str, Any]:
    """Save page PNG, animated GIF, and voice WAV to `runtime/books/<book_id>/`."""
    book_dir = BOOKS_DIR / book_id
    book_dir.mkdir(parents=True, exist_ok=True)

    img_path = book_dir / f"page_{page_num}_image.png"
    gif_path = book_dir / f"page_{page_num}_motion.gif"
    wav_path = book_dir / f"page_{page_num}_voice.wav"

    img_path.write_bytes(image_png)
    gif_path.write_bytes(motion_gif)
    wav_path.write_bytes(voice_wav)

    voice_mime = "audio/wav" if voice_wav[:4] == b"RIFF" else "audio/mpeg"
    return {
        "image_url": f"/api/books/{book_id}/pages/{page_num}/image.png",
        "image_sha256": hashlib.sha256(image_png).hexdigest(),
        "image_data_url": "data:image/png;base64," + base64.b64encode(image_png).decode("ascii"),
        "motion_url": f"/api/books/{book_id}/pages/{page_num}/motion.gif",
        "motion_sha256": hashlib.sha256(motion_gif).hexdigest(),
        "motion_data_url": "data:image/gif;base64," + base64.b64encode(motion_gif).decode("ascii"),
        "voice_url": f"/api/books/{book_id}/pages/{page_num}/voice.wav",
        "voice_sha256": hashlib.sha256(voice_wav).hexdigest(),
        "voice_data_url": f"data:{voice_mime};base64," + base64.b64encode(voice_wav).decode("ascii"),
    }


def _generate_media_parallel(
    nano_banana_prompt: str,
    veo_motion_prompt: str,
    character_dialogue: str,
    visual_state: dict[str, Any],
    page_number: int,
    page_title: str,
    prev_image_bytes: bytes | None,
    prefer_live_api: bool,
    lang: str = "ko",
) -> tuple[bytes, dict[str, Any], bytes, bytes, dict[str, Any]]:
    """Run Nano Banana illustration and Veo motion/voice concurrently in parallel threads."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_img = pool.submit(
            generate_page_illustration,
            nano_banana_prompt=nano_banana_prompt,
            visual_state=visual_state,
            page_number=page_number,
            page_title=page_title,
            prev_image_bytes=prev_image_bytes,
            prefer_live_api=prefer_live_api,
        )
        fut_motion = pool.submit(
            generate_page_motion_and_voice,
            veo_motion_prompt=veo_motion_prompt,
            character_dialogue=character_dialogue,
            visual_state=visual_state,
            page_number=page_number,
            page_title=page_title,
            prev_image_bytes=prev_image_bytes,
            prefer_live_api=prefer_live_api,
            lang=lang,
        )
        img_bytes, img_meta = fut_img.result()
        gif_bytes, wav_bytes, motion_meta = fut_motion.result()
    return img_bytes, img_meta, gif_bytes, wav_bytes, motion_meta


def _save_book_manifest(book: dict[str, Any]) -> None:
    """Write the book manifest (`book.json`) without bloated inline base64 strings."""
    book_id = book["book_id"]
    book_dir = BOOKS_DIR / book_id
    book_dir.mkdir(parents=True, exist_ok=True)

    disk_copy = dict(book)
    disk_pages: list[dict[str, Any]] = []
    for p in book.get("pages", []):
        p_copy = {k: v for k, v in p.items() if not k.endswith("_data_url")}
        disk_pages.append(p_copy)
    disk_copy["pages"] = disk_pages

    manifest_path = book_dir / "book.json"
    manifest_path.write_text(json.dumps(disk_copy, ensure_ascii=False, indent=2), encoding="utf-8")


def create_book(
    preset_id: str = DEFAULT_PRESET_ID,
    custom_title: str | None = None,
    custom_prompt: str | None = None,
    prefer_live_api: bool = False,
    lang: str = "ko",
) -> dict[str, Any]:
    """Create a new interactive book session (preset or custom user prompt) and render Page 1 in parallel."""
    norm_lang = "en" if str(lang or "").strip().lower().startswith("en") else "ko"
    cleaned_custom_prompt = str(custom_prompt or "").strip()

    if cleaned_custom_prompt:
        preset = generate_custom_opening_book(
            custom_prompt=cleaned_custom_prompt,
            lang=norm_lang,
            prefer_live_api=prefer_live_api,
        )
        chosen_id = str(preset.get("preset_id") or DEFAULT_PRESET_ID)
    else:
        chosen_id = preset_id if preset_id in BOOK_PRESETS else DEFAULT_PRESET_ID
        preset = get_localized_preset(chosen_id, norm_lang)

    book_id = _generate_book_id()
    log_function_call(
        "create_book",
        {
            "book_id": book_id,
            "preset_id": chosen_id,
            "custom_title": custom_title,
            "custom_prompt": cleaned_custom_prompt or None,
            "lang": norm_lang,
        },
    )

    t0 = time.perf_counter()
    opening = dict(preset["opening_page"])
    visual_state = dict(preset["initial_visual_state"])
    persona_voice = str(preset.get("character_persona_voice") or "owl_professor")
    visual_state["lang"] = norm_lang
    visual_state["character_persona_voice"] = persona_voice

    img_bytes, img_meta, gif_bytes, wav_bytes, motion_meta = _generate_media_parallel(
        nano_banana_prompt=opening["nano_banana_prompt"],
        veo_motion_prompt=opening["veo_motion_prompt"],
        character_dialogue=opening["character_dialogue"],
        visual_state=visual_state,
        page_number=1,
        page_title=opening["page_title"],
        prev_image_bytes=None,
        prefer_live_api=prefer_live_api,
        lang=norm_lang,
    )
    media_urls = _persist_page_media(book_id, 1, img_bytes, gif_bytes, wav_bytes)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    page_1 = {
        **opening,
        "lang": norm_lang,
        "character_persona_voice": persona_voice,
        "visual_state": visual_state,
        **media_urls,
        "orchestration_metrics": {
            "story_engine": "custom_prompt_seed" if cleaned_custom_prompt else "preset_seed",
            "image_engine": img_meta.get("engine_mode"),
            "motion_engine": motion_meta.get("engine_mode"),
            "tts_engine": motion_meta.get("tts_engine"),
            "voice_name": motion_meta.get("voice_name"),
            "persona": motion_meta.get("persona"),
            "lang": norm_lang,
            "chained_from_previous_page": False,
            "parallel_execution": True,
            "total_turn_ms": elapsed_ms,
        },
    }

    now_iso = datetime.now(timezone.utc).isoformat()
    book: dict[str, Any] = {
        "book_id": book_id,
        "preset_id": chosen_id,
        "mode": preset["mode"],
        "lang": norm_lang,
        "character_persona_voice": persona_voice,
        "title": (custom_title or "").strip() or preset["title"],
        "subtitle": preset["subtitle"],
        "character_name": preset["character_name"],
        "character_bible": preset["character_bible"],
        "custom_prompt": cleaned_custom_prompt or None,
        "status": "in_progress",
        "created_at": now_iso,
        "updated_at": now_iso,
        "page_count": 1,
        "share_url": f"/book/{book_id}",
        "export_url": f"/api/books/{book_id}/export.html",
        "pages": [page_1],
    }
    _save_book_manifest(book)
    return book


def get_book(book_id: str, include_data_urls: bool = True) -> dict[str, Any] | None:
    """Load a book session by `book_id` from disk."""
    safe_id = _sanitize_book_id(book_id)
    if not safe_id:
        return None
    book_dir = BOOKS_DIR / safe_id
    manifest_path = book_dir / "book.json"
    if not manifest_path.exists():
        return None

    book = json.loads(manifest_path.read_text(encoding="utf-8"))
    if include_data_urls:
        for page in book.get("pages", []):
            pnum = page.get("page_number", 1)
            img_p = book_dir / f"page_{pnum}_image.png"
            gif_p = book_dir / f"page_{pnum}_motion.gif"
            wav_p = book_dir / f"page_{pnum}_voice.wav"
            if img_p.exists():
                page["image_data_url"] = "data:image/png;base64," + base64.b64encode(img_p.read_bytes()).decode("ascii")
            if gif_p.exists():
                page["motion_data_url"] = "data:image/gif;base64," + base64.b64encode(gif_p.read_bytes()).decode("ascii")
            if wav_p.exists():
                raw_voice = wav_p.read_bytes()
                v_mime = "audio/wav" if raw_voice[:4] == b"RIFF" else "audio/mpeg"
                page["voice_data_url"] = f"data:{v_mime};base64," + base64.b64encode(raw_voice).decode("ascii")
    return book


def load_page_media_bytes(book_id: str, page_num: int, kind: str) -> bytes | None:
    """Load raw binary media (`image`, `motion`, or `voice`) for a specific book page."""
    safe_id = _sanitize_book_id(book_id)
    if not safe_id or page_num < 1:
        return None
    ext_map = {"image": "image.png", "motion": "motion.gif", "voice": "voice.wav"}
    suffix = ext_map.get(kind)
    if not suffix:
        return None
    target = BOOKS_DIR / safe_id / f"page_{page_num}_{suffix}"
    if not target.exists():
        return None
    return target.read_bytes()


def run_book_turn(
    book_id: str,
    user_utterance: str,
    prefer_live_api: bool = True,
    model_tier: str = "argon_low",
    lang: str | None = None,
) -> dict[str, Any]:
    """Execute a full voice/question turn with cumulative state and parallel media generation."""
    log_function_call(
        "run_book_turn",
        {
            "book_id": book_id,
            "user_utterance": user_utterance,
            "prefer_live_api": prefer_live_api,
            "model_tier": model_tier,
            "lang": lang,
        },
    )
    book = get_book(book_id, include_data_urls=False)
    if book is None:
        return {"status": "execution_blocked", "reasons": [f"Book '{book_id}' was not found."]}

    cleaned = (user_utterance or "").strip()
    if not cleaned:
        return {
            "status": "execution_blocked",
            "reasons": ["Spoken question (`user_utterance`) cannot be empty. Speak or pick a What-If prompt."],
        }

    raw_lang = lang if lang else book.get("lang", "ko")
    effective_lang = "en" if str(raw_lang or "ko").strip().lower().startswith("en") else "ko"
    preset_id = str(book.get("preset_id") or DEFAULT_PRESET_ID)
    localized_preset = get_localized_preset(preset_id, effective_lang)
    book["lang"] = effective_lang
    book["character_name"] = localized_preset.get("character_name", book.get("character_name"))
    book["character_persona_voice"] = localized_preset.get(
        "character_persona_voice", book.get("character_persona_voice", "owl_professor")
    )

    t0 = time.perf_counter()
    story_t0 = time.perf_counter()
    # Step 1: Story & Science Continuity Director (Gemini 4 Argon / Gemini 3.8 Flash)
    story_turn = generate_next_page_story(
        book_state=book,
        user_utterance=cleaned,
        prefer_live_api=prefer_live_api,
        model_tier=model_tier,
        lang=effective_lang,
    )
    story_ms = int((time.perf_counter() - story_t0) * 1000)
    page_num = story_turn["page_number"]
    visual_state = story_turn["visual_state"]
    visual_state["lang"] = effective_lang
    visual_state["character_persona_voice"] = book["character_persona_voice"]

    # Step 2: Load previous page image bytes (Page 1 base scene for 1-by-1 science_lab, previous page for storybook)
    ref_page_num = 1 if book.get("mode") == "science_lab" else (page_num - 1)
    prev_image_bytes = load_page_media_bytes(book_id, ref_page_num, "image")

    # Step 3 & 4 (Parallel): Nano Banana Continuity Illustration + Veo Motion & Voice
    media_t0 = time.perf_counter()
    img_bytes, img_meta, gif_bytes, wav_bytes, motion_meta = _generate_media_parallel(
        nano_banana_prompt=story_turn["nano_banana_prompt"],
        veo_motion_prompt=story_turn["veo_motion_prompt"],
        character_dialogue=story_turn["character_dialogue"],
        visual_state=visual_state,
        page_number=page_num,
        page_title=story_turn["page_title"],
        prev_image_bytes=prev_image_bytes,
        prefer_live_api=prefer_live_api,
        lang=effective_lang,
    )
    media_ms = int((time.perf_counter() - media_t0) * 1000)

    media_urls = _persist_page_media(book_id, page_num, img_bytes, gif_bytes, wav_bytes)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    new_page = {
        **story_turn,
        **media_urls,
        "orchestration_metrics": {
            "story_engine": story_turn["engine_meta"].get("model_used"),
            "model_tier": model_tier,
            "image_engine": img_meta.get("engine_mode"),
            "motion_engine": motion_meta.get("engine_mode"),
            "tts_engine": motion_meta.get("tts_engine"),
            "voice_name": motion_meta.get("voice_name"),
            "persona": motion_meta.get("persona"),
            "lang": effective_lang,
            "chained_from_previous_page": prev_image_bytes is not None,
            "previous_page_number": page_num - 1,
            "accumulated_items_count": len(visual_state.get("accumulated_items", [])),
            "parallel_execution": True,
            "parallel_media_execution": True,
            "story_turn_ms": story_ms,
            "media_parallel_ms": media_ms,
            "total_turn_ms": elapsed_ms,
        },
    }

    book["pages"].append(new_page)
    book["page_count"] = len(book["pages"])
    book["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_book_manifest(book)

    full_book = get_book(book_id, include_data_urls=True) or book
    return {
        "status": "turn_completed",
        "book_id": book_id,
        "latest_page": new_page,
        "book": full_book,
    }


def render_shareable_book_html(book: dict[str, Any]) -> str:
    """Build a standalone, self-contained wide-screen HTML Flipbook for sharing or offline viewing."""
    title_esc = html.escape(str(book.get("title") or "WonderBook Live"))
    subtitle_esc = html.escape(str(book.get("subtitle") or ""))
    mode_esc = html.escape(str(book.get("mode") or "science_lab"))

    pages = book.get("pages", [])
    page_cards: list[str] = []
    for idx, p in enumerate(pages):
        pnum = p.get("page_number", idx + 1)
        ptitle = html.escape(str(p.get("page_title") or f"Page {pnum}"))
        uquest = html.escape(str(p.get("user_question") or ""))
        dialogue = html.escape(str(p.get("character_dialogue") or ""))
        char_name = html.escape(str(p.get("character_name") or "Guide"))
        sci_note = html.escape(str(p.get("science_or_moral_note") or ""))
        vstate = p.get("visual_state") or {}
        acc_labels = ", ".join(str(x) for x in (vstate.get("accumulated_labels") or [])) or "Initial Seed"
        img_src = p.get("image_data_url") or p.get("image_url") or ""
        gif_src = p.get("motion_data_url") or p.get("motion_url") or ""
        wav_src = p.get("voice_data_url") or p.get("voice_url") or ""
        paras_html = "".join(f"<p>{html.escape(str(para))}</p>" for para in (p.get("story_paragraphs") or []))

        prev_bridge_html = ""
        if idx > 0:
            prev_p = pages[idx - 1]
            prev_title = html.escape(str(prev_p.get("page_title") or f"Page {pnum - 1}"))
            prev_img = prev_p.get("image_data_url") or prev_p.get("image_url") or ""
            prev_bridge_html = f"""
            <div class="continuity-bridge">
              <img src="{prev_img}" alt="Prev Page" class="bridge-thumb" />
              <div>
                <div style="font-size:12px;color:#38bdf8;font-weight:700;">🔗 Nano Banana & Story Continuity Bridge (Page {pnum - 1} ➔ Page {pnum})</div>
                <div style="font-size:13px;color:#cbd5e1;">Chained from <strong>{prev_title}</strong> · Cumulative State: <strong>{html.escape(acc_labels)}</strong></div>
              </div>
            </div>
            """

        page_cards.append(
            f"""
            <section class="page-spread" id="page-{pnum}">
              <div class="page-header">
                <span class="badge">Page {pnum}</span>
                <span class="question">Spoken Prompt: "{uquest}"</span>
                <span style="margin-left:auto;font-size:13px;color:#38bdf8;font-weight:600;">Cumulative Items: {html.escape(acc_labels)}</span>
              </div>
              {prev_bridge_html}
              <div class="spread-grid">
                <div class="visual-col">
                  <div class="media-tabs">
                    <strong>🎨 Nano Banana 16:9 HD Illustration & 🎬 Veo Living Motion</strong>
                  </div>
                  <img src="{img_src or gif_src}" alt="{ptitle}" class="scene-img" />
                  <details style="margin-top:10px;">
                    <summary style="cursor:pointer;color:#38bdf8;font-weight:600;">🎬 View Living Motion Frame (Veo Animated Loop)</summary>
                    <img src="{gif_src or img_src}" alt="Motion Frame {ptitle}" class="scene-img" style="margin-top:8px;" />
                  </details>
                </div>
                <div class="text-col">
                  <h2 style="margin-top:0;font-size:24px;">{ptitle}</h2>
                  <div class="story-body">{paras_html}</div>
                  <div class="voice-box">
                    <strong>{char_name}:</strong> "{dialogue}"
                    <div style="margin-top:8px;">
                      <audio controls src="{wav_src}" style="width:100%;height:36px;"></audio>
                    </div>
                  </div>
                  <div class="note-box">
                    <strong>Science / Continuity Note:</strong> {sci_note}
                  </div>
                </div>
              </div>
            </section>
            """
        )

    all_pages_html = "\n".join(page_cards)
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title_esc} — WonderBook Live Published Flipbook</title>
  <style>
    body {{ font-family: 'Pretendard', system-ui, -apple-system, sans-serif; background:#070d19; color:#f8fafc; margin:0; padding:24px; }}
    .container {{ max-width: 1760px; width: 96vw; margin: 0 auto; }}
    header {{ border-bottom: 1px solid #334155; padding-bottom: 18px; margin-bottom: 28px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; }}
    .badge {{ background:#2563eb; color:#fff; padding:5px 14px; border-radius:999px; font-size:13px; font-weight:700; }}
    .page-spread {{ background:#111c33; border:1px solid #293b5f; border-radius:18px; padding:26px; margin-bottom:28px; box-shadow: 0 18px 44px rgba(0,0,0,0.45); }}
    .page-header {{ display:flex; gap:14px; align-items:center; margin-bottom:14px; flex-wrap:wrap; }}
    .question {{ color:#fbbf24; font-weight:700; font-size:16px; }}
    .continuity-bridge {{ display:flex; align-items:center; gap:14px; background:rgba(15,23,42,0.8); border:1px solid rgba(56,189,248,0.35); border-radius:12px; padding:10px 14px; margin-bottom:16px; }}
    .bridge-thumb {{ width:88px; height:50px; object-fit:cover; border-radius:8px; border:1px solid #38bdf8; flex-shrink:0; }}
    .spread-grid {{ display:grid; grid-template-columns: 1.38fr 1fr; gap:28px; align-items:start; }}
    @media (max-width: 1024px) {{ .spread-grid {{ grid-template-columns: 1fr; }} }}
    .scene-img {{ width:100%; aspect-ratio: 16 / 9; object-fit: cover; border-radius:14px; border:1px solid #475569; box-shadow: 0 12px 32px rgba(0,0,0,0.4); }}
    .story-body p {{ line-height: 1.72; font-size: 16px; color: #e2e8f0; margin: 0 0 12px 0; }}
    .voice-box {{ background:#0b1324; border-left:4px solid #38bdf8; padding:14px; border-radius:10px; margin:16px 0; }}
    .note-box {{ background:#172554; border:1px solid #1e40af; padding:14px; border-radius:10px; font-size:14px; color:#bfdbfe; line-height:1.6; }}
    a.btn {{ background:#2563eb; color:#fff; text-decoration:none; padding:10px 18px; border-radius:10px; font-size:14px; font-weight:700; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <span class="badge">Published WonderBook ({mode_esc})</span>
        <h1 style="margin:8px 0 4px 0;">{title_esc}</h1>
        <div style="color:#94a3b8;">{subtitle_esc} · Total Pages: {len(pages)}</div>
      </div>
      <div>
        <a class="btn" href="/">+ Create Another Interactive Book</a>
      </div>
    </header>
    {all_pages_html}
  </div>
</body>
</html>"""


def finish_and_publish_book(book_id: str) -> dict[str, Any]:
    """Finalize a book session into a shareable published flipbook and sync to GCS if configured."""
    log_function_call("finish_and_publish_book", {"book_id": book_id})
    book = get_book(book_id, include_data_urls=True)
    if book is None:
        return {"status": "execution_blocked", "reasons": [f"Book '{book_id}' not found."]}

    book["status"] = "published"
    book["published_at"] = datetime.now(timezone.utc).isoformat()
    _save_book_manifest(book)

    book_dir = BOOKS_DIR / book_id
    export_html_str = render_shareable_book_html(book)
    export_html_bytes = export_html_str.encode("utf-8")
    export_path = book_dir / "export.html"
    export_path.write_bytes(export_html_bytes)

    manifest_bytes = (book_dir / "book.json").read_bytes()
    cloud_cfg = get_cloud_config()
    bucket_name = cloud_cfg.get("artifact_bucket") or ""
    gcs_proof: dict[str, Any] = {"configured": bool(bucket_name), "bucket": bucket_name}

    if bucket_name:
        gcs_proof["book_json"] = _sync_bytes_to_gcs(
            bucket_name=bucket_name,
            object_name=f"books/{book_id}/book.json",
            data=manifest_bytes,
            content_type="application/json",
        )
        gcs_proof["export_html"] = _sync_bytes_to_gcs(
            bucket_name=bucket_name,
            object_name=f"books/{book_id}/export.html",
            data=export_html_bytes,
            content_type="text/html; charset=utf-8",
        )

    storage_proof = {
        "book_id": book_id,
        "local_manifest_path": str(book_dir / "book.json"),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "local_export_html_path": str(export_path),
        "export_html_sha256": hashlib.sha256(export_html_bytes).hexdigest(),
        "page_count": len(book.get("pages", [])),
        "gcs": gcs_proof,
    }
    book["storage_proof"] = storage_proof
    _save_book_manifest(book)

    return {
        "status": "book_published",
        "book_id": book_id,
        "share_url": f"/book/{book_id}",
        "export_url": f"/api/books/{book_id}/export.html",
        "storage_proof": storage_proof,
        "book": book,
    }


def list_books(limit: int = 20) -> list[dict[str, Any]]:
    """List saved books in reverse-chronological order."""
    log_function_call("list_books", {"limit": limit})
    if not BOOKS_DIR.exists():
        return []
    dirs = sorted(
        [d for d in BOOKS_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    results: list[dict[str, Any]] = []
    for d in dirs[: max(1, limit)]:
        mpath = d / "book.json"
        if not mpath.exists():
            continue
        try:
            data = json.loads(mpath.read_text(encoding="utf-8"))
            results.append(
                {
                    "book_id": data.get("book_id", d.name),
                    "preset_id": data.get("preset_id", DEFAULT_PRESET_ID),
                    "lang": data.get("lang", "ko"),
                    "title": data.get("title"),
                    "subtitle": data.get("subtitle"),
                    "mode": data.get("mode"),
                    "status": data.get("status"),
                    "page_count": data.get("page_count", len(data.get("pages", []))),
                    "updated_at": data.get("updated_at"),
                    "share_url": f"/book/{data.get('book_id', d.name)}",
                    "export_url": f"/api/books/{data.get('book_id', d.name)}/export.html",
                }
            )
        except Exception:
            continue
    return results
