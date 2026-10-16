"""WonderBook Live HTTP Server and Multi-Agent Orchestration API Entrypoint.

This module provides the Cloud Run compatible HTTP server (`0.0.0.0:$PORT`) and REST
API for WonderBook Live, a voice-driven interactive storybook and "What-If" science
experiment book platform.

Key Endpoints:
- `GET /healthz`: Cloud Run health check (`200 OK`).
- `GET /`: Serves the functional interactive flipbook UI (`static/index.html`).
- `GET /api/system-status`: Returns configured models (`FAST_STORY_MODEL`, `PRIMARY_MODEL`,
  `FALLBACK_MODEL`, `NANO_BANANA_MODEL`, `VEO_MODEL`), presets, and cloud config.
- `GET /api/books`: Lists created/published books.
- `POST /api/books`: Creates a new interactive book (Storybook or Science Lab mode) with
  Page 1 seed illustration, motion, and voice.
- `GET /api/books/<book_id>`: Returns full book state and all pages.
- `POST /api/books/<book_id>/turn`: Processes a spoken/clicked "What-If" question through
  the 3-stage orchestration pipeline (Fast Story Agent -> Nano Banana Image Continuity ->
  Veo Motion & Character Voice).
- `POST /api/books/<book_id>/finish`: Publishes the book into a shareable flipbook package
  and syncs to Google Cloud Storage (`ARTIFACT_BUCKET`) when configured.
- `GET /book/<book_id>` and `GET /api/books/<book_id>/export.html`: Serves the standalone
  shareable HTML flipbook.
- `GET /api/books/<book_id>/pages/<page_num>/(image.png|motion.gif|voice.wav)`: Serves
  generated page media artifacts.
"""

from __future__ import annotations

import json
import os
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from book_store import (
    create_book,
    finish_and_publish_book,
    get_book,
    list_books,
    load_page_media_bytes,
    render_shareable_book_html,
    run_book_turn,
)
from config import (
    APP_NAME,
    APP_VERSION,
    BOOK_PRESETS,
    DEFAULT_PORT,
    DEFAULT_PRESET_ID,
    FALLBACK_MODEL,
    FAST_STORY_MODEL,
    GEMINI_TTS_MODEL,
    GEMINI_TTS_PRO_MODEL,
    NANO_BANANA_MODEL,
    NANO_BANANA_PRO_MODEL,
    PRIMARY_MODEL,
    STATIC_DIR,
    THINKING_LEVEL,
    VEO_MODEL,
    get_cloud_config,
    get_localized_preset,
    is_gemini_configured,
    log_function_call,
)


def _build_presets_summary(lang: str = "ko") -> list[dict[str, Any]]:
    """Return localized preset summaries for `"ko"` or `"en"`."""
    norm_lang = "en" if str(lang or "").strip().lower().startswith("en") else "ko"
    presets_list: list[dict[str, Any]] = []
    for preset_id in BOOK_PRESETS:
        loc = get_localized_preset(preset_id, norm_lang)
        opening = loc.get("opening_page") or {}
        presets_list.append(
            {
                "preset_id": preset_id,
                "mode": loc["mode"],
                "lang": norm_lang,
                "title": loc["title"],
                "subtitle": loc["subtitle"],
                "character_name": loc["character_name"],
                "character_persona_voice": loc.get("character_persona_voice", "owl_professor"),
                "suggested_followups": list(opening.get("suggested_followups") or []),
            }
        )
    return presets_list


def get_system_status(lang: str = "ko") -> dict[str, Any]:
    """Return current runtime, multi-agent model config, presets, and cloud status."""
    norm_lang = "en" if str(lang or "").strip().lower().startswith("en") else "ko"
    log_function_call("get_system_status", {"lang": norm_lang})
    presets_list = _build_presets_summary(norm_lang)
    return {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "status": "ok",
        "lang": norm_lang,
        "models": {
            "fast_story_model": FAST_STORY_MODEL,
            "primary_model": PRIMARY_MODEL,
            "fallback_model": FALLBACK_MODEL,
            "nano_banana_pro_model": NANO_BANANA_PRO_MODEL,
            "nano_banana_model": NANO_BANANA_MODEL,
            "veo_model": VEO_MODEL,
            "gemini_tts_model": GEMINI_TTS_MODEL,
            "gemini_tts_pro_model": GEMINI_TTS_PRO_MODEL,
            "thinking_level": THINKING_LEVEL,
            "gemini_api_key_configured": is_gemini_configured(),
        },
        "default_preset_id": DEFAULT_PRESET_ID,
        "presets": presets_list,
        "cloud": get_cloud_config(),
    }


class WonderBookRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler serving the WonderBook Live UI, orchestration APIs, and shared books."""

    server_version = "WonderBookLive/2.0"

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        """Serialize and send a UTF-8 JSON response."""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(
        self,
        data: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        """Send raw binary payload with explicit Content-Type."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> tuple[dict[str, Any] | None, str | None]:
        """Read and parse JSON request body safely."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            return {}, None
        if length > 10 * 1024 * 1024:
            return None, "Request body exceeds maximum size limit."
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
            if not isinstance(parsed, dict):
                return None, "JSON body must be an object."
            return parsed, None
        except Exception as exc:
            return None, f"Invalid JSON body: {exc}"

    def do_GET(self) -> None:  # noqa: N802
        """Handle GET requests for health check, UI, book state, media, and shared book view."""
        parsed_url = urlparse(self.path)
        route = parsed_url.path
        query_params = parse_qs(parsed_url.query)
        req_lang = (query_params.get("lang") or ["ko"])[0]
        norm_lang = "en" if str(req_lang or "").strip().lower().startswith("en") else "ko"

        if route == "/healthz":
            self._send_json(
                {
                    "status": "ok",
                    "service": "wonderbook-live",
                    "fast_story_model": FAST_STORY_MODEL,
                    "primary_model": PRIMARY_MODEL,
                    "fallback_model": FALLBACK_MODEL,
                    "nano_banana_pro_model": NANO_BANANA_PRO_MODEL,
                    "nano_banana_model": NANO_BANANA_MODEL,
                    "veo_model": VEO_MODEL,
                    "gemini_tts_model": GEMINI_TTS_MODEL,
                    "gemini_tts_pro_model": GEMINI_TTS_PRO_MODEL,
                }
            )
            return

        if route == "/":
            index_file = STATIC_DIR / "index.html"
            if not index_file.exists():
                self._send_json({"error": "static/index.html not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_bytes(index_file.read_bytes(), "text/html; charset=utf-8")
            return

        static_asset_match = re.match(r"^/static/assets/nano_banana/([a-zA-Z0-9_-]+\.png)$", route)
        if static_asset_match:
            asset_file = STATIC_DIR / "assets" / "nano_banana" / static_asset_match.group(1)
            if not asset_file.exists():
                self._send_json({"error": "Static asset not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_bytes(asset_file.read_bytes(), "image/png")
            return

        if route == "/api/system-status":
            self._send_json(get_system_status(lang=norm_lang))
            return

        if route == "/api/presets":
            self._send_json(
                {
                    "lang": norm_lang,
                    "default_preset_id": DEFAULT_PRESET_ID,
                    "presets": _build_presets_summary(norm_lang),
                }
            )
            return

        if route == "/api/books":
            self._send_json({"books": list_books()})
            return

        # Media route: /api/books/<book_id>/pages/<page_num>/(image.png|motion.gif|voice.wav)
        media_match = re.match(
            r"^/api/books/([a-zA-Z0-9_-]+)/pages/([0-9]+)/(image\.png|motion\.gif|voice\.wav)$",
            route,
        )
        if media_match:
            book_id = media_match.group(1)
            page_num = int(media_match.group(2))
            filename = media_match.group(3)
            kind_map = {
                "image.png": ("image", "image/png"),
                "motion.gif": ("motion", "image/gif"),
                "voice.wav": ("voice", "audio/wav"),
            }
            kind, mime = kind_map[filename]
            media_bytes = load_page_media_bytes(book_id, page_num, kind)
            if media_bytes is None:
                self._send_json({"error": "Media artifact not found"}, status=HTTPStatus.NOT_FOUND)
                return
            if kind == "voice" and media_bytes[:4] != b"RIFF":
                mime = "audio/mpeg"
            self._send_bytes(media_bytes, mime)
            return

        # Standalone HTML export or shared book viewer: /book/<book_id> or /api/books/<book_id>/export.html
        share_match = re.match(r"^(?:/book/([a-zA-Z0-9_-]+)|/api/books/([a-zA-Z0-9_-]+)/export\.html)$", route)
        if share_match:
            book_id = share_match.group(1) or share_match.group(2)
            book = get_book(book_id, include_data_urls=True)
            if book is None:
                self._send_json({"error": f"Book '{book_id}' not found"}, status=HTTPStatus.NOT_FOUND)
                return
            html_str = render_shareable_book_html(book)
            headers = None
            if route.endswith("/export.html"):
                headers = {"Content-Disposition": f'attachment; filename="{book_id}.html"'}
            self._send_bytes(html_str.encode("utf-8"), "text/html; charset=utf-8", extra_headers=headers)
            return

        # Book detail route: /api/books/<book_id>
        book_match = re.match(r"^/api/books/([a-zA-Z0-9_-]+)$", route)
        if book_match:
            book_id = book_match.group(1)
            book = get_book(book_id, include_data_urls=True)
            if book is None:
                self._send_json({"error": f"Book '{book_id}' not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(book)
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        """Handle POST requests for creating books, running voice turns, and publishing."""
        route = urlparse(self.path).path
        payload, err = self._read_json_body()
        if err or payload is None:
            self._send_json(
                {"status": "execution_blocked", "reasons": [err or "Malformed JSON request"]},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        if route == "/api/books":
            preset_id = str(payload.get("preset_id") or DEFAULT_PRESET_ID).strip()
            custom_title = payload.get("custom_title")
            custom_prompt = payload.get("custom_prompt")
            prefer_live = bool(payload.get("prefer_live_api", False))
            lang = str(payload.get("lang") or "ko").strip()
            book = create_book(
                preset_id=preset_id,
                custom_title=str(custom_title) if custom_title else None,
                custom_prompt=str(custom_prompt) if custom_prompt else None,
                prefer_live_api=prefer_live,
                lang=lang,
            )
            self._send_json(book, status=HTTPStatus.OK)
            return

        turn_match = re.match(r"^/api/books/([a-zA-Z0-9_-]+)/turn$", route)
        if turn_match:
            book_id = turn_match.group(1)
            user_utterance = str(payload.get("user_utterance") or payload.get("user_prompt") or "").strip()
            prefer_live = bool(payload.get("prefer_live_api", False))
            model_tier = str(payload.get("model_tier") or payload.get("fast_mode") or "argon_low").strip()
            raw_lang = payload.get("lang")
            turn_lang = str(raw_lang).strip() if raw_lang else None
            result = run_book_turn(
                book_id=book_id,
                user_utterance=user_utterance,
                prefer_live_api=prefer_live,
                model_tier=model_tier,
                lang=turn_lang,
            )
            self._send_json(result, status=HTTPStatus.OK)
            return

        finish_match = re.match(r"^/api/books/([a-zA-Z0-9_-]+)/finish$", route)
        if finish_match:
            book_id = finish_match.group(1)
            result = finish_and_publish_book(book_id)
            self._send_json(result, status=HTTPStatus.OK)
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Suppress default stderr HTTP logs in favor of structured audit logger."""
        return


def run() -> None:
    """Start the WonderBook Live HTTP server on `0.0.0.0:$PORT`."""
    port = int(os.environ.get("PORT", str(DEFAULT_PORT)))
    log_function_call("run", {"host": "0.0.0.0", "port": port})
    server = ThreadingHTTPServer(("0.0.0.0", port), WonderBookRequestHandler)
    print(f"WonderBook Live server listening on http://0.0.0.0:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    run()
