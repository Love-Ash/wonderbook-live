"""End-to-end functional tests for WonderBook Live orchestration skeleton."""

from __future__ import annotations

import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

from app import WonderBookRequestHandler, get_system_status
from book_store import create_book, finish_and_publish_book, run_book_turn
from config import LOG_FILE_PATH


class WonderBookLiveSkeletonTest(unittest.TestCase):
    """Verify multi-agent orchestration, visual continuity chaining, media generation, and book publishing."""

    def test_system_status_and_models(self) -> None:
        """System status should expose upgraded higher-tier models (gemini-4-argon, gemini-3.8-flash, Nano Banana Pro, Veo)."""
        status = get_system_status()
        self.assertEqual(status["status"], "ok")
        models = status["models"]
        self.assertEqual(models["fast_story_model"], "gemini-3.8-flash")
        self.assertEqual(models["primary_model"], "gemini-4-argon")
        self.assertEqual(models["fallback_model"], "gemini-3.8-flash")
        self.assertEqual(models["nano_banana_pro_model"], "gemini-3-pro-image-preview")
        self.assertEqual(models["nano_banana_model"], "gemini-2.5-flash-image")
        self.assertEqual(models["veo_model"], "veo-2.0-generate-001")

    def test_hcl_science_book_multi_turn_and_publish(self) -> None:
        """Test HCl science lab flow with cumulative continuity: metal key -> apple -> cookie -> water -> publish book."""
        book = create_book(preset_id="science_hcl_lab", prefer_live_api=False)
        book_id = book["book_id"]
        self.assertEqual(book["mode"], "science_lab")
        self.assertEqual(book["page_count"], 1)

        # Turn 2: Drop a metal key in HCl
        turn_key = run_book_turn(book_id, "열쇠(금속)를 넣으면 어떻게 되나요?", prefer_live_api=False, model_tier="argon_low")
        self.assertEqual(turn_key["status"], "turn_completed")
        p2 = turn_key["latest_page"]
        self.assertEqual(p2["page_number"], 2)
        self.assertEqual(p2["visual_state"]["item_key"], "metal_key")
        self.assertEqual(p2["visual_state"]["accumulated_items"], ["metal_key"])
        self.assertTrue(p2["orchestration_metrics"]["chained_from_previous_page"])
        self.assertTrue(p2["orchestration_metrics"]["parallel_media_execution"])
        self.assertTrue(p2["image_data_url"].startswith("data:image/png;base64,"))
        self.assertTrue(p2["motion_data_url"].startswith("data:image/gif;base64,"))
        self.assertTrue(
            p2["voice_data_url"].startswith(("data:audio/wav;base64,", "data:audio/mpeg;base64,"))
        )

        # Turn 3: Drop an apple in HCl -> should accumulate ["metal_key", "apple"]
        turn_apple = run_book_turn(book_id, "사과를 넣으면 어떻게 되나요?", prefer_live_api=False, model_tier="flash_ultra")
        self.assertEqual(turn_apple["status"], "turn_completed")
        self.assertEqual(turn_apple["latest_page"]["visual_state"]["item_key"], "apple")
        self.assertEqual(turn_apple["latest_page"]["visual_state"]["accumulated_items"], ["metal_key", "apple"])

        # Turn 4: Drop a cookie/snack in HCl -> should accumulate ["metal_key", "apple", "cookie_snack"]
        turn_cookie = run_book_turn(book_id, "과자를 넣으면 어떻게 되나요?", prefer_live_api=False, model_tier="argon_medium")
        self.assertEqual(turn_cookie["status"], "turn_completed")
        self.assertEqual(turn_cookie["latest_page"]["visual_state"]["item_key"], "cookie_snack")
        self.assertEqual(
            turn_cookie["latest_page"]["visual_state"]["accumulated_items"],
            ["metal_key", "apple", "cookie_snack"],
        )

        # Turn 5: Spray water in HCl -> should accumulate ["metal_key", "apple", "cookie_snack", "water_spray"]
        turn_water = run_book_turn(book_id, "물을 뿌리면 어떻게 되나요?", prefer_live_api=False)
        self.assertEqual(turn_water["status"], "turn_completed")
        self.assertEqual(turn_water["latest_page"]["visual_state"]["item_key"], "water_spray")
        self.assertEqual(
            turn_water["latest_page"]["visual_state"]["accumulated_items"],
            ["metal_key", "apple", "cookie_snack", "water_spray"],
        )
        self.assertEqual(turn_water["book"]["page_count"], 5)

        # Empty utterance should be blocked cleanly
        blocked = run_book_turn(book_id, "   ", prefer_live_api=False)
        self.assertEqual(blocked["status"], "execution_blocked")

        # Finish & publish book into shareable flipbook
        pub = finish_and_publish_book(book_id)
        self.assertEqual(pub["status"], "book_published")
        self.assertEqual(pub["storage_proof"]["page_count"], 5)
        self.assertTrue(LOG_FILE_PATH.exists())

    def test_http_endpoints_and_shared_book_viewer(self) -> None:
        """Verify live HTTP server routes: /healthz, /api/books, /turn, /finish, and /book/<id>."""
        server = ThreadingHTTPServer(("127.0.0.1", 0), WonderBookRequestHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{port}"

        try:
            with urlopen(f"{base_url}/healthz") as resp:
                self.assertEqual(resp.status, 200)
                health = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(health["status"], "ok")

            with urlopen(f"{base_url}/static/assets/nano_banana/fox_base.png") as resp:
                self.assertEqual(resp.status, 200)
                self.assertGreater(len(resp.read()), 500_000)

            # Create a storybook via HTTP POST
            req_create = Request(
                f"{base_url}/api/books",
                data=json.dumps({"preset_id": "fairy_starlight_fox"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req_create) as resp:
                book = json.loads(resp.read().decode("utf-8"))
                book_id = book["book_id"]

            # Run a story turn via HTTP POST
            req_turn = Request(
                f"{base_url}/api/books/{book_id}/turn",
                data=json.dumps(
                    {"user_utterance": "문이 열리고 아기 용이 비눗방울을 뿜으면 어때요?", "prefer_live_api": False}
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req_turn) as resp:
                turn_data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(turn_data["status"], "turn_completed")
                self.assertEqual(turn_data["latest_page"]["page_number"], 2)

            # Publish book via HTTP POST
            req_fin = Request(
                f"{base_url}/api/books/{book_id}/finish",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req_fin) as resp:
                fin_data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(fin_data["status"], "book_published")

            # Fetch shared flipbook HTML
            with urlopen(f"{base_url}/book/{book_id}") as resp:
                html_text = resp.read().decode("utf-8")
                self.assertIn("Published WonderBook", html_text)
                self.assertIn("Page 2", html_text)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
