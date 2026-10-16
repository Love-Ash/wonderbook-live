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

    def test_hcl_science_book_one_by_one_and_publish(self) -> None:
        """Test HCl science lab 1-by-1 independent experiments and book publishing."""
        book = create_book(preset_id="science_hcl_lab", prefer_live_api=False)
        book_id = book["book_id"]
        self.assertEqual(book["mode"], "science_lab")
        self.assertEqual(book["page_count"], 1)

        # Turn 2: Drop a metal key in HCl -> 1-by-1 independent experiment ["metal_key"]
        turn_key = run_book_turn(book_id, "열쇠(금속)를 넣으면 어떻게 되나요?", prefer_live_api=False, model_tier="argon_low")
        self.assertEqual(turn_key["status"], "turn_completed")
        p2 = turn_key["latest_page"]
        self.assertEqual(p2["page_number"], 2)
        self.assertEqual(p2["visual_state"]["item_key"], "metal_key")
        self.assertEqual(p2["visual_state"]["accumulated_items"], ["metal_key"])
        self.assertTrue(p2["orchestration_metrics"]["parallel_media_execution"])
        self.assertTrue(p2["image_data_url"].startswith("data:image/png;base64,"))
        self.assertTrue(p2["motion_data_url"].startswith("data:image/gif;base64,"))
        self.assertTrue(
            p2["voice_data_url"].startswith(("data:audio/wav;base64,", "data:audio/mpeg;base64,"))
        )

        # Turn 3: Drop an apple in HCl -> 1-by-1 independent experiment ["apple"]
        turn_apple = run_book_turn(book_id, "사과를 넣으면 어떻게 되나요?", prefer_live_api=False, model_tier="flash_ultra")
        self.assertEqual(turn_apple["status"], "turn_completed")
        self.assertEqual(turn_apple["latest_page"]["visual_state"]["item_key"], "apple")
        self.assertEqual(turn_apple["latest_page"]["visual_state"]["accumulated_items"], ["apple"])

        # Turn 4: Drop a cookie/snack in HCl -> 1-by-1 independent experiment ["cookie_snack"]
        turn_cookie = run_book_turn(book_id, "과자를 넣으면 어떻게 되나요?", prefer_live_api=False, model_tier="argon_medium")
        self.assertEqual(turn_cookie["status"], "turn_completed")
        self.assertEqual(turn_cookie["latest_page"]["visual_state"]["item_key"], "cookie_snack")
        self.assertEqual(
            turn_cookie["latest_page"]["visual_state"]["accumulated_items"],
            ["cookie_snack"],
        )

        # Turn 5: Spray water in HCl -> 1-by-1 independent experiment ["water_spray"]
        turn_water = run_book_turn(book_id, "물을 뿌리면 어떻게 되나요?", prefer_live_api=False)
        self.assertEqual(turn_water["status"], "turn_completed")
        self.assertEqual(turn_water["latest_page"]["visual_state"]["item_key"], "water_spray")
        self.assertEqual(
            turn_water["latest_page"]["visual_state"]["accumulated_items"],
            ["water_spray"],
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

    def test_storybook_cumulative_continuity_and_cause_effect_queries(self) -> None:
        """Verify storybooks accumulate items & change chapter images, and freeform cause-and-effect questions work."""
        # 1. Storybook cumulative continuity + moon disappearing cause-and-effect
        tiger_book = create_book(preset_id="fairy_tiger_rabbit", prefer_live_api=False)
        tb_id = tiger_book["book_id"]
        p1_img = tiger_book["pages"][0]["image_data_url"]

        t2 = run_book_turn(tb_id, "호랑이에게 달콤한 별빛 꿀떡을 주면 어떻게 되나요?", prefer_live_api=False)
        p2 = t2["latest_page"]
        self.assertEqual(p2["visual_state"]["item_key"], "rice_cake")
        self.assertEqual(p2["visual_state"]["accumulated_items"], ["rice_cake"])
        self.assertNotEqual(p2["image_data_url"], p1_img)

        t3 = run_book_turn(tb_id, "달이 사라지면 어떻게 되어요?", prefer_live_api=False)
        p3 = t3["latest_page"]
        self.assertEqual(p3["visual_state"]["item_key"], "custom_story")
        self.assertEqual(p3["visual_state"]["accumulated_items"], ["rice_cake", "custom_story"])
        self.assertNotEqual(p3["image_data_url"], p2["image_data_url"])
        self.assertIn("달", p3["page_title"] + " ".join(p3["story_paragraphs"]))

        # 2. Space Shuttle 1-by-1 experiment + dynamite explosion cause-and-effect
        shuttle_book = create_book(preset_id="science_space_shuttle", prefer_live_api=False)
        sb_id = shuttle_book["book_id"]
        sp1_img = shuttle_book["pages"][0]["image_data_url"]

        st2 = run_book_turn(sb_id, "우주왕복선안에서 다이너마이트 터트리면 어떻게될까?", prefer_live_api=False)
        sp2 = st2["latest_page"]
        self.assertEqual(sp2["visual_state"]["item_key"], "custom_science")
        self.assertEqual(sp2["visual_state"]["accumulated_items"], ["custom_science"])
        self.assertNotEqual(sp2["image_data_url"], sp1_img)
        combined_text = sp2["page_title"] + " " + " ".join(sp2["story_paragraphs"]) + " " + sp2["science_or_moral_note"]
        self.assertTrue("충격파" in combined_text or "산화제" in combined_text or "기압" in combined_text)

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
