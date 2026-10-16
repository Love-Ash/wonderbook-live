"""Upgraded Story & Science Director Agent with Strict Scenario Isolation, Pure KO/EN Localization, and Character Voice Personas.

This module implements the narrative, scientific reasoning, and cumulative continuity
director for WonderBook Live:
1. Upgraded Model Hierarchy (`gemini-4-argon` & `gemini-3.8-flash`):
   - Supports configurable `model_tier`:
     - `argon_low` (Default): `gemini-4-argon` with `thinking_level="low"` for fast,
       high-intelligence interactive story/science turns, falling back to `gemini-3.8-flash`.
     - `flash_ultra`: `gemini-3.8-flash` first for maximum speed, falling back to `gemini-4-argon`.
     - `argon_medium`: `gemini-4-argon` with `thinking_level="medium"` for deep scientific explanations.
2. Strict Scenario Isolation (`preset_id` / `mode`):
   - `fairy_starlight_fox` (`mode == "storybook"`): 100% fairy-tale forest narrative with Lumi the
     Starlight Fox (never mixes chemistry, beakers, or HCl into the fairy tale, including when
     putting a cookie into the golden oak keyhole or summoning star-candy rain).
   - `science_space_kitchen`: 100% Zero-Gravity Space Station narrative with Captain Orbit.
   - `science_hcl_lab`: 100% Hydrochloric Acid Beaker Lab narrative with Dr. Nova the Owl Professor.
3. Strict Language Separation (`lang="ko"` vs `lang="en"`) & Character Voice Personas:
   - Zero mixed Korean/English sentences in titles, paragraphs, dialogues, notes, labels, or follow-ups.
   - Expressive character vocal acting (`owl_professor`, `baby_fox`, `space_captain`).
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from config import (
    FALLBACK_MODEL,
    FAST_STORY_MODEL,
    MAX_GENAI_RETRIES,
    PRIMARY_MODEL,
    RETRY_BASE_DELAY_SECONDS,
    THINKING_LEVEL,
    get_gemini_api_key,
    get_localized_preset,
    log_function_call,
    log_genai_call,
)

ITEM_DISPLAY_LABELS_KO: dict[str, str] = {
    "metal_key": "금속 열쇠",
    "apple": "빨간 사과",
    "cookie_snack": "초코칩 과자",
    "water_spray": "물 분무기",
    "baking_soda": "베이킹소다",
    "egg_shell": "달걀 껍데기",
    "gold_ring": "24K 순금 반지",
    "zero_g_candle": "무중력 촛불",
    "magic_door": "빛나는 참나무 문",
    "dragon_bubbles": "아기 용과 비눗방울",
    "rainbow_spaceship": "무지개 구름 비행선",
    "star_candy": "별사탕 소나기",
    "rice_cake": "달빛 꿀떡",
    "roasted_pebble": "구운 모닥불 조약돌",
    "tiger_rabbit_path": "달빛 소나무 고갯길",
    "zero_g_cabin": "무중력 실험 챔버",
    "custom_science": "특별 실험 재료",
    "custom_story": "새로운 마법 상상",
}

ITEM_DISPLAY_LABELS_EN: dict[str, str] = {
    "metal_key": "Metal Key",
    "apple": "Red Apple",
    "cookie_snack": "Chocolate-Chip Cookie",
    "water_spray": "Water Spray",
    "baking_soda": "Baking Soda",
    "egg_shell": "Eggshell",
    "gold_ring": "24K Pure Gold Ring",
    "zero_g_candle": "Zero-G Candle Flame",
    "magic_door": "Glowing Oak Door",
    "dragon_bubbles": "Baby Dragon & Bubbles",
    "rainbow_spaceship": "Rainbow Cloud Airship",
    "star_candy": "Star-Candy Shower",
    "rice_cake": "Golden Honey Rice Cake",
    "roasted_pebble": "Roasted Campfire Pebble",
    "tiger_rabbit_path": "Moonlit Mountain Path",
    "zero_g_cabin": "Zero-Gravity Chamber",
    "custom_science": "Custom Lab Reagent",
    "custom_story": "Magical Story Twist",
}

# Default alias kept for backward compatibility
ITEM_DISPLAY_LABELS: dict[str, str] = ITEM_DISPLAY_LABELS_KO

_HANGUL_RE = re.compile(r"[가-힣]")
_ASCII_WORDS_RE = re.compile(r"[A-Za-z]{2,}")


def _resolve_lang(book_state: dict[str, Any], user_utterance: str, lang: str | None = None) -> str:
    """Resolve effective language (`'ko'` or `'en'`) from explicit argument, book state, or utterance."""
    if lang and str(lang).strip():
        return "en" if str(lang).strip().lower().startswith("en") else "ko"

    book_lang = str(book_state.get("lang") or "").strip().lower()
    if book_lang.startswith("en"):
        return "en"
    if book_lang.startswith("ko"):
        # If book is set to ko, check if user_utterance is purely English only when book_lang wasn't explicitly locked
        return "ko"

    # Auto-detect if user_utterance is purely ASCII English letters with no Hangul
    cleaned = (user_utterance or "").strip()
    if cleaned and not _HANGUL_RE.search(cleaned) and len(_ASCII_WORDS_RE.findall(cleaned)) >= 2:
        return "en"
    return "ko"


def get_item_display_label(item_key: str, lang: str = "ko", fallback_text: str = "") -> str:
    """Return a 100% pure Korean (`ko`) or 100% pure English (`en`) item display label."""
    table = ITEM_DISPLAY_LABELS_EN if lang == "en" else ITEM_DISPLAY_LABELS_KO
    if item_key in table:
        return table[item_key]
    if fallback_text:
        return fallback_text[:28]
    return "Magical Element" if lang == "en" else "신기한 재료"


def _resolve_character_persona(book_state: dict[str, Any], preset_id: str, mode: str) -> str:
    """Return the canonical character voice persona identifier for the current book."""
    explicit = str(book_state.get("character_persona_voice") or "").strip()
    if explicit:
        return explicit
    if mode == "storybook" or preset_id in ("fairy_starlight_fox", "fairy_tiger_rabbit") or "fairy" in preset_id:
        return "baby_fox"
    if preset_id in ("science_space_kitchen", "science_space_shuttle") or "space" in preset_id:
        return "space_captain"
    return "owl_professor"


def _build_system_instruction(
    lang: str,
    mode: str,
    preset_id: str,
    character_name: str,
    character_persona_voice: str,
    next_page_num: int,
) -> str:
    """Construct the system instruction enforcing strict language separation, character persona, and scenario isolation."""
    if preset_id == "fairy_tiger_rabbit":
        scenario_rule = (
            "SCENARIO ISOLATION (KOREAN FOLKTALE STORYBOOK): This book takes place 100% on the moonlit pine mountain path "
            "with Dalkong the Clever Rabbit and the friendly Striped Mountain Tiger. NEVER mention hydrochloric acid, beakers, "
            "or laboratory chemistry!"
        )
    elif mode == "storybook" or preset_id == "fairy_starlight_fox" or "fairy" in preset_id:
        scenario_rule = (
            "SCENARIO ISOLATION (FAIRY-TALE STORYBOOK): This book takes place 100% inside an enchanted fairy-tale world. "
            "NEVER mention hydrochloric acid, beakers, pH meters, CO2, or laboratory chemistry! Even if the user drops a "
            "cookie or candy, treat it as a magical fairy-tale event!"
        )
    elif preset_id in ("science_space_kitchen", "science_space_shuttle") or "space" in preset_id:
        scenario_rule = (
            "SCENARIO ISOLATION (ZERO-G SPACE / SPACE SHUTTLE LAB): This book takes place 100% aboard a Zero-Gravity "
            "Space Shuttle Orbiter or Space Station 400km above Earth. Focus on microgravity physics, surface tension spheres, "
            "spherical diffusion flames, and floating objects in orbit."
        )
    else:
        scenario_rule = (
            "SCENARIO ISOLATION (HYDROCHLORIC ACID LAB): This book takes place 100% inside Dr. Nova's Hydrochloric Acid "
            "(1.0M HCl) Beaker Laboratory. Focus on real chemical reactions, gas evolution, pH shifts, and exothermic heat."
        )

    if lang == "en":
        if character_persona_voice == "baby_fox":
            persona_rule = (
                f"CHARACTER VOICE PERSONA ({character_name}): Speak as an adorable, playful, curious baby starlight fox! "
                "Naturally include cute fox vocal expressions like 'Yip-yip!', 'Hehe~', 'Ooh, look at my starry tail!', 'Yip~!'."
            )
        elif character_persona_voice == "space_captain":
            persona_rule = (
                f"CHARACTER VOICE PERSONA ({character_name}): Speak as an energetic, warm orbital astronaut commander! "
                "Naturally include space mission expressions like 'Copy that, cadet!', 'Whoa, check out that zero-G drift!', 'Over!'."
            )
        else:
            persona_rule = (
                f"CHARACTER VOICE PERSONA ({character_name}): Speak as a warm, wise, enthusiastic clockwork owl science professor! "
                "Naturally include charming owl expressions like 'Hoo-hoo!', 'Oh my feathers!', 'Splendid observation, young scientist!'."
            )
        lang_rule = (
            "STRICT LANGUAGE RULE (100% PURE ENGLISH ONLY): Write `page_title`, `story_paragraphs`, `character_dialogue`, "
            "`science_or_moral_note`, `visual_state.item_label`, and `suggested_followups` in 100% PURE ENGLISH. "
            "Do NOT include any Korean characters or bilingual parentheses anywhere!"
        )
        title_example = f"Chapter {next_page_num} — Vivid Pure English Title"
    else:
        if character_persona_voice == "baby_fox":
            persona_rule = (
                f"CHARACTER VOICE PERSONA ({character_name}): 귀엽고 호기심 가득한 아기 별빛 여우 루미의 말투로 연기하세요! "
                "대사(`character_dialogue`)에 '콩콩!', '헤헤~', '우와앙!', '낑~!' 같은 사랑스러운 아기 여우 추임새를 자연스럽게 넣으세요."
            )
        elif character_persona_voice == "space_captain":
            persona_rule = (
                f"CHARACTER VOICE PERSONA ({character_name}): 활기차고 든든한 우주정거장 사령관 오르빗 대장의 말투로 연기하세요! "
                "대사(`character_dialogue`)에 '오버!', '우주 실험실 보고!', '로저!' 같은 생생한 우주 비행사 표현을 자연스럽게 넣으세요."
            )
        else:
            persona_rule = (
                f"CHARACTER VOICE PERSONA ({character_name}): 다정하고 박식한 부엉이 과학자 노바 박사님의 말투로 연기하세요! "
                "대사(`character_dialogue`)에 '부엉~ 부엉!', '호오~!', '부엉!' 같은 정겨운 부엉이 추임새를 자연스럽게 넣으세요."
            )
        lang_rule = (
            "STRICT LANGUAGE RULE (100% 순수 한국어 전용): `page_title`, `story_paragraphs`, `character_dialogue`, "
            "`science_or_moral_note`, `visual_state.item_label`, `suggested_followups`를 100% 순수 한국어로만 작성하세요! "
            "절대로 영어 문장이나 영문 괄호 병기를 섞지 마세요! (단, 과학 노트의 화학식 기호 Fe, HCl, H₂, CO₂ 등은 허용됩니다.)"
        )
        title_example = f"{next_page_num}장 — 생생한 순수 한국어 제목"

    return f"""You are the Story & Science Continuity Director for WonderBook Live.
{scenario_rule}
{lang_rule}
{persona_rule}

Given the book's mode, character bible, cumulative items already in the scene (`accumulated_items`), previous pages, and the user's spoken question (`user_utterance`), generate the NEXT page as valid JSON matching this schema:
{{
  "page_title": "{title_example}",
  "story_paragraphs": [
    "Paragraph 1: Explicitly bridge the previous page's scene and existing items to the user's new spoken question with rich sensory detail.",
    "Paragraph 2: Vividly explain how the new item physically, chemically, or magically interacts with the items ALREADY inside the scene."
  ],
  "character_dialogue": "2-3 expressive, natural sentences spoken aloud by the character directly to the child in the required persona voice.",
  "science_or_moral_note": "Clear scientific explanation or storybook continuity note in the target language.",
  "visual_state": {{
    "item_key": "metal_key | apple | cookie_snack | water_spray | baking_soda | egg_shell | gold_ring | zero_g_candle | dragon_bubbles | rainbow_spaceship | star_candy | custom_science | custom_story",
    "item_label": "Short localized label of the newly added item/reaction",
    "liquid_color": [R, G, B],
    "ph_level": 1.2,
    "temperature_c": 26.5,
    "bubbles_intensity": 0.0,
    "steam_intensity": 0.0,
    "foam_intensity": 0.0
  }},
  "nano_banana_prompt": "Detailed English image-to-image continuity prompt referencing the PREVIOUS page image and listing ALL accumulated items that must remain visible together in the scene.",
  "veo_motion_prompt": "Detailed English motion prompt for Veo describing the animation.",
  "suggested_followups": [
    "Question 1 in target language",
    "Question 2 in target language",
    "Question 3 in target language",
    "Question 4 in target language"
  ]
}}
"""


def _extract_json_dict(raw_text: str) -> dict[str, Any]:
    """Parse a JSON object from model text output."""
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON decode failed: {exc}") from exc

    raise ValueError("Model response did not contain a valid JSON object.")


def _call_story_llm_chain(
    prompt_text: str,
    model_tier: str = "argon_low",
    timeout_s: float = 6.5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Invoke the upgraded Gemini model chain in parallel with fast failover."""
    import concurrent.futures

    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    from google import genai
    from google.genai import types

    if model_tier == "flash_ultra":
        model_chain = [(FAST_STORY_MODEL, "low"), (PRIMARY_MODEL, "low")]
    elif model_tier == "argon_medium":
        model_chain = [(FAST_STORY_MODEL, "low"), (PRIMARY_MODEL, "medium")]
    else:
        model_chain = [(FAST_STORY_MODEL, "low"), (PRIMARY_MODEL, THINKING_LEVEL), (FALLBACK_MODEL, "low")]

    deduped_chain: list[tuple[str, str]] = []
    for m_name, t_lvl in model_chain:
        if m_name and (m_name, t_lvl) not in deduped_chain:
            deduped_chain.append((m_name, t_lvl))

    def _call_single_model(model_name: str, thinking_lvl: str) -> tuple[dict[str, Any], dict[str, Any]]:
        client = genai.Client(api_key=api_key)
        cfg_summary = {
            "response_mime_type": "application/json",
            "thinking_level": thinking_lvl,
            "model_tier": model_tier,
            "parallel_race": True,
        }
        try:
            gen_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_level=thinking_lvl),
            )
            response = client.models.generate_content(
                model=model_name,
                contents=prompt_text,
                config=gen_config,
            )
            raw_text = getattr(response, "text", "") or ""
            log_genai_call(
                model=model_name,
                prompt_summary=prompt_text[:450],
                config_summary=cfg_summary,
                output_summary=raw_text[:600],
                status="ok",
            )
            parsed = _extract_json_dict(raw_text)
            return parsed, {
                "engine_mode": "gemini_live_parallel",
                "model_used": model_name,
                "thinking_level": thinking_lvl,
                "model_tier": model_tier,
                "attempts": 1,
                "fallback_used": False,
            }
        except Exception as exc:
            err_str = str(exc)
            log_genai_call(
                model=model_name,
                prompt_summary=prompt_text[:450],
                config_summary=cfg_summary,
                output_summary=err_str[:450],
                status="error",
            )
            raise

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(deduped_chain))
    futures = {
        executor.submit(_call_single_model, m_name, t_lvl): (m_name, t_lvl)
        for m_name, t_lvl in deduped_chain
    }
    errors: list[str] = []
    try:
        for fut in concurrent.futures.as_completed(futures, timeout=timeout_s):
            m_name, _ = futures[fut]
            try:
                return fut.result()
            except Exception as exc:
                errors.append(f"{m_name}: {str(exc)[:140]}")
    except concurrent.futures.TimeoutError:
        errors.append(f"Timeout after {timeout_s}s")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    raise RuntimeError(f"All parallel story model attempts failed: {errors}")


def _detect_item_key(user_utterance: str, mode: str = "science_lab", preset_id: str = "") -> str:
    """Classify user utterance into a canonical archetype key with strict scenario isolation."""
    text = (user_utterance or "").lower()

    # 0. Check custom cause-and-effect / dramatic "what-if" triggers FIRST so questions like
    # "달이 사라지면 어떻게 되어요?" or "우주왕복선안에서 다이너마이트 터트리면 어떻게될까?" never misclassify!
    is_storybook = mode == "storybook" or preset_id in ("fairy_starlight_fox", "fairy_tiger_rabbit") or "fairy" in preset_id
    if is_storybook:
        if any(
            k in text
            for k in (
                "사라지",
                "없어지",
                "꺼지",
                "어두워",
                "터트리",
                "터지",
                "폭발",
                "다이너마이트",
                "폭탄",
                "폭죽",
                "유령",
                "괴물",
                "늑대",
                "도깨비",
                "얼어",
                "녹아",
                "변신",
                "바뀌",
                "울면",
                "눈물",
                "잠들",
                "노래",
                "춤추",
                "disappear",
                "vanish",
                "dark",
                "dynamite",
                "explode",
                "ghost",
                "monster",
                "wolf",
                "freeze",
                "melt",
                "transform",
                "cry",
                "sing",
                "dance",
            )
        ):
            return "custom_story"

        if any(
            k in text
            for k in ("조약돌", "돌멩이", "모닥불", "뜨거운 돌", "구운 돌", "pebble", "stone", "campfire", "roast")
        ):
            return "roasted_pebble"
        if any(
            k in text
            for k in ("꿀떡", "떡", "찰떡", "송편", "인절미", "rice cake", "ricecake", "tteok", "honey cake")
        ):
            return "rice_cake"
        if any(
            k in text
            for k in (
                "별사탕",
                "사탕",
                "별빛 사탕",
                "캔디",
                "은하수",
                "star candy",
                "star-candy",
                "candy",
                "sweet",
                "shooting star",
            )
        ):
            return "star_candy"
        if any(
            k in text
            for k in ("용", "드래곤", "비눗방울", "아기 용", "거품", "포포", "dragon", "bubble", "door", "문")
        ):
            return "dragon_bubbles"
        if any(
            k in text
            for k in (
                "비행선",
                "우주선",
                "무지개",
                "구름",
                "달나라",
                "날아",
                "하늘",
                "spaceship",
                "airship",
                "rainbow",
                "cloud",
                "fly",
            )
        ):
            return "rainbow_spaceship"
        if any(
            k in text
            for k in ("과자", "쿠키", "초코칩", "열쇠구멍", "cookie", "biscuit", "keyhole", "snack")
        ):
            return "cookie_snack"
        return "custom_story"

    # 2. Science Lab modes (`science_hcl_lab`, `science_space_kitchen`, `science_space_shuttle`):
    if any(
        k in text
        for k in (
            "다이너마이트",
            "폭탄",
            "폭발",
            "터트리",
            "터지",
            "폭죽",
            "화약",
            "로켓",
            "미사일",
            "구멍",
            "깨지",
            "창문",
            "문 열",
            "문을 열",
            "진공",
            "블랙홀",
            "충돌",
            "소행성",
            "운석",
            "번개",
            "전기",
            "레이저",
            "전자레인지",
            "눈물",
            "울면",
            "재채기",
            "방귀",
            "콜라",
            "사이다",
            "멘토스",
            "나트륨",
            "칼륨",
            "드라이아이스",
            "액체질소",
            "용암",
            "마그마",
            "사라지",
            "없어지",
            "dynamite",
            "bomb",
            "explode",
            "explosion",
            "firework",
            "gunpowder",
            "breach",
            "hole",
            "window",
            "vacuum",
            "black hole",
            "meteor",
            "asteroid",
            "lightning",
            "laser",
            "microwave",
            "cry",
            "tear",
            "sneeze",
            "fart",
            "cola",
            "mentos",
            "sodium",
            "liquid nitrogen",
            "lava",
            "disappear",
            "vanish",
        )
    ):
        return "custom_science"

    if any(k in text for k in ("별사탕", "star candy", "star-candy")):
        return "star_candy"
    if any(
        k in text
        for k in ("양초", "촛불", "성냥", "라이터", "불을 붙", "불 붙", "불꽃", "candle", "flame", "match", "lighter")
    ):
        return "zero_g_candle"
    if any(
        k in text
        for k in ("열쇠", "금속", "철", "동전", "못", "구리", "아연", "key", "metal", "iron", "zinc", "coin", "nail")
    ):
        return "metal_key"
    if any(k in text for k in ("사과", "과일", "딸기", "바나나", "apple", "fruit", "berry")):
        return "apple"
    if any(
        k in text
        for k in ("베이킹소다", "탄산수소", "소다", "화산", "발포", "가루", "baking soda", "soda", "fizzy", "bicarbonate")
    ):
        return "baking_soda"
    if any(
        k in text
        for k in ("과자", "쿠키", "스낵", "초콜릿", "초코칩", "빵", "비스킷", "cookie", "snack", "cracker", "biscuit", "bread")
    ):
        return "cookie_snack"
    if any(
        k in text
        for k in ("물", "뿌리", "분무", "얼음", "수돗물", "증류수", "water", "spray", "pour", "dilute", "ice", "droplet")
    ):
        return "water_spray"
    if any(k in text for k in ("달걀", "계란", "껍질", "껍데기", "조개", "분필", "egg", "shell", "chalk", "calcium")):
        return "egg_shell"
    if any(k in text for k in ("금반지", "순금", "금화", "다이아몬드", "반지", "gold", "ring", "treasure")):
        return "gold_ring"
    if any(k in text for k in ("용", "드래곤", "비눗방울", "포포", "dragon", "bubble")):
        return "dragon_bubbles"
    if any(k in text for k in ("구름", "우주선", "무지개", "비행선", "rainbow", "cloud", "spaceship", "airship")):
        return "rainbow_spaceship"
    return "custom_science"


def _classify_utterance_topic(user_utterance: str, mode: str, preset_id: str = "") -> str:
    """Backward-compatible wrapper around `_detect_item_key`."""
    return _detect_item_key(user_utterance, mode=mode, preset_id=preset_id)


def _compute_cumulative_state(
    prev_visual_state: dict[str, Any],
    new_topic: str,
    user_utterance: str,
    lang: str = "ko",
    mode: str = "science_lab",
) -> tuple[list[str], list[str], float, float]:
    """Update items list, localized labels (`ko` or `en`), pH level, and temperature (°C).

    In `storybook` mode, story items accumulate across turns so the fairy-tale world builds up.
    In `science_lab` mode, each experiment is tested cleanly 1-by-1 without awkward multi-item piling.
    """
    if mode == "science_lab":
        cur_items = [new_topic] if new_topic not in ("none", "") else []
        labels = [
            get_item_display_label(it, lang=lang, fallback_text=user_utterance[:24])
            for it in cur_items
        ]
        base_ph = 1.0
        base_temp = 22.0
    else:
        prev_items = list(prev_visual_state.get("accumulated_items") or [])
        if new_topic in ("custom_story", "custom_science"):
            prev_items.append(new_topic)
        elif new_topic not in ("none", "") and new_topic not in prev_items:
            prev_items.append(new_topic)
        cur_items = prev_items
        labels = [
            get_item_display_label(it, lang=lang, fallback_text=user_utterance[:24])
            for it in cur_items
        ]
        base_ph = float(prev_visual_state.get("ph_level", 1.0))
        base_temp = float(prev_visual_state.get("temperature_c", 22.0))

    if new_topic == "metal_key":
        new_ph = min(6.8, round(base_ph + 0.3, 2))
        new_temp = round(base_temp + 6.5, 1)
    elif new_topic == "apple":
        new_ph = min(6.8, round(base_ph + 0.15, 2))
        new_temp = round(base_temp + 0.5, 1)
    elif new_topic == "cookie_snack":
        new_ph = min(6.8, round(base_ph + 0.45, 2))
        new_temp = round(base_temp + 1.8, 1)
    elif new_topic == "water_spray":
        new_ph = min(6.9, round(base_ph + 0.6, 2))
        new_temp = round(base_temp + 2.2, 1)
    elif new_topic == "baking_soda":
        new_ph = min(7.4, round(base_ph + 1.6, 2))
        new_temp = round(base_temp + 3.4, 1)
    elif new_topic == "egg_shell":
        new_ph = min(6.8, round(base_ph + 0.5, 2))
        new_temp = round(base_temp + 1.5, 1)
    elif new_topic == "gold_ring":
        new_ph = base_ph
        new_temp = base_temp
    else:
        new_ph = min(6.8, round(base_ph + 0.25, 2))
        new_temp = round(base_temp + 2.0, 1)

    return cur_items, labels, new_ph, new_temp


def _build_cross_item_interaction(
    prior_keys: list[str],
    new_topic: str,
    mode: str,
    preset_id: str = "",
    lang: str = "ko",
) -> str:
    """Construct a 100% pure Korean (`ko`) or pure English (`en`) sentence explaining scene continuity."""
    is_storybook = mode == "storybook" or preset_id in ("fairy_starlight_fox", "fairy_tiger_rabbit") or "fairy" in preset_id
    is_space = preset_id in ("science_space_kitchen", "science_space_shuttle") or "space" in preset_id

    # Science Lab modes run clean 1-by-1 experiments without piling past reagents into the text
    if not is_storybook:
        return ""

    if lang == "en":
        if not prior_keys:
            if preset_id == "fairy_tiger_rabbit":
                return "Under the glowing full moon on the pine mountain pass, Dalkong the clever rabbit and the striped tiger begin their folktale adventure!"
            return "Right in front of the glowing golden keyhole in the ancient oak tree, Lumi's magical forest adventure comes to life!"

        links_en: list[str] = []
        if "rice_cake" in prior_keys and new_topic != "rice_cake":
            links_en.append("the warm golden honey rice cakes still steam with starlight aroma in the woven basket")
        if "roasted_pebble" in prior_keys and new_topic != "roasted_pebble":
            links_en.append("the glowing ruby-orange campfire pebble sparkles beside the tiger's paws")
        if "dragon_bubbles" in prior_keys and new_topic != "dragon_bubbles":
            links_en.append("the iridescent rainbow bubbles blown by Popo the baby dragon float alongside")
        if "rainbow_spaceship" in prior_keys and new_topic != "rainbow_spaceship":
            links_en.append("the pastel rainbow cloud airship illuminates the night sky with its golden star lantern")
        if "star_candy" in prior_keys and new_topic != "star_candy":
            links_en.append("the sparkling star-candies scattered across the path chime like a music box")
        if "cookie_snack" in prior_keys and new_topic != "cookie_snack":
            links_en.append("warm golden cookie-crumb fairies dance joyfully in the starlight")
        if links_en:
            return "At the same time, " + ", and ".join(links_en) + ", weaving every earlier chapter into one continuous fairy-tale world!"
        return "All the magical wonders from earlier chapters remain right beside our heroes as the storybook adventure continues!"

    # Pure Korean ("ko") for Storybook
    if not prior_keys:
        if preset_id == "fairy_tiger_rabbit":
            return "보름달이 환하게 비추는 소나무 산고갯길에서 꾀돌이 토끼 달콩이와 줄무늬 호랑이의 흥미진진한 동화 이야기가 펼쳐집니다!"
        return "신비로운 별빛 숲의 빛나는 참나무 황금 열쇠구멍 앞에서 루미의 마법 같은 동화 이야기가 펼쳐집니다!"

    story_links_ko: list[str] = []
    if "rice_cake" in prior_keys and new_topic != "rice_cake":
        story_links_ko.append("나무 쟁반 위에서 황금빛 김이 모락모락 피어오르는 달빛 꿀떡의 달콤한 향기가 산길에 가득 퍼지고")
    if "roasted_pebble" in prior_keys and new_topic != "roasted_pebble":
        story_links_ko.append("모닥불 가에서 루비처럼 붉게 달아오른 따끈따끈한 조약돌이 반짝이는 불티를 날리며")
    if "dragon_bubbles" in prior_keys and new_topic != "dragon_bubbles":
        story_links_ko.append("초록빛 아기 용 포포가 불어 준 무지개 비눗방울들이 방울방울 떠올라 밤하늘 길을 밝혀 주고")
    if "rainbow_spaceship" in prior_keys and new_topic != "rainbow_spaceship":
        story_links_ko.append("둥실 떠 있는 파스텔 무지개 구름 비행선의 별빛 랜턴이 포근한 황금빛을 비추며")
    if "star_candy" in prior_keys and new_topic != "star_candy":
        story_links_ko.append("오솔길 바닥에 보석처럼 쌓인 달콤한 별사탕들이 발걸음마다 맑은 오르골 소리를 내고")
    if "cookie_snack" in prior_keys and new_topic != "cookie_snack":
        story_links_ko.append("고소한 별빛 과자 향기와 황금 가루 요정들이 주변을 맴돌며")
    if story_links_ko:
        return "게다가 " + ", ".join(story_links_ko) + " 앞선 장의 마법들이 사라지지 않고 하나의 살아있는 동화로 어우러지고 있답니다!"
    return "이전 장면의 마법과 지혜를 고스란히 간직한 채 다음 동화 이야기로 자연스럽게 이어집니다!"


def _deterministic_story_fallback(
    book_state: dict[str, Any],
    user_utterance: str,
    next_page_num: int,
    prev_visual_state: dict[str, Any],
    lang: str = "ko",
) -> dict[str, Any]:
    """Generate a strictly localized (`ko` or `en`), scenario-isolated page turn with accurate cause-and-effect."""
    norm_lang = "en" if str(lang or "ko").strip().lower().startswith("en") else "ko"
    preset_id = str(book_state.get("preset_id") or prev_visual_state.get("theme") or "science_hcl_lab")
    mode = str(
        book_state.get("mode")
        or ("storybook" if preset_id in ("fairy_starlight_fox", "fairy_tiger_rabbit") or "fairy" in preset_id else "science_lab")
    )
    persona_voice = _resolve_character_persona(book_state, preset_id, mode)

    if norm_lang == "en":
        if preset_id == "fairy_tiger_rabbit":
            default_char = "Dalkong the Clever Mountain Rabbit"
        elif mode == "storybook" or preset_id == "fairy_starlight_fox":
            default_char = "Lumi the Baby Starlight Fox"
        elif preset_id == "science_space_shuttle":
            default_char = "Commander Orbit"
        elif preset_id == "science_space_kitchen" or "space" in preset_id:
            default_char = "Captain Orbit"
        else:
            default_char = "Dr. Nova the Owl Professor"
    else:
        if preset_id == "fairy_tiger_rabbit":
            default_char = "꾀돌이 산토끼 달콩이"
        elif mode == "storybook" or preset_id == "fairy_starlight_fox":
            default_char = "아기 별빛 여우 루미"
        elif preset_id == "science_space_shuttle":
            default_char = "우주왕복선 사령관 오르빗 대장"
        elif preset_id == "science_space_kitchen" or "space" in preset_id:
            default_char = "우주 비행사 오르빗 대장"
        else:
            default_char = "부엉이 과학자 노바 박사"
    character_name = str(book_state.get("character_name") or default_char)
    if norm_lang == "en" and _HANGUL_RE.search(character_name):
        character_name = default_char
    elif norm_lang == "ko" and not _HANGUL_RE.search(character_name):
        character_name = default_char

    pages = book_state.get("pages") or []
    prev_page = pages[-1] if pages else {}
    prev_title = str(prev_page.get("page_title") or ("Chapter 1" if norm_lang == "en" else "1장"))

    topic = _detect_item_key(user_utterance, mode=mode, preset_id=preset_id)
    acc_items, acc_labels, new_ph, new_temp = _compute_cumulative_state(
        prev_visual_state, topic, user_utterance, lang=norm_lang, mode=mode
    )

    is_storybook = mode == "storybook" or preset_id in ("fairy_starlight_fox", "fairy_tiger_rabbit") or "fairy" in preset_id
    if is_storybook:
        prior_keys = [k for k in acc_items[:-1] if k not in ("magic_door", "zero_g_cabin", "tiger_rabbit_path")]
        prior_labels = [get_item_display_label(i, lang=norm_lang) for i in prior_keys]
        cross_interaction = _build_cross_item_interaction(
            prior_keys, topic, mode=mode, preset_id=preset_id, lang=norm_lang
        )
        if cross_interaction:
            cross_interaction = cross_interaction + " "
        if norm_lang == "en":
            prior_bridge = (
                f"Continuing from Chapter {next_page_num - 1} with {', '.join(prior_labels)} still in our scene, "
                if prior_labels
                else f"Continuing from Chapter {next_page_num - 1}, "
            )
        else:
            prior_bridge = (
                f"앞선 {next_page_num - 1}장에서 함께한 {', '.join(prior_labels)}에 이어, "
                if prior_labels
                else f"{next_page_num - 1}장 이야기에 이어, "
            )
    else:
        # 1-by-1 Independent Science Lab Mode: no awkward multi-item prefix piling!
        prior_keys = []
        prior_labels = []
        cross_interaction = ""
        prior_bridge = ""

    lower_q = (user_utterance or "").lower()

    # =========================================================================
    # SCENARIO 0: TIGER & RABBIT KOREAN FOLKTALE (`fairy_tiger_rabbit`)
    # =========================================================================
    if preset_id == "fairy_tiger_rabbit":
        tr_followups_ko = [
            "토끼가 호랑이에게 반짝이는 달빛 꿀떡을 주면 어떻게 돼요?",
            "토끼가 모닥불에서 구운 뜨거운 조약돌을 떡이라고 주면 어때요?",
            "호랑이랑 토끼가 무지개 구름을 타고 달나라로 날아가면 어때요?",
            "하늘에서 달콤한 별사탕 비가 쏟아지면 호랑이가 어떻게 해요?",
        ]
        tr_followups_en = [
            "What if the rabbit gives the tiger glowing golden honey rice cakes?",
            "What if the rabbit tricks the tiger with a warm roasted campfire pebble?",
            "What if the tiger and rabbit ride a rainbow cloud to the moon?",
            "What if sweet star-candy rain falls over the mountain pass?",
        ]
        if topic == "rice_cake":
            if norm_lang == "en":
                title_str = f"Chapter {next_page_num} — The Steaming Golden Starlight Honey Rice Cakes"
                p1 = (
                    f"{prior_bridge}Dalkong the clever white rabbit lifted the lid of her woven basket and offered a wooden tray "
                    "piled high with glowing golden honey rice cakes! As warm, sweet steam drifted across the moonlit pine pass, "
                    "the big striped tiger's golden eyes sparkled with delight."
                )
                p2 = (
                    f"{cross_interaction}Taking one big bite of the chewy honey rice cake, the mountain tiger purred like a giant kitten "
                    "and promised to guard Dalkong all the way over the mountain pass!"
                )
                diag = (
                    "Hop-hop! Look how happy Mr. Tiger is munching the warm golden honey rice cake! "
                    "His scary roar turned into the sweetest purr on the mountain! What shall we do next, hop-hop~!"
                )
            else:
                title_str = f"{next_page_num}장 — 김이 모락모락 피어오르는 황금빛 달빛 꿀떡"
                p1 = (
                    f"{prior_bridge}꾀돌이 산토끼 달콩이가 대나무 바구니를 열고 나무 쟁반 가득 담긴 황금빛 달빛 꿀떡을 내밀었어요! "
                    "따끈따끈한 김과 함께 달콤한 조청 향기가 소나무 고갯길에 퍼지자, 줄무늬 호랑이의 두 눈이 보름달처럼 동그래졌답니다."
                )
                p2 = (
                    f"{cross_interaction}쫀득하고 달콤한 꿀떡을 한 입 크게 베어 문 호랑이는 '어흥, 이렇게 맛있는 떡은 처음이야!' 하고 감탄하며 "
                    "달콩이를 등에 태워 고갯마루까지 든든하게 지켜 주기로 약속했어요!"
                )
                diag = (
                    "깡총깡총! 우와, 호랑이 아저씨가 달빛 꿀떡을 먹고 눈에서 하트가 뿅뿅 나왔어! "
                    "이제 무서운 호랑이가 아니라 내 든든한 산길 친구가 되었단다! 다음엔 또 어떤 상상을 해볼까, 깡총~!"
                )
        elif topic == "roasted_pebble":
            if norm_lang == "en":
                title_str = f"Chapter {next_page_num} — The Sizzling Campfire Pebble Trick"
                p1 = (
                    f"{prior_bridge}Beside the glowing campfire on the mountain path, Dalkong pointed at a round pebble glowing ruby-orange "
                    "and whispered, 'Mr. Tiger, this is a special crispy sesame rice cake—eat it while it's warm!' "
                    "The greedy tiger opened his mouth wide and tried to bite the sizzling pebble!"
                )
                p2 = (
                    f"{cross_interaction}'Ouch, hot-hot-hot!' The surprised tiger puffed out his striped cheeks in comical shock "
                    "as golden sparks danced in the moonlight, while clever Dalkong giggled and offered him cool spring water!"
                )
                diag = (
                    "Hop-hop! Hehe, Mr. Tiger thought the glowing campfire pebble was a crispy sesame rice cake! "
                    "Look at his puffed-up cheeks—he promised never to bully little forest animals again, hop-hop~!"
                )
            else:
                title_str = f"{next_page_num}장 — 모닥불에서 루비처럼 달아오른 조약돌 떡 소동"
                p1 = (
                    f"{prior_bridge}꾀돌이 달콩이는 모닥불 가에서 루비빛으로 발갛게 달아오른 동그란 조약돌을 가리키며 말했어요. "
                    "'호랑이 아저씨, 이 바삭바삭한 불꽃 깨찰떡을 식기 전에 얼른 드세요!' 욕심쟁이 호랑이가 덥석 조약돌을 입에 넣자마자 볼이 풍선처럼 부풀어 올랐답니다!"
                )
                p2 = (
                    f"{cross_interaction}'앗 뜨거워, 어흥!' 호랑이가 두 볼을 부여잡고 펄쩍펄쩍 뛰며 개구쟁이 같은 표정을 짓자, "
                    "달콩이가 까르르 웃으며 시원한 약숫물을 건네주어 둘은 유쾌한 산길 단짝이 되었어요!"
                )
                diag = (
                    "깡총깡총! 헤헤, 호랑이 아저씨가 모닥불 조약돌을 깨찰떡인 줄 알고 덥석 물었다가 볼이 빵빵해졌어! "
                    "다시는 약한 동물들을 괴롭히지 않겠다고 손가락 걸고 약속했단다, 깡총~!"
                )
        elif topic == "rainbow_spaceship":
            if norm_lang == "en":
                title_str = f"Chapter {next_page_num} — Riding the Rainbow Cloud to the Full Moon"
                p1 = (
                    f"{prior_bridge}A fluffy seven-colored rainbow cloud drifted down to the pine mountain pass! "
                    "Dalkong the rabbit and the friendly striped tiger hopped aboard together, soaring up above the lantern-lit trees toward the glowing full moon."
                )
                p2 = (
                    f"{cross_interaction}As the wind tickled the tiger's whiskers, they waved down at the mountain pass and sailed through a sea of twinkling stars!"
                )
                diag = (
                    "Hop-hop! Woohoo! Mr. Tiger and I are flying on a soft rainbow cloud right up to the moon! "
                    "Hold onto Mr. Tiger's striped tail so we don't slip, hop-hop~!"
                )
            else:
                title_str = f"{next_page_num}장 — 무지개 구름을 타고 달나라로 날아간 호랑이와 토끼"
                p1 = (
                    f"{prior_bridge}소나무 고갯마루 위로 일곱 빛깔 폭신한 무지개 구름이 사뿐히 내려왔어요! "
                    "꾀돌이 토끼 달콩이와 줄무늬 호랑이가 나란히 무지개 구름 위에 올라타자, 구름이 두둥실 떠올라 둥근 보름달을 향해 힘차게 날아올랐답니다."
                )
                p2 = (
                    f"{cross_interaction}시원한 밤바람에 호랑이의 수염이 살랑거리고, 발아래로 청사초롱 등불이 빛나는 산고개가 한눈에 내려다보여 둘은 신나게 만세를 불렀어요!"
                )
                diag = (
                    "깡총깡총! 우와앙~ 호랑이 아저씨랑 같이 폭신폭신한 무지개 구름을 타고 달나라로 날아가고 있어! "
                    "구름 위에서 내려다보는 달빛 산고개가 정말 그림 같아, 깡총~!"
                )
        elif topic == "star_candy":
            if norm_lang == "en":
                title_str = f"Chapter {next_page_num} — Sweet Star-Candy Rain Over the Mountain Pass"
                p1 = (
                    f"{prior_bridge}Suddenly the night sky above the pine mountain pass twinkled, and a shower of colorful pastel star-candies rained down like sweet shooting stars! "
                    "The big striped tiger opened his mouth wide—'Arooo!'—catching sparkling star-candies right on his tongue."
                )
                p2 = (
                    f"{cross_interaction}Dalkong collected the glowing pink, mint, and golden star-candies in her bamboo basket, and every candy chimed like a little bell!"
                )
                diag = (
                    "Hop-hop! Look at Mr. Tiger catching sweet star-candies in his big mouth! "
                    "They taste like honey and moonlight—my bamboo basket is overflowing with starry treats, hop-hop~!"
                )
            else:
                title_str = f"{next_page_num}장 — 달빛 고개에 쏟아진 달콤한 별사탕 소나기"
                p1 = (
                    f"{prior_bridge}밤하늘의 은하수가 반짝이더니 소나무 산고개 위로 알록달록한 별사탕 비가 사르르 쏟아져 내렸어요! "
                    "줄무늬 호랑이는 입을 커다랗게 벌리고 '어흥~ 아암!' 하며 떨어지는 별사탕을 쏙쏙 받아먹느라 신이 났답니다."
                )
                p2 = (
                    f"{cross_interaction}달콩이도 대나무 바구니를 활짝 열어 분홍빛과 황금빛 별사탕을 한가득 담았고, 별사탕이 부딪힐 때마다 맑은 종소리가 산골짜기에 울려 퍼졌어요!"
                )
                diag = (
                    "깡총깡총! 헤헤, 호랑이 아저씨가 하늘에서 내리는 별사탕을 입으로 쏙쏙 받아먹고 있어! "
                    "입안에서 달콤한 별빛이 톡톡 터져서 바구니 가득 담아 두었단다, 깡총~!"
                )
        else:
            # Cause-and-effect branches for freeform Tiger & Rabbit questions (e.g., "달이 사라지면 어떻게 되어요?")
            if any(k in lower_q for k in ("달이 사라지", "달이 없어지", "달이 꺼지", "어두워", "밤하늘", "해가", "별이 사라지", "moon", "dark", "disappear", "vanish")):
                if norm_lang == "en":
                    title_str = f"Chapter {next_page_num} — When the Full Moon Vanished into Pitch Darkness"
                    p1 = (
                        f"{prior_bridge}Suddenly, a mischievous shadow cloud swallowed the full moon and the pine mountain pass plunged into pitch-black darkness! "
                        "Because the moonlight vanished, the big striped tiger couldn't see a single step ahead and trembled, 'Oh no, Dalkong, it's too dark to walk!'"
                    )
                    p2 = (
                        f"{cross_interaction}Thinking fast, clever Dalkong opened her bamboo basket to release glowing golden honey-cake fireflies, while the tiger flashed his bright golden eyes like twin headlamps! "
                        "Together, their warm light guided them safely up the dark trail until their cheers blew the shadow cloud away and the full moon shone again!"
                    )
                    diag = (
                        "Hop-hop! When the moon disappeared, the mountain pass turned pitch-black and even big Mr. Tiger got scared! "
                        "So I lit up our bamboo lantern with fireflies and Mr. Tiger used his glowing golden eyes like flashlights until we brought the moon back, hop-hop~!"
                    )
                else:
                    title_str = f"{next_page_num}장 — 보름달이 사라져 깜깜해진 산고개와 호랑이의 황금 눈빛 등불"
                    p1 = (
                        f"{prior_bridge}갑자기 심술궂은 먹구름이 보름달을 꿀꺽 삼켜 버리자, 환하던 소나무 산고개가 한 치 앞도 보이지 않는 칠흑 같은 어둠에 잠겼어요! "
                        "달빛이 사라지자 용감하던 줄무늬 호랑이도 발밑이 보이지 않아 '어흥, 달콩아! 너무 깜깜해서 돌부리에 걸려 넘어지겠어!' 하며 귀를 쫑긋 세우고 덜덜 떨었답니다."
                    )
                    p2 = (
                        f"{cross_interaction}그때 꾀돌이 토끼 달콩이가 기지를 발휘해 바구니 속 반짝이는 꿀떡 별가루와 반딧불이들을 하늘로 날려 보냈고, 호랑이도 보석 같은 황금빛 두 눈을 자동차 전조등처럼 반짝이며 앞길을 비추었어요! "
                        "둘이 힘을 합쳐 어두운 산고개를 환하게 밝히자, 깜짝 놀란 먹구름이 물러가고 보름달이 다시 방긋 웃으며 나타났답니다!"
                    )
                    diag = (
                        "깡총깡총! 달이 갑자기 사라지니까 산고개가 온통 깜깜해져서 덩치 큰 호랑이 아저씨도 무서워하며 내 옆에 꼭 붙었지 뭐야! "
                        "그래서 내가 반딧불이 등불을 켜고 호랑이 아저씨의 반짝이는 황금 눈빛으로 길을 비춰서 무사히 달님을 다시 찾아냈단다, 깡총~!"
                    )
            else:
                if norm_lang == "en":
                    title_str = f"Chapter {next_page_num} — Cause and Effect on the Moonlit Pass: {user_utterance[:28]}"
                    p1 = (
                        f"{prior_bridge}Because you asked '{user_utterance}', a sudden change swept across the moonlit pine mountain pass! "
                        "At first, the big striped tiger was startled by the unexpected turn of events and hid behind a pine tree."
                    )
                    p2 = (
                        f"{cross_interaction}Then clever Dalkong used her quick wits to turn the surprise into a helpful adventure, and the tiger's roar turned into joyful laughter as they solved the problem together!"
                    )
                    diag = (
                        f"Hop-hop! Because of '{user_utterance[:34]}', Mr. Tiger was so surprised at first, "
                        "but we put our heads together and solved it with mountain wisdom, hop-hop~!"
                    )
                else:
                    title_str = f"{next_page_num}장 — 달빛 산고개의 기발한 사건과 결과: {user_utterance[:22]}"
                    p1 = (
                        f"{prior_bridge}소나무 산고개에서 '{user_utterance}' 상황이 실제로 벌어지자, 산속 풍경이 순식간에 술렁이기 시작했어요! "
                        "예상치 못한 변화가 일어나자 줄무늬 호랑이는 처음에 깜짝 놀라 두 눈을 동그랗게 뜨고 소나무 뒤로 몸을 숨겼답니다."
                    )
                    p2 = (
                        f"{cross_interaction}하지만 꾀돌이 토끼 달콩이가 침착하게 원인을 살피고 재치 있는 지혜를 발휘한 결과, 위기가 오히려 호랑이와 토끼를 도와주는 신비로운 축복으로 바뀌어 둘은 기쁘게 고개를 넘었어요!"
                    )
                    diag = (
                        f"깡총깡총! 방금 네가 말해 준 '{user_utterance[:24]}' 일 때문에 호랑이 아저씨가 깜짝 놀랐지만, "
                        "우리가 지혜를 모아 슬기롭게 해결해서 산고개가 더욱 즐거워졌어, 깡총~!"
                    )

        note_str = (
            f"Folktale Cause & Continuity Note: Across Chapters 1 to {next_page_num}, [{', '.join(acc_labels)}] connect every cause and outcome on the mountain pass."
            if norm_lang == "en"
            else f"전래 동화 인과관계 및 맥락 노트: 1장부터 {next_page_num}장까지 [{', '.join(acc_labels)}] 이야기 흐름 속에서 원인과 결과가 생생하게 이어집니다."
        )
        return {
            "page_title": title_str,
            "story_paragraphs": [p1, p2],
            "character_name": character_name,
            "character_persona_voice": persona_voice,
            "character_dialogue": diag,
            "science_or_moral_note": note_str,
            "visual_state": {
                "theme": "fairy_tiger_rabbit",
                "lang": norm_lang,
                "character_persona_voice": persona_voice,
                "item_key": topic,
                "item_label": get_item_display_label(topic, lang=norm_lang, fallback_text=user_utterance[:28]),
                "accumulated_items": acc_items,
                "accumulated_labels": acc_labels,
                "sky_color": [24, 34, 68],
                "sparkle_intensity": 0.92,
                "companion": "mountain_tiger",
                "accent_color": [251, 191, 36],
            },
            "nano_banana_prompt": (
                f"Nano Banana Continuity Edit from Page {next_page_num - 1} ({prev_title}): preserve the friendly striped Korean mountain tiger "
                f"and clever white rabbit on the moonlit pine path while clearly depicting the cause-and-effect outcome of: {user_utterance}."
            ),
            "veo_motion_prompt": (
                f"Animated Korean folktale watercolor scene: clever white rabbit and striped tiger experience {user_utterance}."
            ),
            "suggested_followups": tr_followups_en if norm_lang == "en" else tr_followups_ko,
        }

    # =========================================================================
    # SCENARIO 1: FAIRY-TALE STORYBOOK (`fairy_starlight_fox` / `mode == "storybook"`)
    # 100% isolated from chemistry/beakers!
    # =========================================================================
    if mode == "storybook" or preset_id == "fairy_starlight_fox":
        followups_ko = [
            "문이 열리고 아기 용이 비눗방울을 뿜으면 어때요?",
            "루미가 무지개 구름 비행선을 타면 어떻게 돼요?",
            "열쇠구멍에 달콤한 초코칩 과자를 넣으면 어떻게 돼요?",
            "하늘에서 별사탕 비가 내리면 어떻게 돼요?",
        ]
        followups_en = [
            "What if the door opens and a baby dragon blows bubbles?",
            "What if Lumi rides a rainbow cloud airship?",
            "What if we put a chocolate-chip cookie into the keyhole?",
            "What if sweet star-candy rain falls from the sky?",
        ]

        if topic == "dragon_bubbles":
            if norm_lang == "en":
                title_str = f"Chapter {next_page_num} — Popo the Baby Dragon and the Rainbow Bubbles"
                p1 = (
                    f"{prior_bridge}Lumi adjusted her royal-blue knitted scarf and tapped gently on the glowing oak door. "
                    "With a warm golden chime, the ancient door swung open and out fluttered Popo, a tiny emerald-winged baby dragon! "
                    "Instead of fiery flames, Popo puffed a stream of shimmering rainbow soap bubbles from the tip of his snout."
                )
                p2 = (
                    f"{cross_interaction}Inside every floating bubble danced a tiny reflection of the crescent moon, and whenever "
                    "Lumi brushed a bubble with her glowing star-tipped tail, it rang like a silver bell and showered the moss with harmless stardust."
                )
                diag = (
                    "Yip-yip! Hehe, look who flew out from behind the secret oak door! "
                    "It is Popo the baby dragon blowing rainbow bubbles! When I touch the bubbles with my starry tail, the whole forest sparkles, yip~!"
                )
            else:
                title_str = f"{next_page_num}장 — 참나무 문에서 날아오른 아기 용 포포와 무지개 비눗방울"
                p1 = (
                    f"{prior_bridge}루미가 파란 털실 목도리를 여미며 빛나는 참나무 문을 똑똑 두드리자, 황금빛 문이 스르르 열리며 "
                    "에메랄드빛 날개를 가진 귀여운 아기 용 포포가 날아 나왔어요! 포포는 뜨거운 불꽃 대신 코끝에서 영롱한 무지개 비눗방울들을 퐁퐁 뿜어냈답니다."
                )
                p2 = (
                    f"{cross_interaction}비눗방울 하나하나마다 별빛 숲의 달빛이 담겨 있어서, 루미가 별 모양 꼬리를 흔들 때마다 "
                    "비눗방울 속에서 작은 종소리와 함께 반짝이는 별가루가 사르르 피어올랐어요!"
                )
                diag = (
                    "콩콩! 우와~ 참나무 문틈에서 아기 용 포포가 나와서 무지개 비눗방울을 불어주고 있어! 헤헤, "
                    "내 별빛 꼬리로 비눗방울을 살짝 간지럽히니까 숲 전체가 일곱 빛깔로 반짝거려, 낑~!"
                )
        elif topic == "rainbow_spaceship":
            if norm_lang == "en":
                title_str = f"Chapter {next_page_num} — Boarding the Pastel Rainbow Cloud Airship"
                p1 = (
                    f"{prior_bridge}As your wish echoed across the twilight canopy, the sleepy seven-colored cloud above the ancient oak "
                    "drifted down to the forest clearing and transformed into a cozy pastel rainbow cloud airship adorned with a golden star lantern!"
                )
                p2 = (
                    f"{cross_interaction}Lumi hopped onto the cotton-soft rainbow deck"
                    f"{' alongside Popo the baby dragon' if 'dragon_bubbles' in prior_keys else ''}, looking down at the glowing oak door "
                    "and the sea of golden fireflies as the airship glided gracefully toward the Milky Way sky."
                )
                diag = (
                    f"Yip-yip! Woohoo! {'Riding the bubble breeze from Popo, our ' if 'dragon_bubbles' in prior_keys else 'Our '}"
                    "rainbow cloud airship is floating right up into the starry sky! Hehe, hold onto my blue scarf as we sail to the Milky Way, yip~!"
                )
            else:
                title_str = f"{next_page_num}장 — 별빛 랜턴을 단 무지개 구름 비행선 탑승"
                p1 = (
                    f"{prior_bridge}여러분의 상상이 밤하늘에 울려 퍼지자, 달빛을 머금은 일곱 빛깔 무지개 구름이 참나무 옆 공터로 사뿐히 내려오더니 "
                    "황금 별빛 랜턴이 달린 포근한 구름 비행선으로 변신했어요!"
                )
                p2 = (
                    f"{cross_interaction}루미는 {'아기 용 포포와 나란히 ' if 'dragon_bubbles' in prior_keys else ''}솜사탕처럼 폭신한 무지개 갑판 위에 올라타, "
                    "별빛 숲의 거대한 참나무와 반딧불이 물결을 내려다보며 은하수 하늘길로 힘차게 날아올랐답니다!"
                )
                diag = (
                    f"콩콩! 우와앙~ {'아기 용 포포가 불어 준 비눗방울 바람을 타고 ' if 'dragon_bubbles' in prior_keys else ''}"
                    "무지개 구름 비행선이 두둥실 떠오르고 있어! 헤헤, 나랑 같이 저기 은하수 끝까지 날아가 보자, 낑~!"
                )
        elif topic == "star_candy":
            if norm_lang == "en":
                title_str = f"Chapter {next_page_num} — The Sweet Pastel Star-Candy Shower from the Milky Way"
                p1 = (
                    f"{prior_bridge}The crescent moon above the Whispering Woods curved into a gentle smile, and from the shimmering Milky Way clouds "
                    "fell a magical shower of pink, mint, and golden star-candies drifting down like sweet shooting stars!"
                )
                p2 = (
                    f"{cross_interaction}Every time the pastel star-candies tapped against the ancient oak leaves and mossy roots, "
                    "they played a crystal lullaby melody while Lumi held out the end of her blue scarf to catch strawberry- and lemon-flavored starlight treats."
                )
                diag = (
                    "Yip-yip! Hehe, real star-candy rain is falling from the night sky! "
                    f"{'Even Popo the baby dragon is catching star-candies inside his rainbow bubbles! ' if 'dragon_bubbles' in prior_keys else ''}"
                    "When I crunch one, it tastes like sparkling moonlight, yip~!"
                )
            else:
                title_str = f"{next_page_num}장 — 은하수에서 쏟아지는 달콤한 별사탕 소나기"
                p1 = (
                    f"{prior_bridge}초승달이 활짝 미소 짓더니, 밤하늘의 은하수 구름에서 분홍빛과 민트빛, 황금빛 별사탕들이 "
                    "마법의 유성우처럼 별빛 숲으로 사르르 쏟아져 내리기 시작했어요!"
                )
                p2 = (
                    f"{cross_interaction}별사탕들이 이끼 낀 숲 바닥과 참나무 나뭇잎 위에 내려앉을 때마다 맑은 오르골 멜로디가 울려 퍼졌고, "
                    "루미는 파란 목도리 자락을 활짝 펼쳐 달콤한 딸기맛과 레몬맛 별사탕을 한가득 받았답니다!"
                )
                diag = (
                    "콩콩! 헤헤~ 밤하늘에서 진짜로 달콤한 별사탕 비가 내리고 있어! "
                    f"{'아기 용 포포의 비눗방울 속에도 별사탕이 쏙쏙 들어가서 반짝이는 보석 방울이 되었어! ' if 'dragon_bubbles' in prior_keys else ''}"
                    "입안에 넣으니까 별빛이 톡톡 터지는 맛이야, 낑~!"
                )
        elif topic == "cookie_snack":
            if norm_lang == "en":
                title_str = f"Chapter {next_page_num} — The Star-Shaped Chocolate Cookie in the Golden Keyhole"
                p1 = (
                    f"{prior_bridge}Lumi held up a warm, star-shaped chocolate-chip cookie and gently pressed it into the glowing golden keyhole "
                    "of the ancient oak tree! Instead of a metal key, the sweet cookie fit the magical lock perfectly, filling the twilight forest with the cozy aroma of vanilla and cocoa."
                )
                p2 = (
                    f"{cross_interaction}As the golden keyhole clicked open with a warm amber glow, tiny winged cookie-crumb fairies "
                    "and shimmering cocoa-starlight sprites danced out from the oak trunk, sprinkling sweet golden dust over Lumi's blue scarf!"
                )
                diag = (
                    "Yip-yip! Hehe, putting a crunchy chocolate-chip cookie into the golden keyhole worked like magic! "
                    "The whole oak tree smells like a warm bakery in the stars, and tiny cookie fairies are tickling my ears, yip~!"
                )
            else:
                title_str = f"{next_page_num}장 — 황금 열쇠구멍에 쏙 들어간 별 모양 초코칩 과자"
                p1 = (
                    f"{prior_bridge}루미가 고소하고 달콤한 별 모양 초코칩 과자를 두 손으로 소중히 들고 참나무 밑동의 빛나는 황금 열쇠구멍에 쏙 넣어 보았어요! "
                    "차가운 열쇠 대신 들어간 달콤한 과자가 마법 자물쇠에 꼭 맞물리더니, 숲속 가득 따뜻한 바닐라와 초콜릿 향기가 퍼져 나갔답니다."
                )
                p2 = (
                    f"{cross_interaction}황금 열쇠구멍에서 영롱한 호박빛 광채가 피어오르며 작은 날개를 단 과자 부스러기 요정들이 포르르 날아올라, "
                    "루미의 파란 목도리와 별빛 꼬리 주변에 반짝이는 황금 별가루를 수놓아 주었어요!"
                )
                diag = (
                    "콩콩! 헤헤~ 황금 열쇠구멍에 초코칩 과자를 넣었더니 참나무에서 고소한 빵집 향기가 솔솔 나고 있어! "
                    "반짝이는 과자 요정들이 내 귀를 간지럽히며 춤추고 있어, 정말 신기하다, 낑~!"
                )
        else:
            # Specific Cause-and-Effect branches for freeform Fairy-Tale questions (e.g. "달이 사라지면 어떻게 되어요?")
            if any(k in lower_q for k in ("달이 사라지", "달이 없어지", "달이 꺼지", "어두워", "별이 사라지", "밤하늘", "해가", "moon", "dark", "disappear", "vanish")):
                if norm_lang == "en":
                    title_str = f"Chapter {next_page_num} — When the Moon Vanished and Lumi's Starlight Tail Lit the Dark Woods"
                    p1 = (
                        f"{prior_bridge}Suddenly, the silver crescent moon vanished behind a veil of night mist, and the entire Starlight Forest plunged into pitch-black darkness! "
                        "Because the moonlight disappeared, the glowing flowers closed their petals, the ocean tides fell still, and the little forest birds chirped nervously in the dark."
                    )
                    p2 = (
                        f"{cross_interaction}Right then, Lumi raised her star-tipped tail high like a golden lantern! Calling hundreds of fireflies to gather around the ancient oak door, "
                        "she sent a beam of warm starlight up into the dark sky, waking the sleeping moon and restoring gentle silver light to the forest!"
                    )
                    diag = (
                        "Yip-yip! When the moon disappeared, the whole forest turned pitch-black and the night flowers closed up! "
                        "So I lifted my glowing star-tail like a lantern and teamed up with the fireflies to shine light back into the sky until the moon returned, yip~!"
                    )
                else:
                    title_str = f"{next_page_num}장 — 달이 사라져 깜깜해진 숲과 밤하늘을 다시 밝힌 루미의 별빛 꼬리"
                    p1 = (
                        f"{prior_bridge}밤하늘에서 은빛 달님이 감쪽같이 사라져 버리자, 환하던 별빛 숲이 순식간에 칠흑 같은 어둠에 잠기고 말았어요! "
                        "달빛이 없어지자 달맞이꽃들이 꽃잎을 꼭 닫아 버렸고, 밤하늘의 길잡이를 잃은 아기 새들과 숲속 동물 친구들이 어둠 속에서 길을 잃고 당황했답니다."
                    )
                    p2 = (
                        f"{cross_interaction}그때 아기 여우 루미가 복슬복슬한 꼬리 끝의 황금 별빛을 등대처럼 높이 치켜들었어요! "
                        "루미의 꼬리 빛을 따라 수백 마리의 반딧불이와 참나무 열쇠구멍의 황금 빛줄기가 밤하늘로 솟구쳐 올라가자, 어둠 속에 숨어 있던 달님이 다시 환하게 깨어나 숲속에 은빛 달빛이 되돌아왔답니다!"
                    )
                    diag = (
                        "콩콩! 달님이 갑자기 사라지니까 숲속이 한 치 앞도 안 보이게 깜깜해지고 달맞이꽃들도 오들오들 떨었어! "
                        "그래서 내 꼬리 끝의 별빛 등불을 활짝 켜고 반딧불이 친구들과 힘을 합쳐 밤하늘에 빛을 쏘아 올려서 달님을 다시 깨웠단다, 낑~!"
                    )
            else:
                if norm_lang == "en":
                    title_str = f"Chapter {next_page_num} — Magical Cause and Effect: {user_utterance[:28]}"
                    p1 = (
                        f"{prior_bridge}As soon as '{user_utterance}' happened in the Whispering Woods, the ancient oak tree and the surrounding forest reacted right before Lumi's eyes! "
                        "A ripple of golden starlight swept across the mossy clearing, transforming the scene in direct response to your question."
                    )
                    p2 = (
                        f"{cross_interaction}Lumi adjusted her royal-blue scarf and used her glowing star-tipped tail to guide her forest friends through this wonderful new change!"
                    )
                    diag = (
                        f"Yip-yip! Because of '{user_utterance[:34]}', the whole forest around the oak door changed in the most magical way! "
                        "Let's see what happens next on our adventure, yip~!"
                    )
                else:
                    title_str = f"{next_page_num}장 — 별빛 숲의 마법 같은 원인과 결과: {user_utterance[:22]}"
                    p1 = (
                        f"{prior_bridge}별빛 숲속에서 '{user_utterance}' 일이 실제로 일어나자, 고요하던 참나무 숲 전체에 즉각적인 변화가 찾아왔어요! "
                        "예상치 못한 상황에 숲속 요정들과 나뭇잎들이 술렁이며 새로운 풍경이 눈앞에 펼쳐졌답니다."
                    )
                    p2 = (
                        f"{cross_interaction}아기 여우 루미는 파란 털실 목도리를 여미고 반짝이는 별빛 꼬리를 흔들어, 새롭게 바뀐 숲속 환경을 친구들과 함께 지혜롭고 따뜻하게 이끌어 주었어요!"
                    )
                    diag = (
                        f"콩콩! 방금 물어본 '{user_utterance[:24]}' 일 때문에 우리 별빛 숲에 정말 놀라운 변화가 일어났어! "
                        "내 별빛 꼬리로 비춰 보니까 원인과 결과가 한눈에 보여, 정말 신기하지, 낑~!"
                    )

        note_str = (
            f"Story Cause & Continuity Note: Across Chapters 1 to {next_page_num}, [{', '.join(acc_labels)}] remain connected in Lumi's enchanted forest."
            if norm_lang == "en"
            else f"동화 인과관계 및 맥락 노트: 1장부터 {next_page_num}장까지 [{', '.join(acc_labels)}] 요소가 루미의 별빛 숲 이야기 속에 차곡차곡 이어집니다."
        )
        return {
            "page_title": title_str,
            "story_paragraphs": [p1, p2],
            "character_name": character_name,
            "character_persona_voice": persona_voice,
            "character_dialogue": diag,
            "science_or_moral_note": note_str,
            "visual_state": {
                "theme": "fairy_starlight_fox",
                "lang": norm_lang,
                "character_persona_voice": persona_voice,
                "item_key": topic,
                "item_label": get_item_display_label(topic, lang=norm_lang, fallback_text=user_utterance[:28]),
                "accumulated_items": acc_items,
                "accumulated_labels": acc_labels,
                "sky_color": [32, 26, 84],
                "sparkle_intensity": 0.94,
                "companion": topic,
                "accent_color": [244, 114, 182],
            },
            "nano_banana_prompt": (
                f"Nano Banana Continuity Edit from Page {next_page_num - 1} ({prev_title}): preserve Lumi the baby starlight fox "
                f"in her royal-blue scarf before the ancient oak door while clearly depicting the cause-and-effect outcome of: {user_utterance}."
            ),
            "veo_motion_prompt": (
                f"Animated Ghibli-style watercolor storybook scene: Lumi the Starlight Fox experiences {user_utterance} as magical sparkles drift."
            ),
            "suggested_followups": followups_en if norm_lang == "en" else followups_ko,
        }

    # =========================================================================
    # SCENARIO 2: ZERO-G SPACE STATION / SPACE SHUTTLE LAB (`science_space_kitchen` / `science_space_shuttle`)
    # Clean 1-by-1 Independent Science Experiments + Rich Cause-and-Effect!
    # =========================================================================
    if preset_id in ("science_space_kitchen", "science_space_shuttle"):
        space_theme = "science_space_shuttle" if preset_id == "science_space_shuttle" else "science_space_kitchen"
        space_followups_ko = [
            "우주왕복선 안에서 물을 뿌리면 물방울이 어떻게 되나요?",
            "무중력에서 양초에 불을 붙이면 불꽃 모양이 어때요?",
            "우주왕복선 안에서 초코칩 쿠키와 사과를 띄우면 어떻게 되나요?",
            "무중력 물방울에 발포 비타민을 넣으면 어떻게 되나요?",
        ]
        space_followups_en = [
            "What happens if we spray water inside the Space Shuttle?",
            "What shape does a candle flame make in zero gravity?",
            "What happens if we float a cookie and an apple in the shuttle?",
            "What happens if we add fizzy soda to a floating water sphere?",
        ]

        if topic == "zero_g_candle":
            if norm_lang == "en":
                title_str = f"Experiment {next_page_num} — The Glowing Blue Spherical Flame in Microgravity"
                p1 = (
                    "Inside the sealed orbital combustion chamber, Commander Orbit strikes a match to light a candle in microgravity. "
                    "Instead of stretching upward into a flickering yellow teardrop like on Earth, the flame forms a calm, perfectly round, sapphire-blue sphere!"
                )
                p2 = (
                    "Because there is no gravity-driven buoyancy in orbit, hot air does not rise and cold air does not sink. "
                    "Oxygen molecules diffuse evenly from every direction toward the wick, producing soot-free complete combustion in a glowing blue orb!"
                )
                diag = (
                    "Copy that, cadet! Look at that candle flame—because there is no up or down in zero gravity, "
                    "hot air doesn't rise and the flame burns as a crystal-blue sphere! Over!"
                )
                note_str = "Microgravity Diffusion Flame: Zero buoyancy convection leads to isotropic oxygen diffusion and a soot-free spherical blue flame."
            else:
                title_str = f"실험 {next_page_num} — 무중력 우주왕복선의 둥글고 푸른 구형 불꽃"
                p1 = (
                    "무중력 우주왕복선 연소 챔버 안에서 양초에 불을 붙이자, 지구에서처럼 위로 길쭉한 주황색 불꽃이 아니라 "
                    "사파이어 보석처럼 동그랗고 푸른 구형 불꽃이 허공에 조용히 피어오릅니다!"
                )
                p2 = (
                    "중력이 없는 우주에서는 뜨거운 공기가 위로 올라가는 대류 현상이 일어나지 않기 때문에, "
                    "산소 분자가 모든 방향에서 심지로 고르게 확산되며 그을음 없이 완전 연소하는 파란 공 모양 불꽃이 된답니다!"
                )
                diag = (
                    "오버! 저기 양초 불꽃 좀 봐! 우주에서는 위아래 방향이 없어서 뜨거운 공기가 위로 솟지 않기 때문에 "
                    "불꽃이 공처럼 동그랗고 파랗게 타오른단다! 정말 놀랍지, 로저!"
                )
                note_str = "우주 물리 원리: 무중력 환경에서는 부력에 의한 열대류가 사라지고 산소가 등방성 확산을 하여 푸른 구형 화염이 형성됩니다."
        elif topic == "water_spray":
            if norm_lang == "en":
                title_str = f"Experiment {next_page_num} — Floating Crystal Water Spheres and Surface Tension"
                p1 = (
                    "When Commander Orbit squeezes the water nozzle in microgravity, the water does not fall to the floor! "
                    "Instead, surface tension pulls the liquid into wobbling, crystal-clear spheres that float across the flight deck like liquid marbles."
                )
                p2 = (
                    "As smaller droplets touch in mid-air, hydrogen bonds between water molecules snap them together into a giant lens-like water orb "
                    "that reflects the blue curvature of Earth upside-down!"
                )
                diag = (
                    "Whoa, check out that zero-G water sphere! Surface tension acts like an invisible elastic skin, "
                    "turning our water spray into floating crystal spheres! Over!"
                )
                note_str = "Fluid Physics in Microgravity: Without gravitational hydrostatic pressure, surface tension minimizes surface area into a sphere."
            else:
                title_str = f"실험 {next_page_num} — 수정 구슬처럼 허공에 뭉치는 무중력 물방울과 표면장력"
                p1 = (
                    "오르빗 대장님이 우주왕복선 무중력 선실 허공에 분무기로 물을 뿌리자, 물이 바닥으로 쏟아지지 않고 "
                    "물 분자끼리 서로 끌어당기는 표면장력 덕분에 동그란 수정 구슬 모양으로 뭉쳐 둥실둥실 떠다닙니다!"
                )
                p2 = (
                    "작은 물방울들이 허공에서 서로 스치듯 부딪힐 때마다 탱글탱글하게 하나로 합쳐져 커다란 물 공이 되고, "
                    "그 투명한 물방울 렌즈 속에 창밖의 푸른 지구 모습이 거꾸로 비치고 있답니다!"
                )
                diag = (
                    "오버! 물방울들이 바닥으로 떨어지지 않고 반짝이는 수정 구슬처럼 둥실 떠다니는 것 좀 봐! "
                    "중력이 사라지면 물의 표면장력이 가장 표면적이 작은 둥근 공 모양을 만든단다, 로저!"
                )
                note_str = "우주 유체 역학 원리: 미세중력에서는 중력에 의한 압력이 사라지고 액체의 표면장력이 표면적을 최소화하여 완전한 구형을 이룹니다."
        elif topic in ("cookie_snack", "apple"):
            if norm_lang == "en":
                title_str = f"Experiment {next_page_num} — Floating Space Snacks & Crumb Containment in Zero-G"
                p1 = (
                    "Commander Orbit releases a crunchy chocolate-chip cookie and a glossy red apple in the middle of the Space Shuttle deck! "
                    "Both snacks float weightlessly at eye level, spinning slowly like tiny delicious planets."
                )
                p2 = (
                    "However, in zero gravity, crunchy cookie crumbs do not fall to the floor—they float everywhere and can drift into air vents or astronauts' eyes! "
                    "That is why real astronauts eat bite-sized or tortilla-wrapped space snacks and catch stray crumbs with airflow filters."
                )
                diag = (
                    "Copy that! Floating cookies and apples look fun to catch with your mouth, "
                    "but in a real Space Shuttle we have to watch out for floating crumbs getting into the instrument panels! Over!"
                )
                note_str = "Space Food Engineering: In microgravity, dry crumbs remain suspended in cabin air, requiring bite-sized coatings or suction filters."
            else:
                title_str = f"실험 {next_page_num} — 무중력에서 둥실 떠다니는 우주 간식과 과자 부스러기 주의보"
                p1 = (
                    "오르빗 대장님이 우주왕복선 선실 한가운데에 바삭한 초코칩 쿠키와 빨간 사과를 살짝 놓아주자, "
                    "바닥으로 떨어지지 않고 작은 행성처럼 빙글빙글 돌며 눈앞에 둥실둥실 떠다닙니다!"
                )
                p2 = (
                    "하지만 무중력 상태에서 바삭한 과자를 베어 물면 부스러기들이 바닥에 가라앉지 않고 선실 공중에 흩날려 환기구나 계기판 속으로 들어갈 수 있어요! "
                    "그래서 실제 우주비행사들은 부스러기가 날리지 않도록 한입 크기 간식을 먹거나 또띠아를 사용한답니다."
                )
                diag = (
                    "오버! 둥실 떠다니는 사과와 쿠키를 입으로 쏙 받아먹는 건 정말 재미있지만, "
                    "과자 부스러기가 무중력 선실에 떠다니면 조종석 기계를 고장 낼 수 있어서 조심해야 한단다, 로저!"
                )
                note_str = "우주 식품 공학 원리: 미세중력에서는 미세 분말과 과자 부스러기가 가라앉지 않고 부유하므로 한입 크기 코팅 식품과 공기 필터가 필수적입니다."
        elif topic == "baking_soda":
            if norm_lang == "en":
                title_str = f"Experiment {next_page_num} — Fizzing Bubbles trapped Inside a Zero-G Water Sphere"
                p1 = (
                    "When we inject fizzy baking soda into a floating sphere of water in microgravity, carbon dioxide gas bubbles burst to life inside the orb! "
                    "On Earth, bubbles immediately rise to the top and pop because of buoyancy."
                )
                p2 = (
                    "In zero gravity, there is no buoyancy! The fizzy bubbles stay trapped inside the floating water sphere, turning it into a growing, foamy crystal planet suspended in mid-air!"
                )
                diag = (
                    "Copy that! Because there is no buoyancy in zero gravity, the gas bubbles don't rise to the surface—they stay right inside the floating water ball! Over!"
                )
                note_str = "Two-Phase Microgravity Fluid Dynamics: Without gravitational buoyancy, gas bubbles remain suspended inside a liquid sphere instead of rising."
            else:
                title_str = f"실험 {next_page_num} — 무중력 물방울 속에 갇힌 보글보글 탄산 기포 구슬"
                p1 = (
                    "허공에 둥실 떠 있는 무중력 물방울 속에 발포 소다를 넣자마자, 물방울 안에서 보글보글 이산화탄소 기포들이 폭발적으로 생겨납니다! "
                    "지구에서라면 가벼운 기포가 부력 때문에 즉시 수면 위로 떠올라 터져 버렸을 거예요."
                )
                p2 = (
                    "하지만 중력이 없는 우주왕복선 안에서는 위아래 부력이 작용하지 않아서, 기포들이 물방울 밖으로 빠져나가지 못하고 투명한 물 공 안에 갇힌 채 거품 행성처럼 부풀어 오른답니다!"
                )
                diag = (
                    "오버! 무중력에서는 부력이 사라져서 탄산 기포들이 위로 떠오르지 못하고 물방울 구슬 안에 보석처럼 갇혀서 빙글빙글 도는구나, 로저!"
                )
                note_str = "우주 이상 유체 역학: 미세중력에서는 밀도 차에 의한 부력 분리가 일어나지 않아 기포가 액체 구 내부에 포집된 상태로 팽창합니다."
        else:
            # Rich Cause-and-Effect branches for freeform Space Shuttle questions!
            if any(k in lower_q for k in ("다이너마이트", "폭탄", "폭발", "터트리", "터지", "폭죽", "화약", "미사일", "dynamite", "bomb", "explode", "explosion", "firework")):
                if norm_lang == "en":
                    title_str = f"Experiment {next_page_num} — Cause & Effect: What If Dynamite Exploded Inside the Space Shuttle?"
                    p1 = (
                        "Wait—detonating real dynamite inside the sealed Space Shuttle would be catastrophic! Because dynamite carries its own chemical oxidizer, it can detonate instantly even in space. "
                        "Inside the pressurized cabin, the shockwave and superheated gas would slam into the aluminum hull and cockpit windows without anywhere to escape!"
                    )
                    p2 = (
                        "If the blast breached the hull, the 1-atmosphere cabin air would rush out into the vacuum of space in explosive decompression, while debris would fly in straight lines forever in zero-G! "
                        "Instead, Commander Orbit runs a micro-scale containment simulation inside a triple-shielded blast sphere, showing how the shockwave expands as a perfect 360-degree sphere in microgravity!"
                    )
                    diag = (
                        " Mayday prevented, cadet! If we detonated real dynamite inside the Space Shuttle, the shockwave would rupture the pressurized cabin hull and vent all our oxygen into the vacuum of space! "
                        "That's why I ran it inside our holographic containment sphere—see how a zero-G shockwave expands in a perfect sphere! Over!"
                    )
                    note_str = (
                        "Spacecraft Pressure & Shockwave Physics: Explosives contain internal oxidizers and detonate in space; inside a pressurized cabin (101.3 kPa) vs vacuum (0 kPa), a hull breach triggers rapid decompression and isotropic zero-G shrapnel dispersion."
                    )
                else:
                    title_str = f"실험 {next_page_num} — 우주왕복선 안에서 다이너마이트를 터뜨리면 벌어지는 물리적 결과"
                    p1 = (
                        "비상 경보! 밀폐된 우주왕복선 선실 안에서 진짜 다이너마이트를 터뜨리면 엄청난 대참사가 일어납니다! "
                        "다이너마이트는 자체 산화제를 포함하고 있어 폭발하는 순간 고온·고압의 가스와 충격파가 밀폐된 선실 공기를 타고 사방으로 맹렬하게 퍼져 나가며, "
                        "무중력 상태라 파편들이 바닥으로 떨어지지 않고 총알처럼 직선으로 끝없이 날아와 조종석 유리창과 기체 외벽을 타격합니다!"
                    )
                    p2 = (
                        "만약 폭발 충격으로 우주왕복선 외벽에 구멍이라도 나면, 선실 내부(1기압)와 우주 진공(0기압)의 압력 차이 때문에 순식간에 공기가 밖으로 빨려 나가는 '급격한 감압(폭발적 감압)' 현상이 일어나 산소가 사라지게 돼요! "
                        "그래서 오르빗 대장님이 3중 방폭 시뮬레이션 챔버 안에서 초소형 충격파 실험으로 안전하게 시연한 결과, 중력이 없는 우주에서는 폭발 화염과 충격파가 위아래 치우침 없이 완벽한 360도 공 모양(구형)으로 퍼져 나가는 것을 확인했답니다!"
                    )
                    diag = (
                        "오버! 큰일 날 뻔했다, 대원! 밀폐된 우주왕복선 안에서 다이너마이트를 그대로 터뜨리면 충격파로 선체 외벽이 파손되고, 우주 진공으로 공기가 한순간에 빠져나가는 급격한 감압 현상이 일어난단다! "
                        "대신 방폭 시뮬레이션 돔으로 확인해 보니 무중력 폭발 충격파가 완벽한 구형으로 퍼지는 원리를 안전하게 관찰할 수 있지, 로저!"
                    )
                    note_str = (
                        "우주선 압력 역학 및 충격파 원리: 밀폐 가압 선실(1기압) 내 폭발 시 충격파 반사와 무중력 직선 파편 비산이 발생하며, 선체 손상 시 외부 진공(0기압)으로 공기가 급격히 유출되는 급격한 감압(Rapid Decompression)이 발생합니다."
                    )
            elif any(k in lower_q for k in ("창문", "문 열", "문을 열", "구멍", "깨지", "진공", "산소", "우주복", "window", "door", "hole", "breach", "vacuum", "oxygen")):
                if norm_lang == "en":
                    title_str = f"Experiment {next_page_num} — Cause & Effect: Opening a Hatch or Window to Space Vacuum"
                    p1 = (
                        "What happens if you open a window or hatch inside the Space Shuttle while in orbit? Outside the hull is the hard vacuum of space at 0 Pa, while inside the cabin is pressurized air at 101.3 kPa! "
                        "The moment a gap opens, air rushes violently out into space at nearly the speed of sound, pulling every loose floating object toward the breach!"
                    )
                    p2 = (
                        "At the same time, as cabin pressure drops toward zero, liquid water and moisture boil instantly at room temperature because boiling point drops in a vacuum, and sound waves vanish completely because sound cannot travel without air! "
                        "Commander Orbit demonstrates this safely inside a mini vacuum bell-jar on the mid-deck."
                    )
                    diag = (
                        "Hold the hatch, cadet! Opening a window in orbit would suck all our air into the vacuum of space and make water boil at room temperature! "
                        "Watch our vacuum chamber demo instead—without air, even sound disappears completely! Over!"
                    )
                    note_str = "Vacuum Decompression & Phase Change: Pressure differential (1 atm → 0 atm) drives sonic airflow; below 6.1 kPa (triple point), liquid water boils instantly at ambient temperature."
                else:
                    title_str = f"실험 {next_page_num} — 우주왕복선 창문이나 해치를 열면 벌어지는 진공 감압 현상"
                    p1 = (
                        "우주 궤도를 비행 중인 우주왕복선에서 창문이나 출입문(해치)을 열면 어떻게 될까요? "
                        "선실 안은 지구처럼 숨 쉴 수 있는 1기압의 공기로 채워져 있지만, 창밖 우주는 공기가 전혀 없는 0기압의 진공 상태예요! "
                        "따라서 문이 열리는 순간 선실 안의 공기가 초음속에 가까운 바람을 일으키며 우주 공간으로 순식간에 빠져나가고, 둥실 떠 있던 물건들도 모두 문밖으로 빨려 나가게 됩니다!"
                    )
                    p2 = (
                        "게다가 기압이 0으로 떨어지면 물의 끓는점이 낮아져서 차가운 물방울도 즉시 부글부글 끓으며 수증기로 증발하고, 공기가 사라지기 때문에 바로 옆에서 외쳐도 소리가 전혀 들리지 않는 완벽한 침묵의 공간이 된답니다! "
                        "오르빗 대장님이 소형 진공 벨자 챔버로 이 놀라운 압력 차이와 저온 비등 현상을 안전하게 보여주었어요."
                    )
                    diag = (
                        "오버! 우주에서 창문을 열면 1기압의 선실 공기가 0기압의 우주 진공으로 순식간에 빨려 나가고, 기압이 낮아져서 상온의 물도 부글부글 끓어 증발해 버린단다! "
                        "공기가 없으면 소리조차 전달되지 않는다는 점도 꼭 기억해 두렴, 로저!"
                    )
                    note_str = "진공 감압 및 상변화 원리: 1기압 선실과 0기압 우주 진공의 압력 차로 인한 급격한 기체 유출 및 삼중점(0.006기압) 이하에서의 상온 액체 비등 현상."
            elif any(k in lower_q for k in ("울면", "눈물", "콜라", "탄산", "사이다", "멘토스", "방귀", "재채기", "던지", "밀면", "cry", "tear", "cola", "soda", "mentos", "sneeze", "fart", "throw", "push")):
                if norm_lang == "en":
                    title_str = f"Experiment {next_page_num} — Zero-G Human & Fluid Reaction: {user_utterance[:28]}"
                    p1 = (
                        f"When we test '{user_utterance}' inside the zero-gravity Space Shuttle, everyday Earth physics turns upside down! "
                        "If you cry in space, tears do not roll down your cheeks—surface tension traps them into a growing wobbling water goggle right over your eyes! "
                        "And if you open fizzy soda, sneeze, or push an object, Newton's Third Law (Action-Reaction) and zero buoyancy take over immediately."
                    )
                    p2 = (
                        "Without gravity, gas bubbles stay mixed as foamy froth inside liquids, and any force you push outward pushes your own body backward in the opposite direction across the cabin!"
                    )
                    diag = (
                        f"Copy that! Testing '{user_utterance[:34]}' in zero gravity shows how surface tension and Newton's action-reaction law rule life aboard the Space Shuttle! Over!"
                    )
                    note_str = "Microgravity Biomechanics & Newton's 3rd Law: Every expelled mass or push generates an equal and opposite momentum change (Δp = -m·v) while fluids cling via surface tension."
                else:
                    title_str = f"실험 {next_page_num} — 무중력 우주왕복선의 인체·유체 인과관계: {user_utterance[:22]}"
                    p1 = (
                        f"우주왕복선 안에서 '{user_utterance}' 실험을 하면 지구와는 정반대의 놀라운 결과가 나타납니다! "
                        "만약 우주에서 눈물을 흘리면 중력이 없어 눈물이 볼을 타고 흐르지 않고 표면장력 때문에 눈가에 커다란 물방울 안경처럼 고이게 되며, "
                        "탄산음료를 열면 부력이 없어 기포가 위로 빠지지 않고 거품 반죽처럼 통째로 뿜어져 나온답니다!"
                    )
                    p2 = (
                        "또한 무중력 허공에서 재채기를 하거나 물건을 앞으로 던지면, 뉴턴의 운동 제3법칙(작용·반작용의 법칙)에 따라 밀어낸 힘만큼 내 몸이 반대 방향 뒤쪽으로 둥실 밀려가며 회전하게 돼요!"
                    )
                    diag = (
                        f"오버! 우주왕복선에서 '{user_utterance[:24]}' 실험을 해보니 중력이 사라진 탓에 표면장력과 작용·반작용의 법칙이 곧바로 나타나는구나! 정말 흥미로운 결과야, 로저!"
                    )
                    note_str = "무중력 작용·반작용 및 유체 원리: 미세중력에서는 중력 침강과 부력이 사라져 표면장력이 지배하며, 모든 힘의 작용에 대해 반대 방향의 반작용 가속도(F = -F')가 그대로 나타납니다."
            else:
                if norm_lang == "en":
                    title_str = f"Experiment {next_page_num} — Space Shuttle Zero-G Cause & Effect: {user_utterance[:28]}"
                    p1 = (
                        f"Inside the Space Shuttle flight deck, Commander Orbit tests '{user_utterance}' under microgravity conditions! "
                        "Because the shuttle is in continuous free-fall orbit around Earth at 28,000 km/h, gravitational weight vanishes and objects neither sink nor fall."
                    )
                    p2 = (
                        "As a direct result, surface tension, inertia, and action-reaction forces dominate the outcome—Liquids pull into spheres, gases expand isotropically in all directions, and every push sends objects gliding in a straight line!"
                    )
                    diag = (
                        f"Copy that, cadet! When we test '{user_utterance[:34]}' in zero gravity, inertia and surface tension create a reaction you can never see on Earth! Over!"
                    )
                    note_str = f"Orbital Microgravity Cause & Effect: Free-fall weightlessness eliminates sedimentation, convection, and hydrostatic pressure during '{user_utterance[:30]}'."
                else:
                    title_str = f"실험 {next_page_num} — 우주왕복선 무중력 인과관계 탐구: {user_utterance[:22]}"
                    p1 = (
                        f"오르빗 대장님이 지구가 내려다보이는 우주왕복선 실험 데크에서 '{user_utterance}' 실험을 진행합니다! "
                        "우주왕복선 안은 지구 궤도를 초속 7.8킬로미터로 자유 낙하하는 무중력 상태이기 때문에, 위아래로 끌어당기는 무게와 부력이 사라집니다."
                    )
                    p2 = (
                        "그 결과, 지구에서는 바닥으로 가라앉거나 위로 솟구치던 물질들이 우주왕복선 안에서는 모든 방향으로 고르게 퍼져 나가고, 작은 힘만 작용해도 관성과 반작용의 법칙에 따라 허공을 가로질러 움직이는 뚜렷한 인과관계를 보여줍니다!"
                    )
                    diag = (
                        f"오버! 우주왕복선 안에서 '{user_utterance[:24]}' 실험을 수행한 결과, 중력과 대류가 사라지면서 관성과 표면장력이 지배하는 생생한 우주 물리 현상이 나타났어, 로저!"
                    )
                    note_str = f"우주 물리 인과관계 노트: 자유 낙하 미세중력 환경에서는 침강·부력·열대류가 제거되어 '{user_utterance[:24]}' 실험 시 등방성 확산과 관성 운동이 나타납니다."

        return {
            "page_title": title_str,
            "story_paragraphs": [p1, p2],
            "character_name": character_name,
            "character_persona_voice": persona_voice,
            "character_dialogue": diag,
            "science_or_moral_note": note_str,
            "visual_state": {
                "theme": space_theme,
                "lang": norm_lang,
                "character_persona_voice": persona_voice,
                "item_key": topic,
                "item_label": get_item_display_label(topic, lang=norm_lang, fallback_text=user_utterance[:28]),
                "accumulated_items": acc_items,
                "accumulated_labels": acc_labels,
                "ph_level": new_ph,
                "temperature_c": new_temp,
                "liquid_color": [96, 185, 250],
                "liquid_level": 0.65,
                "bubbles_intensity": 0.65,
                "steam_intensity": 0.25,
                "foam_intensity": 0.20,
                "accent_color": [167, 139, 250],
            },
            "nano_banana_prompt": (
                f"Nano Banana 16:9 Space Shuttle Science Illustration: Commander Orbit inside the {space_theme} cabin "
                f"demonstrating the exact cause-and-effect outcome of: {user_utterance}."
            ),
            "veo_motion_prompt": (
                f"Animated microgravity space scene with Commander Orbit demonstrating {user_utterance}."
            ),
            "suggested_followups": space_followups_en if norm_lang == "en" else space_followups_ko,
        }

    # =========================================================================
    # SCENARIO 3: DR. NOVA'S HYDROCHLORIC ACID LAB (`science_hcl_lab`)
    # Clean 1-by-1 Independent Chemical Experiment Mode!
    # =========================================================================
    hcl_followups_ko = [
        "열쇠(금속)를 넣으면 어떻게 되나요?",
        "사과를 넣으면 어떻게 되나요?",
        "과자를 넣으면 어떻게 되나요?",
        "물을 뿌리면 어떻게 되나요?",
        "베이킹소다를 한 스푼 넣으면 어떻게 되나요?",
        "달걀 껍데기를 넣으면 어떻게 되나요?",
        "순금 반지를 넣으면 어떻게 되나요?",
    ]
    hcl_followups_en = [
        "What happens if we drop a metal key in?",
        "What happens if we drop a red apple in?",
        "What happens if we drop a crunchy cookie in?",
        "What happens if we spray water in?",
        "What happens if we add a spoonful of baking soda?",
        "What happens if we drop an eggshell in?",
        "What happens if we drop a pure gold ring in?",
    ]

    if topic == "metal_key":
        if norm_lang == "en":
            title_str = f"Experiment {next_page_num} — Fizzing Metal Key and Silver Hydrogen Plumes"
            p1 = (
                "We carefully drop a gleaming iron-and-brass key into the five-hundred-milliliter hydrochloric acid beaker! "
                "The instant the metal touches the bottom, iron and zinc atoms donate electrons to the hydrogen ions in the strong acid, "
                "sending hundreds of silver hydrogen gas bubbles streaming upward in a shimmering column!"
            )
            p2 = (
                f"Because metal-acid oxidation is an exothermic reaction, the digital thermometer climbs to {new_temp} °C, "
                f"while dissolved ferrous chloride ions tint the solution a rich emerald green at pH {new_ph}."
            )
            diag = (
                f"Hoo-hoo! Look at those silver hydrogen bubbles fizzing right off the metal key! "
                f"The hydrochloric acid is oxidizing the metal, releasing hydrogen gas and warming the beaker to {new_temp} degrees Celsius, hoo-hoo!"
            )
            note_str = (
                f"Reaction Equation: Fe(s) + 2HCl(aq) → FeCl₂(aq) + H₂(g)↑ (Exothermic redox reaction) | "
                f"Live Telemetry: pH {new_ph}, Temp {new_temp} °C."
            )
            f_list = [
                "What happens if we drop a red apple in?",
                "What happens if we drop a crunchy cookie in?",
                "What happens if we spray water from above?",
                "What happens if we add a spoonful of baking soda?",
            ]
        else:
            title_str = f"실험 {next_page_num} — 보글보글 금속 열쇠와 은빛 수소 기포 기둥"
            p1 = (
                "반짝이는 철 합금 열쇠를 오백 밀리리터 염산 비커 속으로 조심스럽게 떨어뜨렸어요! "
                "열쇠가 바닥에 닿는 순간, 금속 표면의 철과 아연 원자들이 강산의 수소 이온에게 전자를 건네주며 수백 개의 은빛 수소 기체 방울이 보글보글 기둥처럼 솟구쳐 오릅니다!"
            )
            p2 = (
                f"산화 환원 반응에서 발생한 발열 에너지 때문에 비커 온도는 섭씨 {new_temp}도로 훌쩍 올라가고, "
                f"녹아 나온 염화철 이온 덕분에 용액이 신비로운 진초록 에메랄드빛(산성도 {new_ph})으로 물들고 있답니다."
            )
            diag = (
                f"부엉~ 부엉! 방금 열쇠가 들어가자마자 은빛 수소 거품이 기둥처럼 솟구치는 거 봤니? "
                f"염산이 금속 표면과 반응해 수소 기체를 만들면서 온도도 {new_temp}도까지 따뜻해졌단다, 부엉!"
            )
            note_str = (
                f"핵심 과학 노트: Fe(s) + 2HCl(aq) → FeCl₂(aq) + H₂(g)↑ (금속-산 산화 환원 발열 반응) | "
                f"실험 계측: pH {new_ph}, 온도 {new_temp} °C."
            )
            f_list = [
                "사과를 넣으면 어떻게 되나요?",
                "바삭한 초코칩 과자를 넣으면 거품이 생길까요?",
                "위에서 물을 뿌리면 염산이 어떻게 변하나요?",
                "베이킹소다를 한 스푼 넣으면 화산처럼 폭발하나요?",
            ]
        return {
            "page_title": title_str,
            "story_paragraphs": [p1, p2],
            "character_name": character_name,
            "character_persona_voice": persona_voice,
            "character_dialogue": diag,
            "science_or_moral_note": note_str,
            "visual_state": {
                "theme": prev_visual_state.get("theme", "science_hcl_lab"),
                "lang": norm_lang,
                "character_persona_voice": persona_voice,
                "item_key": "metal_key",
                "item_label": get_item_display_label("metal_key", lang=norm_lang),
                "accumulated_items": acc_items,
                "accumulated_labels": acc_labels,
                "ph_level": new_ph,
                "temperature_c": new_temp,
                "liquid_color": [74, 212, 165],
                "liquid_level": 0.66,
                "bubbles_intensity": 0.94,
                "steam_intensity": 0.58,
                "foam_intensity": 0.25,
                "accent_color": [52, 211, 153],
            },
            "nano_banana_prompt": (
                "Nano Banana 16:9 Lab Scene: preserve the glass beaker and Dr. Nova owl professor "
                "while a brass key fizzes with silver H2 bubbles at the bottom of the emerald acid."
            ),
            "veo_motion_prompt": (
                "Cinematic close-up of the glass laboratory beaker: streams of silver hydrogen bubbles rise from the submerged brass key while Dr. Nova gestures warmly."
            ),
            "suggested_followups": f_list,
        }

    if topic == "apple":
        if norm_lang == "en":
            title_str = f"Experiment {next_page_num} — Amber Pigment Swirls and Pectin Digestion with a Red Apple"
            p1 = (
                "A glossy red apple splashes into the hydrochloric acid beaker! "
                f"While it does not fizz like reactive metal, the natural red anthocyanin pigments in the apple skin react with the strong acid (pH {new_ph}) "
                "to send warm amber and orange ribbons swirling through the liquid."
            )
            p2 = (
                "Just like human stomach acid digesting fruit after a meal, the hydronium ions hydrolyze the long pectin chains "
                "binding the apple's cell walls, softening the fruit's surface into a tender baked-apple texture."
            )
            diag = (
                "Hoo-hoo! Oh my feathers, watch how the red apple bobs in the acid and turns a warm amber-orange hue! "
                "This is the exact same way your stomach acid digests a crisp apple after lunch, hoo-hoo!"
            )
            note_str = (
                f"Biochemical Reaction: Pectin polysaccharide + H₂O --(H⁺ catalyst)--> soluble galacturonic acid + anthocyanin pH shift | "
                f"Live Telemetry: pH {new_ph}, Temp {new_temp} °C."
            )
            f_list = [
                "What happens if we drop a crunchy cookie in?",
                "What happens if we drop a metal key in?",
                "What happens if we spray water from above?",
                "What happens if we add a spoonful of baking soda?",
            ]
        else:
            title_str = f"실험 {next_page_num} — 호박빛 색소 물결과 펙틴 소화가 일어나는 빨간 사과"
            p1 = (
                "탐스러운 빨간 사과를 염산 비커 속으로 풍덩 넣었어요! "
                f"금속처럼 격렬한 수소 기포가 터지지는 않지만, 사과 껍질 속 붉은 천연 안토시아닌 색소가 강산 용액(산성도 {new_ph})과 반응해 따뜻한 주황빛과 호박색 물결을 피워 올립니다."
            )
            p2 = (
                "마치 우리 뱃속의 위액이 과일을 소화시키는 것처럼, 산성 용액이 사과 세포벽을 단단하게 묶어 주던 펙틴 다당류 사슬을 가수분해하여 사과 겉면이 구운 사과처럼 말랑말랑해진답니다."
            )
            diag = (
                "부엉~ 부엉! 호오, 사과가 비커 속에 둥실 떠서 껍질 색깔이 노르스름한 호박빛으로 변하고 있어! "
                "우리 위장 속 위산이 사과를 부드럽게 소화시키는 원리와 똑같단다, 부엉!"
            )
            note_str = (
                f"생화학 노트: 펙틴 다당류 + H₂O --(H⁺ 촉매)--> 가용성 당류 분해 및 안토시아닌 산성 변색 | "
                f"실험 계측: pH {new_ph}, 온도 {new_temp} °C."
            )
            f_list = [
                "바삭한 과자를 넣으면 어떻게 되나요?",
                "금속 열쇠를 넣으면 어떻게 되나요?",
                "위에서 물을 시원하게 뿌리면 어떻게 되나요?",
                "베이킹소다를 넣으면 어떻게 되나요?",
            ]
        return {
            "page_title": title_str,
            "story_paragraphs": [p1, p2],
            "character_name": character_name,
            "character_persona_voice": persona_voice,
            "character_dialogue": diag,
            "science_or_moral_note": note_str,
            "visual_state": {
                "theme": prev_visual_state.get("theme", "science_hcl_lab"),
                "lang": norm_lang,
                "character_persona_voice": persona_voice,
                "item_key": "apple",
                "item_label": get_item_display_label("apple", lang=norm_lang),
                "accumulated_items": acc_items,
                "accumulated_labels": acc_labels,
                "ph_level": new_ph,
                "temperature_c": new_temp,
                "liquid_color": [228, 158, 122],
                "liquid_level": 0.68,
                "bubbles_intensity": 0.38,
                "steam_intensity": 0.28,
                "foam_intensity": 0.18,
                "accent_color": [248, 113, 113],
            },
            "nano_banana_prompt": (
                "Nano Banana 16:9 Lab Scene: preserve the glass beaker and Dr. Nova "
                "as a glossy red apple floats in the acid with warm amber pigment swirls."
            ),
            "veo_motion_prompt": (
                "A glossy red apple bobs gently in the laboratory beaker as warm amber pectin swirls diffuse through the liquid."
            ),
            "suggested_followups": f_list,
        }

    if topic == "cookie_snack":
        if norm_lang == "en":
            title_str = f"Experiment {next_page_num} — Creamy Carbon Dioxide Foam Crown from a Crunchy Cookie"
            p1 = (
                "We drop a crunchy chocolate-chip cookie into the hydrochloric acid beaker! "
                "As the acid rapidly soaks into the porous baked snack, the sodium bicarbonate leavening agent inside the cookie reacts immediately "
                "with the acid to crown the top of the liquid with a thick, creamy layer of carbon dioxide foam!"
            )
            p2 = (
                f"At the same time, acid hydrolysis breaks down the starch and sugar matrix into golden crumbs that swirl "
                f"through the beaker while nudging the solution pH up to {new_ph}."
            )
            diag = (
                "Hoo-hoo! Look at that fluffy white foam crown rising right to the top of our beaker like a cappuccino! "
                "The baking soda baked inside the cookie met the hydrochloric acid and released fizzy carbon dioxide gas, hoo-hoo!"
            )
            note_str = (
                f"Reaction Equation: NaHCO₃ (cookie leavening) + HCl(aq) → NaCl(aq) + H₂O(l) + CO₂(g)↑ + starch hydrolysis | "
                f"Live Telemetry: pH {new_ph}, Temp {new_temp} °C."
            )
            f_list = [
                "What happens if we spray water from above?",
                "What happens if we drop a metal key in?",
                "What happens if we drop a red apple in?",
                "What happens if we add a full spoonful of baking soda?",
            ]
        else:
            title_str = f"실험 {next_page_num} — 초코칩 과자가 피워 올린 몽글몽글 이산화탄소 거품 왕관"
            p1 = (
                "달콤하고 바삭한 초코칩 과자를 염산 비커 속에 넣었어요! "
                "구멍이 송송 뚫린 다공성 과자 속으로 염산이 빠르게 스며들면서, 과자를 부풀릴 때 넣었던 팽창제인 탄산수소나트륨이 강산과 즉시 반응해 보드라운 이산화탄소 크림 거품 층을 비커 맨 위에 몽글몽글 피워 올립니다!"
            )
            p2 = (
                "동시에 과자의 주성분인 녹말과 당분이 산성 가수분해를 거쳐 고소한 금빛 부스러기로 풀어지며 용액 위쪽에 하얀 거품 왕관을 완성합니다."
            )
            diag = (
                "부엉~ 부엉! 호오, 과자가 들어가자마자 비커 맨 위에 카푸치노 우유 거품 같은 하얀 거품 왕관이 생겼구나! "
                "과자 속 베이킹소다가 염산과 만나 이산화탄소 기체를 뿜어낸 거란다, 부엉!"
            )
            note_str = (
                f"핵심 과학 노트: NaHCO₃(과자 팽창제) + HCl(aq) → NaCl(aq) + H₂O(l) + CO₂(g)↑ 및 전분 산 가수분해 | "
                f"실험 계측: pH {new_ph}, 온도 {new_temp} °C."
            )
            f_list = [
                "위에서 물을 뿌리면 어떻게 되나요?",
                "금속 열쇠를 넣으면 어떻게 되나요?",
                "빨간 사과를 넣으면 어떻게 되나요?",
                "베이킹소다를 한 스푼 넣으면 어떻게 되나요?",
            ]
        return {
            "page_title": title_str,
            "story_paragraphs": [p1, p2],
            "character_name": character_name,
            "character_persona_voice": persona_voice,
            "character_dialogue": diag,
            "science_or_moral_note": note_str,
            "visual_state": {
                "theme": prev_visual_state.get("theme", "science_hcl_lab"),
                "lang": norm_lang,
                "character_persona_voice": persona_voice,
                "item_key": "cookie_snack",
                "item_label": get_item_display_label("cookie_snack", lang=norm_lang),
                "accumulated_items": acc_items,
                "accumulated_labels": acc_labels,
                "ph_level": new_ph,
                "temperature_c": new_temp,
                "liquid_color": [218, 178, 108],
                "liquid_level": 0.69,
                "bubbles_intensity": 0.78,
                "steam_intensity": 0.26,
                "foam_intensity": 0.90,
                "accent_color": [251, 191, 36],
            },
            "nano_banana_prompt": (
                "Nano Banana 16:9 Lab Scene: preserve the glass beaker and Dr. Nova "
                "with a crumbling chocolate-chip cookie and a thick creamy CO2 foam layer on top."
            ),
            "veo_motion_prompt": (
                "A chocolate-chip cookie drops into the laboratory beaker, splashes, and forms a frothy white CO2 foam crown."
            ),
            "suggested_followups": f_list,
        }

    if topic == "water_spray":
        if norm_lang == "en":
            title_str = f"Experiment {next_page_num} — Sparkling Water Spray Dilution and Hydration Heat"
            p1 = (
                "A fine mist of pure water sprays from the laboratory nozzle into our hydrochloric acid beaker! "
                f"As sparkling droplets ripple across the surface, water molecules surround the hydrogen ions in a hydration reaction, raising the liquid level and clearing the solution to a bright sky-cyan tint at pH {new_ph}."
            )
            p2 = (
                f"Because hydrating a strong acid releases exothermic hydration energy, the digital thermometer ticks up to {new_temp} °C while diluting the acid's concentration."
            )
            diag = (
                f"Hoo-hoo! Spraying fresh water diluted the acid and raised the pH to {new_ph}! "
                "Always remember, young scientist: diluting strong acid releases hydration heat, hoo-hoo!"
            )
            note_str = (
                f"Physical Chemistry Note: HCl(aq) + H₂O(l) → H₃O⁺(aq) + Cl⁻(aq) (Exothermic hydration, ΔH_hyd < 0) | "
                f"Live Telemetry: pH {new_ph}, Temp {new_temp} °C."
            )
            f_list = [
                "What happens if we add a spoonful of baking soda?",
                "What happens if we drop an eggshell in?",
                "What happens if we drop a pure gold ring in?",
                "What happens if we drop a metal key in?",
            ]
        else:
            title_str = f"실험 {next_page_num} — 시원한 물 분무기 희석과 수화열 현상"
            p1 = (
                "분무기 노즐로 맑은 물을 비커 위에서 고르게 뿌려 주었어요! "
                f"반짝이는 물방울들이 수면 위에 동심원 파문을 그리며 내려앉자, 물 분자들이 수소 이온을 에워싸는 수화 반응이 일어나며 수위가 높아지고 용액이 투명한 하늘빛(산성도 {new_ph})으로 맑아집니다."
            )
            p2 = (
                f"특히 강산이 물과 섞일 때는 에너지를 방출하는 발열 수화 반응이 일어나기 때문에 비커 온도가 섭씨 {new_temp}도로 살짝 따뜻해지는 것을 온도계로 확인할 수 있답니다!"
            )
            diag = (
                f"부엉~ 부엉! 시원하게 물을 뿌려 주니까 비커 속 용액이 묽어지면서 산성도가 {new_ph}로 올라갔구나! "
                "강산이 물 분자와 결합할 때는 따뜻한 수화열이 나온다는 점도 꼭 기억해 두렴, 부엉!"
            )
            note_str = (
                f"핵심 과학 노트: HCl(aq) + H₂O(l) → H₃O⁺(옥소늄 이온) + Cl⁻(aq) (발열 수화 반응 및 산 희석) | "
                f"실험 계측: pH {new_ph}, 온도 {new_temp} °C."
            )
            f_list = [
                "베이킹소다를 한 스푼 넣으면 어떻게 되나요?",
                "달걀 껍데기를 넣으면 어떻게 되나요?",
                "순금 반지를 넣으면 녹을까요, 안 녹을까요?",
                "금속 열쇠를 넣으면 어떻게 되나요?",
            ]
        return {
            "page_title": title_str,
            "story_paragraphs": [p1, p2],
            "character_name": character_name,
            "character_persona_voice": persona_voice,
            "character_dialogue": diag,
            "science_or_moral_note": note_str,
            "visual_state": {
                "theme": prev_visual_state.get("theme", "science_hcl_lab"),
                "lang": norm_lang,
                "character_persona_voice": persona_voice,
                "item_key": "water_spray",
                "item_label": get_item_display_label("water_spray", lang=norm_lang),
                "accumulated_items": acc_items,
                "accumulated_labels": acc_labels,
                "ph_level": new_ph,
                "temperature_c": new_temp,
                "liquid_color": [125, 220, 248],
                "liquid_level": 0.72,
                "bubbles_intensity": 0.38,
                "steam_intensity": 0.45,
                "foam_intensity": 0.20,
                "accent_color": [56, 189, 248],
            },
            "nano_banana_prompt": (
                "Nano Banana 16:9 Lab Scene: sparkling cyan water droplets spray from a nozzle into the glass beaker, creating concentric surface ripples."
            ),
            "veo_motion_prompt": (
                "Sparkling water droplets spray from a laboratory nozzle into the glass beaker, creating concentric shimmering ripples."
            ),
            "suggested_followups": f_list,
        }

    if topic == "baking_soda":
        if norm_lang == "en":
            title_str = f"Experiment {next_page_num} — The Erupting Emerald Baking Soda Foam Volcano"
            p1 = (
                "A spoonful of white baking soda (sodium bicarbonate) cascades into the hydrochloric acid beaker—whoosh! "
                "The alkaline powder meets the strong acid in a rapid acid-base neutralization reaction, sending a dazzling plume of white and emerald carbon dioxide foam surging up to the beaker rim!"
            )
            p2 = (
                f"The turbulent cloud of CO₂ bubbles churns through the solution and lifts the pH all the way to {new_ph}, "
                "transforming harsh hydrochloric acid toward gentle salt water!"
            )
            diag = (
                f"Hoo-hoo! What a magnificent carbon dioxide foam volcano! "
                f"Acid and base just neutralized each other into salt water and carbon dioxide gas, jumping the pH to {new_ph}, hoo-hoo!"
            )
            note_str = (
                f"Neutralization Equation: NaHCO₃(s) + HCl(aq) → NaCl(aq) + H₂O(l) + CO₂(g)↑ | "
                f"Live Telemetry: pH {new_ph}, Temp {new_temp} °C."
            )
            f_list = [
                "What happens if we spray water in?",
                "What happens if we drop an eggshell in?",
                "What happens if we drop a pure gold ring in?",
                "What happens if we drop a metal key in?",
            ]
        else:
            title_str = f"실험 {next_page_num} — 솟구치는 에메랄드 거품 화산과 베이킹소다 중화 반응"
            p1 = (
                "하얀 베이킹소다(탄산수소나트륨) 가루 한 스푼을 비커 속으로 솔솔 넣는 순간, 슈우욱 소리와 함께 폭발적인 산 염기 중화 반응이 일어납니다! "
                "염기성 가루와 강산인 염산이 만나 새하얗고 푸른 이산화탄소 거품 기둥이 비커 입구 위로 화산처럼 솟구쳐 오릅니다!"
            )
            p2 = (
                f"급격히 발생한 이산화탄소 기포 구름이 비커 속 용액을 휘저으며 산성도를 {new_ph}까지 끌어올려, "
                "강산의 성질을 부드러운 소금물 쪽으로 중화시켜 준답니다!"
            )
            diag = (
                f"부엉~ 부엉! 우와아, 비커 위로 하얀 이산화탄소 거품 화산이 멋지게 솟구쳐 올랐구나! "
                f"산성과 염기성이 만나 소금물과 이산화탄소가 되는 중화 반응 덕분에 산성도가 {new_ph}까지 올라갔단다, 부엉!"
            )
            note_str = (
                f"핵심 과학 노트: NaHCO₃(s) + HCl(aq) → NaCl(aq) + H₂O(l) + CO₂(g)↑ (급격한 기체 발생 및 산 염기 중화 반응) | "
                f"실험 계측: pH {new_ph}, 온도 {new_temp} °C."
            )
            f_list = [
                "물을 뿌리면 어떻게 되나요?",
                "달걀 껍데기를 넣으면 어떻게 되나요?",
                "순금 반지를 넣으면 어떻게 되나요?",
                "금속 열쇠를 넣으면 어떻게 되나요?",
            ]
        return {
            "page_title": title_str,
            "story_paragraphs": [p1, p2],
            "character_name": character_name,
            "character_persona_voice": persona_voice,
            "character_dialogue": diag,
            "science_or_moral_note": note_str,
            "visual_state": {
                "theme": prev_visual_state.get("theme", "science_hcl_lab"),
                "lang": norm_lang,
                "character_persona_voice": persona_voice,
                "item_key": "baking_soda",
                "item_label": get_item_display_label("baking_soda", lang=norm_lang),
                "accumulated_items": acc_items,
                "accumulated_labels": acc_labels,
                "ph_level": new_ph,
                "temperature_c": new_temp,
                "liquid_color": [94, 234, 212],
                "liquid_level": 0.85,
                "bubbles_intensity": 1.0,
                "steam_intensity": 0.65,
                "foam_intensity": 1.0,
                "accent_color": [45, 212, 191],
            },
            "nano_banana_prompt": (
                "Nano Banana 16:9 Lab Scene: preserve the glass beaker and Dr. Nova "
                "while a spectacular white and cyan CO2 volcano foam plume erupts over the rim."
            ),
            "veo_motion_prompt": (
                "Spectacular fizzing CO2 volcano foam erupts from the glass beaker as baking soda neutralizes the acid while Dr. Nova cheers."
            ),
            "suggested_followups": f_list,
        }

    if topic in ("egg_shell", "gold_ring", "zero_g_candle"):
        if topic == "egg_shell":
            if norm_lang == "en":
                title_str = f"Experiment {next_page_num} — The Bobbing Submarine Egg and Calcium Carbonate Bubbles"
                p1 = "When we lower an eggshell into the hydrochloric acid, the calcium carbonate (CaCO₃) in the shell reacts with the acid to coat the entire shell in pearl-like carbon dioxide bubbles!"
                p2 = "Once enough gas bubbles cling to the shell like tiny life jackets, buoyancy lifts the egg to the surface, where the bubbles pop and the egg dives back down like a dancing submarine!"
                diag = "Hoo-hoo! Look at those pearl-like bubbles lifting the eggshell up and down like a little submarine! Calcium carbonate and hydrochloric acid make carbon dioxide gas, hoo-hoo!"
                note_str = f"Reaction Equation: CaCO₃(s) + 2HCl(aq) → CaCl₂(aq) + H₂O(l) + CO₂(g)↑ (Buoyancy oscillation) | pH {new_ph}, {new_temp} °C."
            else:
                title_str = f"실험 {next_page_num} — 오르락내리락 춤추는 잠수함 달걀 껍데기"
                p1 = "달걀을 껍데기째 염산 비커 속에 넣자, 달걀 껍데기의 주성분인 탄산칼슘이 염산과 반응해 표면 전체에 진주알 같은 이산화탄소 기포가 다닥다닥 맺힙니다!"
                p2 = "기포가 충분히 맺히면 부력 때문에 달걀이 두둥실 수면 위로 떠올랐다가, 수면에서 기포가 톡톡 터지면 다시 가라앉아 오르락내리락 춤추는 잠수함 달걀이 된답니다!"
                diag = "부엉~ 부엉! 호오, 달걀 껍데기에서 생긴 기포들이 구명조끼처럼 달걀을 둥실 띄웠다가 다시 내려가게 하고 있어! 정말 신기한 부력 잠수함이구나, 부엉!"
                note_str = f"핵심 과학 노트: CaCO₃(달걀 껍데기) + 2HCl → CaCl₂ + H₂O + CO₂(g)↑ (부력 진동 현상) | pH {new_ph}, 온도 {new_temp} °C."
        elif topic == "gold_ring":
            if norm_lang == "en":
                title_str = f"Experiment {next_page_num} — The Noble Metal Secret: 24K Pure Gold Ring in Acid"
                p1 = "We drop a gleaming 24K pure gold ring into the hydrochloric acid beaker! Unlike the iron key, not a single bubble forms on the golden band—it shines completely unharmed at the bottom!"
                p2 = "Gold (Au) is a noble metal with a much lower ionization tendency than hydrogen, so ordinary hydrochloric acid cannot oxidize or dissolve it at all!"
                diag = "Hoo-hoo! Not a single bubble on the pure gold ring! Iron fizzes away, but noble gold resists hydrochloric acid completely, hoo-hoo!"
                note_str = "Metal Reactivity Series: K > Ca > Na > Mg > Al > Zn > Fe > (H) > Cu > Ag > Pt > Au (Noble gold does not react with HCl)."
            else:
                title_str = f"실험 {next_page_num} — 염산에도 끄떡없는 황금의 비밀, 순금 반지 실험"
                p1 = "반짝이는 순금 반지를 염산 비커 속에 넣었어요! 철 열쇠가 들어갔을 때와 달리, 순금 반지 표면에서는 기포가 단 한 방울도 생기지 않고 영롱한 황금빛을 그대로 유지합니다!"
                p2 = "금은 수소보다 이온화 경향이 훨씬 낮은 귀금속이라서 일반 염산에는 절대 녹지 않아요. 금을 녹이려면 질산과 염산을 섞은 왕수가 필요하답니다!"
                diag = "부엉~ 부엉! 호오, 철 열쇠는 보글보글 기포를 내며 녹는데 순금 반지는 염산 속에서도 전혀 반응하지 않고 반짝이고 있구나! 이게 바로 귀금속의 반응성 차이란다, 부엉!"
                note_str = "핵심 과학 노트: 금속의 이온화 경향 (Fe > H > Cu > Ag > Pt > Au)에 의해 순금은 염산에 산화되지 않습니다."
        else:
            if norm_lang == "en":
                title_str = f"Experiment {next_page_num} — The Glowing Spherical Flame & Hydrogen Pop Test"
                p1 = "When we bring a flame near the hydrogen gas rising from an acid-metal reaction, the hydrogen reacts rapidly with oxygen in the air with a cheerful 'pop!' sound to form pure water vapor!"
                p2 = "In a sealed low-convection chamber, the flame burns as a calm glowing sphere while demonstrating how chemical energy converts into heat and light."
                diag = "Hoo-hoo! Hydrogen gas from our acid beaker meets oxygen and makes a little 'pop!' as it turns into water drops, hoo-hoo!"
                note_str = "Combustion Equation: 2H₂(g) + O₂(g) → 2H₂O(g) + Energy (Hydrogen pop test)."
            else:
                title_str = f"실험 {next_page_num} — 불꽃 반응과 수소 기체의 퍽 소리 연소 실험"
                p1 = "염산 반응 비커 위에서 모은 수소 기체에 작은 불꽃을 가까이 가져가면, 공기 중의 산소와 순간적으로 결합하며 경쾌한 '퍽!' 소리와 함께 깨끗한 물방울(수증기)로 변합니다!"
                p2 = "또한 대류가 차단된 특수 챔버 안에서 촛불을 켜면 불꽃이 공처럼 둥글고 푸르게 타오르며 연소 화학의 원리를 보여준답니다!"
                diag = "부엉~ 부엉! 수소 기체가 산소와 만나면서 '퍽!' 소리를 내며 깨끗한 물방울로 변하는 걸 보니 화학 반응의 인과관계가 한눈에 보이는구나, 부엉!"
                note_str = "핵심 과학 노트: 2H₂(g) + O₂(g) → 2H₂O(l) (수소 기체의 폭명성 연소 반응)."

        return {
            "page_title": title_str,
            "story_paragraphs": [p1, p2],
            "character_name": character_name,
            "character_persona_voice": persona_voice,
            "character_dialogue": diag,
            "science_or_moral_note": note_str,
            "visual_state": {
                "theme": prev_visual_state.get("theme", "science_hcl_lab"),
                "lang": norm_lang,
                "character_persona_voice": persona_voice,
                "item_key": topic,
                "item_label": get_item_display_label(topic, lang=norm_lang),
                "accumulated_items": acc_items,
                "accumulated_labels": acc_labels,
                "ph_level": new_ph,
                "temperature_c": new_temp,
                "liquid_color": [98, 215, 235],
                "liquid_level": 0.72,
                "bubbles_intensity": 0.65 if topic != "gold_ring" else 0.15,
                "steam_intensity": 0.30,
                "foam_intensity": 0.25,
                "accent_color": [56, 189, 248],
            },
            "nano_banana_prompt": f"Nano Banana 16:9 Lab Scene: depict {title_str} inside the glass beaker with Dr. Nova.",
            "veo_motion_prompt": f"Animated science illustration of {title_str} with Dr. Nova explaining.",
            "suggested_followups": hcl_followups_en[:4] if norm_lang == "en" else hcl_followups_ko[:4],
        }

    # Custom HCl science experiment turn with rich cause-and-effect branches!
    if any(k in lower_q for k in ("다이너마이트", "폭탄", "폭발", "터트리", "터지", "나트륨", "칼륨", "화약", "dynamite", "bomb", "explode", "sodium", "potassium")):
        if norm_lang == "en":
            title_str = f"Experiment {next_page_num} — High-Energy Reaction & Blast Shield Containment: {user_utterance[:26]}"
            p1 = (
                f"Whoa! Dropping an explosive or alkali metal like in '{user_utterance}' directly into hydrochloric acid triggers an ultra-fast exothermic reaction! "
                "A sudden surge of gas pressure and heat would shatter an unprotected glass beaker and splash strong acid outward."
            )
            p2 = (
                "Dr. Nova immediately lowers the polycarbonate blast shield! Inside the reinforced containment dome, we observe how rapid gas expansion and neutralization release intense heat and shockwaves safely."
            )
            diag = (
                "Hoo-hoo! Safety goggles on! An explosive reaction in hydrochloric acid creates a massive pressure wave that would shatter glass, "
                "so our triple-layer blast shield contained the shockwave safely, hoo-hoo!"
            )
            note_str = "Chemical Safety & Exothermic Kinetics: Rapid gas evolution (ΔV/Δt) and exothermic heat in strong acid create overpressure shockwaves requiring blast containment."
        else:
            title_str = f"실험 {next_page_num} — 폭발성 물질과 강산의 급격한 압력 팽창 실험: {user_utterance[:20]}"
            p1 = (
                f"잠깐! 염산 비커 안에서 '{user_utterance}' 실험을 실제로 하면 순식간에 엄청난 고온 가스와 충격파가 발생합니다! "
                "폭발 압력이나 알칼리 금속의 급격한 수소 폭발이 일어나면 유리 비커가 산산조각 나고 위험한 강산 용액이 사방으로 튈 수 있어요!"
            )
            p2 = (
                "그래서 노바 박사님이 즉시 3중 방폭 안전 유리돔을 작동시켰어요! 방폭 돔 안에서 관찰한 결과, 급격한 기체 부피 팽창이 산성 용액을 밀어 올리며 강렬한 충격파와 수증기 기둥을 만드는 인과관계를 안전하게 확인했답니다!"
            )
            diag = (
                "부엉~ 부엉! 보안경 착용 필수! 염산 비커에서 폭발 반응이 일어나면 급격한 기체 팽창 때문에 유리가 깨지고 강산이 튈 수 있어서, "
                "방폭 안전 돔 안에서 충격파와 기체 팽창 원리를 안전하게 실험했단다, 부엉!"
            )
            note_str = "화학 안전 및 급격한 기체 팽창 원리: 밀폐 용기 및 강산(HCl) 내부에서 폭발적 발열·기체 생성 반응이 일어나면 충격파에 의해 용기가 파손되므로 방폭 차폐가 필수적입니다."
    elif any(k in lower_q for k in ("드라이아이스", "액체질소", "얼음", "얼리", "끓", "가열", "전자레인지", "dry ice", "liquid nitrogen", "ice", "freeze", "boil", "heat")):
        if norm_lang == "en":
            title_str = f"Experiment {next_page_num} — Extreme Temperature & Phase Change in Acid: {user_utterance[:26]}"
            p1 = (
                f"When we test '{user_utterance}' in the hydrochloric acid beaker, dramatic thermal transfer and phase changes occur immediately! "
                "If dry ice or ice meets the acid, rapid sublimation or cooling sends thick white fog cascading over the beaker rim while slowing down reaction kinetics."
            )
            p2 = (
                "Conversely, heating strong hydrochloric acid drives hydrogen chloride gas (HCl↑) out of solution as sharp acidic vapor, which is why lab fume hoods must capture the steam!"
            )
            diag = (
                f"Hoo-hoo! Look at that thick white fog rolling over the beaker! Changing the temperature with '{user_utterance[:28]}' directly changes molecular speed and gas solubility, hoo-hoo!"
            )
            note_str = "Thermodynamics & Reaction Kinetics: Temperature directly governs activation energy (Arrhenius law) and gas solubility in aqueous HCl."
        else:
            title_str = f"실험 {next_page_num} — 염산 비커의 극저온·가열 상변화 인과관계: {user_utterance[:20]}"
            p1 = (
                f"염산 비커에서 '{user_utterance}' 실험을 진행하자 즉각적인 열교환과 상태 변화가 일어납니다! "
                "드라이아이스나 얼음을 넣으면 급격한 승화와 냉각으로 인해 구름처럼 새하얀 안개(응결된 수증기)가 비커 밖으로 폭포수처럼 쏟아져 내리며 분자 운동이 느려집니다."
            )
            p2 = (
                "반대로 염산을 뜨겁게 가열하면 물보다 끓는점이 낮은 염화수소(HCl) 기체가 매캐한 산성 증기가 되어 날아가므로, 실험실 환기 후드를 꼭 켜야 한답니다!"
            )
            diag = (
                f"부엉~ 부엉! 비커 위로 하얀 안개 구름이 폭포처럼 쏟아지는 것 좀 보렴! '{user_utterance[:22]}' 실험처럼 온도가 바뀌면 분자들의 움직임과 상태가 즉시 변한단다, 부엉!"
            )
            note_str = "열역학 및 반응 속도 원리: 온도의 급격한 변화는 분자 운동 에너지(아레니우스 법칙)와 기체 용해도를 변화시켜 승화 안개 또는 염화수소 기체 휘발을 유발합니다."
    else:
        if norm_lang == "en":
            title_str = f"Experiment {next_page_num} — Chemical Cause & Effect in HCl Acid: {user_utterance[:28]}"
            p1 = (
                f"We test your question—'{user_utterance}'—directly inside the 1M hydrochloric acid beaker! "
                f"As the test sample meets the strong hydrogen ions (H⁺) and chloride ions (Cl⁻) in the solution, any carbonate, active metal, or organic pigment on its surface reacts immediately."
            )
            p2 = (
                f"Our digital sensors record the direct cause-and-effect outcome: surface micro-bubbles form, soluble ions dissolve into the liquid, and the beaker stabilizes at pH {new_ph} and {new_temp} °C!"
            )
            diag = (
                f"Hoo-hoo! When we test '{user_utterance[:34]}' in hydrochloric acid, the strong H⁺ ions immediately test its chemical bonds! "
                f"The solution is now at pH {new_ph} and {new_temp} degrees Celsius, hoo-hoo!"
            )
            note_str = f"Acid-Matter Interaction Telemetry: H⁺ protonation and ion exchange in 1M HCl(aq) | Live pH {new_ph}, Temp {new_temp} °C."
        else:
            title_str = f"실험 {next_page_num} — 염산 용액과의 화학적 원인과 결과: {user_utterance[:22]}"
            p1 = (
                f"궁금해한 '{user_utterance}' 실험을 1몰(1M) 묽은 염산 비커에서 곧바로 진행해 보았어요! "
                f"물질이 염산 속의 강력한 수소 이온(H⁺) 및 염화 이온(Cl⁻)과 접촉하자마자, 물질 표면의 성분에 따라 산성 가수분해와 이온 교환 반응이 시작됩니다."
            )
            p2 = (
                f"그 결과 표면에서 미세한 반응 기포와 색 변화가 나타나며, 디지털 계측기에 기록된 용액의 산성도는 pH {new_ph}, 온도는 섭씨 {new_temp}도로 변화하는 뚜렷한 인과관계를 확인할 수 있답니다!"
            )
            diag = (
                f"부엉~ 부엉! 염산 비커에서 '{user_utterance[:24]}' 실험을 해보니 수소 이온들이 물질 표면을 공격해 즉각적인 화학 변화를 일으키는구나! "
                f"결과 산성도는 pH {new_ph}, 온도는 섭씨 {new_temp}도란다, 부엉!"
            )
            note_str = f"산-물질 반응 인과관계 노트: 강산(HCl)의 수소 이온(H⁺)에 의한 표면 산화·가수분해 반응 | 계측 결과: pH {new_ph}, 온도 {new_temp} °C."

    return {
        "page_title": title_str,
        "story_paragraphs": [p1, p2],
        "character_name": character_name,
        "character_persona_voice": persona_voice,
        "character_dialogue": diag,
        "science_or_moral_note": note_str,
        "visual_state": {
            "theme": prev_visual_state.get("theme", "science_hcl_lab"),
            "lang": norm_lang,
            "character_persona_voice": persona_voice,
            "item_key": topic,
            "item_label": get_item_display_label(topic, lang=norm_lang, fallback_text=user_utterance[:28]),
            "accumulated_items": acc_items,
            "accumulated_labels": acc_labels,
            "ph_level": new_ph,
            "temperature_c": new_temp,
            "liquid_color": [98, 205, 238],
            "liquid_level": 0.72,
            "bubbles_intensity": 0.78,
            "steam_intensity": 0.36,
            "foam_intensity": 0.50,
            "accent_color": [129, 140, 248],
        },
        "nano_banana_prompt": (
            f"Nano Banana 16:9 Lab Scene: preserve the glass beaker and Dr. Nova while clearly visualizing the chemical cause-and-effect of '{user_utterance}'."
        ),
        "veo_motion_prompt": (
            f"Animated laboratory beaker reaction for '{user_utterance}' with rising bubbles and Dr. Nova narrating."
        ),
        "suggested_followups": hcl_followups_en[:4] if norm_lang == "en" else hcl_followups_ko[:4],
    }


def _deterministic_turn_response(
    book_state: dict[str, Any],
    user_utterance: str,
    next_page_num: int,
    prev_visual_state: dict[str, Any],
    lang: str = "ko",
) -> dict[str, Any]:
    """Alias for `_deterministic_story_fallback` preserving existing callers."""
    return _deterministic_story_fallback(
        book_state=book_state,
        user_utterance=user_utterance,
        next_page_num=next_page_num,
        prev_visual_state=prev_visual_state,
        lang=lang,
    )


def _matches_target_lang(text: str, lang: str) -> bool:
    """Verify that generated user-facing text honors strict Korean (`ko`) vs English (`en`) separation."""
    if not text or not text.strip():
        return False
    has_hangul = bool(_HANGUL_RE.search(text))
    if lang == "en":
        return not has_hangul
    # For Korean (`ko`), must contain Hangul and must not have long English phrases (>3 consecutive ASCII words)
    if not has_hangul:
        return False
    if re.search(r"(?:[A-Za-z]{3,}\s+){3,}[A-Za-z]{3,}", text):
        return False
    return True


def generate_next_page_story(
    book_state: dict[str, Any],
    user_utterance: str,
    prefer_live_api: bool = True,
    model_tier: str = "argon_low",
    lang: str | None = None,
) -> dict[str, Any]:
    """Generate the next page's narrative, persona dialogue, science note, and cumulative continuity state.

    Args:
        book_state: Current book dictionary containing `mode`, `preset_id`, `character_bible`, and `pages`.
        user_utterance: Spoken or clicked "What-If" question from the user.
        prefer_live_api: Whether to attempt live Gemini API call first when configured.
        model_tier: Selected model tier (`argon_low`, `flash_ultra`, `argon_medium`).
        lang: Target output language (`"ko"` for 100% pure Korean, `"en"` for 100% pure English).

    Returns:
        Dictionary containing the structured story/science page turn and `engine_meta`.
    """
    cleaned_utterance = (user_utterance or "").strip()
    pages = book_state.get("pages") or []
    next_page_num = len(pages) + 1
    preset_id = str(book_state.get("preset_id") or "science_hcl_lab")
    mode = str(
        book_state.get("mode")
        or ("storybook" if preset_id in ("fairy_starlight_fox", "fairy_tiger_rabbit") or "fairy" in preset_id else "science_lab")
    )
    prev_visual_state = dict(pages[-1].get("visual_state") or {}) if pages else {}
    effective_lang = _resolve_lang(book_state, cleaned_utterance, lang=lang)
    persona_voice = _resolve_character_persona(book_state, preset_id, mode)

    log_function_call(
        "generate_next_page_story",
        {
            "book_id": book_state.get("book_id"),
            "preset_id": preset_id,
            "mode": mode,
            "lang": effective_lang,
            "character_persona_voice": persona_voice,
            "next_page_num": next_page_num,
            "user_utterance": cleaned_utterance,
            "model_tier": model_tier,
            "prev_accumulated_items": prev_visual_state.get("accumulated_items", []),
            "prefer_live_api": prefer_live_api,
        },
    )

    if not cleaned_utterance:
        raise ValueError("user_utterance cannot be empty.")

    baseline = _deterministic_story_fallback(
        book_state=book_state,
        user_utterance=cleaned_utterance,
        next_page_num=next_page_num,
        prev_visual_state=prev_visual_state,
        lang=effective_lang,
    )
    acc_labels = baseline["visual_state"].get("accumulated_labels", [])
    continuity_note = (
        f"Chained from Page {next_page_num - 1} via Nano Banana | Scene Focus: {', '.join(acc_labels)}"
        if effective_lang == "en"
        else f"이전 {next_page_num - 1}장 장면에서 생성 | 핵심 요소: {', '.join(acc_labels)}"
    )

    import os

    topic_key = baseline["visual_state"].get("item_key", "")
    is_custom_topic = topic_key in ("custom_science", "custom_story")
    force_live_llm = os.environ.get("FORCE_LIVE_STORY_LLM") == "1"

    if prefer_live_api and get_gemini_api_key() and (is_custom_topic or force_live_llm):
        prev_summaries = [
            f"Page {p.get('page_number')}: Q='{p.get('user_question')}' -> {p.get('page_title')}"
            for p in pages[-4:]
        ]
        sys_instruction = _build_system_instruction(
            lang=effective_lang,
            mode=mode,
            preset_id=preset_id,
            character_name=baseline["character_name"],
            character_persona_voice=persona_voice,
            next_page_num=next_page_num,
        )
        prompt_text = (
            f"{sys_instruction}\n"
            f"Book Title: {book_state.get('title')}\n"
            f"Preset ID: {preset_id}\n"
            f"Book Mode: {mode}\n"
            f"Target Language: {'100% Pure English (en)' if effective_lang == 'en' else '100% 순수 한국어 (ko)'}\n"
            f"Character Name: {baseline['character_name']} (Persona: {persona_voice})\n"
            f"Character & Visual Continuity Bible: {book_state.get('character_bible')}\n"
            f"Current Scene Items: {', '.join(acc_labels)}\n"
            f"Previous Pages Summary: {' | '.join(prev_summaries)}\n"
            f"Target Page Number: {next_page_num}\n"
            f"User's Spoken Question (`user_utterance`): {cleaned_utterance}\n"
            "CRITICAL: Directly and vividly explain the exact cause-and-effect outcome of the user's question (`user_utterance`) in `page_title`, `story_paragraphs`, `character_dialogue`, and `science_or_moral_note`. Never output generic filler!\n"
        )
        try:
            llm_json, call_meta = _call_story_llm_chain(prompt_text, model_tier=model_tier, timeout_s=6.5)
            merged_visual = dict(baseline["visual_state"])
            if isinstance(llm_json.get("visual_state"), dict):
                for k, v in llm_json["visual_state"].items():
                    if k not in ("accumulated_items", "accumulated_labels", "lang", "character_persona_voice", "theme") and v is not None:
                        merged_visual[k] = v
            if not _matches_target_lang(str(merged_visual.get("item_label") or ""), effective_lang):
                merged_visual["item_label"] = baseline["visual_state"]["item_label"]

            raw_title = str(llm_json.get("page_title") or "")
            final_title = raw_title if _matches_target_lang(raw_title, effective_lang) else baseline["page_title"]

            raw_paragraphs = llm_json.get("story_paragraphs")
            if (
                isinstance(raw_paragraphs, list)
                and len(raw_paragraphs) >= 2
                and all(_matches_target_lang(str(p), effective_lang) for p in raw_paragraphs)
            ):
                final_paragraphs = [str(p) for p in raw_paragraphs]
            else:
                final_paragraphs = baseline["story_paragraphs"]

            raw_dialogue = str(llm_json.get("character_dialogue") or "")
            final_dialogue = (
                raw_dialogue if _matches_target_lang(raw_dialogue, effective_lang) else baseline["character_dialogue"]
            )

            raw_note = str(llm_json.get("science_or_moral_note") or "")
            final_note = (
                raw_note if _matches_target_lang(raw_note, effective_lang) else baseline["science_or_moral_note"]
            )

            raw_followups = llm_json.get("suggested_followups")
            if (
                isinstance(raw_followups, list)
                and len(raw_followups) >= 2
                and all(_matches_target_lang(str(f), effective_lang) for f in raw_followups[:4])
            ):
                final_followups = [str(f) for f in raw_followups[:4]]
            else:
                final_followups = baseline["suggested_followups"]

            return {
                "page_number": next_page_num,
                "lang": effective_lang,
                "user_question": cleaned_utterance,
                "page_title": final_title,
                "story_paragraphs": final_paragraphs,
                "character_name": baseline["character_name"],
                "character_persona_voice": persona_voice,
                "character_dialogue": final_dialogue,
                "science_or_moral_note": final_note,
                "visual_continuity_note": continuity_note,
                "visual_state": merged_visual,
                "nano_banana_prompt": str(llm_json.get("nano_banana_prompt") or baseline["nano_banana_prompt"]),
                "veo_motion_prompt": str(llm_json.get("veo_motion_prompt") or baseline["veo_motion_prompt"]),
                "suggested_followups": final_followups,
                "engine_meta": call_meta,
            }
        except Exception as exc:
            fallback_reason = f"Live Gemini call fell back to cumulative director ({exc})"
    else:
        fallback_reason = "Ultra-Fast Cumulative Story & Science Director active (instant parallel path)."

    baseline["page_number"] = next_page_num
    baseline["lang"] = effective_lang
    baseline["user_question"] = cleaned_utterance
    baseline["visual_continuity_note"] = continuity_note
    baseline["engine_meta"] = {
        "engine_mode": "deterministic_cumulative_director",
        "model_used": f"{PRIMARY_MODEL} / {FAST_STORY_MODEL} (instant-parallel)",
        "configured_primary_model": PRIMARY_MODEL,
        "configured_fast_model": FAST_STORY_MODEL,
        "configured_fallback_model": FALLBACK_MODEL,
        "model_tier": model_tier,
        "lang": effective_lang,
        "note": fallback_reason,
    }
    return baseline


def generate_custom_opening_book(
    custom_prompt: str,
    lang: str = "ko",
    prefer_live_api: bool = False,
    model_tier: str = "argon_low",
) -> dict[str, Any]:
    """Create a tailored opening book configuration from a user's freeform prompt.

    Examples:
        - "호랑이랑 토끼가 나오는 동화책 시작해줘" -> maps to `fairy_tiger_rabbit`
        - "우주왕복선 안에서 할 수 있는 실험책 만들어줘" -> maps to `science_space_shuttle`
        - Any custom fairy tale or science experiment prompt -> builds a custom opening page & preset
    """
    from config import BOOK_PRESETS, BOOK_PRESETS_EN

    cleaned = (custom_prompt or "").strip()
    norm_lang = "en" if str(lang or "").strip().lower().startswith("en") else "ko"
    if cleaned and not _HANGUL_RE.search(cleaned) and len(_ASCII_WORDS_RE.findall(cleaned)) >= 2 and not lang:
        norm_lang = "en"

    low = cleaned.lower()
    presets = BOOK_PRESETS_EN if norm_lang == "en" else BOOK_PRESETS

    # 1. Match Tiger & Rabbit Folktale
    if any(k in low for k in ("호랑이", "토끼", "달콩", "떡 하나", "tiger", "rabbit", "bunny")):
        return dict(presets["fairy_tiger_rabbit"])

    # 2. Match Space Shuttle / Zero-G Experiment Book
    if any(k in low for k in ("우주왕복선", "왕복선", "오비터", "우주선", "무중력", "우주", "shuttle", "spaceship", "spacecraft", "zero-g", "zero gravity", "astronaut")):
        if any(k in low for k in ("왕복선", "오비터", "우주선", "shuttle", "spaceship", "spacecraft")):
            return dict(presets["science_space_shuttle"])
        return dict(presets["science_space_shuttle"])

    # 3. Match Starlight Fox
    if any(k in low for k in ("여우", "루미", "별빛 숲", "fox", "lumi")):
        return dict(presets["fairy_starlight_fox"])

    # 4. Match HCl / Acid / Beaker Lab
    if any(k in low for k in ("염산", "비커", "산성", "부엉이", "노바", "hcl", "acid", "beaker", "chemistry")):
        return dict(presets["science_hcl_lab"])

    # 5. Classify whether the arbitrary custom prompt is a Science Experiment book or a Fairy-Tale Storybook
    is_science = any(
        k in low
        for k in ("실험", "과학", "화학", "물리", "관찰", "반응", "만들", "연구", "experiment", "science", "lab", "physics", "test")
    ) and not any(k in low for k in ("동화", "이야기", "옛날", "공주", "왕자", "용", "마법", "fairy", "story", "tale"))

    short_topic = cleaned
    for suffix in (
        "나오는 동화책 시작해줘",
        "동화책 시작해줘",
        "동화책 만들어줘",
        "할 수 있는 실험책 만들어줘",
        "실험책 만들어줘",
        "실험책 시작해줘",
        "책 시작해줘",
        "책 만들어줘",
        "시작해줘",
        "만들어줘",
    ):
        if short_topic.endswith(suffix):
            short_topic = short_topic[: -len(suffix)].strip()
            break
    short_topic = short_topic.strip(" ,.~!?") or cleaned[:28]

    if is_science:
        base_preset = dict(presets["science_hcl_lab"])
        if norm_lang == "en":
            return {
                **base_preset,
                "preset_id": "science_hcl_lab",
                "mode": "science_lab",
                "lang": "en",
                "title": f"WonderLab Custom Science Book: {short_topic[:36]}",
                "subtitle": f"Interactive Experiment Book created from '{cleaned[:48]}'",
                "opening_page": {
                    **dict(base_preset["opening_page"]),
                    "page_title": f"Chapter 1 — Opening the '{short_topic[:32]}' Experiment Lab",
                    "story_paragraphs": [
                        f"Welcome to your custom interactive science lab inspired by your idea: '{cleaned}'! "
                        "Dr. Nova the Owl Professor has prepared the transparent reaction beaker, digital sensors, and safety gear on the lab bench.",
                        "Ask any 'What-If' question below or drop a reagent into the chamber to see every step—entering, splashing, and bubbling—come to life!",
                    ],
                    "character_dialogue": (
                        f"Hoo-hoo! I set up a special experiment book all about '{short_topic[:32]}' just for you! "
                        "What should we drop in or test first, hoo-hoo?"
                    ),
                },
            }
        return {
            **base_preset,
            "preset_id": "science_hcl_lab",
            "mode": "science_lab",
            "lang": "ko",
            "title": f"나만의 맞춤 실험책: {short_topic[:26]}",
            "subtitle": f"'{cleaned[:38]}' 주제로 시작하는 인터랙티브 과학 실험책",
            "opening_page": {
                **dict(base_preset["opening_page"]),
                "page_title": f"1장 — '{short_topic[:24]}' 호기심 과학 실험실 문을 열다",
                "story_paragraphs": [
                    f"여러분이 직접 요청한 '{cleaned}' 주제로 특별한 대화형 과학 실험책이 열렸습니다! "
                    "척척박사 부엉이 노바 박사님이 실험대 위에 투명한 반응 비커와 정밀 센서들을 모두 준비해 두었어요.",
                    "재료를 넣으면 비커 속으로 풍덩 들어가서 보글보글 거품이 피어오르는 전 과정을 눈으로 직접 관찰할 수 있답니다!",
                ],
                "character_dialogue": (
                    f"부엉~ 부엉! 네가 말한 '{short_topic[:22]}' 주제로 멋진 실험실을 열었단다! "
                    "가장 먼저 어떤 재료를 넣고 실험해 볼까, 부엉?"
                ),
            },
        }
    else:
        base_preset = dict(presets["fairy_tiger_rabbit"] if any(k in low for k in ("산", "숲", "동물", "고개", "전래")) else presets["fairy_starlight_fox"])
        chosen_id = base_preset["preset_id"]
        if norm_lang == "en":
            return {
                **base_preset,
                "preset_id": chosen_id,
                "mode": "storybook",
                "lang": "en",
                "title": f"Custom Storybook: {short_topic[:36]}",
                "subtitle": f"Interactive Fairy-Tale Book created from '{cleaned[:48]}'",
                "opening_page": {
                    **dict(base_preset["opening_page"]),
                    "page_title": f"Chapter 1 — The Beginning of '{short_topic[:32]}'",
                    "story_paragraphs": [
                        f"Under a sky shimmering with golden starlight, a brand-new fairy tale begins from your wish: '{cleaned}'! "
                        "Every lantern along the forest path glows warmly as our storybook heroes step onto the stage.",
                        "Speak or click any 'What-If' imagination below, and watch each magical moment unfold step by step!",
                    ],
                    "character_dialogue": (
                        f"Yip-yip! Hop-hop! Our brand-new storybook about '{short_topic[:32]}' is open! "
                        "What magical adventure should we start on the next page?"
                    ),
                },
            }
        return {
            **base_preset,
            "preset_id": chosen_id,
            "mode": "storybook",
            "lang": "ko",
            "title": f"나만의 맞춤 동화책: {short_topic[:26]}",
            "subtitle": f"'{cleaned[:38]}' 상상으로 시작하는 인터랙티브 동화책",
            "opening_page": {
                **dict(base_preset["opening_page"]),
                "page_title": f"1장 — '{short_topic[:24]}' 마법 동화의 첫 페이지",
                "story_paragraphs": [
                    f"여러분이 직접 상상해 준 '{cleaned}' 이야기가 달빛과 별빛이 반짝이는 동화책 속에서 활짝 펼쳐졌습니다! "
                    "숲속의 따뜻한 등불들이 켜지며 주인공들이 여러분의 목소리에 귀를 기울이고 있어요.",
                    "다음 장에서 어떤 신기한 일이 벌어질지 자유롭게 말하거나 아래 상상 버튼을 눌러 보세요!",
                ],
                "character_dialogue": (
                    f"우와앙! 네가 원하던 '{short_topic[:22]}' 동화책이 드디어 시작되었어! "
                    "우리 이제 어떤 신나는 모험을 떠나 볼까?"
                ),
            },
        }

