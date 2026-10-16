"""Centralized configuration, localized preset catalog, and audit logging for WonderBook Live.

This module centralizes all configurable settings for the WonderBook Live interactive
voice-driven storybook and "What-If" science experiment book platform:
1. Upgraded Multi-Agent Model Configuration:
   - `PRIMARY_MODEL` (`gemini-4-argon`): Primary multimodal reasoning & story continuity
     director (`thinking_level="low"` for fast interactive turns, `"medium"` for deep mode).
   - `FAST_STORY_MODEL` (`gemini-3.8-flash`): High-speed, high-quality Flash model for
     low-latency narrative and scientific reaction generation.
   - `FALLBACK_MODEL` (`gemini-3.8-flash`): Automatic fallback model when `503` or
     transient overload occurs on primary model.
   - `NANO_BANANA_PRO_MODEL` (`gemini-3-pro-image-preview`) & `NANO_BANANA_MODEL`
     (`gemini-2.5-flash-image`): Visual continuity image generation and image-to-image
     editing models chaining the previous page's illustration and cumulative scene items.
   - `VEO_MODEL` (`veo-2.0-generate-001`): Living motion video model.
   - `GEMINI_TTS_MODEL` (`gemini-2.5-flash-preview-tts`) & `GEMINI_TTS_PRO_MODEL`
     (`gemini-2.5-pro-preview-tts`): Expressive character persona voice synthesis models.
2. Strictly Separated Pure Korean (`ko`) and Pure English (`en`) Preset Catalogs
   (`BOOK_PRESETS`, `BOOK_PRESETS_EN`, `get_localized_preset`) with character voice personas
   (`owl_professor`, `baby_fox`, `space_captain`) and cumulative scene state.
3. Structured `INFO`-level logging (`log_function_call`, `log_genai_call`) that logs all
   function calls and `genai` invocations while stripping raw binary/base64 payloads
   and API credentials.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

APP_NAME = "WonderBook Live"
APP_VERSION = "2.0.0"

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
RUNTIME_DIR = BASE_DIR / "runtime"
BOOKS_DIR = RUNTIME_DIR / "books"
LOG_FILE_PATH = RUNTIME_DIR / "wonderbook_audit.log"

for _directory in (RUNTIME_DIR, BOOKS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


def _load_dotenv_if_present() -> None:
    """Load key=value pairs from local .env into os.environ without external dependencies."""
    env_file = BASE_DIR / ".env"
    if not env_file.is_file():
        return
    try:
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except OSError:
        pass


_load_dotenv_if_present()

# ---------------------------------------------------------------------------
# Upgraded Multi-Agent Model & Retry Configuration
# ---------------------------------------------------------------------------
PRIMARY_MODEL = os.environ.get("PRIMARY_MODEL", "gemini-4-argon").strip() or "gemini-4-argon"
FAST_STORY_MODEL = os.environ.get("FAST_STORY_MODEL", "gemini-3.8-flash").strip() or "gemini-3.8-flash"
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "gemini-3.8-flash").strip() or "gemini-3.8-flash"
NANO_BANANA_PRO_MODEL = (
    os.environ.get("NANO_BANANA_PRO_MODEL", "gemini-3-pro-image-preview").strip() or "gemini-3-pro-image-preview"
)
NANO_BANANA_MODEL = os.environ.get("NANO_BANANA_MODEL", "gemini-2.5-flash-image").strip() or "gemini-2.5-flash-image"
VEO_MODEL = os.environ.get("VEO_MODEL", "veo-2.0-generate-001").strip() or "veo-2.0-generate-001"
GEMINI_TTS_MODEL = (
    os.environ.get("WONDERBOOK_TTS_MODEL", "gemini-2.5-flash-preview-tts").strip()
    or "gemini-2.5-flash-preview-tts"
)
GEMINI_TTS_PRO_MODEL = (
    os.environ.get("WONDERBOOK_TTS_PRO_MODEL", "gemini-2.5-pro-preview-tts").strip()
    or "gemini-2.5-pro-preview-tts"
)
THINKING_LEVEL = os.environ.get("THINKING_LEVEL", "low").strip() or "low"

MAX_GENAI_RETRIES = int(os.environ.get("MAX_GENAI_RETRIES", "2"))
RETRY_BASE_DELAY_SECONDS = float(os.environ.get("RETRY_BASE_DELAY_SECONDS", "0.35"))
DEFAULT_PORT = int(os.environ.get("PORT", "8080"))

# ---------------------------------------------------------------------------
# Preset Book Scenarios — Pure Korean ("ko", Default) & Pure English ("en")
# ---------------------------------------------------------------------------
DEFAULT_PRESET_ID = "science_hcl_lab"

BOOK_PRESETS: dict[str, dict[str, Any]] = {
    "science_hcl_lab": {
        "preset_id": "science_hcl_lab",
        "mode": "science_lab",
        "lang": "ko",
        "title": "노바 박사의 신기한 염산 실험실",
        "subtitle": "염산 비커에 무엇이든 넣어보고 실시간 화학 반응을 관찰해요",
        "character_name": "부엉이 과학자 노바 박사",
        "character_persona_voice": "owl_professor",
        "character_bible": (
            "A bright, high-detail illustrated chemistry laboratory bench with a clear 500mL borosilicate glass "
            "beaker at the center filled with shimmering pale-emerald hydrochloric acid (1.0M HCl), digital pH "
            "and temperature sensors on the left, and friendly clockwork-owl scientist Dr. Nova wearing cyan safety goggles."
        ),
        "initial_visual_state": {
            "theme": "science_hcl_lab",
            "lang": "ko",
            "character_persona_voice": "owl_professor",
            "liquid_color": [110, 231, 183],
            "liquid_level": 0.60,
            "item_key": "none",
            "item_label": "맑은 염산 용액 (실험 준비 완료)",
            "accumulated_items": [],
            "accumulated_labels": [],
            "ph_level": 1.0,
            "temperature_c": 22.0,
            "bubbles_intensity": 0.15,
            "steam_intensity": 0.08,
            "foam_intensity": 0.0,
            "accent_color": [56, 189, 248],
        },
        "opening_page": {
            "page_number": 1,
            "user_question": "실험실 문을 열고 첫 번째 비커를 살펴봐요",
            "page_title": "1장 — 에메랄드빛으로 속삭이는 염산 비커",
            "story_paragraphs": [
                "반짝이는 과학 실험대 한가운데에 연한 에메랄드빛으로 은은하게 빛나는 오백 밀리리터 유리 비커가 놓여 있어요. 비커 안에는 우리 위장 속 소화액에도 들어 있는 강한 산성 용액인 염산이 담겨 있고, 옆에 나란히 놓인 디지털 측정기는 산성도 일 점 영과 섭씨 이십이 도를 가리키며 조용히 깜빡이고 있답니다.",
                "청록색 안전 고글을 쓴 태엽 부엉이 과학자 노바 박사님이 날개를 활짝 펴며 여러분을 반갑게 맞이해요. 이 신기한 비커 속에는 여러분이 궁금해하는 어떤 재료든 안전하게 계속 넣어볼 수 있고, 먼저 넣은 재료들은 사라지지 않고 다음 장에서도 그대로 남아 새로운 재료와 함께 반응한답니다!",
            ],
            "character_name": "부엉이 과학자 노바 박사",
            "character_persona_voice": "owl_professor",
            "character_dialogue": (
                "부엉~ 부엉! 안녕 꼬마 과학자 친구! 내 에메랄드빛 염산 비커에 어서 와! "
                "여기에 열쇠를 넣으면 어떻게 될까? 사과나 달콤한 과자, 아니면 물을 뿌려볼까? 궁금한 걸 말해봐, 부엉!"
            ),
            "science_or_moral_note": (
                "핵심 과학 노트: 염산(HCl)은 수소 이온(H+)이 풍부한 강산성 용액이에요. "
                "금속이나 탄산염과 만나면 기체를 내뿜으며 역동적으로 반응한답니다."
            ),
            "visual_continuity_note": "1장 기준 장면: 500mL 유리 비커 속 1.0M 염산 용액 (pH 1.0, 22.0 °C)과 부엉이 과학자 노바 박사.",
            "nano_banana_prompt": (
                "High-craft children's science atlas illustration: a clear 500mL glass beaker of glowing pale-emerald "
                "hydrochloric acid on a laboratory table with digital pH 1.0 and 22.0C meters and Dr. Nova the friendly "
                "owl scientist wearing safety goggles."
            ),
            "veo_motion_prompt": (
                "Gentle micro-bubbles rising inside a clear glass beaker of hydrochloric acid while Dr. Nova the owl professor waves warmly."
            ),
            "suggested_followups": [
                "열쇠(금속)를 넣으면 어떻게 되나요?",
                "사과를 넣으면 어떻게 되나요?",
                "과자를 넣으면 어떻게 되나요?",
                "물을 뿌리면 어떻게 되나요?",
                "베이킹소다를 한 스푼 넣으면 어떻게 되나요?",
                "달걀 껍데기를 넣으면 어떻게 되나요?",
                "순금 반지를 넣으면 어떻게 되나요?",
            ],
        },
    },
    "fairy_starlight_fox": {
        "preset_id": "fairy_starlight_fox",
        "mode": "storybook",
        "lang": "ko",
        "title": "아기 별빛 여우 루미와 속삭이는 숲",
        "subtitle": "내가 말하는 상상대로 다음 페이지가 펼쳐지는 살아있는 동화책",
        "character_name": "아기 별빛 여우 루미",
        "character_persona_voice": "baby_fox",
        "character_bible": (
            "Warm, high-craft storybook watercolor and gouache illustration featuring Lumi, a small silver-and-amber "
            "baby fox with a glowing star-tipped tail and a royal-blue knitted scarf, inside an enchanted twilight forest."
        ),
        "initial_visual_state": {
            "theme": "fairy_starlight_fox",
            "lang": "ko",
            "character_persona_voice": "baby_fox",
            "sky_color": [22, 32, 72],
            "item_key": "magic_door",
            "item_label": "빛나는 참나무 비밀 문",
            "accumulated_items": ["magic_door"],
            "accumulated_labels": ["빛나는 참나무 문"],
            "sparkle_intensity": 0.65,
            "companion": "none",
            "accent_color": [251, 191, 36],
        },
        "opening_page": {
            "page_number": 1,
            "user_question": "별빛 숲에서 루미의 이야기를 시작해요",
            "page_title": "1장 — 참나무 밑동의 빛나는 황금 열쇠구멍",
            "story_paragraphs": [
                "고요한 속삭임의 숲 가장 깊은 곳, 은빛과 호박빛 털을 가진 아기 별빛 여우 루미가 커다란 고목 참나무 앞에 걸음을 멈추었어요. 파란 털실 목도리를 두른 루미가 별 모양 꼬리를 살랑거리자, 나무 밑동 한가운데에 숨겨져 있던 작은 황금 열쇠구멍이 따뜻한 황금빛을 내뿜기 시작했답니다.",
                "나뭇가지 위로는 폭신한 무지개 구름이 졸린 듯 둥실 떠 있고, 황금빛 반딧불이들이 루미의 귀 주변을 맴돌며 작은 종소리를 울렸어요. 루미는 복슬복슬한 귀를 쫑긋 세우며 여러분의 목소리에 귀를 기울이고 있어요. 여러분이 들려주는 상상이 다음 장 그림과 이야기 속에 마법처럼 차곡차곡 이어진답니다!",
            ],
            "character_name": "아기 별빛 여우 루미",
            "character_persona_voice": "baby_fox",
            "character_dialogue": (
                "콩콩! 헤헤, 안녕 친구야! 내 꼬리 끝의 별빛이 참나무의 비밀 문을 비추고 있어! "
                "이 문 뒤에 아기 용이 숨어 있으면 어떨까? 아니면 하늘에서 별사탕 비가 내리면 어떨까? 네가 상상하는 걸 말해줘, 낑~!"
            ),
            "science_or_moral_note": (
                "동화 맥락 노트: 루미의 파란 목도리와 별빛 꼬리, 그리고 참나무 문은 다음 페이지에서도 그대로 이어지며 여러분의 상상이 하나씩 더해져요."
            ),
            "visual_continuity_note": "1장 기준 장면: 빛나는 참나무 비밀 문 앞에 선 파란 목도리의 아기 별빛 여우 루미.",
            "nano_banana_prompt": (
                "High-craft children's storybook watercolor illustration of Lumi, a cute silver-and-amber baby fox wearing "
                "a royal-blue scarf with a glowing star tail, standing before a glowing golden keyhole door in an ancient oak tree."
            ),
            "veo_motion_prompt": (
                "Lumi the baby fox wagging her glowing star-tipped tail as golden fireflies drift around the magical oak tree door."
            ),
            "suggested_followups": [
                "문이 열리고 아기 용이 비눗방울을 뿜으면 어때요?",
                "루미가 무지개 구름 비행선을 타면 어떻게 돼요?",
                "열쇠구멍에 달콤한 초코칩 과자를 넣으면 어떻게 돼요?",
                "하늘에서 별사탕 비가 내리면 어떻게 돼요?",
            ],
        },
    },
    "science_space_kitchen": {
        "preset_id": "science_space_kitchen",
        "mode": "science_lab",
        "lang": "ko",
        "title": "오르빗 대장의 무중력 우주 실험실",
        "subtitle": "중력이 없는 우주정거장에서 불꽃과 물방울은 어떻게 움직일까?",
        "character_name": "우주 비행사 오르빗 대장",
        "character_persona_voice": "space_captain",
        "character_bible": (
            "High-detail illustrated science picture-book cabin inside a panoramic orbital space station with Earth "
            "visible through the cupola window, featuring Captain Orbit in a white-and-orange spacesuit."
        ),
        "initial_visual_state": {
            "theme": "science_space_kitchen",
            "lang": "ko",
            "character_persona_voice": "space_captain",
            "liquid_color": [96, 165, 250],
            "liquid_level": 0.50,
            "item_key": "zero_g_cabin",
            "item_label": "무중력 우주 실험 챔버",
            "accumulated_items": [],
            "accumulated_labels": [],
            "ph_level": 7.0,
            "temperature_c": 21.0,
            "bubbles_intensity": 0.40,
            "steam_intensity": 0.0,
            "foam_intensity": 0.0,
            "accent_color": [167, 139, 250],
        },
        "opening_page": {
            "page_number": 1,
            "user_question": "무중력 우주정거장 실험실의 문을 열어요",
            "page_title": "1장 — 지구 위 사백 킬로미터, 둥실둥실 무중력 실험실",
            "story_paragraphs": [
                "푸른 지구가 둥근 전망창 너머로 찬란하게 빛나는 이곳은 지상 사백 킬로미터 상공의 무중력 우주정거장 실험실이에요! 중력이 거의 느껴지지 않는 미세중력 공간이라서, 바닥으로 떨어지는 물건 없이 모든 물방울과 실험 도구들이 허공에 보석처럼 둥실둥실 떠다닌답니다.",
                "하얗고 주황빛이 감도는 우주복을 입은 오르빗 대장님이 공중에서 가볍게 제비돌기를 하며 손짓하고 있어요. 우주 공간에서 물을 뿌리거나 양초에 불을 붙이고, 사과와 과자를 띄워 보면 지구에서와는 전혀 다른 놀라운 과학 현상이 눈앞에 펼쳐질 거예요!",
            ],
            "character_name": "우주 비행사 오르빗 대장",
            "character_persona_voice": "space_captain",
            "character_dialogue": (
                "오버! 여기는 무중력 우주 실험실의 오르빗 대장이다! 둥실둥실 허공에 떠 있는 기분이 정말 최고야! "
                "우주에서 물을 뿌리거나 양초에 불을 붙이면 어떻게 될까? 네가 해보고 싶은 실험을 마음껏 명령해 줘, 로저!"
            ),
            "science_or_moral_note": (
                "우주 물리 노트: 궤도를 도는 우주정거장의 미세중력 환경에서는 부력과 대류 현상이 사라지고, 액체의 표면장력이 모양을 결정한답니다."
            ),
            "visual_continuity_note": "1장 기준 장면: 지구가 내려다보이는 우주정거장 전망창과 오르빗 대장.",
            "nano_banana_prompt": (
                "High-craft children's science illustration of Captain Orbit floating inside a space station cupola "
                "with Earth glowing outside the window."
            ),
            "veo_motion_prompt": (
                "Captain Orbit floating gently in zero gravity while small water droplets drift like crystal marbles."
            ),
            "suggested_followups": [
                "무중력에서 물을 뿌리면 물방울이 어떻게 되나요?",
                "무중력에서 양초에 불을 붙이면 불꽃 모양이 어때요?",
                "우주에서 사과와 과자를 동시에 띄우면 어떻게 되나요?",
                "우주에서 발포 비타민이나 소다를 물방울에 넣으면 어떻게 되나요?",
            ],
        },
    },
    "fairy_tiger_rabbit": {
        "preset_id": "fairy_tiger_rabbit",
        "mode": "storybook",
        "lang": "ko",
        "title": "호랑이와 꾀돌이 토끼의 달빛 산길 동화",
        "subtitle": "떡 하나 주면 안 잡아먹지! 내가 말하는 대로 다음 장이 펼쳐지는 전래 동화책",
        "character_name": "꾀돌이 산토끼 달콩이",
        "character_persona_voice": "baby_fox",
        "character_bible": (
            "Warm Korean traditional fairy-tale and Studio Ghibli watercolor painting style. "
            "On a moonlit pine forest mountain path with glowing warm lanterns and autumn persimmon trees, "
            "a big friendly cute striped Korean mountain tiger with expressive golden eyes sits across from "
            "a clever adorable white rabbit holding a woven basket."
        ),
        "initial_visual_state": {
            "theme": "fairy_tiger_rabbit",
            "lang": "ko",
            "character_persona_voice": "baby_fox",
            "sky_color": [24, 34, 68],
            "item_key": "tiger_rabbit_path",
            "item_label": "달빛 소나무 고갯길과 떡바구니",
            "accumulated_items": ["tiger_rabbit_path"],
            "accumulated_labels": ["달빛 소나무 고갯길"],
            "sparkle_intensity": 0.72,
            "companion": "mountain_tiger",
            "accent_color": [251, 191, 36],
        },
        "opening_page": {
            "page_number": 1,
            "user_question": "호랑이랑 토끼가 나오는 동화책 시작해줘",
            "page_title": "1장 — 달빛 고갯길에서 만난 얼룩 호랑이와 꾀돌이 토끼",
            "story_paragraphs": [
                "둥근 보름달이 소나무 숲 고갯길을 환하게 비추던 밤이었어요. 하얀 귀를 쫑긋 세운 꾀돌이 산토끼 달콩이가 향긋한 바구니를 들고 고개를 넘어가는데, 바위 뒤에서 커다란 줄무늬 산호랑이가 '어흥!' 하고 나타나 길을 막아섰답니다.",
                "배가 고픈 호랑이는 금빛 눈을 반짝이며 입맛을 다셨지만, 영리한 달콩이는 조금도 무서워하지 않고 바구니를 살짝 열었어요. 과연 달콩이는 호랑이에게 달콤한 달빛 꿀떡을 건넬까요, 아니면 모닥불 속의 따끈따끈한 조약돌 떡을 건넬까요? 여러분이 상상하는 이야기를 들려주세요!",
            ],
            "character_name": "꾀돌이 산토끼 달콩이",
            "character_persona_voice": "baby_fox",
            "character_dialogue": (
                "깡총깡총! 헤헤, 호랑이 아저씨가 '떡 하나 주면 안 잡아먹지!' 하고 길을 막았어! "
                "내 바구니 속의 달빛 꿀떡을 줄까, 아니면 모닥불에 구운 따끈한 조약돌을 맛있는 떡이라고 속여서 줄까? 네가 다음 이야기를 정해줘, 깡총~!"
            ),
            "science_or_moral_note": (
                "전래 동화 지혜 노트: 위급한 순간에도 당황하지 않고 재치와 지혜를 발휘하면 커다란 호랑이와도 유쾌한 친구가 될 수 있어요."
            ),
            "visual_continuity_note": "1장 기준 장면: 달빛 소나무 고갯길에서 마주 앉은 줄무늬 산호랑이와 꾀돌이 산토끼 달콩이.",
            "nano_banana_prompt": (
                "Masterpiece 16:9 widescreen children's storybook illustration, warm Korean traditional fairy-tale and Studio Ghibli watercolor painting style. "
                "On a moonlit pine forest mountain path with glowing warm lanterns, a big friendly cute striped Korean mountain tiger sits across from a clever white rabbit holding a woven basket."
            ),
            "veo_motion_prompt": (
                "The clever white rabbit twitching its ears happily while the friendly striped Korean tiger tilts its head curiously under moonlight and fireflies."
            ),
            "suggested_followups": [
                "토끼가 호랑이에게 반짝이는 달빛 꿀떡을 주면 어떻게 돼요?",
                "토끼가 모닥불에서 구운 뜨거운 조약돌을 떡이라고 주면 어때요?",
                "호랑이랑 토끼가 무지개 구름을 타고 달나라로 날아가면 어때요?",
                "하늘에서 달콤한 별사탕 비가 쏟아지면 호랑이가 어떻게 해요?",
            ],
        },
    },
    "science_space_shuttle": {
        "preset_id": "science_space_shuttle",
        "mode": "science_lab",
        "lang": "ko",
        "title": "우주왕복선 오비터 무중력 실험책",
        "subtitle": "지구를 도는 우주왕복선 조종석과 실험 데크에서 직접 해보는 무중력 과학 실험",
        "character_name": "우주왕복선 사령관 오르빗 대장",
        "character_persona_voice": "space_captain",
        "character_bible": (
            "Masterpiece 16:9 widescreen children's science picture book illustration. Inside the bright futuristic "
            "Space Shuttle Orbiter flight deck and mid-deck science lab, wide panoramic cockpit windows showing the "
            "glowing blue curvature of Earth and stars outside, featuring Captain Orbit in a blue-and-white shuttle flight suit."
        ),
        "initial_visual_state": {
            "theme": "science_space_shuttle",
            "lang": "ko",
            "character_persona_voice": "space_captain",
            "liquid_color": [96, 165, 250],
            "liquid_level": 0.55,
            "item_key": "zero_g_cabin",
            "item_label": "우주왕복선 오비터 무중력 데크",
            "accumulated_items": [],
            "accumulated_labels": [],
            "ph_level": 7.0,
            "temperature_c": 21.0,
            "bubbles_intensity": 0.45,
            "steam_intensity": 0.0,
            "foam_intensity": 0.0,
            "accent_color": [56, 189, 248],
        },
        "opening_page": {
            "page_number": 1,
            "user_question": "우주왕복선 안에서 할 수 있는 실험책 만들어줘",
            "page_title": "1장 — 지구 궤도에 진입한 우주왕복선 오비터 실험 데크",
            "story_paragraphs": [
                "엄청난 로켓 추진력을 뚫고 대기권을 벗어난 우주왕복선 오비터가 지상 사백 킬로미터 지구 궤도에 성공적으로 진입했어요! 넓은 조종석 창문 너머로 푸른 지구와 찬란한 별빛이 펼쳐지고, 우주왕복선 내부 실험 데크는 모든 물체가 둥실둥실 떠오르는 무중력 상태가 되었답니다.",
                "파랗고 하얀 우주왕복선 비행복을 입은 오르빗 대장님이 무중력 실험 구체들 사이에서 유쾌하게 경례를 보냅니다. 이 우주왕복선 안에서 물을 뿌리거나 양초에 불을 붙이고, 발포 비타민과 과일을 띄우면 어떤 신기한 과학 현상이 일어날까요?",
            ],
            "character_name": "우주왕복선 사령관 오르빗 대장",
            "character_persona_voice": "space_captain",
            "character_dialogue": (
                "오버! 여기는 지구 궤도를 비행 중인 우주왕복선 오비터 실험 데크다! "
                "네가 원하던 우주왕복선 무중력 실험책이 지금 막 펼쳐졌어! 물방울 표면장력 실험부터 구형 불꽃 실험까지 무엇이든 명령해 줘, 로저!"
            ),
            "science_or_moral_note": (
                "우주왕복선 과학 노트: 우주왕복선은 지구 주위를 초속 약 7.8킬로미터로 자유 낙하 비행하기 때문에 내부에서 미세중력(무중력) 상태가 만들어집니다."
            ),
            "visual_continuity_note": "1장 기준 장면: 푸른 지구가 보이는 우주왕복선 오비터 조종석 실험 데크와 오르빗 대장.",
            "nano_banana_prompt": (
                "Masterpiece 16:9 widescreen children's science picture book illustration inside the Space Shuttle Orbiter flight deck and mid-deck science lab, "
                "wide cockpit windows showing glowing blue Earth outside, friendly astronaut captain floating weightlessly beside scientific spheres."
            ),
            "veo_motion_prompt": (
                "Inside the Space Shuttle Orbiter flight deck, the astronaut captain and glowing scientific spheres float weightlessly as Earth rotates outside."
            ),
            "suggested_followups": [
                "우주왕복선 안에서 물을 뿌리면 물방울이 어떻게 되나요?",
                "우주왕복선 안에서 양초에 불을 붙이면 불꽃 모양이 어때요?",
                "우주왕복선 안에서 사과와 과자를 동시에 띄우면 어떻게 되나요?",
                "우주왕복선 물방울 속에 발포 비타민(소다)을 넣으면 어떻게 되나요?",
            ],
        },
    },
}

BOOK_PRESETS_EN: dict[str, dict[str, Any]] = {
    "science_hcl_lab": {
        "preset_id": "science_hcl_lab",
        "mode": "science_lab",
        "lang": "en",
        "title": "Dr. Nova's Wonder Acid Lab",
        "subtitle": "Drop anything into the hydrochloric acid beaker and watch live chemical reactions",
        "character_name": "Dr. Nova the Owl Professor",
        "character_persona_voice": "owl_professor",
        "character_bible": (
            "A bright, high-detail illustrated chemistry laboratory bench with a clear 500mL borosilicate glass "
            "beaker at the center filled with shimmering pale-emerald hydrochloric acid (1.0M HCl), digital pH "
            "and temperature sensors on the left, and friendly clockwork-owl scientist Dr. Nova wearing cyan safety goggles."
        ),
        "initial_visual_state": {
            "theme": "science_hcl_lab",
            "lang": "en",
            "character_persona_voice": "owl_professor",
            "liquid_color": [110, 231, 183],
            "liquid_level": 0.60,
            "item_key": "none",
            "item_label": "Pure 1.0M HCl Solution (Ready for Experiment)",
            "accumulated_items": [],
            "accumulated_labels": [],
            "ph_level": 1.0,
            "temperature_c": 22.0,
            "bubbles_intensity": 0.15,
            "steam_intensity": 0.08,
            "foam_intensity": 0.0,
            "accent_color": [56, 189, 248],
        },
        "opening_page": {
            "page_number": 1,
            "user_question": "Open the laboratory door and inspect the first beaker",
            "page_title": "Chapter 1 — The Whispering Emerald Acid Beaker",
            "story_paragraphs": [
                "Right in the center of the gleaming laboratory bench sits a crystal-clear five-hundred-milliliter borosilicate glass beaker shimmering with pale-emerald hydrochloric acid. It holds the same strong acid found inside human stomach juice, while the digital sensors beside it quietly blink at pH 1.0 and 22.0 degrees Celsius.",
                "Dr. Nova, the friendly clockwork owl professor wearing cyan safety goggles, spreads his brass-trimmed wings to welcome you. Every item you choose to drop into this beaker will stay inside across pages and interact directly with the next reagent you test!",
            ],
            "character_name": "Dr. Nova the Owl Professor",
            "character_persona_voice": "owl_professor",
            "character_dialogue": (
                "Hoo-hoo! Welcome, young scientist! Step right up to my shimmering emerald hydrochloric acid beaker! "
                "What happens if we drop a metal key inside? Or a crisp red apple, a crunchy cookie, or a splash of water? "
                "Tell me what you want to test first, hoo-hoo!"
            ),
            "science_or_moral_note": (
                "Core Science Note: Hydrochloric acid (1.0M HCl) is a strong acid rich in hydrogen ions (H+). "
                "When it meets reactive metals or carbonates, it releases gas bubbles and heat!"
            ),
            "visual_continuity_note": "Chapter 1 Seed Frame: 500mL laboratory beaker with 1.0M HCl (pH 1.0, 22.0 °C) and Dr. Nova.",
            "nano_banana_prompt": (
                "High-craft children's science atlas illustration: a clear 500mL glass beaker of glowing pale-emerald "
                "hydrochloric acid on a laboratory table with digital pH 1.0 and 22.0C meters and Dr. Nova the friendly "
                "owl scientist wearing safety goggles."
            ),
            "veo_motion_prompt": (
                "Gentle micro-bubbles rising inside a clear glass beaker of hydrochloric acid while Dr. Nova the owl professor waves warmly."
            ),
            "suggested_followups": [
                "What happens if we drop a metal key in?",
                "What happens if we drop a red apple in?",
                "What happens if we drop a crunchy cookie in?",
                "What happens if we spray water in?",
                "What happens if we add a spoonful of baking soda?",
                "What happens if we drop an eggshell in?",
                "What happens if we drop a pure gold ring in?",
            ],
        },
    },
    "fairy_starlight_fox": {
        "preset_id": "fairy_starlight_fox",
        "mode": "storybook",
        "lang": "en",
        "title": "Lumi the Starlight Fox & The Whispering Woods",
        "subtitle": "A living fairy-tale picture book where every idea you speak becomes the next page",
        "character_name": "Lumi the Baby Starlight Fox",
        "character_persona_voice": "baby_fox",
        "character_bible": (
            "Warm, high-craft storybook watercolor and gouache illustration featuring Lumi, a small silver-and-amber "
            "baby fox with a glowing star-tipped tail and a royal-blue knitted scarf, inside an enchanted twilight forest."
        ),
        "initial_visual_state": {
            "theme": "fairy_starlight_fox",
            "lang": "en",
            "character_persona_voice": "baby_fox",
            "sky_color": [22, 32, 72],
            "item_key": "magic_door",
            "item_label": "The Whispering Tree Door",
            "accumulated_items": ["magic_door"],
            "accumulated_labels": ["Glowing Oak Door"],
            "sparkle_intensity": 0.65,
            "companion": "none",
            "accent_color": [251, 191, 36],
        },
        "opening_page": {
            "page_number": 1,
            "user_question": "Begin Lumi's adventure in the Starlight Woods",
            "page_title": "Chapter 1 — The Glowing Golden Keyhole in the Ancient Oak",
            "story_paragraphs": [
                "Deep inside the Whispering Woods, little Lumi the Starlight Fox paused in front of a towering ancient oak tree. Wrapped in her cozy royal-blue knitted scarf, she wagged her star-tipped tail, and a tiny golden keyhole carved into the mossy trunk began to glow with warm amber light.",
                "High above the branches floated a sleepy pastel rainbow cloud while golden fireflies chimed softly around Lumi's fluffy ears. Every magical friend, airship, or starlight shower you speak aloud will stay in Lumi's forest as our storybook unfolds page by page!",
            ],
            "character_name": "Lumi the Baby Starlight Fox",
            "character_persona_voice": "baby_fox",
            "character_dialogue": (
                "Yip-yip! Hehe, hello friend! Look how the starlight at the tip of my tail lights up the secret oak door! "
                "What if a friendly baby dragon is hiding behind this door? Or what if sweet star-candy rain falls from the sky? "
                "Tell me what you imagine, yip~!"
            ),
            "science_or_moral_note": (
                "Story Continuity Note: Lumi's royal-blue scarf, glowing star tail, and the ancient oak door stay in the scene across every page as your magical ideas accumulate."
            ),
            "visual_continuity_note": "Chapter 1 Seed Frame: Lumi the Starlight Fox in her blue scarf before the glowing oak door.",
            "nano_banana_prompt": (
                "High-craft children's storybook watercolor illustration of Lumi, a cute silver-and-amber baby fox wearing "
                "a royal-blue scarf with a glowing star tail, standing before a glowing golden keyhole door in an ancient oak tree."
            ),
            "veo_motion_prompt": (
                "Lumi the baby fox wagging her glowing star-tipped tail as golden fireflies drift around the magical oak tree door."
            ),
            "suggested_followups": [
                "What if the door opens and a baby dragon blows bubbles?",
                "What if Lumi rides a rainbow cloud airship?",
                "What if we put a chocolate-chip cookie into the keyhole?",
                "What if sweet star-candy rain falls from the sky?",
            ],
        },
    },
    "science_space_kitchen": {
        "preset_id": "science_space_kitchen",
        "mode": "science_lab",
        "lang": "en",
        "title": "Captain Orbit's Zero-Gravity Space Lab",
        "subtitle": "How do flames, water spheres, and snacks behave aboard an orbital space station?",
        "character_name": "Captain Orbit",
        "character_persona_voice": "space_captain",
        "character_bible": (
            "High-detail illustrated science picture-book cabin inside a panoramic orbital space station with Earth "
            "visible through the cupola window, featuring Captain Orbit in a white-and-orange spacesuit."
        ),
        "initial_visual_state": {
            "theme": "science_space_kitchen",
            "lang": "en",
            "character_persona_voice": "space_captain",
            "liquid_color": [96, 165, 250],
            "liquid_level": 0.50,
            "item_key": "zero_g_cabin",
            "item_label": "Zero-Gravity Floating Chamber",
            "accumulated_items": [],
            "accumulated_labels": [],
            "ph_level": 7.0,
            "temperature_c": 21.0,
            "bubbles_intensity": 0.40,
            "steam_intensity": 0.0,
            "foam_intensity": 0.0,
            "accent_color": [167, 139, 250],
        },
        "opening_page": {
            "page_number": 1,
            "user_question": "Open the hatch to the Zero-Gravity Space Station Lab",
            "page_title": "Chapter 1 — Floating Inside the Orbital Cupola Cabin",
            "story_paragraphs": [
                "Welcome aboard the WonderBook Space Station, orbiting four hundred kilometers above the glowing blue curve of Earth! Inside this panoramic cupola laboratory, microgravity means nothing falls to the floor—every droplet of water and scientific tool floats in mid-air like a jewel.",
                "Captain Orbit does a gentle somersault in his white-and-orange spacesuit and salutes you with a grin. If we spray water, light a candle flame, or float apples and cookies up here in space, physics behaves in ways you never see on the ground!",
            ],
            "character_name": "Captain Orbit",
            "character_persona_voice": "space_captain",
            "character_dialogue": (
                "Copy that, cadet! Captain Orbit reporting from the Zero-Gravity Space Lab four hundred kilometers above Earth! "
                "Up here, everything floats like magic! What happens if we spray water, light a candle, or float an apple and cookie in microgravity? "
                "Give the command, over!"
            ),
            "science_or_moral_note": (
                "Microgravity Physics Note: In free-fall orbit, buoyancy and convection vanish while surface tension pulls liquids into shimmering spheres."
            ),
            "visual_continuity_note": "Chapter 1 Seed Frame: Orbital cupola cabin with Earth horizon in background and Captain Orbit.",
            "nano_banana_prompt": (
                "High-craft children's science illustration of Captain Orbit floating inside a space station cupola "
                "with Earth glowing outside the window."
            ),
            "veo_motion_prompt": (
                "Captain Orbit floating gently in zero gravity while small water droplets drift like crystal marbles."
            ),
            "suggested_followups": [
                "What happens if we spray water in zero gravity?",
                "What shape does a candle flame make in zero gravity?",
                "What happens if we float an apple and a cookie together?",
                "What happens if we add fizzy soda to a floating water sphere?",
            ],
        },
    },
    "fairy_tiger_rabbit": {
        "preset_id": "fairy_tiger_rabbit",
        "mode": "storybook",
        "lang": "en",
        "title": "Clever Rabbit & Mountain Tiger's Moonlit Tale",
        "subtitle": "A Korean folktale storybook where your imagination guides every twist on the mountain path",
        "character_name": "Dalkong the Clever Mountain Rabbit",
        "character_persona_voice": "baby_fox",
        "character_bible": (
            "Warm Korean traditional fairy-tale and Studio Ghibli watercolor painting style. "
            "On a moonlit pine forest mountain path with glowing warm lanterns and autumn persimmon trees, "
            "a big friendly cute striped Korean mountain tiger with expressive golden eyes sits across from "
            "a clever adorable white rabbit holding a woven basket."
        ),
        "initial_visual_state": {
            "theme": "fairy_tiger_rabbit",
            "lang": "en",
            "character_persona_voice": "baby_fox",
            "sky_color": [24, 34, 68],
            "item_key": "tiger_rabbit_path",
            "item_label": "Moonlit Pine Mountain Path & Basket",
            "accumulated_items": ["tiger_rabbit_path"],
            "accumulated_labels": ["Moonlit Mountain Path"],
            "sparkle_intensity": 0.72,
            "companion": "mountain_tiger",
            "accent_color": [251, 191, 36],
        },
        "opening_page": {
            "page_number": 1,
            "user_question": "Start a fairy-tale book about a tiger and a clever rabbit",
            "page_title": "Chapter 1 — The Striped Tiger and the Clever Rabbit on Moonlit Pass",
            "story_paragraphs": [
                "Under a glowing full moon on a pine-scented Korean mountain pass, Dalkong the clever white rabbit hopped along carrying a warm woven basket. Suddenly, a big striped mountain tiger bounded out from behind a boulder with a rumbling 'Roar!' and blocked the lantern-lit trail.",
                "The hungry tiger licked his whiskers with sparkling golden eyes, saying, 'Give me one rice cake and I won't eat you!' Unfazed, Dalkong twitched her fluffy ears and peeked into her basket. Will she share a glowing golden starlight rice cake, or trick the tiger with a sizzling campfire pebble? Tell us what happens next!",
            ],
            "character_name": "Dalkong the Clever Mountain Rabbit",
            "character_persona_voice": "baby_fox",
            "character_dialogue": (
                "Hop-hop! Hehe, Mr. Tiger just blocked the moonlit path asking for a tasty rice cake! "
                "Should I share my glowing golden honey rice cake, or trick him with a warm roasted campfire pebble? "
                "Tell me what happens next, hop-hop~!"
            ),
            "science_or_moral_note": (
                "Folktale Wisdom Note: Quick wit, calm courage, and creative thinking can turn even a fierce mountain tiger into a friendly companion."
            ),
            "visual_continuity_note": "Chapter 1 Seed Frame: Striped Korean mountain tiger and clever white rabbit on the moonlit pine path.",
            "nano_banana_prompt": (
                "Masterpiece 16:9 widescreen children's storybook illustration, warm Korean traditional fairy-tale and Studio Ghibli watercolor painting style. "
                "On a moonlit pine forest mountain path with glowing warm lanterns, a big friendly cute striped Korean mountain tiger sits across from a clever white rabbit holding a woven basket."
            ),
            "veo_motion_prompt": (
                "The clever white rabbit twitching its ears happily while the friendly striped Korean tiger tilts its head curiously under moonlight and fireflies."
            ),
            "suggested_followups": [
                "What if the rabbit gives the tiger glowing golden honey rice cakes?",
                "What if the rabbit tricks the tiger with a warm roasted campfire pebble?",
                "What if the tiger and rabbit ride a rainbow cloud to the moon?",
                "What if sweet star-candy rain falls over the mountain pass?",
            ],
        },
    },
    "science_space_shuttle": {
        "preset_id": "science_space_shuttle",
        "mode": "science_lab",
        "lang": "en",
        "title": "Space Shuttle Orbiter Zero-G Lab",
        "subtitle": "Hands-on microgravity physics experiments aboard the Space Shuttle flight deck",
        "character_name": "Commander Orbit",
        "character_persona_voice": "space_captain",
        "character_bible": (
            "Masterpiece 16:9 widescreen children's science picture book illustration. Inside the bright futuristic "
            "Space Shuttle Orbiter flight deck and mid-deck science lab, wide panoramic cockpit windows showing the "
            "glowing blue curvature of Earth and stars outside, featuring Commander Orbit in a blue-and-white shuttle flight suit."
        ),
        "initial_visual_state": {
            "theme": "science_space_shuttle",
            "lang": "en",
            "character_persona_voice": "space_captain",
            "liquid_color": [96, 165, 250],
            "liquid_level": 0.55,
            "item_key": "zero_g_cabin",
            "item_label": "Space Shuttle Orbiter Flight Deck",
            "accumulated_items": [],
            "accumulated_labels": [],
            "ph_level": 7.0,
            "temperature_c": 21.0,
            "bubbles_intensity": 0.45,
            "steam_intensity": 0.0,
            "foam_intensity": 0.0,
            "accent_color": [56, 189, 248],
        },
        "opening_page": {
            "page_number": 1,
            "user_question": "Create a science experiment book inside a Space Shuttle",
            "page_title": "Chapter 1 — Aboard the Space Shuttle Orbiter in Earth Orbit",
            "story_paragraphs": [
                "After thundering into the sky on twin solid rocket boosters, our Space Shuttle Orbiter is now coasting in low-Earth orbit four hundred kilometers above the glowing blue planet! Through the wide panoramic flight deck windows, Earth's horizon curves against the stars while every scientific instrument floats weightlessly.",
                "Commander Orbit floats effortlessly in a blue-and-white shuttle flight suit beside shimmering zero-gravity test spheres. What experiment shall we run inside the Space Shuttle first—spraying floating water orbs, lighting a zero-gravity candle flame, or launching fizzy tablets into a floating water droplet?",
            ],
            "character_name": "Commander Orbit",
            "character_persona_voice": "space_captain",
            "character_dialogue": (
                "Copy that, mission specialist! Welcome aboard the Space Shuttle Orbiter flight deck! "
                "Your custom Space Shuttle experiment book is live in zero gravity! Shall we test floating water spheres or a spherical candle flame first? Over!"
            ),
            "science_or_moral_note": (
                "Space Shuttle Physics Note: The Space Shuttle Orbiter travels at 28,000 km/h in continuous free-fall around Earth, creating microgravity inside the cabin."
            ),
            "visual_continuity_note": "Chapter 1 Seed Frame: Space Shuttle Orbiter flight deck overlooking Earth with Commander Orbit.",
            "nano_banana_prompt": (
                "Masterpiece 16:9 widescreen children's science picture book illustration inside the Space Shuttle Orbiter flight deck and mid-deck science lab, "
                "wide cockpit windows showing glowing blue Earth outside, friendly astronaut captain floating weightlessly beside scientific spheres."
            ),
            "veo_motion_prompt": (
                "Inside the Space Shuttle Orbiter flight deck, the astronaut captain and glowing scientific spheres float weightlessly as Earth rotates outside."
            ),
            "suggested_followups": [
                "What happens if we spray water inside the Space Shuttle?",
                "What shape does a candle flame make inside the Space Shuttle?",
                "What happens if we float an apple and a cookie in the cockpit?",
                "What happens if we drop a fizzy tablet into a floating water sphere?",
            ],
        },
    },
}


def get_localized_preset(preset_id: str, lang: str = "ko") -> dict[str, Any]:
    """Return a deep copy of the requested preset in Pure Korean (`ko`) or Pure English (`en`)."""
    norm_lang = "en" if str(lang or "ko").strip().lower().startswith("en") else "ko"
    catalog = BOOK_PRESETS_EN if norm_lang == "en" else BOOK_PRESETS
    chosen_id = preset_id if preset_id in catalog else DEFAULT_PRESET_ID
    return copy.deepcopy(catalog[chosen_id])


def _configure_audit_logger() -> logging.Logger:
    """Create and configure the application-wide INFO logger."""
    logger = logging.getLogger("wonderbook_live")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    try:
        file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        pass

    return logger


AUDIT_LOGGER = _configure_audit_logger()


def _sanitize_value(value: Any, max_str_len: int = 260) -> Any:
    """Sanitize parameters for safe logging without binary blobs or API keys."""
    if isinstance(value, (bytes, bytearray)):
        return f"<bytes len={len(value)}>"
    if isinstance(value, str):
        if value.startswith("data:image/") or value.startswith("data:audio/"):
            return f"<data-url len={len(value)}>"
        if len(value) > max_str_len:
            return value[:max_str_len] + f"...<truncated total={len(value)}>"
        return value
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for k, v in value.items():
            key_str = str(k)
            if any(sec in key_str.lower() for sec in ("api_key", "secret", "token", "password", "authorization")):
                sanitized[key_str] = "<redacted-secret>"
            elif any(blob in key_str.lower() for blob in ("base64", "bytes", "data_url", "raw_image", "raw_audio")):
                sanitized[key_str] = f"<redacted-binary len={len(v) if hasattr(v, '__len__') else 'n/a'}>"
            else:
                sanitized[key_str] = _sanitize_value(v, max_str_len=max_str_len)
        return sanitized
    if isinstance(value, (list, tuple)):
        if len(value) > 15:
            return [_sanitize_value(v, max_str_len=max_str_len) for v in value[:15]] + [
                f"...<+{len(value) - 15} items>"
            ]
        return [_sanitize_value(v, max_str_len=max_str_len) for v in value]
    return value


def log_function_call(func_name: str, params: dict[str, Any] | None = None) -> None:
    """Log a function invocation and its sanitized parameters at INFO level."""
    safe_params = _sanitize_value(params or {})
    AUDIT_LOGGER.info(
        "FUNC_CALL | function=%s | params=%s",
        func_name,
        json.dumps(safe_params, ensure_ascii=False, default=str),
    )


def log_genai_call(
    model: str,
    prompt_summary: str,
    config_summary: dict[str, Any],
    output_summary: Any,
    status: str = "ok",
) -> None:
    """Log a Gemini / Nano Banana / Veo API call at INFO level without binary payloads."""
    payload = {
        "model": model,
        "status": status,
        "config": _sanitize_value(config_summary),
        "prompt": _sanitize_value(prompt_summary, max_str_len=400),
        "output": _sanitize_value(output_summary, max_str_len=500),
    }
    AUDIT_LOGGER.info("GENAI_CALL | %s", json.dumps(payload, ensure_ascii=False, default=str))


def get_gemini_api_key() -> str:
    """Retrieve the Gemini API key from environment variables without hardcoding."""
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = os.environ.get(env_name, "").strip()
        if val:
            return val
    return ""


def is_gemini_configured() -> bool:
    """Return True if a Gemini API key is configured in the environment."""
    return bool(get_gemini_api_key())


def get_cloud_config() -> dict[str, Any]:
    """Return Google Cloud runtime and artifact bucket settings."""
    return {
        "project_id": os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
        "region": os.environ.get("GOOGLE_CLOUD_REGION", "global"),
        "service_name": os.environ.get("K_SERVICE", "wonderbook-live"),
        "artifact_bucket": os.environ.get("ARTIFACT_BUCKET", "").strip(),
    }
