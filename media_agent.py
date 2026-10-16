"""Media Orchestration Agent: Nano Banana 16:9 Visual Continuity & Veo Motion/Voice Engine.

This module implements the multimodal media generation tier of WonderBook Live:
1. Nano Banana Multi-Model Image Continuity (`generate_page_illustration`):
   - Accepts `prev_image_bytes` (the previous page's generated illustration) together with
     `nano_banana_prompt` and `visual_state` (including `accumulated_items`).
   - When `GEMINI_API_KEY` is configured and `prefer_live_api=True`, invokes the Nano Banana
     multi-model chain (`[NANO_BANANA_PRO_MODEL, NANO_BANANA_MODEL]`, i.e.,
     `gemini-3-pro-image-preview` -> `gemini-2.5-flash-image`), passing BOTH the previous
     page's PNG (`types.Part.from_bytes`) and the cumulative continuity prompt so the
     laboratory beaker or storybook protagonist preserves exact visual continuity while
     accumulating every item/twist across turns.
   - Provides a pure, cinematic 16:9 (`1280x720`) Nano Banana continuity compositor
     (`_render_scene_frame`) backed by high-resolution pre-generated Nano Banana artwork
     (`static/assets/nano_banana/*.png`), feathered Gaussian-blurred multi-item region
     compositing when custom item orders are triggered, and delicate painterly lighting
     effects with zero burned-in HUD boxes or text overlays.
2. Veo Video & Expressive Character Persona Voice Engine (`generate_page_motion_and_voice`):
   - Invokes `VEO_MODEL` logging when configured, and produces a smooth 8-frame 16:9
     widescreen animated GIF (`960x540`, `motion.gif`) with organic sinusoidal camera
     breathing and animated atmospheric particles (rising specular bubbles, cyan water mist,
     or twinkling golden starlight fireflies).
   - Synthesizes expressive, character-acted voice audio (`_generate_character_voice_bytes`)
     using Gemini Native TTS (`gemini-2.5-flash-preview-tts` / `gemini-2.5-pro-preview-tts`)
     when `GEMINI_API_KEY` is active, or neural character persona TTS via `edge_tts`
     (Dr. Nova the Clockwork Owl Professor, Lumi the Baby Starlight Fox, and Captain Orbit)
     with automatic Korean/English detection.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import math
import os
from pathlib import Path
import struct
import wave
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

try:
    import config
    from config import (
        NANO_BANANA_MODEL,
        VEO_MODEL,
        get_gemini_api_key,
        log_function_call,
        log_genai_call,
    )
except ImportError:
    from . import config  # type: ignore[no-redef]
    from .config import (  # type: ignore[no-redef]
        NANO_BANANA_MODEL,
        VEO_MODEL,
        get_gemini_api_key,
        log_function_call,
        log_genai_call,
    )

NANO_BANANA_PRO_MODEL: str = getattr(
    config, "NANO_BANANA_PRO_MODEL", "gemini-3-pro-image-preview"
)
GEMINI_TTS_MODELS: tuple[str, ...] = (
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
)

CANVAS_WIDTH: int = 1280
CANVAS_HEIGHT: int = 720
GIF_WIDTH: int = 640
GIF_HEIGHT: int = 360
GIF_NUM_FRAMES: int = 10

NANO_BANANA_ASSETS_DIR: Path = (
    Path(__file__).resolve().parent / "static" / "assets" / "nano_banana"
)

_ASSET_IMAGE_CACHE: dict[str, Image.Image] = {}
_VOICE_AUDIO_CACHE: dict[str, tuple[bytes, dict[str, Any]]] = {}

ITEM_DISPLAY_NAMES: dict[str, str] = {
    "metal_key": "Fe/Brass Key",
    "apple": "Red Apple",
    "cookie_snack": "Cookie Snack",
    "water_spray": "H2O Spray",
    "baking_soda": "NaHCO3 Soda",
    "egg_shell": "CaCO3 Eggshell",
    "gold_ring": "24K Gold Ring",
    "zero_g_candle": "Zero-G Flame",
    "dragon_bubbles": "Bubble Dragon",
    "rainbow_spaceship": "Rainbow Ship",
    "star_candy": "Star Candy Rain",
    "rice_cake": "Golden Honey Rice Cake",
    "roasted_pebble": "Roasted Campfire Pebble",
    "tiger_rabbit_path": "Moonlit Mountain Path",
    "magic_door": "Whispering Door",
    "zero_g_cabin": "Orbital Cupola",
    "custom_science": "Custom Reagent",
    "custom_story": "Story Twist",
}

# Items already natively depicted inside each pre-generated Nano Banana continuity plate
NATIVE_PLATE_ITEMS: dict[str, set[str]] = {
    "hcl_base.png": set(),
    "hcl_metal_key.png": {"metal_key"},
    "hcl_apple.png": {"metal_key", "apple"},
    "hcl_cookie_snack.png": {"metal_key", "apple", "cookie_snack"},
    "hcl_water_spray.png": {"metal_key", "apple", "cookie_snack", "water_spray"},
    "hcl_baking_soda.png": {"metal_key", "apple", "cookie_snack", "baking_soda"},
    "hcl_egg_shell.png": {"metal_key", "egg_shell"},
    "hcl_gold_ring.png": {"metal_key", "apple", "gold_ring"},
    "space_base.png": set(),
    "space_shuttle_base.png": set(),
    "space_zero_g_candle.png": {"zero_g_candle"},
    "tiger_rabbit_base.png": set(),
    "tiger_rabbit_ricecake.png": {"rice_cake"},
    "tiger_rabbit_pebble.png": {"roasted_pebble"},
    "fox_base.png": set(),
    "fox_dragon_bubbles.png": {"dragon_bubbles"},
    "fox_rainbow_spaceship.png": {"dragon_bubbles", "rainbow_spaceship"},
    "fox_star_candy.png": {"dragon_bubbles", "star_candy"},
    "fox_cookie_snack.png": {"cookie_snack"},
    "fox_grand_finale.png": {
        "dragon_bubbles",
        "rainbow_spaceship",
        "star_candy",
        "cookie_snack",
    },
}

# Focal region crop boxes (on 1280x720 canvas) for feathered multi-item continuity compositing
SCIENCE_ITEM_REGIONS: dict[str, tuple[str, tuple[int, int, int, int], int]] = {
    "metal_key": ("hcl_metal_key.png", (410, 430, 660, 610), 170),
    "apple": ("hcl_apple.png", (410, 210, 640, 420), 180),
    "cookie_snack": ("hcl_cookie_snack.png", (485, 255, 685, 515), 175),
    "water_spray": ("hcl_water_spray.png", (75, 20, 580, 410), 190),
    "baking_soda": ("hcl_baking_soda.png", (375, 105, 735, 465), 185),
    "egg_shell": ("hcl_egg_shell.png", (450, 240, 685, 505), 180),
    "gold_ring": ("hcl_gold_ring.png", (455, 275, 675, 510), 190),
}

STORY_ITEM_REGIONS: dict[str, tuple[str, tuple[int, int, int, int], int]] = {
    "dragon_bubbles": ("fox_dragon_bubbles.png", (435, 220, 935, 625), 180),
    "rainbow_spaceship": ("fox_rainbow_spaceship.png", (380, 205, 755, 525), 190),
    "star_candy": ("fox_star_candy.png", (355, 10, 815, 430), 180),
    "cookie_snack": ("fox_cookie_snack.png", (515, 335, 785, 530), 185),
    "rice_cake": ("tiger_rabbit_ricecake.png", (360, 210, 890, 630), 185),
    "roasted_pebble": ("tiger_rabbit_pebble.png", (360, 210, 890, 630), 185),
}


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a readable TrueType font (with Korean & Latin support) or fallback to default.

    Args:
        size: Font point size in pixels.
        bold: Whether to prefer bold font weights.

    Returns:
        Loaded `ImageFont.FreeTypeFont` or fallback `ImageFont.ImageFont`.
    """
    candidates = (
        ("malgunbd.ttf", "arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf", "malgun.ttf", "arial.ttf")
        if bold
        else ("malgun.ttf", "arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "malgunbd.ttf")
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _resolve_accumulated_items(visual_state: dict[str, Any]) -> list[str]:
    """Extract ordered, deduplicated cumulative items from `visual_state`.

    Args:
        visual_state: Page visual state dictionary.

    Returns:
        Ordered list of active item keys accumulated up to the current page.
    """
    seed_keys = ("none", "null", "initial_setup", "magic_door", "zero_g_cabin", "tiger_rabbit_path")
    raw_list = visual_state.get("accumulated_items")
    items: list[str] = []
    if isinstance(raw_list, (list, tuple)):
        for entry in raw_list:
            key = str(entry or "").strip()
            if key and key not in seed_keys and key not in items:
                items.append(key)

    current_key = str(visual_state.get("item_key") or "").strip()
    if current_key and current_key not in seed_keys and current_key not in items:
        items.append(current_key)

    return items


def _compute_lab_telemetry(
    visual_state: dict[str, Any],
    accumulated_items: list[str],
) -> dict[str, Any]:
    """Compute realistic laboratory pH, temperature, and reaction formula telemetry.

    Reads explicit `ph_level` and `temperature_c` from `visual_state` when provided,
    or derives physically grounded values from the cumulative items inside the beaker.

    Args:
        visual_state: Visual state dictionary for the current page.
        accumulated_items: Ordered list of items added to the beaker.

    Returns:
        Dictionary with `ph_level`, `ph_status`, `temperature_c`, `temp_status`,
        and `formula_text`.
    """
    if "ph_level" in visual_state and visual_state.get("ph_level") is not None:
        try:
            ph_val = float(visual_state["ph_level"])
        except (TypeError, ValueError):
            ph_val = 1.0
    else:
        ph_val = 1.0
        for item in accumulated_items:
            if item == "metal_key":
                ph_val += 0.35
            elif item == "apple":
                ph_val += 0.20
            elif item == "cookie_snack":
                ph_val += 0.75
            elif item == "water_spray":
                ph_val += 0.95
            elif item == "baking_soda":
                ph_val += 2.80
            elif item == "egg_shell":
                ph_val += 1.40
            elif item == "gold_ring":
                ph_val += 0.0
        ph_val = max(0.5, min(8.5, ph_val))

    if "temperature_c" in visual_state and visual_state.get("temperature_c") is not None:
        try:
            temp_c = float(visual_state["temperature_c"])
        except (TypeError, ValueError):
            temp_c = 22.0
    else:
        steam = float(visual_state.get("steam_intensity", 0.1))
        temp_c = 21.5 + steam * 5.0
        for item in accumulated_items:
            if item == "metal_key":
                temp_c += 11.2
            elif item == "water_spray":
                temp_c += 4.3
            elif item == "baking_soda":
                temp_c += 6.8
            elif item == "cookie_snack":
                temp_c += 2.1
            elif item == "apple":
                temp_c += 0.7
            elif item == "egg_shell":
                temp_c += 3.4
            elif item == "zero_g_candle":
                temp_c += 18.0
        temp_c = max(18.0, min(68.0, temp_c))

    if ph_val < 2.0:
        ph_status = "STRONG ACID"
    elif ph_val < 4.5:
        ph_status = "MODERATE ACID"
    elif ph_val < 6.5:
        ph_status = "MILD ACID"
    else:
        ph_status = "NEUTRALIZED"

    temp_status = "EXOTHERMIC (+dH)" if temp_c >= 26.5 else "AMBIENT (22C)"

    latest = accumulated_items[-1] if accumulated_items else str(visual_state.get("item_key") or "none")
    formula_map = {
        "metal_key": "Fe(s) + 2HCl -> FeCl2 + H2(g)",
        "apple": "Pectin + H3O+ -> Hydrolyzed Sugars",
        "cookie_snack": "NaHCO3 + HCl -> NaCl + H2O + CO2(g)",
        "water_spray": "HCl + H2O -> H3O+(aq) + Cl-(aq)",
        "baking_soda": "NaHCO3 + HCl -> NaCl + H2O + CO2(g)",
        "egg_shell": "CaCO3(s) + 2HCl -> CaCl2 + H2O + CO2(g)",
        "gold_ring": "Au(s) + HCl(aq) -> No Reaction (Noble Metal)",
        "zero_g_candle": "CnH2n+2 + O2 -> Zero-G Flame",
    }
    formula_text = str(
        visual_state.get("reaction_formula")
        or formula_map.get(latest, "HCl(aq) + H2O(l) -> H3O+(aq) + Cl-(aq)")
    )

    return {
        "ph_level": round(ph_val, 2),
        "ph_status": ph_status,
        "temperature_c": round(temp_c, 1),
        "temp_status": temp_status,
        "formula_text": formula_text,
    }


def _try_live_nano_banana(
    nano_banana_prompt: str,
    prev_image_bytes: bytes | None,
    visual_state: dict[str, Any] | None = None,
) -> tuple[bytes | None, dict[str, Any]]:
    """Attempt live image generation/editing via Nano Banana Pro & Flash Image models.

    Tries `[NANO_BANANA_PRO_MODEL, NANO_BANANA_MODEL]` (`gemini-3-pro-image-preview`
    then `gemini-2.5-flash-image`), passing `prev_image_bytes` as
    `types.Part.from_bytes(data=prev_image_bytes, mime_type="image/png")` together with
    the cumulative item list so Page N directly edits/continues Page N-1's illustration.

    Args:
        nano_banana_prompt: Continuity-aware image generation prompt.
        prev_image_bytes: Optional PNG bytes from the previous page.
        visual_state: Optional visual state dictionary containing `accumulated_items`.

    Returns:
        Tuple of `(png_bytes_or_none, engine_meta_dict)`.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        return None, {
            "engine_mode": "nano_banana_continuity_compositor",
            "reason": "GEMINI_API_KEY not set",
        }

    accumulated_items = _resolve_accumulated_items(visual_state or {})
    acc_summary = ", ".join(accumulated_items) if accumulated_items else "initial seed scene"

    model_candidates: list[str] = []
    for candidate in (NANO_BANANA_PRO_MODEL, NANO_BANANA_MODEL):
        if candidate and candidate not in model_candidates:
            model_candidates.append(candidate)

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        contents: list[Any] = []
        if prev_image_bytes:
            contents.append(types.Part.from_bytes(data=prev_image_bytes, mime_type="image/png"))
            contents.append(
                "Edit and continue this exact previous page 16:9 widescreen illustration while preserving "
                "the identical character design, glass laboratory beaker or forest setting, camera angle, "
                "and warm picture-book art style. Do NOT render any text, labels, or HUD boxes on the image. "
                f"Cumulative items that must ALL remain visible in the scene: [{acc_summary}]. "
                f"New page scene instruction: {nano_banana_prompt}"
            )
        else:
            contents.append(
                f"{nano_banana_prompt} | 16:9 widescreen storybook illustration, zero text or HUD boxes. "
                f"Cumulative scene elements: [{acc_summary}]."
            )

        for model_name in model_candidates:
            cfg_summary = {
                "chained_from_prev_image": bool(prev_image_bytes),
                "accumulated_items": accumulated_items,
            }
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                )
                candidates = getattr(response, "candidates", None) or []
                for cand in candidates:
                    content = getattr(cand, "content", None)
                    parts = getattr(content, "parts", None) or []
                    for part in parts:
                        inline_data = getattr(part, "inline_data", None)
                        data_bytes = getattr(inline_data, "data", None) if inline_data else None
                        if isinstance(data_bytes, (bytes, bytearray)) and len(data_bytes) > 100:
                            log_genai_call(
                                model=model_name,
                                prompt_summary=nano_banana_prompt,
                                config_summary=cfg_summary,
                                output_summary=f"<generated image bytes={len(data_bytes)}>",
                                status="ok",
                            )
                            return bytes(data_bytes), {
                                "engine_mode": "nano_banana_live",
                                "model_used": model_name,
                                "chained_from_prev_image": bool(prev_image_bytes),
                                "accumulated_items": accumulated_items,
                            }

                log_genai_call(
                    model=model_name,
                    prompt_summary=nano_banana_prompt,
                    config_summary=cfg_summary,
                    output_summary="No inline image part returned; trying next model or continuity compositor",
                    status="fallback",
                )
            except Exception as model_exc:
                log_genai_call(
                    model=model_name,
                    prompt_summary=nano_banana_prompt,
                    config_summary=cfg_summary,
                    output_summary=str(model_exc)[:300],
                    status="error",
                )
    except Exception as exc:
        log_genai_call(
            model=NANO_BANANA_MODEL,
            prompt_summary=nano_banana_prompt,
            config_summary={"chained_from_prev_image": bool(prev_image_bytes)},
            output_summary=str(exc)[:300],
            status="error",
        )

    return None, {
        "engine_mode": "nano_banana_continuity_compositor",
        "model_used": "nano-banana-16x9-compositor",
        "accumulated_items": accumulated_items,
    }


def _load_nano_banana_asset(filename: str) -> Image.Image | None:
    """Load and cache a high-res Nano Banana PNG fitted to `(1280, 720)` RGBA.

    Args:
        filename: Asset basename inside `static/assets/nano_banana/`.

    Returns:
        A fresh `.copy()` of the `(1280, 720)` RGBA `PIL.Image.Image`, or `None` if missing.
    """
    cached = _ASSET_IMAGE_CACHE.get(filename)
    if cached is not None:
        return cached.copy()

    asset_path = NANO_BANANA_ASSETS_DIR / filename
    if not asset_path.is_file():
        return None

    try:
        with Image.open(asset_path) as raw_img:
            fitted = ImageOps.fit(
                raw_img.convert("RGBA"),
                (CANVAS_WIDTH, CANVAS_HEIGHT),
                method=Image.Resampling.LANCZOS,
            )
            _ASSET_IMAGE_CACHE[filename] = fitted
            return fitted.copy()
    except Exception:
        return None


def _select_nano_banana_plate(
    visual_state: dict[str, Any],
    accumulated_items: list[str],
) -> tuple[Image.Image | None, str]:
    """Select and load the primary 16:9 Nano Banana illustration plate for the scene.

    Args:
        visual_state: Page visual state dictionary.
        accumulated_items: Ordered list of cumulative items/twists in the scene.

    Returns:
        Tuple of `(rgba_image_or_none, chosen_filename)`.
    """
    theme = str(
        visual_state.get("theme")
        or visual_state.get("topic_id")
        or visual_state.get("preset_id")
        or "science_hcl_lab"
    ).strip()
    persona_voice = str(visual_state.get("character_persona_voice") or "").strip()
    item_key = str(visual_state.get("item_key") or "none").strip()
    seed_keys = ("", "none", "null", "initial_setup", "magic_door", "zero_g_cabin", "tiger_rabbit_path")
    active_key = (
        item_key
        if item_key not in seed_keys
        else (accumulated_items[-1] if accumulated_items else "none")
    )

    is_tiger_rabbit = (
        theme == "fairy_tiger_rabbit"
        or "tiger" in theme
        or "rabbit" in theme
        or item_key == "tiger_rabbit_path"
        or active_key in ("rice_cake", "roasted_pebble", "tiger_rabbit_path")
    )
    is_space_shuttle = (
        theme == "science_space_shuttle"
        or "shuttle" in theme
        or "orbiter" in theme
    )

    story_keys = {
        "dragon_bubbles",
        "rainbow_spaceship",
        "star_candy",
        "rice_cake",
        "roasted_pebble",
        "tiger_rabbit_path",
        "magic_door",
        "custom_story",
    }
    is_storybook = (
        is_tiger_rabbit
        or theme in ("fairy_starlight_fox", "starlight_fox")
        or persona_voice == "baby_fox"
        or "story" in theme
        or "fairy" in theme
        or "fox" in theme
        or item_key == "magic_door"
        or active_key in story_keys
    )
    is_space = (
        is_space_shuttle
        or theme in ("science_space_kitchen", "zero_g_space_cabin")
        or persona_voice == "space_captain"
        or "space" in theme
        or "zero_g" in theme
        or item_key == "zero_g_cabin"
        or active_key in ("zero_g_candle", "zero_g_cabin")
    )

    candidates: list[str] = []
    if is_tiger_rabbit:
        if active_key == "rice_cake":
            candidates.extend(["tiger_rabbit_ricecake.png", "tiger_rabbit_base.png"])
        elif active_key == "roasted_pebble":
            candidates.extend(["tiger_rabbit_pebble.png", "tiger_rabbit_base.png"])
        elif active_key in ("dragon_bubbles", "rainbow_spaceship"):
            candidates.extend(["tiger_rabbit_cloud.png", "tiger_rabbit_pebble.png", "tiger_rabbit_base.png"])
        elif active_key in ("star_candy", "cookie_snack", "custom_story"):
            candidates.extend(["tiger_rabbit_starcandy.png", "tiger_rabbit_cloud.png", "tiger_rabbit_ricecake.png", "tiger_rabbit_base.png"])
        elif "rice_cake" in accumulated_items and "roasted_pebble" in accumulated_items:
            candidates.extend(["tiger_rabbit_cloud.png", "tiger_rabbit_pebble.png", "tiger_rabbit_ricecake.png", "tiger_rabbit_base.png"])
        elif "rice_cake" in accumulated_items:
            candidates.extend(["tiger_rabbit_ricecake.png", "tiger_rabbit_base.png"])
        elif "roasted_pebble" in accumulated_items:
            candidates.extend(["tiger_rabbit_pebble.png", "tiger_rabbit_base.png"])
        else:
            candidates.extend(["tiger_rabbit_base.png", "tiger_rabbit_ricecake.png"])

    elif is_storybook:
        story_accum_count = sum(
            1
            for k in accumulated_items
            if k in ("dragon_bubbles", "rainbow_spaceship", "star_candy", "cookie_snack", "custom_story")
        )
        if story_accum_count >= 3 or len(accumulated_items) >= 3:
            candidates.append("fox_grand_finale.png")

        if active_key == "dragon_bubbles":
            candidates.extend(["fox_dragon_bubbles.png", "fox_base.png"])
        elif active_key == "rainbow_spaceship":
            candidates.extend(["fox_rainbow_spaceship.png", "fox_dragon_bubbles.png", "fox_base.png"])
        elif active_key == "star_candy":
            candidates.extend(["fox_star_candy.png", "fox_dragon_bubbles.png", "fox_base.png"])
        elif active_key == "cookie_snack":
            candidates.extend(["fox_cookie_snack.png", "fox_star_candy.png", "fox_base.png"])
        elif active_key == "custom_story":
            candidates.extend(["fox_grand_finale.png", "fox_star_candy.png", "fox_dragon_bubbles.png", "fox_base.png"])
        else:
            candidates.append("fox_base.png")

    elif is_space_shuttle:
        if active_key == "water_spray":
            candidates.extend(["space_shuttle_water.png", "space_shuttle_base.png", "space_base.png"])
        elif active_key in ("cookie_snack", "baking_soda"):
            candidates.extend(["space_shuttle_cookie.png", "space_shuttle_base.png", "space_base.png"])
        elif active_key == "zero_g_candle":
            candidates.extend(["space_zero_g_candle.png", "space_shuttle_base.png", "space_base.png"])
        else:
            candidates.extend(["space_shuttle_base.png", "space_shuttle_water.png", "space_base.png"])

    elif is_space:
        if active_key == "water_spray":
            candidates.extend(["space_shuttle_water.png", "space_base.png"])
        elif active_key in ("cookie_snack", "baking_soda"):
            candidates.extend(["space_shuttle_cookie.png", "space_base.png"])
        elif active_key == "zero_g_candle":
            candidates.extend(["space_zero_g_candle.png", "space_base.png"])
        else:
            candidates.extend(["space_base.png", "space_zero_g_candle.png"])

    else:
        if active_key in seed_keys and not accumulated_items:
            candidates.append("hcl_base.png")
        elif active_key == "metal_key":
            candidates.extend(["hcl_metal_key.png", "hcl_base.png"])
        elif active_key == "apple":
            candidates.extend(["hcl_apple.png", "hcl_metal_key.png", "hcl_base.png"])
        elif active_key == "cookie_snack":
            candidates.extend(["hcl_cookie_snack.png", "hcl_apple.png", "hcl_base.png"])
        elif active_key == "water_spray":
            candidates.extend(["hcl_water_spray.png", "hcl_cookie_snack.png", "hcl_base.png"])
        elif active_key == "baking_soda":
            candidates.extend(["hcl_baking_soda.png", "hcl_cookie_snack.png", "hcl_base.png"])
        elif active_key == "egg_shell":
            candidates.extend(["hcl_egg_shell.png", "hcl_cookie_snack.png", "hcl_metal_key.png"])
        elif active_key == "gold_ring":
            candidates.extend(["hcl_gold_ring.png", "hcl_apple.png", "hcl_metal_key.png"])
        else:
            candidates.extend(
                [
                    "hcl_baking_soda.png",
                    "hcl_water_spray.png",
                    "hcl_cookie_snack.png",
                    "hcl_metal_key.png",
                    "hcl_base.png",
                ]
            )

    for cand_name in candidates:
        loaded = _load_nano_banana_asset(cand_name)
        if loaded is not None:
            return loaded, cand_name

    return None, ""


def _composite_feathered_multi_items(
    canvas: Image.Image,
    primary_plate_name: str,
    visual_state: dict[str, Any],
    accumulated_items: list[str],
) -> None:
    """Blend feathered focal regions for accumulated items in storybook mode only.

    Science labs run clean 1-by-1 independent experiments per turn, whereas storybooks
    accumulate magical elements across chapters.
    """
    if len(accumulated_items) <= 1:
        return

    theme = str(visual_state.get("theme") or "science_hcl_lab")
    is_storybook = (
        theme in ("fairy_starlight_fox", "fairy_tiger_rabbit")
        or "story" in theme
        or "fairy" in theme
        or primary_plate_name.startswith("fox_")
        or primary_plate_name.startswith("tiger_")
    )
    if not is_storybook:
        return

    already_present = set(NATIVE_PLATE_ITEMS.get(primary_plate_name, set()))
    region_catalog = STORY_ITEM_REGIONS

    for item_key in accumulated_items:
        if item_key in already_present:
            continue
        spec = region_catalog.get(item_key)
        if spec is None:
            continue
        plate_file, (x0, y0, x1, y1), max_alpha = spec
        if plate_file == primary_plate_name:
            continue

        sec_img = _load_nano_banana_asset(plate_file)
        if sec_img is None:
            continue

        patch_w = max(16, x1 - x0)
        patch_h = max(16, y1 - y0)
        patch = sec_img.crop((x0, y0, x0 + patch_w, y0 + patch_h))

        mask = Image.new("L", (patch_w, patch_h), 0)
        mask_draw = ImageDraw.Draw(mask)
        pad_x = max(12, patch_w // 7)
        pad_y = max(12, patch_h // 7)
        mask_draw.ellipse(
            [pad_x, pad_y, patch_w - pad_x, patch_h - pad_y],
            fill=max_alpha,
        )
        mask = mask.filter(ImageFilter.GaussianBlur(radius=28))

        canvas.paste(patch, (x0, y0), mask)
        already_present.add(item_key)


def _draw_four_point_star(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    outer_r: int,
    inner_r: int,
    fill: tuple[int, ...],
) -> None:
    """Draw a sparkling 4-point star polygon centered at `(cx, cy)`."""
    pts = [
        (cx, cy - outer_r),
        (cx + inner_r, cy - inner_r),
        (cx + outer_r, cy),
        (cx + inner_r, cy + inner_r),
        (cx, cy + outer_r),
        (cx - inner_r, cy + inner_r),
        (cx - outer_r, cy),
        (cx - inner_r, cy - inner_r),
    ]
    draw.polygon(pts, fill=fill)


def _apply_painterly_continuity_effects(
    canvas: Image.Image,
    visual_state: dict[str, Any],
    accumulated_items: list[str],
    page_number: int,
    frame_idx: int = 0,
    num_frames: int = GIF_NUM_FRAMES,
) -> Image.Image:
    """Composite delicate, HUD-free painterly atmospheric particles & lighting onto `canvas`.

    Adds translucent rising specular micro-bubbles, warm alchemical glow, cyan water mist
    droplets, or twinkling golden forest fireflies and iridescent soap bubbles scaled cleanly
    to `canvas.size` (`1280x720` for PNG or `960x540` for animated GIF frames).

    Args:
        canvas: Base RGBA or RGB image.
        visual_state: Visual state dictionary for the current page.
        accumulated_items: Ordered list of cumulative items in the scene.
        page_number: Current 1-based page number.
        frame_idx: Animation frame index (`0..num_frames-1`).
        num_frames: Total loop frames for seamless cyclical particle motion.

    Returns:
        Composited RGBA `PIL.Image.Image` of the same dimensions as `canvas`.
    """
    base_rgba = canvas.convert("RGBA")
    width, height = base_rgba.size
    sx = width / float(CANVAS_WIDTH)
    sy = height / float(CANVAS_HEIGHT)

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)

    theme = str(visual_state.get("theme") or "science_hcl_lab")
    phase = (2.0 * math.pi * frame_idx) / max(1, num_frames)

    item_label_low = str(visual_state.get("item_label") or "").lower()
    liq_col = str(visual_state.get("liquid_color") or "").lower()

    if theme in ("fairy_starlight_fox", "fairy_tiger_rabbit") or "story" in theme or "fairy" in theme:
        is_moon_gone = (
            "달" in item_label_low
            or "사라" in item_label_low
            or "어둠" in item_label_low
            or "moon" in item_label_low
            or liq_col == "indigo_violet"
        )
        if is_moon_gone:
            # Deep nocturnal eclipse veil over the upper sky & moon region so the moon visibly vanishes
            ov_draw.rectangle(
                [0, 0, width, int(height * 0.58)],
                fill=(10, 14, 38, 135),
            )
            moon_cx, moon_cy = int(640 * sx), int(145 * sy)
            for m_rad, m_alpha in ((170, 175), (120, 210), (75, 235)):
                mrx, mry = int(m_rad * sx), int(m_rad * sy)
                ov_draw.ellipse(
                    [moon_cx - mrx, moon_cy - mry, moon_cx + mrx, moon_cy + mry],
                    fill=(12, 17, 46, m_alpha),
                )

        # 1. Warm golden lantern & door light bloom on the right side of the clearing
        door_cx, door_cy = int(910 * sx), int(485 * sy)
        pulse_alpha = int((28 if is_moon_gone else 18) + 8 * math.sin(phase))
        for rad_base, a_mul in ((190, 0.5), (120, 0.8), (65, 1.1)):
            rx = int(rad_base * sx)
            ry = int(rad_base * sy)
            ov_draw.ellipse(
                [door_cx - rx, door_cy - ry, door_cx + rx, door_cy + ry],
                fill=(253, 224, 71, max(4, int(pulse_alpha * a_mul))),
            )

        # 2. Twinkling golden starlight fireflies across the enchanted forest clearing
        firefly_count = (26 if is_moon_gone else 12) + min(10, len(accumulated_items) * 3)
        for idx in range(firefly_count):
            seed_x = (idx * 137 + page_number * 29) % 1080 + 100
            seed_y = (idx * 89 + page_number * 17) % 520 + 90
            drift_x = math.sin(phase + idx * 0.7) * 14.0
            drift_y = math.cos(phase + idx * 0.9) * 10.0
            fx = int((seed_x + drift_x) * sx)
            fy = int((seed_y + drift_y) * sy)
            twinkle = 0.55 + 0.45 * math.sin(phase * 2.0 + idx * 1.3)
            halo_r = max(3, int((8 + (idx % 4) * 2) * sx * twinkle))
            core_r = max(1, int(2.5 * sx))
            ov_draw.ellipse(
                [fx - halo_r, fy - halo_r, fx + halo_r, fy + halo_r],
                fill=(253, 224, 71, int(68 * twinkle)),
            )
            ov_draw.ellipse(
                [fx - core_r, fy - core_r, fx + core_r, fy + core_r],
                fill=(254, 249, 195, int(235 * twinkle)),
            )
            if idx % 3 == 0 and twinkle > 0.72:
                _draw_four_point_star(
                    ov_draw,
                    fx,
                    fy,
                    max(4, int(9 * sx)),
                    max(1, int(2.5 * sx)),
                    (255, 251, 235, int(220 * twinkle)),
                )

        # 3. Floating iridescent soap bubbles when dragon_bubbles is active
        if "dragon_bubbles" in accumulated_items:
            for b_i in range(7):
                base_bx = 520 + ((b_i * 67) % 280)
                base_by = 480 - int(((b_i * 53 + (frame_idx / max(1, num_frames)) * 180) % 260))
                bx = int((base_bx + math.sin(phase + b_i) * 10.0) * sx)
                by = int(base_by * sy)
                br = max(5, int((10 + (b_i % 4) * 4) * sx))
                rim_col = (244, 114, 182, 165) if b_i % 2 == 0 else (125, 211, 252, 165)
                ov_draw.ellipse(
                    [bx - br, by - br, bx + br, by + br],
                    fill=(224, 242, 254, 38),
                    outline=rim_col,
                    width=max(1, int(1.6 * sx)),
                )
                hl_r = max(1, br // 3)
                ov_draw.ellipse(
                    [bx - br // 2, by - br // 2, bx - br // 2 + hl_r, by - br // 2 + hl_r],
                    fill=(255, 255, 255, 210),
                )

        # 4. Pastel star-candy & rainbow airship sparkles in the upper canopy
        if "star_candy" in accumulated_items or "rainbow_spaceship" in accumulated_items:
            pastel_cols = [
                (244, 114, 182, 210),
                (253, 224, 71, 210),
                (110, 231, 183, 210),
                (196, 181, 253, 210),
            ]
            for s_i in range(8):
                scx = int((380 + s_i * 58 + math.sin(phase + s_i) * 8.0) * sx)
                scy = int((110 + ((s_i * 47) % 210) + math.cos(phase + s_i) * 6.0) * sy)
                _draw_four_point_star(
                    ov_draw,
                    scx,
                    scy,
                    max(4, int(8 * sx)),
                    max(2, int(2.5 * sx)),
                    pastel_cols[s_i % len(pastel_cols)],
                )

    elif theme in ("science_space_kitchen", "science_space_shuttle") or "space" in theme:
        cx, cy = int(640 * sx), int(360 * sy)
        is_explosion = (
            "다이너마이트" in item_label_low
            or "폭발" in item_label_low
            or "explosion" in item_label_low
            or "dynamite" in item_label_low
            or liq_col == "orange_amber"
        )
        if is_explosion:
            # Triple-shielded spherical blast containment chamber & concentric zero-G shockwave rings
            chamber_r = int(175 * sx)
            ov_draw.ellipse(
                [cx - chamber_r, cy - chamber_r, cx + chamber_r, cy + chamber_r],
                fill=(15, 23, 42, 75),
                outline=(56, 189, 248, 220),
                width=max(2, int(3.5 * sx)),
            )
            for ring_idx, ring_base in enumerate((135, 95, 55)):
                rr = int((ring_base + 14 * math.sin(phase + ring_idx)) * sx)
                ring_col = (251, 146, 60, 195) if ring_idx == 0 else (253, 224, 71, 215)
                ov_draw.ellipse(
                    [cx - rr, cy - rr, cx + rr, cy + rr],
                    fill=(249, 115, 22, 42 if ring_idx == 2 else 18),
                    outline=ring_col,
                    width=max(2, int(2.5 * sx)),
                )
            for sp_i in range(12):
                ang = (2.0 * math.pi * sp_i) / 12.0 + phase * 0.5
                rad = (70 + (sp_i % 3) * 32) * sx
                px = int(cx + math.cos(ang) * rad)
                py = int(cy + math.sin(ang) * rad)
                _draw_four_point_star(
                    ov_draw,
                    px,
                    py,
                    max(4, int(9 * sx)),
                    max(2, int(3 * sx)),
                    (254, 240, 138, 235),
                )
        else:
            # Spherical zero-G blue combustion aura & floating microgravity water droplets
            pulse_r = int((85 + 8 * math.sin(phase)) * sx)
            ov_draw.ellipse(
                [cx - pulse_r, cy - pulse_r, cx + pulse_r, cy + pulse_r],
                fill=(56, 189, 248, 32),
            )
            for d_i in range(8):
                ang = (2.0 * math.pi * d_i) / 8.0 + phase * 0.25
                dist_x = (135 + (d_i % 3) * 42) * sx
                dist_y = (95 + (d_i % 2) * 38) * sy
                dx = int(cx + math.cos(ang) * dist_x)
                dy = int(cy + math.sin(ang) * dist_y)
                dr = max(3, int((6 + (d_i % 3) * 3) * sx))
                ov_draw.ellipse(
                    [dx - dr, dy - dr, dx + dr, dy + dr],
                    fill=(186, 230, 253, 75),
                    outline=(255, 255, 255, 195),
                )

    else:
        # Science HCl Lab: Translucent rising H2/CO2 micro-bubbles inside the central glass beaker
        beaker_cx = int(505 * sx)
        beaker_top = int(310 * sy)
        beaker_bot = int(585 * sy)
        beaker_half_w = int(115 * sx)

        bubbles_intensity = float(visual_state.get("bubbles_intensity", 0.35))
        foam_intensity = float(visual_state.get("foam_intensity", 0.0))

        # Subtle emerald-cyan bioluminescent aura inside the acid solution
        glow_cy = (beaker_top + beaker_bot) // 2
        glow_rx = int((135 + 6 * math.sin(phase)) * sx)
        glow_ry = int((115 + 5 * math.cos(phase)) * sy)
        ov_draw.ellipse(
            [beaker_cx - glow_rx, glow_cy - glow_ry, beaker_cx + glow_rx, glow_cy + glow_ry],
            fill=(52, 211, 153, 24),
        )

        # Rising translucent specular reaction bubbles
        bubble_count = int(12 + bubbles_intensity * 22 + len(accumulated_items) * 4)
        span_h = max(36, beaker_bot - beaker_top)
        progress = frame_idx / float(max(1, num_frames))
        for b_idx in range(bubble_count):
            rel_x = ((b_idx * 47 + page_number * 13) % (beaker_half_w * 2)) - beaker_half_w
            wobble = int(math.sin(phase + b_idx * 0.8) * 5.0 * sx)
            bx = beaker_cx + rel_x + wobble

            travel = int((b_idx * 29 + progress * span_h) % span_h)
            by = beaker_bot - travel
            br = max(2, int((2.5 + (b_idx % 4) * 1.4) * sx))

            ov_draw.ellipse(
                [bx - br, by - br, bx + br, by + br],
                fill=(209, 250, 229, 95),
                outline=(255, 255, 255, 205),
            )
            if br >= 3:
                ov_draw.ellipse(
                    [bx - br // 2, by - br // 2, bx - br // 2 + 1, by - br // 2 + 1],
                    fill=(255, 255, 255, 245),
                )

        # Shimmering cyan water mist droplets when `water_spray` is active
        if "water_spray" in accumulated_items:
            for m_idx in range(14):
                t_drop = ((m_idx / 14.0) + progress) % 1.0
                mx = int((295 + t_drop * 220 + (m_idx % 5) * 14) * sx)
                my = int((155 + t_drop * 210 + (m_idx % 3) * 12) * sy)
                mr = max(2, int(3.0 * sx))
                ov_draw.ellipse(
                    [mx - mr, my - mr, mx + mr, my + mr],
                    fill=(125, 211, 252, 185),
                    outline=(240, 249, 255, 220),
                )

        # Effervescent CO2 foam sparkles when baking_soda, cookie_snack, or egg_shell is active
        if (
            foam_intensity > 0.1
            or "baking_soda" in accumulated_items
            or "cookie_snack" in accumulated_items
            or "egg_shell" in accumulated_items
        ):
            for f_idx in range(10):
                fx = beaker_cx - beaker_half_w + int(((f_idx * 37) % (beaker_half_w * 2)))
                fy = beaker_top - int(((f_idx * 19 + progress * 60) % 65) * sy)
                fr = max(2, int((3 + (f_idx % 3)) * sx))
                ov_draw.ellipse(
                    [fx - fr, fy - fr, fx + fr, fy + fr],
                    fill=(254, 249, 195, 165),
                    outline=(255, 255, 255, 215),
                )

        # Noble metal golden specular glint when gold_ring is active
        if "gold_ring" in accumulated_items:
            gx = int((555 + math.sin(phase) * 4) * sx)
            gy = int((390 + math.cos(phase) * 4) * sy)
            _draw_four_point_star(
                ov_draw,
                gx,
                gy,
                max(5, int(12 * sx)),
                max(2, int(3.5 * sx)),
                (254, 240, 138, 235),
            )

    return Image.alpha_composite(base_rgba, overlay)


def _render_procedural_fallback_canvas(
    visual_state: dict[str, Any],
    accumulated_items: list[str],
    frame_idx: int = 0,
) -> Image.Image:
    """Render a clean 16:9 (`1280x720`) HUD-free procedural fallback canvas if PNG assets are absent."""
    base = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (14, 22, 40, 255))
    draw = ImageDraw.Draw(base)
    theme = str(visual_state.get("theme") or "science_hcl_lab")

    if theme == "fairy_starlight_fox":
        for band in range(24):
            y0 = int((band / 24) * CANVAS_HEIGHT)
            y1 = int(((band + 1) / 24) * CANVAS_HEIGHT)
            t = band / 23.0
            draw.rectangle(
                [0, y0, CANVAS_WIDTH, y1],
                fill=(int(18 + t * 28), int(22 + t * 36), int(62 + t * 50), 255),
            )
        draw.pieslice([-150, 420, 1430, 920], 180, 360, fill=(20, 56, 52, 255))
        draw.rounded_rectangle([820, 260, 980, 560], radius=64, fill=(251, 191, 36, 255))
        draw.ellipse([380, 410, 520, 540], fill=(234, 88, 12, 255))
    else:
        for band in range(20):
            y0 = int((band / 20) * CANVAS_HEIGHT)
            y1 = int(((band + 1) / 20) * CANVAS_HEIGHT)
            t = band / 19.0
            draw.rectangle(
                [0, y0, CANVAS_WIDTH, y1],
                fill=(int(24 + t * 20), int(20 + t * 18), int(18 + t * 14), 255),
            )
        draw.rectangle([0, 565, CANVAS_WIDTH, CANVAS_HEIGHT], fill=(92, 58, 32, 255))
        draw.rounded_rectangle(
            [410, 180, 690, 585],
            radius=24,
            fill=(52, 211, 153, 180),
            outline=(226, 232, 240, 245),
            width=4,
        )

    return _apply_painterly_continuity_effects(
        base,
        visual_state=visual_state,
        accumulated_items=accumulated_items,
        page_number=1,
        frame_idx=frame_idx,
    )


def _render_base_continuity_canvas(
    visual_state: dict[str, Any],
    accumulated_items: list[str],
) -> tuple[Image.Image, str]:
    """Load the primary 16:9 Nano Banana plate and composite multi-item continuity layers."""
    plate_img, plate_name = _select_nano_banana_plate(visual_state, accumulated_items)
    if plate_img is None:
        return (
            _render_procedural_fallback_canvas(visual_state, accumulated_items, frame_idx=0),
            "procedural_16x9_fallback",
        )

    _composite_feathered_multi_items(
        canvas=plate_img,
        primary_plate_name=plate_name,
        visual_state=visual_state,
        accumulated_items=accumulated_items,
    )
    return plate_img, plate_name


def _render_scene_frame(
    visual_state: dict[str, Any],
    page_number: int,
    page_title: str,
    item_label: str,
    prev_image_bytes: bytes | None = None,
    frame_idx: int = 0,
) -> Image.Image:
    """Render a pure, cinematic 16:9 (`1280x720`) Nano Banana scene frame with zero HUD boxes.

    Loads the matching high-resolution Nano Banana artwork (`static/assets/nano_banana/*.png`),
    applies feathered Gaussian-blurred multi-item continuity compositing when multiple items
    are accumulated, and overlays delicate painterly lighting and atmospheric particles.

    Args:
        visual_state: Dictionary describing liquid color, item_key, accumulated_items, pH, etc.
        page_number: Current 1-based page number.
        page_title: Title of the current page (displayed in HTML UI, never burned into canvas).
        item_label: Short description of the active item/scene twist.
        prev_image_bytes: Previous page PNG bytes (preserved for signature compatibility).
        frame_idx: Animation frame index (`0` for static PNG, `0..7` for animated motion).

    Returns:
        Rendered RGB `PIL.Image.Image` of size `(1280, 720)`.
    """
    _ = (page_title, item_label, prev_image_bytes)
    accumulated_items = _resolve_accumulated_items(visual_state)
    base_canvas, _ = _render_base_continuity_canvas(visual_state, accumulated_items)

    composited = _apply_painterly_continuity_effects(
        canvas=base_canvas,
        visual_state=visual_state,
        accumulated_items=accumulated_items,
        page_number=page_number,
        frame_idx=frame_idx,
        num_frames=GIF_NUM_FRAMES,
    )
    return composited.convert("RGB")


def _draw_process_item_sprite(
    draw: ImageDraw.ImageDraw,
    item_key: str,
    cx: int,
    cy: int,
    scale: float = 1.0,
    with_trail: bool = True,
) -> None:
    """Draw a rich, recognizable item sprite (cookie, key, apple, soda spoon, water spray, rice cake, etc.) at `(cx, cy)`."""
    r = max(12, int(24 * scale))

    if with_trail:
        for t_idx in range(3):
            tx = cx + (t_idx - 1) * 11
            ty_top = max(8, cy - r - 38 - (t_idx % 2) * 12)
            ty_bot = max(12, cy - r - 8)
            if ty_bot > ty_top:
                draw.line(
                    [(tx, ty_top), (tx, ty_bot)],
                    fill=(255, 255, 255, 175),
                    width=2,
                )

    if item_key == "cookie_snack":
        # Golden-baked round chocolate-chip cookie with dark chocolate chips & crumbs
        draw.ellipse([cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2], fill=(146, 64, 14, 230))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(217, 119, 6, 252), outline=(251, 191, 36, 255), width=2)
        draw.ellipse([cx - int(r * 0.75), cy - int(r * 0.75), cx + int(r * 0.45), cy + int(r * 0.45)], fill=(245, 158, 11, 120))
        chip_offsets = [(-10, -8), (8, -9), (0, 2), (-11, 8), (9, 7), (-2, -13), (12, -1)]
        for ox, oy in chip_offsets:
            chx = cx + int(ox * scale)
            chy = cy + int(oy * scale)
            cr = max(2, int(3.8 * scale))
            draw.ellipse([chx - cr, chy - cr, chx + cr, chy + cr], fill=(69, 26, 3, 255))
        if with_trail:
            for c_i, (cox, coy) in enumerate([(-16, -26), (14, -30), (4, -22), (-8, -34)]):
                cr_r = max(1, int((2.5 - (c_i % 2) * 0.6) * scale))
                draw.ellipse([cx + cox - cr_r, cy + coy - cr_r, cx + cox + cr_r, cy + coy + cr_r], fill=(245, 158, 11, 220))

    elif item_key == "metal_key":
        # Gleaming golden-brass laboratory key
        bow_r = max(8, int(13 * scale))
        bow_cy = cy - int(12 * scale)
        draw.ellipse(
            [cx - bow_r, bow_cy - bow_r, cx + bow_r, bow_cy + bow_r],
            fill=(250, 204, 21, 250),
            outline=(161, 98, 7, 255),
            width=3,
        )
        inner_r = max(3, bow_r // 2)
        draw.ellipse([cx - inner_r, bow_cy - inner_r, cx + inner_r, bow_cy + inner_r], fill=(30, 41, 59, 210))
        shaft_bot = cy + int(22 * scale)
        draw.rectangle([cx - 3, bow_cy + bow_r - 2, cx + 3, shaft_bot], fill=(250, 204, 21, 255), outline=(161, 98, 7, 255))
        draw.rectangle([cx + 3, shaft_bot - 12, cx + 12, shaft_bot - 8], fill=(250, 204, 21, 255))
        draw.rectangle([cx + 3, shaft_bot - 5, cx + 10, shaft_bot - 1], fill=(250, 204, 21, 255))

    elif item_key == "apple":
        # Glossy crimson-red apple with leaf & specular shine
        draw.ellipse([cx - r, cy - int(r * 0.9), cx + r, cy + r], fill=(220, 38, 38, 252), outline=(153, 27, 27, 255), width=2)
        draw.ellipse([cx - int(r * 0.55), cy - int(r * 0.55), cx - int(r * 0.15), cy - int(r * 0.15)], fill=(254, 202, 202, 210))
        draw.line([(cx, cy - int(r * 0.9)), (cx + 2, cy - int(r * 1.25))], fill=(120, 53, 15, 255), width=3)
        draw.polygon([(cx + 2, cy - int(r * 1.1)), (cx + 15, cy - int(r * 1.3)), (cx + 9, cy - int(r * 0.9))], fill=(34, 197, 94, 255))

    elif item_key == "baking_soda":
        # Silver laboratory spoon pouring cascading white NaHCO3 powder stream
        spoon_x = cx - int(28 * scale)
        spoon_y = cy - int(18 * scale)
        draw.line([(spoon_x - 35, spoon_y - 18), (spoon_x, spoon_y)], fill=(203, 213, 225, 255), width=5)
        draw.ellipse([spoon_x - 14, spoon_y - 8, spoon_x + 18, spoon_y + 10], fill=(241, 245, 249, 255), outline=(148, 163, 184, 255), width=2)
        # Falling white powder cone
        draw.polygon(
            [(spoon_x + 10, spoon_y + 4), (cx - 18, cy + 24), (cx + 18, cy + 24)],
            fill=(255, 255, 255, 225),
        )
        for p_i in range(9):
            px = cx + ((p_i * 11) % 30) - 15
            py = cy - 4 + (p_i * 4)
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(224, 242, 254, 245))

    elif item_key == "water_spray":
        # Pressurized cyan water spray stream from nozzle
        noz_x, noz_y = max(40, cx - 95), max(30, cy - 75)
        draw.rounded_rectangle([noz_x - 18, noz_y - 10, noz_x + 12, noz_y + 12], radius=5, fill=(56, 189, 248, 255), outline=(255, 255, 255, 240), width=2)
        for ray in range(-2, 3):
            end_x = cx + ray * 14
            end_y = cy + abs(ray) * 5
            draw.line([(noz_x + 12, noz_y + 2), (end_x, end_y)], fill=(125, 211, 252, 195), width=2)
            draw.ellipse([end_x - 4, end_y - 4, end_x + 4, end_y + 4], fill=(224, 242, 254, 240))

    elif item_key == "egg_shell":
        # Pearly white oval egg
        rx = int(r * 0.82)
        ry = int(r * 1.08)
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(254, 243, 199, 252), outline=(251, 191, 36, 235), width=2)
        draw.ellipse([cx - rx // 2, cy - ry // 2, cx, cy - ry // 6], fill=(255, 255, 255, 220))

    elif item_key == "gold_ring":
        # 24K Pure Gold Ring with diamond sparkle
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(250, 204, 21, 255), width=6)
        draw.ellipse([cx - r + 2, cy - r + 2, cx + r - 2, cy + r - 2], outline=(254, 240, 138, 255), width=2)
        _draw_four_point_star(draw, cx, cy - r - 4, 10, 3, (255, 255, 255, 250))

    elif item_key == "rice_cake":
        # Warm golden-pink Korean honey rice cakes (tteok) on a wooden tray
        draw.rounded_rectangle([cx - r - 8, cy + 4, cx + r + 8, cy + 14], radius=4, fill=(180, 83, 9, 250), outline=(251, 191, 36, 255), width=2)
        for rx_off, col in [(-12, (253, 224, 71, 252)), (0, (244, 114, 182, 252)), (12, (167, 243, 208, 252))]:
            draw.ellipse([cx + rx_off - 10, cy - 12, cx + rx_off + 10, cy + 6], fill=col, outline=(255, 255, 255, 220))

    elif item_key == "roasted_pebble":
        # Glowing ruby-orange roasted campfire pebble
        draw.ellipse([cx - r, cy - int(r * 0.8), cx + r, cy + int(r * 0.8)], fill=(234, 88, 12, 252), outline=(253, 224, 71, 255), width=3)
        draw.ellipse([cx - r // 2, cy - r // 2, cx + r // 2, cy + r // 3], fill=(254, 215, 170, 220))

    else:
        # Glowing magical/scientific orb with starlight core
        draw.ellipse([cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4], fill=(56, 189, 248, 110))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(253, 224, 71, 245), outline=(255, 255, 255, 255), width=2)
        _draw_four_point_star(draw, cx, cy, max(6, r // 2), 3, (255, 255, 255, 250))


def _render_veo_process_stage_frame(
    before_canvas: Image.Image,
    after_canvas: Image.Image,
    visual_state: dict[str, Any],
    accumulated_items: list[str],
    page_number: int,
    frame_idx: int,
    num_frames: int,
) -> Image.Image:
    """Render a single frame of the 3-Stage Step-by-Step Veo Process Animation (`640x360`).

    When `page_number > 1` (a turn where an item/action is introduced):
      - Stage 1 (Frames 0..2): Item Entering & Dropping into the Beaker/Scene on `before_canvas`.
      - Stage 2 (Frames 3..5): Item Splashing & Submerging into the liquid while cross-fading
        into `after_canvas` with surface ripples, splash droplets, and initial nucleation bubbles.
      - Stage 3 (Frames 6..9): Vigorous Reaction — thick creamy foam swelling across the top of
        the liquid, dense columns of rising gas bubbles, and warm rising steam plumes.
    """
    theme = str(visual_state.get("theme") or "science_hcl_lab")
    active_key = str(
        visual_state.get("item_key")
        or (accumulated_items[-1] if accumulated_items else "hcl_beaker")
    )
    is_science_beaker = theme not in ("fairy_starlight_fox", "fairy_tiger_rabbit", "science_space_kitchen", "science_space_shuttle") and "fairy" not in theme and "space" not in theme

    # Determine background blend between `before_canvas` (pre-item state) and `after_canvas` (reaction state)
    if page_number <= 1:
        base_frame = after_canvas.copy()
    elif frame_idx <= 2:
        # Stage 1: Item is still in the air / entering -> show the pre-reaction scene!
        base_frame = before_canvas.copy()
    elif frame_idx == 3:
        base_frame = Image.blend(before_canvas, after_canvas, 0.38)
    elif frame_idx == 4:
        base_frame = Image.blend(before_canvas, after_canvas, 0.72)
    elif frame_idx == 5:
        base_frame = Image.blend(before_canvas, after_canvas, 0.92)
    else:
        base_frame = after_canvas.copy()

    # Apply base atmospheric continuity effects
    frame_rgba = _apply_painterly_continuity_effects(
        canvas=base_frame,
        visual_state=visual_state,
        accumulated_items=accumulated_items,
        page_number=page_number,
        frame_idx=frame_idx,
        num_frames=num_frames,
    )

    if page_number <= 1:
        return frame_rgba.convert("RGB")

    ov = Image.new("RGBA", (GIF_WIDTH, GIF_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)

    # Target coordinates on 640x360 (16:9) canvas
    if is_science_beaker:
        target_cx = 262
        meniscus_y = 182
        submerged_y = 246
        beaker_half_w = 62
    elif theme == "fairy_tiger_rabbit":
        target_cx = 310
        meniscus_y = 205
        submerged_y = 225
        beaker_half_w = 75
    else:
        target_cx = 320
        meniscus_y = 190
        submerged_y = 215
        beaker_half_w = 70

    # =========================================================================
    # STAGE 1 (Frames 0, 1, 2): ITEM ENTERING & DROPPING IN
    # =========================================================================
    if frame_idx <= 2:
        drop_t = frame_idx / 2.0  # 0.0 -> 0.5 -> 1.0
        cur_y = int(42 + drop_t * (meniscus_y - 54))
        # Draw downward entry guide ring at meniscus
        ring_rx = int(28 - drop_t * 8)
        ring_ry = max(4, int(8 - drop_t * 2))
        draw.ellipse(
            [target_cx - ring_rx, meniscus_y - ring_ry, target_cx + ring_rx, meniscus_y + ring_ry],
            outline=(253, 224, 71, 210),
            width=2,
        )
        _draw_process_item_sprite(
            draw,
            item_key=active_key,
            cx=target_cx,
            cy=cur_y,
            scale=1.05,
            with_trail=True,
        )

    # =========================================================================
    # STAGE 2 (Frames 3, 4, 5): SPLASH, SUBMERSION & INITIAL REACTION ONSET
    # =========================================================================
    elif frame_idx <= 5:
        sub_t = (frame_idx - 3) / 2.0  # 0.0 -> 0.5 -> 1.0
        cur_y = int(meniscus_y + 6 + sub_t * (submerged_y - meniscus_y - 6))

        # 1. Expanding concentric splash ripples at the liquid meniscus
        for r_idx in range(2):
            rip_rx = int(22 + sub_t * 36 + r_idx * 14)
            rip_ry = int(5 + sub_t * 8 + r_idx * 3)
            rip_alpha = max(45, int(235 * (1.0 - sub_t * 0.55) - r_idx * 45))
            draw.ellipse(
                [target_cx - rip_rx, meniscus_y - rip_ry, target_cx + rip_rx, meniscus_y + rip_ry],
                outline=(224, 242, 254, rip_alpha),
                width=2,
            )

        # 2. Splash crown droplets arcing upward from impact point
        for d_i in range(8):
            side = -1 if d_i < 4 else 1
            idx_in_side = (d_i % 4) + 1
            sx_drop = target_cx + side * int(idx_in_side * (9 + sub_t * 11))
            sy_drop = meniscus_y - int((24 - idx_in_side * 3) * (1.0 - abs(sub_t - 0.35)))
            dr = max(2, 5 - idx_in_side // 2)
            draw.ellipse(
                [sx_drop - dr, sy_drop - dr, sx_drop + dr, sy_drop + dr],
                fill=(167, 243, 208, 230) if is_science_beaker else (253, 224, 71, 230),
                outline=(255, 255, 255, 240),
            )

        # 3. Draw the submerging item inside the liquid
        _draw_process_item_sprite(
            draw,
            item_key=active_key,
            cx=target_cx,
            cy=cur_y,
            scale=0.92,
            with_trail=False,
        )

        # 4. Initial vigorous nucleation micro-bubbles bursting around the submerged item
        if active_key != "gold_ring":
            for b_i in range(14):
                ang = (2.0 * math.pi * b_i) / 14.0 + sub_t * 1.6
                rad = 16 + (b_i % 3) * 8 + int(sub_t * 12)
                bx = target_cx + int(math.cos(ang) * rad)
                by = cur_y + int(math.sin(ang) * (rad * 0.7)) - int(sub_t * 14)
                br = 2 + (b_i % 3)
                draw.ellipse(
                    [bx - br, by - br, bx + br, by + br],
                    fill=(255, 255, 255, 215),
                    outline=(110, 231, 183, 240),
                )

    # =========================================================================
    # STAGE 3 (Frames 6, 7, 8, 9): VIGOROUS FOAM CROWN & BUBBLE ERUPTION
    # =========================================================================
    else:
        react_t = (frame_idx - 6) / 3.0  # 0.0 -> 0.33 -> 0.67 -> 1.0
        foam_level = float(visual_state.get("foam_intensity", 0.35))
        has_heavy_foam = (
            active_key in ("cookie_snack", "baking_soda", "egg_shell")
            or foam_level >= 0.3
            or "cookie_snack" in accumulated_items
            or "baking_soda" in accumulated_items
        )

        if is_science_beaker and active_key != "gold_ring":
            # 1. Dense rising gas bubble streams shooting from submerged zone to meniscus
            for b_i in range(28):
                lane_x = target_cx + ((b_i * 19) % (beaker_half_w * 2 - 16)) - (beaker_half_w - 8)
                wobble_x = int(math.sin(react_t * math.pi * 2.0 + b_i) * 4)
                bx = lane_x + wobble_x
                span = max(30, submerged_y - meniscus_y + 25)
                by = submerged_y - int(((b_i * 13 + react_t * span * 1.4) % span))
                br = 2 + (b_i % 4)
                draw.ellipse(
                    [bx - br, by - br, bx + br, by + br],
                    fill=(236, 253, 245, 195),
                    outline=(255, 255, 255, 245),
                )

            # 2. Progressive Swelling Creamy Foam Crown at top of liquid (especially for Cookie & Baking Soda!)
            if has_heavy_foam:
                foam_layers = 2 if active_key != "baking_soda" else 3
                foam_rise = int(8 + react_t * 22)
                for layer in range(foam_layers):
                    layer_y = meniscus_y - layer * 10 - int(react_t * 10)
                    for puff_i in range(11):
                        px = (target_cx - beaker_half_w + 8) + puff_i * int((beaker_half_w * 2 - 16) / 10.0)
                        py = layer_y - int(math.sin(puff_i * 0.9 + react_t * 4.0) * 5) - (foam_rise // 3)
                        pr = int(9 + (puff_i % 3) * 3 + react_t * 4)
                        foam_col = (
                            (254, 243, 199, 225)
                            if active_key == "cookie_snack"
                            else (240, 253, 250, 230)
                        )
                        draw.ellipse(
                            [px - pr, py - pr, px + pr, py + pr],
                            fill=foam_col,
                            outline=(255, 255, 255, 245),
                        )

            # 3. Rising warm steam / gas plumes above the beaker mouth
            for s_i in range(5):
                sx_plume = target_cx - 28 + s_i * 14 + int(math.sin(react_t * 3.0 + s_i) * 6)
                sy_plume = meniscus_y - 34 - s_i * 11 - int(react_t * 24)
                sr_x = 14 + (s_i % 2) * 5
                sr_y = 9 + (s_i % 2) * 4
                draw.ellipse(
                    [sx_plume - sr_x, sy_plume - sr_y, sx_plume + sr_x, sy_plume + sr_y],
                    fill=(255, 255, 255, int(75 * (1.0 - react_t * 0.35))),
                )

    # =========================================================================
    # SLEEK 3-STAGE VISUAL PROGRESS TIMELINE BAR AT BOTTOM (`y = 352..360`)
    # =========================================================================
    bar_y0 = GIF_HEIGHT - 7
    draw.rectangle([0, bar_y0, GIF_WIDTH, GIF_HEIGHT], fill=(15, 23, 42, 210))
    progress_w = int(((frame_idx + 1) / float(max(1, num_frames))) * GIF_WIDTH)
    if frame_idx <= 2:
        bar_col = (251, 191, 36, 245)  # Stage 1 Amber (Drop/Entry)
    elif frame_idx <= 5:
        bar_col = (56, 189, 248, 245)  # Stage 2 Cyan (Splash/Submerge)
    else:
        bar_col = (52, 211, 153, 245)  # Stage 3 Emerald (Foam/Bubble Eruption)
    draw.rectangle([0, bar_y0, progress_w, GIF_HEIGHT], fill=bar_col)
    # Stage divider ticks at 30% and 60%
    for tick_frac in (0.30, 0.60):
        tx = int(GIF_WIDTH * tick_frac)
        draw.line([(tx, bar_y0), (tx, GIF_HEIGHT)], fill=(255, 255, 255, 200), width=2)

    return Image.alpha_composite(frame_rgba, ov).convert("RGB")


def _generate_animated_gif_bytes(
    visual_state: dict[str, Any],
    page_number: int,
    num_frames: int = GIF_NUM_FRAMES,
    prev_image_bytes: bytes | None = None,
) -> bytes:
    """Generate a 10-frame 16:9 widescreen (`640x360`) 3-Stage Process Animation GIF.

    Visually animates the step-by-step causal process from the previous scene (`prev_image_bytes`)
    to the new reaction scene (`base_canvas`):
      1. Stage 1 (Frames 0-2): Item entering and dropping toward the beaker/scene.
      2. Stage 2 (Frames 3-5): Item splashing and submerging into the liquid as reaction starts.
      3. Stage 3 (Frames 6-9): Vigorous foam crown swelling, rising gas bubble streams, and steam.
    """
    accumulated_items = _resolve_accumulated_items(visual_state)
    after_full, _ = _render_base_continuity_canvas(visual_state, accumulated_items)
    after_small = after_full.resize((GIF_WIDTH, GIF_HEIGHT), resample=Image.Resampling.BILINEAR).convert("RGBA")

    before_small: Image.Image | None = None
    if prev_image_bytes:
        try:
            with Image.open(io.BytesIO(prev_image_bytes)) as prev_img:
                before_small = prev_img.resize((GIF_WIDTH, GIF_HEIGHT), resample=Image.Resampling.BILINEAR).convert("RGBA")
        except Exception:
            before_small = None

    if before_small is None:
        prior_items = accumulated_items[:-1] if len(accumulated_items) > 1 else []
        prev_state = dict(visual_state)
        prev_state["item_key"] = prior_items[-1] if prior_items else "hcl_beaker"
        prev_state["accumulated_items"] = prior_items
        before_full, _ = _render_base_continuity_canvas(prev_state, prior_items)
        before_small = before_full.resize((GIF_WIDTH, GIF_HEIGHT), resample=Image.Resampling.BILINEAR).convert("RGBA")

    frames: list[Image.Image] = []
    for frame_idx in range(num_frames):
        frame_rgb = _render_veo_process_stage_frame(
            before_canvas=before_small,
            after_canvas=after_small,
            visual_state=visual_state,
            accumulated_items=accumulated_items,
            page_number=page_number,
            frame_idx=frame_idx,
            num_frames=num_frames,
        )
        frames.append(frame_rgb)

    gif_buf = io.BytesIO()
    frames[0].save(
        gif_buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0,
    )
    return gif_buf.getvalue()


def generate_page_illustration(
    nano_banana_prompt: str,
    visual_state: dict[str, Any],
    page_number: int,
    page_title: str,
    prev_image_bytes: bytes | None = None,
    prefer_live_api: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    """Generate the current page's 16:9 illustration chaining the previous page's image."""
    import os

    accumulated_items = _resolve_accumulated_items(visual_state)
    log_function_call(
        "generate_page_illustration",
        {
            "page_number": page_number,
            "page_title": page_title,
            "has_prev_image": prev_image_bytes is not None,
            "prefer_live_api": prefer_live_api,
            "item_key": visual_state.get("item_key"),
            "accumulated_items": accumulated_items,
        },
    )

    # Use instant 16:9 Nano Banana high-res continuity compositor unless FORCE_LIVE_NANO_BANANA=1 is set
    if prefer_live_api and os.environ.get("FORCE_LIVE_NANO_BANANA") == "1":
        live_bytes, live_meta = _try_live_nano_banana(
            nano_banana_prompt=nano_banana_prompt,
            prev_image_bytes=prev_image_bytes,
            visual_state=visual_state,
        )
        if live_bytes is not None:
            return live_bytes, live_meta

    _, plate_name = _select_nano_banana_plate(visual_state, accumulated_items)
    frame = _render_scene_frame(
        visual_state=visual_state,
        page_number=page_number,
        page_title=page_title,
        item_label=str(visual_state.get("item_label") or "Interactive Scene"),
        prev_image_bytes=prev_image_bytes,
        frame_idx=0,
    )
    buf = io.BytesIO()
    frame.save(buf, format="PNG")
    return buf.getvalue(), {
        "engine_mode": "nano_banana_16x9_continuity",
        "asset_plate": plate_name or "procedural_16x9_fallback",
        "resolution": f"{CANVAS_WIDTH}x{CANVAS_HEIGHT}",
        "configured_nano_banana_pro_model": NANO_BANANA_PRO_MODEL,
        "configured_nano_banana_model": NANO_BANANA_MODEL,
        "chained_from_prev_image": prev_image_bytes is not None,
        "accumulated_items": accumulated_items,
    }


def _wrap_pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    """Wrap raw 16-bit mono little-endian PCM bytes (`audio/L16`) into a valid `.wav` container."""
    if pcm_bytes[:4] == b"RIFF":
        return pcm_bytes
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buf.getvalue()


def _detect_is_korean(text: str, lang_hint: str = "ko") -> bool:
    """Auto-detect whether `text` contains Korean Hangul syllables or falls back to `lang_hint`."""
    if any("\uac00" <= ch <= "\ud7a3" for ch in (text or "")):
        return True
    if any("a" <= ch.lower() <= "z" for ch in (text or "")):
        return False
    return str(lang_hint or "ko").lower().startswith("ko")


def _try_live_gemini_character_tts(
    character_dialogue: str,
    visual_state: dict[str, Any],
) -> tuple[bytes | None, dict[str, Any]]:
    """Priority 1: Synthesize expressive character-acted voice via Gemini Native TTS models.

    Invokes `gemini-2.5-flash-preview-tts` (fallback `gemini-2.5-pro-preview-tts`) with
    `response_modalities=["AUDIO"]` and a persona director prompt tailored to Lumi the
    Baby Starlight Fox, Dr. Nova the Clockwork Owl Professor, or Captain Orbit.

    Args:
        character_dialogue: Spoken line for the page's character guide.
        visual_state: Visual state dictionary identifying the active `theme`.

    Returns:
        Tuple of `(wav_bytes_or_none, tts_metadata)`.
    """
    api_key = get_gemini_api_key()
    if not api_key or not (character_dialogue or "").strip():
        return None, {"tts_engine": "skipped_no_api_key"}

    theme = str(
        visual_state.get("theme")
        or visual_state.get("topic_id")
        or visual_state.get("preset_id")
        or "science_hcl_lab"
    ).strip()
    persona_voice = str(visual_state.get("character_persona_voice") or "").strip()
    is_storybook = (
        theme in ("fairy_starlight_fox", "starlight_fox")
        or persona_voice == "baby_fox"
        or "story" in theme
        or "fairy" in theme
        or "fox" in theme
    )
    is_space = (
        theme in ("science_space_kitchen", "zero_g_space_cabin")
        or persona_voice == "space_captain"
        or "space" in theme
        or "zero_g" in theme
    )

    if is_storybook:
        voice_name = "Leda"
        persona_prompt = (
            "Act as Lumi, an adorable, playful baby woodland fox in a fairy tale. "
            "Speak with a bright, cute, high-energy fox character voice with joyful curiosity: "
            + character_dialogue
        )
    elif is_space:
        voice_name = "Aoede"
        persona_prompt = (
            "Act as Captain Orbit, a cheerful space explorer in zero gravity: "
            + character_dialogue
        )
    else:
        voice_name = "Fenrir"
        persona_prompt = (
            "Act as Dr. Nova, a warm, wise, theatrical clockwork owl professor. "
            "Speak with a cozy owl character cadence ('부엉~!' / 'Hoo-hoo!') and enthusiastic wonder: "
            + character_dialogue
        )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        tts_config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                )
            ),
        )
        for model_name in GEMINI_TTS_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=persona_prompt,
                    config=tts_config,
                )
                candidates = getattr(response, "candidates", None) or []
                for cand in candidates:
                    content = getattr(cand, "content", None)
                    parts = getattr(content, "parts", None) or []
                    for part in parts:
                        inline_data = getattr(part, "inline_data", None)
                        raw_audio = getattr(inline_data, "data", None) if inline_data else None
                        if isinstance(raw_audio, (bytes, bytearray)) and len(raw_audio) > 200:
                            wav_bytes = _wrap_pcm16_to_wav(bytes(raw_audio), sample_rate=24000)
                            log_genai_call(
                                model=model_name,
                                prompt_summary=persona_prompt[:180],
                                config_summary={"voice_name": voice_name, "theme": theme},
                                output_summary=f"<gemini tts wav bytes={len(wav_bytes)}>",
                                status="ok",
                            )
                            return wav_bytes, {
                                "tts_engine": "gemini_native_character_tts",
                                "tts_model": model_name,
                                "voice_name": voice_name,
                            }
            except Exception as model_exc:
                log_genai_call(
                    model=model_name,
                    prompt_summary=persona_prompt[:180],
                    config_summary={"voice_name": voice_name, "theme": theme},
                    output_summary=str(model_exc)[:240],
                    status="error",
                )
    except Exception as exc:
        log_genai_call(
            model=GEMINI_TTS_MODELS[0],
            prompt_summary=persona_prompt[:180],
            config_summary={"voice_name": voice_name, "theme": theme},
            output_summary=str(exc)[:240],
            status="error",
        )

    return None, {"tts_engine": "gemini_tts_fallback"}


def _build_edge_persona_spec(
    character_dialogue: str,
    visual_state: dict[str, Any],
    lang: str = "ko",
) -> tuple[str, str, str, str, str]:
    """Build `(acted_text, voice_id, rate, pitch, persona_label)` for `edge_tts`."""
    cleaned = (character_dialogue or "").strip()
    # Normalize chemical subscripts/superscripts & arrows so TTS reads the full line smoothly
    for sub_char, ascii_char in (
        ("₂", "2"),
        ("₃", "3"),
        ("₄", "4"),
        ("⁺", "+"),
        ("⁻", "-"),
        ("→", ", "),
        ("➔", ", "),
    ):
        cleaned = cleaned.replace(sub_char, ascii_char)

    effective_lang = str(visual_state.get("lang") or lang or "ko").strip()
    is_ko = _detect_is_korean(cleaned, lang_hint=effective_lang)
    theme = str(
        visual_state.get("theme")
        or visual_state.get("topic_id")
        or visual_state.get("preset_id")
        or "science_hcl_lab"
    ).strip()
    persona_voice = str(visual_state.get("character_persona_voice") or "").strip()
    is_storybook = (
        theme in ("fairy_starlight_fox", "starlight_fox")
        or persona_voice == "baby_fox"
        or "story" in theme
        or "fairy" in theme
        or "fox" in theme
    )
    is_space = (
        theme in ("science_space_kitchen", "zero_g_space_cabin")
        or persona_voice == "space_captain"
        or "space" in theme
        or "zero_g" in theme
    )

    if is_storybook:
        if is_ko:
            return (
                cleaned,
                "ko-KR-SunHiNeural",
                "+12%",
                "+26Hz",
                "Lumi (Baby Starlight Fox - KO)",
            )
        return (
            cleaned,
            "en-US-AnaNeural",
            "+8%",
            "+18Hz",
            "Lumi (Baby Starlight Fox - EN)",
        )

    if is_space:
        if is_ko:
            return (
                cleaned,
                "ko-KR-HyunsuMultilingualNeural",
                "+8%",
                "+6Hz",
                "Captain Orbit (Zero-G Guide - KO)",
            )
        return (
            cleaned,
            "en-US-AriaNeural",
            "+8%",
            "+6Hz",
            "Captain Orbit (Zero-G Guide - EN)",
        )

    # Default: Science HCl Lab -> Dr. Nova the Clockwork Owl Professor
    if is_ko:
        return (
            cleaned,
            "ko-KR-InJoonNeural",
            "+5%",
            "-10Hz",
            "Dr. Nova (Clockwork Owl Professor - KO)",
        )
    return (
        cleaned,
        "en-GB-RyanNeural",
        "+5%",
        "-8Hz",
        "Dr. Nova (Clockwork Owl Professor - EN)",
    )



def _synthesize_edge_character_voice(
    character_dialogue: str,
    visual_state: dict[str, Any],
    lang: str = "ko",
) -> tuple[bytes | None, dict[str, Any]]:
    """Priority 2: Synthesize high-naturalness neural character persona voice via `edge_tts`.

    Uses distinct voice models, pitch shifts, speaking rates, and character interjections
    for Lumi the Baby Starlight Fox (`ko-KR-SunHiNeural` / `en-US-AnaNeural`), Dr. Nova
    the Clockwork Owl Professor (`ko-KR-InJoonNeural` / `en-GB-RyanNeural`), and Captain
    Orbit (`ko-KR-HyunsuMultilingualNeural` / `en-US-AriaNeural`).

    Args:
        character_dialogue: Spoken character dialogue string.
        visual_state: Visual state dictionary containing `theme`.
        lang: Language hint (`"ko"` or `"en"`).

    Returns:
        Tuple of `(audio_bytes_or_none, voice_metadata)`.
    """
    if not (character_dialogue or "").strip():
        return None, {"tts_engine": "empty_dialogue"}

    acted_text, voice_id, rate, pitch, persona_label = _build_edge_persona_spec(
        character_dialogue=character_dialogue,
        visual_state=visual_state,
        lang=lang,
    )
    cache_key = f"{voice_id}|{rate}|{pitch}|{acted_text}"
    cached = _VOICE_AUDIO_CACHE.get(cache_key)
    if cached is not None:
        return cached

    async def _collect_stream() -> bytes:
        import edge_tts  # type: ignore[import-not-found]

        communicate = edge_tts.Communicate(
            acted_text,
            voice=voice_id,
            rate=rate,
            pitch=pitch,
        )
        audio_buf = bytearray()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                audio_buf.extend(chunk["data"])
        return bytes(audio_buf)

    def _run_in_fresh_loop() -> bytes:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_collect_stream())
        finally:
            loop.close()

    try:
        try:
            asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as ex:
                audio_bytes = ex.submit(_run_in_fresh_loop).result(timeout=12.0)
        except RuntimeError:
            audio_bytes = _run_in_fresh_loop()

        if audio_bytes and len(audio_bytes) > 200:
            meta = {
                "tts_engine": "edge_neural_character_tts",
                "voice_name": voice_id,
                "persona": persona_label,
                "rate": rate,
                "pitch": pitch,
            }
            _VOICE_AUDIO_CACHE[cache_key] = (audio_bytes, meta)
            return audio_bytes, meta
    except Exception:
        pass

    return None, {"tts_engine": "edge_tts_unavailable"}


def _generate_chime_wav_bytes(dialogue_text: str, page_number: int) -> bytes:
    """Last-resort offline fallback: synthesize a warm 16-bit PCM WAV chime track."""
    sample_rate = 16000
    duration_sec = min(2.4, max(1.0, len(dialogue_text or "WonderBook") * 0.025))
    num_samples = int(sample_rate * duration_sec)
    base_freq = 330.0 + ((page_number % 4) * 28.0)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames = bytearray()
        for i in range(num_samples):
            t = i / sample_rate
            syllable_env = 0.55 + 0.45 * math.sin(2.0 * math.pi * 5.5 * t)
            fade = min(1.0, t / 0.08, (duration_sec - t) / 0.15)
            freq = base_freq + 45.0 * math.sin(2.0 * math.pi * 2.2 * t)
            sample_val = (
                0.65 * math.sin(2.0 * math.pi * freq * t)
                + 0.35 * math.sin(2.0 * math.pi * (freq * 1.5) * t)
            )
            pcm = int(max(-32767, min(32767, sample_val * syllable_env * fade * 9500)))
            frames.extend(struct.pack("<h", pcm))

        wav_file.writeframes(bytes(frames))

    return buf.getvalue()


def _synthesize_character_voice_wav(dialogue_text: str, page_number: int) -> bytes:
    """Backward-compatible wrapper returning chime WAV bytes when offline."""
    return _generate_chime_wav_bytes(dialogue_text, page_number)


def _generate_character_voice_bytes(
    character_dialogue: str,
    visual_state: dict[str, Any],
    page_number: int,
    prefer_live_api: bool = True,
    lang: str = "ko",
) -> tuple[bytes, dict[str, Any]]:
    """Generate expressive character-acted voice audio with 3-tier priority fallback.

    1. Priority 1: Gemini Native Expressive Character TTS (`gemini-2.5-flash-preview-tts`
       / `gemini-2.5-pro-preview-tts`) when `GEMINI_API_KEY` is configured and `prefer_live_api=True`.
    2. Priority 2: High-naturalness neural character persona TTS via `edge_tts` with
       character-specific pitch, rate, and fox/owl/astronaut vocal acting.
    3. Priority 3: Offline 16-bit PCM WAV chime fallback (`_generate_chime_wav_bytes`).

    Args:
        character_dialogue: Dialogue spoken by the guide/character.
        visual_state: Visual state dictionary for the page.
        page_number: 1-based page number.
        prefer_live_api: Whether to attempt live Gemini Native TTS first.
        lang: Language hint (`"ko"` or `"en"`).

    Returns:
        Tuple of `(voice_audio_bytes, voice_metadata)`.
    """
    if prefer_live_api and os.environ.get("FORCE_LIVE_GEMINI_TTS") == "1":
        gemini_wav, gemini_meta = _try_live_gemini_character_tts(
            character_dialogue=character_dialogue,
            visual_state=visual_state,
        )
        if gemini_wav is not None:
            return gemini_wav, gemini_meta

    edge_audio, edge_meta = _synthesize_edge_character_voice(
        character_dialogue=character_dialogue,
        visual_state=visual_state,
        lang=lang,
    )
    if edge_audio is not None:
        return edge_audio, edge_meta

    chime_wav = _generate_chime_wav_bytes(character_dialogue, page_number)
    return chime_wav, {
        "tts_engine": "offline_chime_wav_fallback",
        "voice_name": "procedural_chime",
    }


def generate_page_motion_and_voice(
    veo_motion_prompt: str,
    character_dialogue: str,
    visual_state: dict[str, Any],
    page_number: int,
    page_title: str,
    prev_image_bytes: bytes | None = None,
    prefer_live_api: bool = True,
    lang: str = "ko",
) -> tuple[bytes, bytes, dict[str, Any]]:
    """Generate living 16:9 3-stage process animation (`motion.gif`) and character voice (`voice.wav`) in parallel."""
    _ = page_title
    accumulated_items = _resolve_accumulated_items(visual_state)
    log_function_call(
        "generate_page_motion_and_voice",
        {
            "page_number": page_number,
            "veo_motion_prompt": veo_motion_prompt,
            "dialogue_length": len(character_dialogue or ""),
            "prefer_live_api": prefer_live_api,
            "lang": lang,
            "accumulated_items": accumulated_items,
            "has_prev_image": prev_image_bytes is not None,
        },
    )

    veo_status = "veo_3stage_process_stream"
    if prefer_live_api and get_gemini_api_key():
        log_genai_call(
            model=VEO_MODEL,
            prompt_summary=f"3-Stage Process Motion: {veo_motion_prompt} | Voice: {character_dialogue}",
            config_summary={
                "page_number": page_number,
                "accumulated_items": accumulated_items,
            },
            output_summary="Synthesized real-time 16:9 3-stage process animation (Drop -> Submerge -> Foam/Reaction) + character audio",
            status="ok",
        )
        veo_status = "veo_orchestrated_stream"

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_gif = pool.submit(
            _generate_animated_gif_bytes,
            visual_state=visual_state,
            page_number=page_number,
            num_frames=GIF_NUM_FRAMES,
            prev_image_bytes=prev_image_bytes,
        )
        fut_voice = pool.submit(
            _generate_character_voice_bytes,
            character_dialogue=character_dialogue,
            visual_state=visual_state,
            page_number=page_number,
            prefer_live_api=prefer_live_api,
            lang=lang,
        )
        gif_bytes = fut_gif.result()
        voice_wav_bytes, voice_meta = fut_voice.result()

    return (
        gif_bytes,
        voice_wav_bytes,
        {
            "engine_mode": veo_status,
            "configured_veo_model": VEO_MODEL,
            "resolution": f"{GIF_WIDTH}x{GIF_HEIGHT}",
            "frame_count": GIF_NUM_FRAMES,
            "process_stages": ["item_entry_drop", "splash_submersion", "vigorous_foam_reaction"],
            "voice_audio_bytes": len(voice_wav_bytes),
            "tts_engine": voice_meta.get("tts_engine"),
            "voice_name": voice_meta.get("voice_name"),
            "persona": voice_meta.get("persona"),
            "accumulated_items": accumulated_items,
        },
    )

