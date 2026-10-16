# WonderBook Live — Google Stitch UI Design Specification (`STITCH_DESIGN_GUIDE.md`)

This document specifies the **Google Stitch** component architecture, adaptive dual-mode design tokens (`science_lab` vs. `storybook`), typography scale, and ready-to-paste **Google Stitch UI Generation Prompts** implemented in [`static/index.html`](static/index.html).

---

## 1. Google Stitch Component Hierarchy (`data-stitch-*` Architecture)

The [`static/index.html`](static/index.html) interface is structured into 5 modular top-level Stitch sections (`data-stitch-section`) and 6 sub-components (`data-stitch-component`) with zero external build dependencies:

| Stitch Attribute | DOM Selector / ID | Responsibility |
| :--- | :--- | :--- |
| `data-stitch-section="top-nav"` | `header.top-nav` | Brand identity, active mode pill (`SCIENCE LAB MODE` / `STORYBOOK MODE`), Scenario Preset Selector (`#presetSelect`), Model Speed/Quality Tier Selector (`#fastModeSelect`), Live Pipeline Status Pills, and primary **"Finish & Publish Book (완본 저장·공유)"** CTA (`#publishBtn`). |
| `data-stitch-component="preset-selector"` | `#presetSelect`, `#newBookBtn` | Switches between `science_hcl_lab`, `fairy_starlight_fox`, and `science_space_kitchen` or starts a fresh book session. |
| `data-stitch-component="model-tier-selector"` | `#fastModeSelect` | Selects `fast_mode`: `argon_low` (`gemini-4-argon` low thinking + `gemini-3.8-flash`), `flash_ultra` (`gemini-3.8-flash` Ultra-Fast), or `argon_medium` (`gemini-4-argon` medium thinking). |
| `data-stitch-component="pipeline-pills"` | `#pillStoryModel`, `#pillImageModel`, `#pillMotionModel`, `#pillTurnLatency` | Real-time telemetry badges displaying the active story model, Nano Banana visual continuity status, Veo motion/voice status, and last turn latency (`ms`). |
| `data-stitch-section="book-spread"` | `main.book-outer-frame` | Open hardbound 2-page picture-book / lab-notebook spread with 3D cover rim, center spine shadow (`.book-spine`), and page-flip keyframe transitions (`.turning-page`). |
| `data-stitch-component="visual-stage"` | `section.book-page-left` | Left page housing the **View Switcher Tabs** (`🎬 Living Motion (Veo + Continuity)` vs `🎨 Hi-Res Illustration (Nano Banana)`), main visual canvas (`#sceneImage`), loading shimmer overlay (`#visualShimmer`), **Cumulative Scene State Strip**, and collapsible Prompt Inspector (`#continuityNote`). |
| `data-stitch-component="cumulative-state-strip"` | `#chainBadge`, `#phBadge`, `#tempBadge`, `#cumulativeItemsStrip` | Displays the **Nano Banana Continuity Chain Badge** (`Chained from Page N-1`), live `pH` & `Temp (°C)` sensors (in Science Lab mode), and numbered badges for all accumulated items (`🔑 Metal Key`, `🍎 Apple`, `🍪 Cookie`, `💧 Water Spray`). |
| `data-stitch-component="narrative-page"` | `section.book-page-right` | Right page displaying the child's Spoken Question Stamp (`#userQuestionBadge`), Page Title (`#pageTitle`), bilingual Story Paragraphs (`#storyParagraphs`), Character Voice Card, Science/Continuity Card, and Page Flip controls (`◀ Previous Page`, `Page X of Y`, `Next Page ▶`). |
| `data-stitch-component="character-voice-card"` | `#characterAvatar`, `#characterName`, `#characterDialogue`, `#voiceAudio`, `#autoSpeakToggle` | Character persona badge (`Dr. Nova 🦉`, `Lumi 🦊`, `Captain Orbit 👩‍🚀`), spoken dialogue quote, native `<audio>` player, **🔊 Browser TTS Voice** button, and **Auto-Speak on Page Turn** checkbox. |
| `data-stitch-component="science-note-card"` | `#scienceNoteHeaderLabel`, `#scienceNote` | Highlights chemical reaction equations (e.g., $\text{Fe(s)} + 2\text{HCl(aq)} \rightarrow \text{FeCl}_2\text{(aq)} + \text{H}_2\text{(g)}\uparrow$) and safety insights or story continuity notes. |
| `data-stitch-section="voice-dock"` | `section.voice-dock` | Interactive command dock featuring the **Push-to-Talk Microphone Button** (`#micBtn`) with pulsing animation rings, language selector (`ko-KR` / `en-US`), text input (`#utteranceInput`), **Ask & Turn Page ▶** button (`#askBtn`), and **1-Click "What-If" Curiosity Chips** (`#followupChips`). |
| `data-stitch-section="continuity-filmstrip"` | `section.continuity-filmstrip-section` | Horizontal scrollable timeline (`#filmstripTimeline`) showing every generated page (`Page 1 ➔ Chained ➔ Page 2 ➔ ...`) with thumbnail preview and 1-click page jumping. |
| `data-stitch-section="publish-modal"` | `#publishModal` | Modal dialog presenting the published share link (`/book/<book_id>`), 1-click copy button, standalone HTML download (`/api/books/<book_id>/export.html`), and SHA-256 / Cloud Storage verification proof. |

---

## 2. Adaptive Dual-Mode Color Tokens

The root `<html>` element switches `data-theme="science_lab"` and `data-theme="storybook"` automatically based on the active book's `mode`:

| CSS Custom Property | `science_lab` Mode (Bioluminescent Lab) | `storybook` Mode (Enchanted Starlight) |
| :--- | :--- | :--- |
| `--bg-canvas` | `#060b16` (Deep Obsidian Lab Slate) | `#0f0a1e` (Midnight Starlight Indigo) |
| `--surface-book-cover` | `#0f2942` (Borosilicate Navy Cover) | `#3b1d5e` (Royal Velvet Hardcover) |
| `--surface-page-left` | `linear-gradient(155deg, #0d192b, #0f2038)` | `linear-gradient(155deg, #1b1332, #241942)` |
| `--surface-page-right` | `linear-gradient(205deg, #0f1d33, #12243e)` | `linear-gradient(205deg, #21173b, #2a1d49)` |
| `--accent-primary` | `#10b981` (Bioluminescent Emerald) | `#f59e0b` (Starlight Amber Gold) |
| `--accent-secondary` | `#38bdf8` (Electric Lab Cyan) | `#f472b6` (Fairy Rose Pink) |
| `--accent-tertiary` | `#fbbf24` (Reaction Warning Gold) | `#fbbf24` (Firefly Warm Gold) |
| `--text-primary` | `#f1f5f9` (Crisp Alabaster) | `#fdf8f5` (Warm Parchment Ivory) |

---

## 3. Typography & Responsive Grid Scale

- **Display / Page Titles (`--font-display`)**:
  - `science_lab`: `'Pretendard', 'Segoe UI', system-ui, sans-serif` (`22px`, `font-weight: 800`, `-0.015em` tracking)
  - `storybook`: `'Georgia', 'Pretendard', serif` (`22px`, `font-weight: 800`)
- **Narrative Body (`--font-body`)**: `15px` (`line-height: 1.68`) with left accent border callouts for comfortable bilingual (Korean + English) reading.
- **Telemetry & Chemical Metrics (`--font-mono`)**: `'JetBrains Mono', 'Cascadia Code', Consolas, monospace` (`11px–12px`) for `pH`, `°C`, turn latency (`ms`), and SHA-256 verification digests.
- **Responsive Breakpoints**:
  - `> 980px`: Full 2-column open-book spread (`grid-template-columns: 1fr 1fr`) with central 38px spine shadow overlay.
  - `<= 980px`: Single-column stacked book pages with preserved visual stage and touch-friendly What-If curiosity chips.

---

## 4. Ready-to-Paste Google Stitch Prompts

Use the prompts below in **Google Stitch** (`stitch.withgoogle.com`) to generate or iterate on any section of **WonderBook Live**:

### Prompt A — Full-Page WonderBook Live Open-Book Interface
> **Prompt for Google Stitch:**
> "Design a desktop-first interactive web application called **WonderBook Live (Living Storybook & Safe What-If Science Lab)** styled as an open hardbound two-page picture book and interactive chemistry lab notebook.
> 1. **Top Orchestration Bar**: Brand emblem + 'WonderBook Live' title, Scenario selector ('What-If HCl Beaker Lab', 'Lumi the Starlight Fox', 'Zero-G Space Kitchen'), Model Speed Tier selector ('gemini-4-argon Low Thinking', 'gemini-3.8-flash Ultra-Fast', 'gemini-4-argon Medium Thinking'), live status pills for Story Agent, Nano Banana Continuity, Veo Motion, and turn latency in ms, plus a glowing primary CTA button 'Finish & Publish Book (완본 저장·공유)'.
> 2. **Two-Page Open Book Spread** with a subtle center book-spine shadow:
>    - **Left Page (Visual Stage)**: Segmented tabs to toggle between '🎬 Living Motion (Veo + Continuity)' and '🎨 Hi-Res Illustration (Nano Banana)', a 4:3 visual canvas with a subtle corner badge, a Cumulative Scene State strip showing 'Chained from Page N-1', live pH and Temperature (°C) pills, accumulated item badges ('🔑 Metal Key', '🍎 Apple', '🍪 Cookie', '💧 Water Spray'), and a collapsible prompt inspector.
>    - **Right Page (Narrative & Voice)**: A highlighted 'Spoken Question Stamp', editorial page title and bilingual story paragraphs, a Character Voice Card with avatar ('Dr. Nova 🦉'), spoken dialogue quote, embedded audio player, 'Browser TTS' button, and 'Auto-Speak on Page Turn' checkbox, followed by a 'Science Reaction Equation & Safety Insight' card and Previous/Next Page flip buttons.
> 3. **Bottom Interactive Voice & What-If Command Dock**: A pulsing red Push-to-Talk microphone button, language selector (ko-KR / en-US), wide text input with 'Ask & Turn Page ▶' CTA, and a row of 1-click 'What-If' curiosity chips.
> 4. **Bottom Visual Continuity Filmstrip**: Horizontal timeline of page thumbnails connected by '➔ Chained' arrows."

### Prompt B — Publish & Share Verification Modal
> **Prompt for Google Stitch:**
> "Design a sleek dark-mode modal dialog titled '🎉 WonderBook Published & Verified!' for sharing a completed interactive picture book. Include a read-only monospace share URL input with a '📋 Copy Share Link' button, two side-by-side action buttons ('📖 Open Published Flipbook Viewer' and '⬇️ Download Standalone HTML Book'), and a technical verification box showing Book ID, Page Count, Manifest SHA-256 digest, Export HTML SHA-256 digest, and Google Cloud Storage sync status."
