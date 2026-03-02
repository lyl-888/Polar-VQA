import argparse
import json
import re
from pathlib import Path


COLOR_WORDS = [
    "red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "black",
    "white", "gray", "grey", "cyan", "magenta", "maroon", "navy", "teal", "violet",
    "gold", "silver", "beige", "tan", "olive", "turquoise", "lavender", "cream",
]

COLOR_PHRASES = [
    r"in shades of [^.,;:]+",
    r"in shade of [^.,;:]+",
    r"in tones of [^.,;:]+",
    r"in tone of [^.,;:]+",
    r"with colors? [^.,;:]+",
]

TEXTURE_WORDS = [
    "texture", "textured", "pattern", "patterned", "mosaic", "striped", "checkered",
    "checked", "polka-dot", "polka", "dotted", "speckled", "spotted", "grainy",
    "wrinkled", "rough", "smooth", "matte", "glossy",
]

LIGHTING_WORDS = [
    "bright", "dim", "dark", "shadow", "shadows", "shadowy", "glare", "highlight",
    "shiny", "reflective", "reflection", "reflections",
]


def _compile_word_regex(words):
    escaped = [re.escape(w) for w in words]
    return re.compile(r"\b(?:%s)\b" % "|".join(escaped), flags=re.IGNORECASE)


COLOR_WORD_RE = _compile_word_regex(COLOR_WORDS)
TEXTURE_WORD_RE = _compile_word_regex(TEXTURE_WORDS)
LIGHTING_WORD_RE = _compile_word_regex(LIGHTING_WORDS)
COLOR_PHRASE_RES = [re.compile(p, flags=re.IGNORECASE) for p in COLOR_PHRASES]


def clean_caption(text, remove_colors=True, remove_textures=True, remove_lighting=True):
    original = text
    stats = {"colors": 0, "textures": 0, "lighting": 0}

    if not isinstance(text, str):
        return text, stats

    text = text.replace("\n", " ").strip()

    if remove_colors:
        for pattern in COLOR_PHRASE_RES:
            text, n = pattern.subn("", text)
            stats["colors"] += n
        text, n = COLOR_WORD_RE.subn("", text)
        stats["colors"] += n

    if remove_textures:
        text, n = TEXTURE_WORD_RE.subn("", text)
        stats["textures"] += n

    if remove_lighting:
        text, n = LIGHTING_WORD_RE.subn("", text)
        stats["lighting"] += n

    # Clean up whitespace and punctuation
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = text.strip()
    text = text.strip(" ,.;:!-")

    if len(text) < 3:
        text = "An object is shown."

    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    changed = text != original
    return text, stats, changed


def process_file(input_path, output_path, remove_colors, remove_textures, remove_lighting, dry_run=False):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    changed_count = 0
    stats_total = {"colors": 0, "textures": 0, "lighting": 0}

    for item in data:
        conversations = item.get("conversations", [])
        for turn in conversations:
            if turn.get("from") != "gpt":
                continue
            total += 1
            cleaned, stats, changed = clean_caption(
                turn.get("value", ""),
                remove_colors=remove_colors,
                remove_textures=remove_textures,
                remove_lighting=remove_lighting,
            )
            stats_total["colors"] += stats["colors"]
            stats_total["textures"] += stats["textures"]
            stats_total["lighting"] += stats["lighting"]
            if changed:
                changed_count += 1
                turn["value"] = cleaned

    if not dry_run:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return total, changed_count, stats_total


def main():
    parser = argparse.ArgumentParser(description="Clean captions for Polar-only alignment.")
    parser.add_argument("--input", nargs="+", required=True, help="Input JSON files.")
    parser.add_argument("--output-suffix", default="_cleaned", help="Suffix for output files.")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only, do not write files.")
    parser.add_argument("--keep-colors", action="store_true", help="Do not remove color words.")
    parser.add_argument("--keep-textures", action="store_true", help="Do not remove texture/pattern words.")
    parser.add_argument("--keep-lighting", action="store_true", help="Do not remove lighting/reflectance words.")
    args = parser.parse_args()

    remove_colors = not args.keep_colors
    remove_textures = not args.keep_textures
    remove_lighting = not args.keep_lighting

    for input_file in args.input:
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        output_path = input_path.with_name(
            f"{input_path.stem}{args.output_suffix}{input_path.suffix}"
        )

        total, changed, stats = process_file(
            input_path,
            output_path,
            remove_colors=remove_colors,
            remove_textures=remove_textures,
            remove_lighting=remove_lighting,
            dry_run=args.dry_run,
        )

        print(f"[{input_path.name}] total captions: {total}, changed: {changed}")
        print(
            f"  removed - colors: {stats['colors']}, textures: {stats['textures']}, lighting: {stats['lighting']}"
        )
        if not args.dry_run:
            print(f"  output: {output_path}")


if __name__ == "__main__":
    main()
