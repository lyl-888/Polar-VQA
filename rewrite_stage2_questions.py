import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


COORD_PREFIX_RE = re.compile(
    r"^\s*Focus\s+on\s+region\s*\[[^\]]+\]\.\s*",
    flags=re.IGNORECASE,
)
# Match both:
# - "of the <noun phrase>"
# - "of <noun phrase>"
OF_NOUN_PHRASE_RE = re.compile(r"\bof\s+(?:the\s+)?([^?.!]+)", flags=re.IGNORECASE)


def remove_coord_prefix(question: str) -> str:
    """Remove leading coordinate-style prompt prefix."""
    return COORD_PREFIX_RE.sub("", question).strip()


def extract_noun_from_detail(detail_question: str) -> Optional[str]:
    """
    Extract phrase after 'of' / 'of the' in detail question.
    Example:
      What are the color and material of the chair? -> chair
    """
    q = remove_coord_prefix(detail_question)
    matches = OF_NOUN_PHRASE_RE.findall(q)
    if not matches:
        return None
    # Use the last "of the ..." span to avoid capturing earlier clause fragments.
    phrase = matches[-1].strip()
    phrase = phrase.rstrip(" .")
    # Avoid "the the xxx"
    if phrase.lower().startswith("the "):
        phrase = phrase[4:].strip()
    return phrase if phrase else None


def ensure_in_the_image(question: str) -> str:
    """
    Ensure question ends with 'in the image'.
    If ends with '?', insert before '?'.
    """
    q = remove_coord_prefix(question).strip()
    if not q:
        return q

    # Keep original punctuation style; insert phrase before final punctuation.
    tail = ""
    if q[-1] in ".?!":
        tail = q[-1]
        core = q[:-1].rstrip()
    else:
        core = q

    if "in the image" in core.lower():
        return core + tail

    return core + " in the image" + tail


def get_group_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
    """Group content/detail/spatial triplets belonging to the same region/sample."""
    bbox = item.get("bbox_norm")
    bbox_tuple = tuple(bbox) if isinstance(bbox, list) else bbox
    return (
        item.get("image"),
        item.get("input_path"),
        item.get("gt_path"),
        item.get("scene_id"),
        item.get("type"),
        bbox_tuple,
    )


def get_human_conv(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    convs = item.get("conversations", [])
    for conv in convs:
        if conv.get("from") == "human":
            return conv
    return None


def build_detail_noun_map(data: List[Dict[str, Any]]) -> Dict[Tuple[Any, ...], str]:
    detail_map: Dict[Tuple[Any, ...], str] = {}
    for item in data:
        if item.get("subtype") != "detail":
            continue
        human = get_human_conv(item)
        if not human:
            continue
        noun = extract_noun_from_detail(str(human.get("value", "")))
        if noun:
            detail_map[get_group_key(item)] = noun
    return detail_map


def rewrite_questions(data: List[Dict[str, Any]]) -> int:
    changed = 0
    detail_noun_map = build_detail_noun_map(data)

    for item in data:
        human = get_human_conv(item)
        if not human:
            continue

        old_q = str(human.get("value", "")).strip()
        if not old_q:
            continue

        subtype = item.get("subtype")
        new_q: str

        if subtype == "content":
            noun = detail_noun_map.get(get_group_key(item))
            if noun:
                new_q = f"Describe the {noun} in the image."
            else:
                # Fallback if paired detail is missing or cannot parse noun.
                # Keep original sentence style, only remove coords + append phrase.
                new_q = ensure_in_the_image(old_q)
        else:
            # Keep spatial/detail semantics, remove coords, append "in the image".
            new_q = ensure_in_the_image(old_q)

        if new_q != old_q:
            human["value"] = new_q
            changed += 1

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite stage2 QA questions: remove coordinate prefix, "
            "append 'in the image', and regenerate content questions from detail."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=r"C:\Users\LY\Desktop\train\没招\merged_stage2_new_format.json",
        help="Path to input JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help=(
            "Path to output JSON file. If omitted, overwrite input file "
            "(a .bak backup will be created unless --no-backup is set)."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable backup when overwriting input file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of samples.")

    changed = rewrite_questions(data)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Done. Updated {changed} questions.")
        print(f"Saved to: {output_path}")
        return

    # Overwrite input file.
    if not args.no_backup:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        backup_path.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Backup created: {backup_path}")

    with input_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Done. Updated {changed} questions.")
    print(f"Overwritten: {input_path}")


if __name__ == "__main__":
    main()

