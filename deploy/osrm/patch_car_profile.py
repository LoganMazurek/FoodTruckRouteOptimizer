import re
import sys


def insert_after_block(text, pattern, insert):
    if insert.strip() in text:
        return text

    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        raise ValueError("Pattern not found for insertion.")

    return text[:match.end()] + insert + text[match.end():]


def insert_before(text, pattern, insert):
    if insert.strip() in text:
        return text

    match = re.search(pattern, text)
    if not match:
        raise ValueError("Pattern not found for insertion.")

    return text[:match.start()] + insert + text[match.start():]


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: patch_car_profile.py <input_car.lua> <output_car.lua>")

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(input_path, "r", encoding="utf-8") as handle:
        text = handle.read()

    newline = "\r\n" if "\r\n" in text else "\n"

    surface_blacklist_block = (
        f"{newline}    surface_blacklist = Set {{{newline}"
        f"      'gravel',{newline}"
        f"      'unpaved',{newline}"
        f"      'dirt',{newline}"
        f"      'compacted'{newline}"
        f"    }},{newline}"
    )

    highway_blacklist_block = (
        f"{newline}    highway_blacklist = Set {{{newline}"
        f"      'service',{newline}"
        f"      'track',{newline}"
        f"      'unclassified'{newline}"
        f"    }},{newline}"
    )

    surface_pattern = r"surface_speeds\s*=\s*\{.*?\n\s*\},\n"
    highway_pattern = r"restricted_highway_whitelist\s*=\s*Set\s*\{.*?\n\s*\},\n"

    text = insert_after_block(text, surface_pattern, surface_blacklist_block)
    text = insert_after_block(text, highway_pattern, highway_blacklist_block)

    filter_block = (
        f"{newline}  if data.highway and profile.highway_blacklist and profile.highway_blacklist[data.highway] then{newline}"
        f"    return{newline}"
        f"  end{newline}{newline}"
        f"  local surface = way:get_value_by_key('surface'){newline}"
        f"  if surface and profile.surface_blacklist and profile.surface_blacklist[surface] then{newline}"
        f"    return{newline}"
        f"  end{newline}{newline}"
    )

    filter_anchor = r"\n\s*if \(not data\.highway"
    text = insert_before(text, filter_anchor, filter_block)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(text)


if __name__ == "__main__":
    main()
