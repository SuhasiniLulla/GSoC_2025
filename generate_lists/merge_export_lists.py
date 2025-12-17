import argparse
import json
from pathlib import Path
from typing import Dict, Any

import typer



def merge_json_files_by_key(
    input_dir: Path,
    top_level_key: str = "genes",
) -> Dict[str, Any]:
    merged_items: Dict[str, Any] = {}

    for json_file in sorted(input_dir.glob("*.json")):
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict) or top_level_key not in data:
                typer.echo(
                    f"Warning: {json_file} does not contain top-level key "
                    f"'{top_level_key}'. Skipping."
                )
                continue

            if not isinstance(data[top_level_key], dict):
                typer.echo(
                    f"Warning: '{top_level_key}' in {json_file} is not an object. Skipping."
                )
                continue

            for item_id, payload in data[top_level_key].items():
                if item_id in merged_items:
                    typer.echo(
                        f"Warning: duplicate '{item_id}' found in {json_file}. Overwriting."
                    )
                merged_items[item_id] = payload

        except json.JSONDecodeError as exc:
            typer.echo(f"Error decoding JSON from {json_file}: {exc}")

        except OSError as exc:
            typer.echo(f"Error reading {json_file}: {exc}")

    return {top_level_key: merged_items}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge JSON files by a shared top-level object key."
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=Path,
        help="Directory containing JSON files",
    )
    parser.add_argument(
        "-k",
        "--key",
        default="genes",
        help="Top-level key to merge (default: genes)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("merged.json"),
        help="Output JSON file",
    )

    return parser.parse_args()



def main() -> None:
    args = parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input path is not a directory: {args.input_dir}")

    merged_object = merge_json_files_by_key(
        input_dir=args.input_dir,
        top_level_key=args.key,
    )

    with args.output.open("w", encoding="utf-8") as outfile:
        json.dump(merged_object, outfile, indent=4)

    typer.echo(
        f"Successfully merged JSON files from {args.input_dir} "
        f"using key '{args.key}' into {args.output}"
    )



if __name__ == "__main__":
    main()
