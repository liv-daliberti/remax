"""Check or regenerate the evidence-bound README performance preview."""

import argparse
from pathlib import Path
from remax.results import compare, comparison_markdown

ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- remax-performance:start -->"
END = "<!-- remax-performance:end -->"


def render():
    report = compare(root=ROOT)
    assert all(
        len(r["terminal_seeds"]) == 5
        for r in report["rows"]
        if r["domain"] in report["domains"]
    )
    return BEGIN + "\n" + comparison_markdown(report) + "\n" + END


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--write", action="store_true")
    args = parser.parse_args()
    path = ROOT / "README.md"
    before, rest = path.read_text().split(BEGIN, 1)
    current, after = rest.split(END, 1)
    expected = render()
    if args.write:
        path.write_text(before + expected + after)
    elif BEGIN + current + END != expected:
        raise SystemExit("README performance differs from frozen evidence")
    else:
        print("README performance matches the frozen five-seed results.")


if __name__ == "__main__":
    main()
