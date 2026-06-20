"""
Patch the vendored AERIS-GB model.py so swin_shift() works for window
sizes other than (60, 60).

The shipped AERIS-GB src/aeris_wp_inference/model.py:swin_shift() has
two 60-hardcodes that prevent the 6h `p_1Bd66c*` checkpoint family
(window_size=[30,30]) from running:

  Line 22-23:  assert ws_y == 60
               assert ws_x == 60
  Line 116:    rearrange(..., ws_y=60, ..., ws_x=60)
  Line 118:    rearrange(..., ws_y=60, ..., ws_x=60)

The function ALREADY has `ws_y, ws_x = window_size[0], window_size[1]`
in scope (line 21), so the rest of the function uses the variables
correctly. Only the asserts + the two trailing rearranges hardcode 60.

We rewrite those four lines:
  - assert ws_y == 60     ->  # (patched out — relaxed by aeris_swin_shift_window.py)
  - assert ws_x == 60     ->  # (patched out)
  - rearrange ws_y=60     ->  ws_y=ws_y
  - rearrange ws_x=60     ->  ws_x=ws_x

Idempotent: re-runs are no-ops once patched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = (REPO / "data" / "cache" / "external" / "aeris-gb"
                  / "src" / "aeris_wp_inference" / "model.py")

PATCH_MARKER = "# PATCHED by aeris_swin_shift_window.py"

REPLACEMENTS = [
    (
        "    assert ws_y == 60\n",
        f"    # assert ws_y == 60  {PATCH_MARKER}\n",
    ),
    (
        "    assert ws_x == 60\n",
        f"    # assert ws_x == 60  {PATCH_MARKER}\n",
    ),
    (
        '        out = rearrange(tensor, "b (wcl_y ws_y) (wcl_x ws_x) d -> b (wcl_y ws_y wcl_x ws_x) d", wcl_y=wcl_y, ws_y=60, wcl_x=wcl_x, ws_x=60)\n',
        f'        out = rearrange(tensor, "b (wcl_y ws_y) (wcl_x ws_x) d -> b (wcl_y ws_y wcl_x ws_x) d", wcl_y=wcl_y, ws_y=ws_y, wcl_x=wcl_x, ws_x=ws_x)  {PATCH_MARKER}\n',
    ),
    (
        '        out = rearrange(tensor, "b (wcl_y ws_y) (wcl_x ws_x) h d -> b (wcl_y wcl_x ws_y ws_x) h d", wcl_y=wcl_y, ws_y=60, wcl_x=wcl_x, ws_x=60)\n',
        f'        out = rearrange(tensor, "b (wcl_y ws_y) (wcl_x ws_x) h d -> b (wcl_y wcl_x ws_y ws_x) h d", wcl_y=wcl_y, ws_y=ws_y, wcl_x=wcl_x, ws_x=ws_x)  {PATCH_MARKER}\n',
    ),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=Path, default=DEFAULT_TARGET,
                   help=f"Path to model.py (default: {DEFAULT_TARGET})")
    args = p.parse_args()

    if not args.target.exists():
        print(f"[aeris_swin_shift] target not found: {args.target}",
              file=sys.stderr)
        return 1

    text = args.target.read_text()

    # Idempotent check — once any replacement is applied the marker shows
    # up, so re-running is harmless.
    if PATCH_MARKER in text:
        print(f"[aeris_swin_shift] already patched: {args.target}")
        return 0

    n = 0
    for old, new in REPLACEMENTS:
        if old not in text:
            print(f"[aeris_swin_shift] WARNING: didn't find expected line:"
                  f"\n  {old!r}\n in {args.target}. Vendor may have changed.",
                  file=sys.stderr)
            return 2
        text = text.replace(old, new, 1)
        n += 1

    args.target.write_text(text)
    print(f"[aeris_swin_shift] patched {n} lines in {args.target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
