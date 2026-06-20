"""
Patch vendor AERIS-GB model.py:swin_shift() to work at DP > 1.

The original swin_shift computes peer ranks as GROUP-LOCAL indices via
WP_grid[((my_y-1) % wp_y, (my_x-1) % wp_x)], then passes them to
torch.distributed.send(dst=...) / recv(src=...). The `dst`/`src` kwargs
expect GLOBAL ranks. With DP=1 the global and group-local ranks coincide
(group_global_ranks = [0..SP-1]) so the bug is invisible. With DP > 1,
DP group g's SP group contains global ranks [g*SP .. (g+1)*SP - 1], so
passing local rank 8 means "global rank 8" — which is in DP group 0, not
the current group → ValueError: "Global rank N is not part of group ...".

Fix: pass group_dst=/group_src= (the PyTorch 2.x kwarg for group-local
ranks) instead of dst=/src=. Identical math, correct semantics, works
for any DP factor.

Idempotent: re-runs are no-ops once patched.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = (REPO / "data" / "cache" / "external" / "aeris-gb"
                  / "src" / "aeris_wp_inference" / "model.py")

PATCH_MARKER = "# PATCHED by aeris_swin_shift_DP.py"

# Six call sites to rewrite: 4 corner/vertical (the first conditional
# block based on my_x%2) + 2 horizontal (the second based on my_y%2).
REPLACEMENTS = [
    # corner / vertical (my_x % 2 == 0 branch)
    (
        "        torch.distributed.send(corner, dst=corner_dst, group=sp_group)\n"
        "        torch.distributed.recv(recv_corner, src=corner_src, group=sp_group)\n"
        "        torch.distributed.send(vertical_slice, dst=vertical_dst, group=sp_group)\n"
        "        torch.distributed.recv(recv_vertical_slice, src=vertical_src, group=sp_group)\n",
        f"        torch.distributed.send(corner, group_dst=int(corner_dst), group=sp_group)  {PATCH_MARKER}\n"
        f"        torch.distributed.recv(recv_corner, group_src=int(corner_src), group=sp_group)  {PATCH_MARKER}\n"
        f"        torch.distributed.send(vertical_slice, group_dst=int(vertical_dst), group=sp_group)  {PATCH_MARKER}\n"
        f"        torch.distributed.recv(recv_vertical_slice, group_src=int(vertical_src), group=sp_group)  {PATCH_MARKER}\n",
    ),
    # corner / vertical (my_x % 2 == 1 branch)
    (
        "        torch.distributed.recv(recv_corner, src=corner_src, group=sp_group)\n"
        "        torch.distributed.send(corner, dst=corner_dst, group=sp_group)\n"
        "        torch.distributed.recv(recv_vertical_slice, src=vertical_src, group=sp_group)\n"
        "        torch.distributed.send(vertical_slice, dst=vertical_dst, group=sp_group)\n",
        f"        torch.distributed.recv(recv_corner, group_src=int(corner_src), group=sp_group)  {PATCH_MARKER}\n"
        f"        torch.distributed.send(corner, group_dst=int(corner_dst), group=sp_group)  {PATCH_MARKER}\n"
        f"        torch.distributed.recv(recv_vertical_slice, group_src=int(vertical_src), group=sp_group)  {PATCH_MARKER}\n"
        f"        torch.distributed.send(vertical_slice, group_dst=int(vertical_dst), group=sp_group)  {PATCH_MARKER}\n",
    ),
    # horizontal (my_y % 2 == 0 branch)
    (
        "        torch.distributed.send(horizontal_slice, dst=horizontal_dst, group=sp_group)\n"
        "        torch.distributed.recv(recv_horizontal_slice, src=horizontal_src, group=sp_group)\n"
        "    else:\n"
        "        torch.distributed.recv(recv_horizontal_slice, src=horizontal_src, group=sp_group)\n"
        "        torch.distributed.send(horizontal_slice, dst=horizontal_dst, group=sp_group)\n",
        f"        torch.distributed.send(horizontal_slice, group_dst=int(horizontal_dst), group=sp_group)  {PATCH_MARKER}\n"
        f"        torch.distributed.recv(recv_horizontal_slice, group_src=int(horizontal_src), group=sp_group)  {PATCH_MARKER}\n"
        f"    else:\n"
        f"        torch.distributed.recv(recv_horizontal_slice, group_src=int(horizontal_src), group=sp_group)  {PATCH_MARKER}\n"
        f"        torch.distributed.send(horizontal_slice, group_dst=int(horizontal_dst), group=sp_group)  {PATCH_MARKER}\n",
    ),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = p.parse_args()

    if not args.target.exists():
        print(f"[swin_shift_DP] target not found: {args.target}", file=sys.stderr)
        return 1

    text = args.target.read_text()

    if PATCH_MARKER in text:
        print(f"[swin_shift_DP] already patched: {args.target}")
        return 0

    n = 0
    for old, new in REPLACEMENTS:
        if old not in text:
            print(f"[swin_shift_DP] WARNING: missing expected block:\n{old[:200]}",
                  file=sys.stderr)
            return 2
        text = text.replace(old, new, 1)
        n += 1

    args.target.write_text(text)
    print(f"[swin_shift_DP] patched {n} send/recv blocks in {args.target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
