from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from geosynthbench.io.deserialize import world_from_jsonl
from geosynthbench.io.jsonl_utils import read_jsonl_record
from geosynthbench.render.renderer import mask_layers_to_mask_image, render_world_textured_with_mask

# --- Import your texture pipeline entrypoint(s)
# Adjust these imports to match your code.


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True, help="Path to t0.jsonl")
    ap.add_argument("--idx", type=int, default=0, help="Which jsonl record to load")
    ap.add_argument("--out", type=Path, default=Path("out"), help="Output directory")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for textures")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rec = read_jsonl_record(args.jsonl, args.idx)
    world = world_from_jsonl(rec["t0"], base_dir=args.jsonl.parent)

    if world.terrain is not None:
        elevation_map = world.terrain.elevation_m
        if world.terrain._slope_cache is not None:
            slope_map = world.terrain._slope_cache
        else:
            slope_map = world.terrain.slope()
        Image.fromarray(elevation_map).save(args.out / "elevation.png")
        Image.fromarray(slope_map).save(args.out / "slope.png")
    else:
        print("[WARN] No terrain found in world, skipping elevation and slope maps")

    rgb, mask_layers = render_world_textured_with_mask(world, np.random.default_rng(args.seed))
    sem_mask = mask_layers_to_mask_image(mask_layers)

    # Save semantic mask and RGB image
    Image.fromarray(sem_mask, mode="L").save(args.out / "mask_semantic.png")
    Image.fromarray(rgb, mode="RGB").save(args.out / "rgb_textured.png")

    print(f"[OK] wrote {args.out / 'mask_semantic.png'}")
    print(f"[OK] wrote {args.out / 'rgb_textured.png'}")


if __name__ == "__main__":
    main()
