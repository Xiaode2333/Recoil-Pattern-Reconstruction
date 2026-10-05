#!/usr/bin/env python3
"""Compatibility CLI and API facade for the portable reconstruction package."""
from recoil_reconstruction import models,common,ammo,tracking,reticle,output,engine
for _module in (models,common,ammo,tracking,reticle,output,engine):
    globals().update({key:value for key,value in vars(_module).items() if not key.startswith("__")})
if __name__ == "__main__":
    try:
        raise SystemExit(engine.main())
    except (RuntimeError, ValueError, cv2.error) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(2)
