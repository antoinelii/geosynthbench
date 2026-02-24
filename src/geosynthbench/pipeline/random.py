import hashlib
from typing import Literal

SeedKind = Literal["world", "render", "task"]


def derive_seed32(base_seed: int, task_id: str, k: int, kind: SeedKind) -> int:
    """
    Deterministically derive a 32-bit seed from (base_seed, task_id, k, kind).
    Stable across platforms. Avoids Python's hash().
    """
    # Domain separation via 'kind' prevents accidental correlation.
    msg = f"geosynthbench|v1|{kind}|base={base_seed}|task={task_id}|k={k}".encode("utf-8")
    digest = hashlib.blake2b(msg, digest_size=8).digest()  # 64 bits
    seed64 = int.from_bytes(digest, "little", signed=False)
    return seed64 & 0xFFFFFFFF  # 32-bit seed for numpy/random/etc.


def derive_seeds(base_seed: int, task_id: str, k: int) -> tuple[int, int, int]:
    world_seed = derive_seed32(base_seed, task_id, k, "world")
    render_seed = derive_seed32(base_seed, task_id, k, "render")
    task_seed = derive_seed32(base_seed, task_id, k, "task")
    return world_seed, render_seed, task_seed
