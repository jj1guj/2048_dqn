from __future__ import annotations

import gymnasium as gym


def _register_fast2048_env() -> None:
    env_id = "fast2048-rust/Fast2048-v0"
    if env_id in gym.registry:
        return

    gym.register(
        id=env_id,
        entry_point="fast2048_env:Fast2048Env",
    )


_register_fast2048_env()