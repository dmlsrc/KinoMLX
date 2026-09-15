"""Pinned Krea generation oracle fixtures and numerical acceptance limits."""

RAW_REVISION = "6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603"
TURBO_REVISION = "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7"

FIXTURE_SHA256 = {
    "turbo": {
        "image-input.safetensors": (
            "4bb00dbfa68cb340eaf897f67176be3996a344f5b28c5f49bfa63bddea5bdc7f"
        ),
        "denoise-trajectories.safetensors": (
            "ab32899a70958a607071fddb962be5342ff223ff48668d3c8697937707c9de95"
        ),
        "official-vae-boundaries.safetensors": (
            "6d08fd67552fc98a4ad2a1b796792a267b01217feb13d2e04cb37d116a117d74"
        ),
        "official.png": "a0e72e4b26c537ea36b6677e16be4c09f734d052775fab675b89c307beb4eaed",
        "image-report.json": ("3ab0645ef6b6acf039b08afb75f955d65c4eb85d98ede7ede7901094f9a789f0"),
    },
    "raw": {
        "image-input.safetensors": (
            "982bbc22222bcbbe44ec0f45b9f7aa6d65d7f243a72302478309bde855c17d64"
        ),
        "denoise-trajectories.safetensors": (
            "d1400bb82cb89d6eede1d9a719e97ff065360eac59ae691c3b37874ae8710f34"
        ),
        "official-vae-boundaries.safetensors": (
            "ad6bb33075086ff7330e1799a9ba52b2ca0ada1a3e451405c68e20917351520c"
        ),
        "official.png": "2931ac2e9eb605ff6803da807d8f069b256d7a5e36f12cc5b993727e6170cbb2",
        "image-report.json": ("b780b5ad7b884f2db90166359f0da87a4d6fafd4331ca45b30451707b298a5ff"),
    },
}

SCHEDULE_MAX_ABS_LIMIT = 1.2e-7
MODEL_OUTPUT_NORMALIZED_RMSE_LIMIT = 0.03125
ONE_STEP_STATE_NORMALIZED_RMSE_LIMIT = 0.0078125
DECODED_PIXEL_MAX_ABS_LIMIT = 5
DECODED_PIXEL_MEAN_ABS_LIMIT = 0.3


__all__ = [
    "DECODED_PIXEL_MAX_ABS_LIMIT",
    "DECODED_PIXEL_MEAN_ABS_LIMIT",
    "FIXTURE_SHA256",
    "MODEL_OUTPUT_NORMALIZED_RMSE_LIMIT",
    "ONE_STEP_STATE_NORMALIZED_RMSE_LIMIT",
    "RAW_REVISION",
    "SCHEDULE_MAX_ABS_LIMIT",
    "TURBO_REVISION",
]
