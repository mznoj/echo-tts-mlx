"""Echo-TTS MLX — Diffusion TTS inference on Apple Silicon."""

__version__ = "1.0.0"

from .autoencoder import MlxFishS1DAC
from .config import ModelConfig
from .model import MlxEchoDiT
from .pipeline import EchoTTS

__all__ = ["EchoTTS", "ModelConfig", "MlxEchoDiT", "MlxFishS1DAC", "__version__"]
