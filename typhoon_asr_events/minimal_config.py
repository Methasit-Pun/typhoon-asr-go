"""
Minimal Configuration for Typhoon ASR Events

Simple configuration that works with the built-in typhoon_asr package.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class MinimalConfig:
    """
    Minimal configuration for event-driven extensions.
    Leverages typhoon_asr's built-in parameters where possible.
    """
    
    # Core ASR settings (passed to typhoon_asr.transcribe)
    model_name: str = "scb10x/typhoon-asr-realtime"
    device: str = "auto"
    with_timestamps: bool = False
    
    # Event system settings
    enable_events: bool = True
    enable_aggregation: bool = True
    
    # Redis settings (optional)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    cache_ttl: int = 3600
    
    # Simple aggregation settings
    sentence_timeout: float = 5.0
    confidence_threshold: float = 0.7
    
    @classmethod
    def from_env(cls) -> 'MinimalConfig':
        """Load configuration from environment variables."""
        return cls(
            model_name=os.getenv('TYPHOON_MODEL_NAME', cls.model_name),
            device=os.getenv('TYPHOON_DEVICE', cls.device),
            with_timestamps=os.getenv('TYPHOON_WITH_TIMESTAMPS', '').lower() == 'true',
            redis_host=os.getenv('REDIS_HOST', cls.redis_host),
            redis_port=int(os.getenv('REDIS_PORT', str(cls.redis_port))),
            redis_password=os.getenv('REDIS_PASSWORD'),
            cache_ttl=int(os.getenv('CACHE_TTL', str(cls.cache_ttl))),
            sentence_timeout=float(os.getenv('SENTENCE_TIMEOUT', str(cls.sentence_timeout))),
            confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', str(cls.confidence_threshold)))
        )
    
    def to_typhoon_args(self) -> dict:
        """Convert to arguments for typhoon_asr.transcribe()"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'with_timestamps': self.with_timestamps
        }