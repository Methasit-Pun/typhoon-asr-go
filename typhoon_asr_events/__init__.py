"""
Typhoon ASR Events - Minimal Event-Driven Wrapper

A lightweight event-driven wrapper around the existing typhoon_asr package.
This extends the built-in functionality with event processing and Redis caching
without duplicating the core ASR functionality.

Quick Start:
    from typhoon_asr_events import TyphoonASREventSystem
    
    async def main():
        system = TyphoonASREventSystem()
        result = await system.process_audio_file("audio.wav")
        print(result['text'])
        await system.shutdown()
    
    import asyncio
    asyncio.run(main())
"""

__version__ = "1.0.0"

# Re-export existing typhoon_asr functionality
try:
    from typhoon_asr import transcribe as _base_transcribe
    TYPHOON_ASR_AVAILABLE = True
except ImportError:
    TYPHOON_ASR_AVAILABLE = False
    def _base_transcribe(*args, **kwargs):
        raise ImportError("typhoon_asr package not installed. Please install: pip install typhoon-asr")

# Import our minimal extensions
from .event_wrapper import TyphoonASREventSystem, transcribe_with_events
from .simple_aggregator import SimpleTranscriptionAggregator
from .minimal_config import MinimalConfig

# Main exports - minimal but powerful
__all__ = [
    # Main event-driven interface
    'TyphoonASREventSystem',
    'transcribe_with_events',
    
    # Simple aggregation
    'SimpleTranscriptionAggregator',
    
    # Configuration
    'MinimalConfig',
    
    # Re-exported base functionality
    'transcribe',
]

# Re-export base transcribe function
transcribe = _base_transcribe