"""
Event Wrapper for Typhoon ASR

Lightweight wrapper that adds event-driven functionality to the existing
typhoon_asr.transcribe() function without duplicating core functionality.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, AsyncGenerator
from pathlib import Path

# Import the base transcribe function
try:
    from typhoon_asr import transcribe as _base_transcribe
    TYPHOON_ASR_AVAILABLE = True
except ImportError:
    TYPHOON_ASR_AVAILABLE = False
    def _base_transcribe(*args, **kwargs):
        raise ImportError("typhoon_asr package not installed. Please install: pip install typhoon-asr")

from .minimal_config import MinimalConfig
from .simple_aggregator import SimpleTranscriptionAggregator


class TyphoonASREventSystem:
    """
    Minimal event-driven wrapper around typhoon_asr.transcribe().
    
    Adds session management, aggregation, and caching without
    duplicating the core ASR functionality.
    """
    
    def __init__(self, config: Optional[MinimalConfig] = None):
        """
        Initialize the event system.
        
        Args:
            config: Optional configuration. Uses defaults if None.
        """
        self.config = config or MinimalConfig()
        self._aggregator = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Lazy initialization of components."""
        if self._initialized:
            return
        
        if self.config.enable_aggregation:
            self._aggregator = SimpleTranscriptionAggregator(
                redis_host=self.config.redis_host,
                redis_port=self.config.redis_port,
                redis_password=self.config.redis_password,
                cache_ttl=self.config.cache_ttl,
                sentence_timeout=self.config.sentence_timeout
            )
        
        self._initialized = True
    
    async def process_audio_file(self, 
                               audio_file: str, 
                               session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an audio file using typhoon_asr with optional event processing.
        
        Args:
            audio_file: Path to audio file
            session_id: Optional session ID for aggregation
            
        Returns:
            Transcription result (enhanced with session info if applicable)
        """
        await self._ensure_initialized()
        
        if not TYPHOON_ASR_AVAILABLE:
            raise ImportError("typhoon_asr package not installed. Please install: pip install typhoon-asr")
        
        # Use the built-in typhoon_asr.transcribe function
        typhoon_args = self.config.to_typhoon_args()
        result = _base_transcribe(audio_file, **typhoon_args)
        
        # Add session management if enabled
        if self.config.enable_aggregation and self._aggregator and session_id:
            aggregated = await self._aggregator.add_transcription(session_id, result)
            if aggregated:
                # Return aggregated result
                return {
                    **aggregated,
                    'original_result': result,
                    'aggregated': True
                }
        
        # Return original result with minimal enhancements
        return {
            'text': result.get('text', ''),
            'full_text': result.get('text', ''),  # Compatibility
            'sentences': [result.get('text', '')] if result.get('text', '').strip() else [],
            'confidence': 1.0,  # typhoon_asr doesn't provide confidence, assume high
            'processing_time': result.get('processing_time', 0.0),
            'audio_duration': result.get('audio_duration', 0.0),
            'timestamps': result.get('timestamps', []),
            'session_id': session_id,
            'aggregated': False,
            'original_result': result
        }
    
    async def process_audio_stream(self, 
                                 audio_files: List[str],
                                 session_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process multiple audio files as a stream.
        
        Args:
            audio_files: List of audio file paths
            session_id: Optional session ID for aggregation
            
        Yields:
            Transcription results as they become available
        """
        session_id = session_id or f"stream_{int(time.time())}"
        
        for audio_file in audio_files:
            try:
                result = await self.process_audio_file(audio_file, session_id)
                yield result
                
                # Small delay to simulate streaming
                await asyncio.sleep(0.1)
                
            except Exception as e:
                yield {
                    'error': str(e),
                    'file': audio_file,
                    'session_id': session_id,
                    'success': False
                }
    
    async def get_session_history(self, 
                                session_id: str, 
                                limit: int = 10) -> List[Dict[str, Any]]:
        """Get historical results for a session."""
        await self._ensure_initialized()
        
        if self._aggregator:
            return await self._aggregator.get_session_history(session_id, limit)
        
        return []
    
    async def cleanup_old_sessions(self, max_age_hours: float = 24.0):
        """Clean up old session data."""
        await self._ensure_initialized()
        
        if self._aggregator:
            await self._aggregator.cleanup_old_sessions(max_age_hours)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get basic system statistics."""
        return {
            'typhoon_asr_available': TYPHOON_ASR_AVAILABLE,
            'config': {
                'model_name': self.config.model_name,
                'device': self.config.device,
                'enable_aggregation': self.config.enable_aggregation
            },
            'initialized': self._initialized,
            'aggregator_enabled': self._aggregator is not None
        }
    
    async def shutdown(self):
        """Clean shutdown of the system."""
        if self._aggregator:
            await self._aggregator.shutdown()
        
        self._initialized = False


# Convenience functions for simple usage
async def transcribe_with_events(audio_file: str, 
                               config: Optional[MinimalConfig] = None,
                               session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Simple function to transcribe with event processing.
    
    Args:
        audio_file: Path to audio file
        config: Optional configuration
        session_id: Optional session ID
        
    Returns:
        Transcription result
    """
    system = TyphoonASREventSystem(config)
    try:
        return await system.process_audio_file(audio_file, session_id)
    finally:
        await system.shutdown()


def transcribe_simple(audio_file: str, 
                     model_name: str = "scb10x/typhoon-asr-realtime",
                     device: str = "auto",
                     with_timestamps: bool = False) -> str:
    """
    Simple synchronous transcription using typhoon_asr directly.
    
    Args:
        audio_file: Path to audio file
        model_name: Model name for typhoon_asr
        device: Device to use
        with_timestamps: Whether to include timestamps
        
    Returns:
        Transcribed text
    """
    if not TYPHOON_ASR_AVAILABLE:
        raise ImportError("typhoon_asr package not installed. Please install: pip install typhoon-asr")
    
    result = _base_transcribe(
        audio_file, 
        model_name=model_name, 
        device=device, 
        with_timestamps=with_timestamps
    )
    
    return result.get('text', '')