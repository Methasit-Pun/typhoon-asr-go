"""
Simple Transcription Aggregator

Minimal aggregation functionality that works with typhoon_asr results.
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class TranscriptionChunk:
    """Simple transcription chunk data."""
    text: str
    timestamp: float
    processing_time: float
    audio_duration: float
    confidence: float = 1.0  # Default high confidence for typhoon_asr


class SimpleTranscriptionAggregator:
    """
    Simple aggregator that combines transcription results.
    Uses typhoon_asr output format directly.
    """
    
    def __init__(self, 
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 redis_password: Optional[str] = None,
                 cache_ttl: int = 3600,
                 sentence_timeout: float = 5.0):
        
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_password = redis_password
        self.cache_ttl = cache_ttl
        self.sentence_timeout = sentence_timeout
        
        self._redis_client = None
        self._session_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._session_timestamps: Dict[str, float] = {}
        
    async def _get_redis_client(self):
        """Get Redis client if available."""
        if not REDIS_AVAILABLE:
            return None
            
        if self._redis_client is None:
            try:
                self._redis_client = aioredis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    password=self.redis_password,
                    decode_responses=True
                )
                await self._redis_client.ping()
            except Exception:
                self._redis_client = None
                
        return self._redis_client
    
    async def add_transcription(self, 
                              session_id: str, 
                              typhoon_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Add a transcription result from typhoon_asr.transcribe().
        
        Args:
            session_id: Session identifier
            typhoon_result: Result dict from typhoon_asr.transcribe()
            
        Returns:
            Aggregated result if ready, None otherwise
        """
        # Convert typhoon_asr result to our format
        chunk = TranscriptionChunk(
            text=typhoon_result.get('text', ''),
            timestamp=time.time(),
            processing_time=typhoon_result.get('processing_time', 0.0),
            audio_duration=typhoon_result.get('audio_duration', 0.0),
            confidence=self._estimate_confidence(typhoon_result.get('text', ''))
        )
        
        # Add to buffer
        self._session_buffers[session_id].append(chunk)
        self._session_timestamps[session_id] = time.time()
        
        # Check if we should aggregate
        return await self._try_aggregate(session_id)
    
    async def _try_aggregate(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Try to aggregate transcriptions for a session."""
        buffer = self._session_buffers[session_id]
        if not buffer:
            return None
        
        # Check timeout
        last_update = self._session_timestamps.get(session_id, 0)
        if time.time() - last_update < self.sentence_timeout:
            return None
        
        # Aggregate all chunks
        texts = [chunk.text for chunk in buffer if chunk.text.strip()]
        if not texts:
            return None
        
        combined_text = " ".join(texts)
        sentences = self._simple_sentence_split(combined_text)
        
        avg_confidence = sum(chunk.confidence for chunk in buffer) / len(buffer)
        total_duration = sum(chunk.audio_duration for chunk in buffer)
        total_processing = sum(chunk.processing_time for chunk in buffer)
        
        result = {
            'session_id': session_id,
            'full_text': combined_text,
            'sentences': sentences,
            'confidence': avg_confidence,
            'chunk_count': len(buffer),
            'audio_duration': total_duration,
            'processing_time': total_processing,
            'timestamp': time.time()
        }
        
        # Cache if Redis available
        await self._cache_result(session_id, result)
        
        # Clear buffer
        buffer.clear()
        
        return result
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting for Thai/English text."""
        if not text.strip():
            return []
        
        # Simple split on common sentence endings
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in '.!?។':  # Thai and English sentence endings
                sentences.append(current.strip())
                current = ""
        
        # Add remaining text
        if current.strip():
            sentences.append(current.strip())
        
        return sentences if sentences else [text.strip()]
    
    def _estimate_confidence(self, text: str) -> float:
        """Simple confidence estimation based on text characteristics."""
        if not text.strip():
            return 0.0
        
        # Simple heuristics
        word_count = len(text.split())
        if word_count >= 3:
            return 0.9
        elif word_count >= 1:
            return 0.7
        else:
            return 0.5
    
    async def _cache_result(self, session_id: str, result: Dict[str, Any]):
        """Cache result in Redis if available."""
        redis_client = await self._get_redis_client()
        if not redis_client:
            return
        
        try:
            cache_key = f"typhoon_asr:{session_id}:{int(result['timestamp'])}"
            cache_data = json.dumps(result, ensure_ascii=False)
            await redis_client.setex(cache_key, self.cache_ttl, cache_data)
        except Exception:
            pass  # Fail silently if caching fails
    
    async def get_session_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get cached results for a session."""
        redis_client = await self._get_redis_client()
        if not redis_client:
            return []
        
        try:
            pattern = f"typhoon_asr:{session_id}:*"
            keys = await redis_client.keys(pattern)
            keys.sort(reverse=True)  # Most recent first
            
            results = []
            for key in keys[:limit]:
                cached_data = await redis_client.get(key)
                if cached_data:
                    results.append(json.loads(cached_data))
            
            return results
        except Exception:
            return []
    
    async def cleanup_old_sessions(self, max_age_hours: float = 24.0):
        """Clean up old session buffers."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        old_sessions = [
            session_id for session_id, timestamp in self._session_timestamps.items()
            if current_time - timestamp > max_age_seconds
        ]
        
        for session_id in old_sessions:
            if session_id in self._session_buffers:
                del self._session_buffers[session_id]
            if session_id in self._session_timestamps:
                del self._session_timestamps[session_id]
    
    async def shutdown(self):
        """Clean shutdown."""
        if self._redis_client:
            await self._redis_client.close()
        
        self._session_buffers.clear()
        self._session_timestamps.clear()