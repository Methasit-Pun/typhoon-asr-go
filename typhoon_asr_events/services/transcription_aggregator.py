"""
Transcription Aggregation Service

Handles transcription.aggregate events to combine partial transcriptions,
detect sentence boundaries, and cache results in Redis.
"""

import asyncio
import logging
import time
import re
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

import redis
import redis.asyncio as aioredis

from ..core.event_system import Event, EventHandler, EventTypes


@dataclass
class PartialTranscription:
    """Represents a partial transcription result."""
    chunk_id: str
    text: str
    confidence: float
    timestamp: float
    processing_time: float
    word_timestamps: Optional[List[Dict[str, Any]]] = None
    sequence_number: Optional[int] = None


@dataclass
class AggregatedTranscription:
    """Represents aggregated transcription results."""
    session_id: str
    full_text: str
    sentences: List[str]
    average_confidence: float
    total_chunks: int
    start_time: float
    end_time: float
    processing_stats: Dict[str, Any]


class SentenceBoundaryDetector:
    """Detects sentence boundaries in streaming text."""
    
    def __init__(self):
        # Thai sentence endings and patterns
        self.sentence_endings = ['.', '!', '?', '।', '။']
        # Common Thai particles that might indicate sentence breaks
        self.thai_particles = ['ครับ', 'ค่ะ', 'นะ', 'จ้า', 'เถอะ']
        
    def detect_boundaries(self, text: str) -> List[str]:
        """
        Detect sentence boundaries in Thai/English mixed text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of sentences
        """
        if not text.strip():
            return []
        
        sentences = []
        current_sentence = ""
        
        # Split by clear sentence endings first
        parts = re.split(r'([.!?।။])', text)
        
        for i in range(0, len(parts) - 1, 2):
            sentence_part = parts[i].strip()
            ending = parts[i + 1] if i + 1 < len(parts) else ""
            
            if sentence_part:
                current_sentence += sentence_part + ending
                
                # Check if this forms a complete sentence
                if self._is_complete_sentence(current_sentence):
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
        
        # Handle remaining text
        if current_sentence.strip():
            # Check for Thai particles that might indicate sentence end
            words = current_sentence.split()
            if words and any(particle in words[-1] for particle in self.thai_particles):
                sentences.append(current_sentence.strip())
            else:
                # Keep as incomplete sentence if we have existing sentences
                if sentences:
                    sentences.append(current_sentence.strip())
                else:
                    # Single fragment, treat as sentence
                    sentences.append(current_sentence.strip())
        
        return sentences
    
    def _is_complete_sentence(self, text: str) -> bool:
        """Check if text represents a complete sentence."""
        text = text.strip()
        if not text:
            return False
            
        # Has clear ending punctuation
        if any(text.endswith(ending) for ending in self.sentence_endings):
            return True
            
        # Has Thai particle at the end
        words = text.split()
        if words and any(particle in words[-1] for particle in self.thai_particles):
            return True
            
        # Minimum length heuristic (3+ words typically form sentences)
        if len(words) >= 3:
            return True
            
        return False


class TranscriptionAggregator(EventHandler):
    """
    Service for aggregating partial transcriptions into complete text.
    
    Handles transcription.completed events and generates text.ready events.
    """
    
    def __init__(self,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 redis_password: Optional[str] = None,
                 cache_ttl: int = 3600,
                 max_buffer_size: int = 100,
                 sentence_timeout: float = 5.0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize transcription aggregator.
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port  
            redis_db: Redis database number
            redis_password: Redis password if required
            cache_ttl: Cache time-to-live in seconds
            max_buffer_size: Maximum transcriptions to buffer per session
            sentence_timeout: Timeout for incomplete sentences (seconds)
            logger: Optional logger instance
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_password = redis_password
        self.cache_ttl = cache_ttl
        self.max_buffer_size = max_buffer_size
        self.sentence_timeout = sentence_timeout
        self.logger = logger or logging.getLogger(__name__)
        
        self._redis_client = None
        self._session_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_buffer_size))
        self._session_metadata: Dict[str, Dict] = {}
        self._boundary_detector = SentenceBoundaryDetector()
        
        # Stats tracking
        self._aggregation_stats = {
            'total_sessions': 0,
            'total_transcriptions_processed': 0,
            'total_sentences_detected': 0,
            'average_confidence': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    @property
    def handled_events(self) -> List[str]:
        """Events this handler processes."""
        return [EventTypes.TRANSCRIPTION_COMPLETED]
    
    async def _get_redis_client(self):
        """Get or create Redis client."""
        if self._redis_client is None:
            self._redis_client = aioredis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                decode_responses=True
            )
            
            # Test connection
            try:
                await self._redis_client.ping()
                self.logger.info("Connected to Redis successfully")
            except Exception as e:
                self.logger.error(f"Failed to connect to Redis: {e}")
                raise
        
        return self._redis_client
    
    async def handle(self, event: Event) -> Optional[Event]:
        """
        Handle transcription completed events.
        
        Args:
            event: Event containing transcription results
            
        Returns:
            Event with aggregated text or None if aggregation not complete
        """
        if event.event_type != EventTypes.TRANSCRIPTION_COMPLETED:
            return None
        
        try:
            # Extract transcription data
            partial_transcription = self._extract_partial_transcription(event)
            if not partial_transcription:
                return None
            
            # Determine session ID
            session_id = event.correlation_id or 'default_session'
            
            # Add to session buffer
            await self._add_to_session(session_id, partial_transcription)
            
            # Check if we should aggregate and emit results
            aggregated = await self._try_aggregate_session(session_id)
            
            if aggregated:
                # Cache results
                await self._cache_results(session_id, aggregated)
                
                # Update stats
                self._update_stats(aggregated)
                
                # Create text ready event
                return self._create_text_ready_event(event, aggregated)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Aggregation error: {e}")
            return self._create_error_event(event, str(e))
    
    def _extract_partial_transcription(self, event: Event) -> Optional[PartialTranscription]:
        """Extract partial transcription from event."""
        try:
            data = event.data
            
            # Skip if below confidence threshold
            if not data.get('meets_threshold', True):
                return None
            
            return PartialTranscription(
                chunk_id=data.get('chunk_id', event.event_id),
                text=data.get('text', ''),
                confidence=data.get('confidence', 0.0),
                timestamp=event.timestamp,
                processing_time=data.get('processing_time', 0.0),
                word_timestamps=data.get('word_timestamps'),
                sequence_number=data.get('sequence_number')
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting transcription: {e}")
            return None
    
    async def _add_to_session(self, session_id: str, transcription: PartialTranscription):
        """Add transcription to session buffer."""
        # Initialize session metadata if new
        if session_id not in self._session_metadata:
            self._session_metadata[session_id] = {
                'start_time': time.time(),
                'chunk_count': 0,
                'total_confidence': 0.0,
                'last_activity': time.time()
            }
            self._aggregation_stats['total_sessions'] += 1
        
        # Add to buffer
        self._session_buffers[session_id].append(transcription)
        
        # Update metadata
        metadata = self._session_metadata[session_id]
        metadata['chunk_count'] += 1
        metadata['total_confidence'] += transcription.confidence
        metadata['last_activity'] = time.time()
        
        self._aggregation_stats['total_transcriptions_processed'] += 1
        
        self.logger.debug(f"Added transcription to session {session_id}: {transcription.text[:50]}...")
    
    async def _try_aggregate_session(self, session_id: str) -> Optional[AggregatedTranscription]:
        """Try to aggregate session transcriptions."""
        buffer = self._session_buffers[session_id]
        metadata = self._session_metadata[session_id]
        
        if not buffer:
            return None
        
        # Combine all text
        combined_text = " ".join(t.text for t in buffer if t.text.strip())
        
        if not combined_text.strip():
            return None
        
        # Detect sentence boundaries
        sentences = self._boundary_detector.detect_boundaries(combined_text)
        
        # Check if we have complete sentences or timeout
        current_time = time.time()
        time_since_last = current_time - metadata['last_activity']
        
        has_complete_sentences = len(sentences) > 0 and self._has_complete_sentences(sentences)
        has_timeout = time_since_last > self.sentence_timeout
        
        if not (has_complete_sentences or has_timeout):
            return None
        
        # Create aggregated result
        aggregated = AggregatedTranscription(
            session_id=session_id,
            full_text=combined_text,
            sentences=sentences,
            average_confidence=metadata['total_confidence'] / metadata['chunk_count'],
            total_chunks=metadata['chunk_count'],
            start_time=metadata['start_time'],
            end_time=current_time,
            processing_stats={
                'total_processing_time': sum(t.processing_time for t in buffer),
                'chunks_processed': len(buffer),
                'sentence_count': len(sentences),
                'aggregation_trigger': 'complete_sentences' if has_complete_sentences else 'timeout'
            }
        )
        
        self._aggregation_stats['total_sentences_detected'] += len(sentences)
        
        # Clear processed transcriptions from buffer
        if has_complete_sentences:
            # Keep incomplete sentence fragments for next aggregation
            last_sentence = sentences[-1] if sentences else ""
            remaining_text = combined_text
            for sentence in sentences[:-1]:
                remaining_text = remaining_text.replace(sentence, "", 1)
            
            # Clear buffer and add back incomplete part if any
            buffer.clear()
            if remaining_text.strip() and remaining_text.strip() != last_sentence.strip():
                # Create partial transcription for remaining text
                remaining_transcription = PartialTranscription(
                    chunk_id=f"remaining_{session_id}",
                    text=remaining_text.strip(),
                    confidence=aggregated.average_confidence,
                    timestamp=current_time,
                    processing_time=0.0
                )
                buffer.append(remaining_transcription)
        else:
            # Timeout case - clear all
            buffer.clear()
            
        return aggregated
    
    def _has_complete_sentences(self, sentences: List[str]) -> bool:
        """Check if sentences list contains complete sentences."""
        if not sentences:
            return False
            
        # Check if at least one sentence has proper ending
        for sentence in sentences:
            if any(sentence.strip().endswith(ending) for ending in ['.', '!', '?', '।', '။']):
                return True
            
            # Check for Thai particles
            words = sentence.split()
            if words and any(particle in words[-1] for particle in ['ครับ', 'ค่ะ', 'นะ', 'จ้า', 'เถอะ']):
                return True
        
        return False
    
    async def _cache_results(self, session_id: str, aggregated: AggregatedTranscription):
        """Cache aggregated results in Redis."""
        try:
            redis_client = await self._get_redis_client()
            
            # Cache the aggregated transcription
            cache_key = f"transcription:{session_id}:{int(aggregated.end_time)}"
            cache_data = json.dumps(asdict(aggregated), ensure_ascii=False, indent=None)
            
            await redis_client.setex(cache_key, self.cache_ttl, cache_data)
            
            # Update session index
            index_key = f"session_index:{session_id}"
            await redis_client.zadd(index_key, {cache_key: aggregated.end_time})
            await redis_client.expire(index_key, self.cache_ttl)
            
            self.logger.debug(f"Cached results for session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Cache error: {e}")
    
    async def get_session_history(self, session_id: str, limit: int = 10) -> List[AggregatedTranscription]:
        """
        Get historical aggregated transcriptions for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of results to return
            
        Returns:
            List of aggregated transcriptions
        """
        try:
            redis_client = await self._get_redis_client()
            
            # Get recent cache keys for session
            index_key = f"session_index:{session_id}"
            cache_keys = await redis_client.zrevrange(index_key, 0, limit - 1)
            
            results = []
            for cache_key in cache_keys:
                cached_data = await redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    results.append(AggregatedTranscription(**data))
                    self._aggregation_stats['cache_hits'] += 1
                else:
                    self._aggregation_stats['cache_misses'] += 1
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving session history: {e}")
            return []
    
    def _create_text_ready_event(self, original_event: Event, aggregated: AggregatedTranscription) -> Event:
        """Create text ready event."""
        return Event(
            event_type=EventTypes.TEXT_READY,
            data={
                'session_id': aggregated.session_id,
                'full_text': aggregated.full_text,
                'sentences': aggregated.sentences,
                'confidence': aggregated.average_confidence,
                'chunk_count': aggregated.total_chunks,
                'processing_stats': aggregated.processing_stats,
                'duration': aggregated.end_time - aggregated.start_time
            },
            source='TranscriptionAggregator',
            correlation_id=original_event.correlation_id or original_event.event_id
        )
    
    def _create_error_event(self, original_event: Event, error_message: str) -> Event:
        """Create processing error event."""
        return Event(
            event_type=EventTypes.PROCESSING_ERROR,
            data={
                'error': error_message,
                'original_event_type': original_event.event_type,
                'source_service': 'TranscriptionAggregator'
            },
            source='TranscriptionAggregator',
            correlation_id=original_event.correlation_id or original_event.event_id
        )
    
    def _update_stats(self, aggregated: AggregatedTranscription):
        """Update aggregation statistics."""
        # Update average confidence (running average)
        current_avg = self._aggregation_stats['average_confidence']
        total_sessions = self._aggregation_stats['total_sessions']
        
        if total_sessions > 0:
            self._aggregation_stats['average_confidence'] = (
                (current_avg * (total_sessions - 1) + aggregated.average_confidence) / total_sessions
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        return {
            **self._aggregation_stats,
            'active_sessions': len(self._session_buffers),
            'total_buffered_transcriptions': sum(len(buffer) for buffer in self._session_buffers.values()),
            'redis_connected': self._redis_client is not None,
            'cache_ttl': self.cache_ttl,
            'max_buffer_size': self.max_buffer_size,
            'sentence_timeout': self.sentence_timeout
        }
    
    async def cleanup_old_sessions(self, max_age_hours: float = 24.0):
        """Clean up old inactive sessions."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        sessions_to_remove = []
        for session_id, metadata in self._session_metadata.items():
            if current_time - metadata['last_activity'] > max_age_seconds:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            if session_id in self._session_buffers:
                del self._session_buffers[session_id]
            if session_id in self._session_metadata:
                del self._session_metadata[session_id]
            
            self.logger.info(f"Cleaned up old session: {session_id}")
    
    async def shutdown(self):
        """Clean shutdown of the aggregator."""
        self.logger.info("Shutting down transcription aggregator")
        
        if self._redis_client:
            await self._redis_client.close()
            
        self._session_buffers.clear()
        self._session_metadata.clear()