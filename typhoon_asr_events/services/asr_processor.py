"""
ASR Processing Service

Handles the asr.process.request event with real-time transcription,
confidence scoring, and streaming capabilities using Typhoon ASR.
"""

import asyncio
import logging
import time
import io
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

import torch
import librosa
import soundfile as sf
import numpy as np
import nemo.collections.asr as nemo_asr

from ..core.event_system import Event, EventHandler, EventTypes


@dataclass
class AudioChunk:
    """Represents an audio chunk for processing."""
    data: np.ndarray
    sample_rate: int
    chunk_id: str
    timestamp: float
    duration: float


@dataclass 
class TranscriptionResult:
    """Result of ASR processing."""
    text: str
    confidence: float
    chunk_id: str
    processing_time: float
    word_timestamps: Optional[List[Dict[str, Any]]] = None
    alternatives: Optional[List[Dict[str, Any]]] = None


class ASRProcessor(EventHandler):
    """
    ASR Processing Service for handling real-time transcription requests.
    
    Processes audio.chunk.ready events and generates transcription.completed events.
    """
    
    def __init__(self, 
                 model_name: str = "scb10x/typhoon-asr-realtime",
                 device: str = "auto",
                 batch_size: int = 1,
                 confidence_threshold: float = 0.7,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize ASR processor.
        
        Args:
            model_name: Name of the Typhoon ASR model
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            batch_size: Batch size for processing multiple chunks
            confidence_threshold: Minimum confidence for accepting results
            logger: Optional logger instance
        """
        self.model_name = model_name
        self.device = self._determine_device(device)
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        self._model = None
        self._model_loaded = False
        self._processing_stats = {
            'total_chunks': 0,
            'successful_transcriptions': 0,
            'failed_transcriptions': 0,
            'average_processing_time': 0.0,
            'total_processing_time': 0.0
        }
    
    @property
    def handled_events(self) -> List[str]:
        """Events this handler processes."""
        return [EventTypes.ASR_PROCESS_REQUEST]
    
    def _determine_device(self, device: str) -> str:
        """Determine the appropriate device for processing."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    async def _load_model(self):
        """Load the Typhoon ASR model asynchronously."""
        if self._model_loaded:
            return
            
        self.logger.info(f"Loading Typhoon ASR model: {self.model_name} on {self.device}")
        
        try:
            # Load model in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None, 
                self._load_model_sync
            )
            self._model_loaded = True
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load ASR model: {e}")
            raise
    
    def _load_model_sync(self):
        """Synchronous model loading."""
        return nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_name,
            map_location=self.device
        )
    
    async def handle(self, event: Event) -> Optional[Event]:
        """
        Handle ASR process request events.
        
        Args:
            event: Event containing audio chunk data
            
        Returns:
            Event with transcription results or None on failure
        """
        if event.event_type != EventTypes.ASR_PROCESS_REQUEST:
            return None
        
        try:
            # Ensure model is loaded
            await self._load_model()
            
            # Extract audio chunk from event
            audio_chunk = self._extract_audio_chunk(event)
            if not audio_chunk:
                return self._create_error_event(event, "Invalid audio chunk data")
            
            # Process transcription
            result = await self._process_transcription(audio_chunk)
            
            # Update stats
            self._update_stats(result)
            
            # Create success event
            return self._create_transcription_event(event, result)
            
        except Exception as e:
            self.logger.error(f"ASR processing error: {e}")
            self._processing_stats['failed_transcriptions'] += 1
            return self._create_error_event(event, str(e))
    
    def _extract_audio_chunk(self, event: Event) -> Optional[AudioChunk]:
        """Extract audio chunk from event data."""
        try:
            data = event.data
            
            # Support different input formats
            if 'audio_file' in data:
                # File-based input
                return self._load_audio_file(data['audio_file'], data.get('chunk_id', event.event_id))
            
            elif 'audio_data' in data:
                # Raw audio data input
                return AudioChunk(
                    data=np.array(data['audio_data']),
                    sample_rate=data.get('sample_rate', 16000),
                    chunk_id=data.get('chunk_id', event.event_id),
                    timestamp=event.timestamp,
                    duration=len(data['audio_data']) / data.get('sample_rate', 16000)
                )
            
            elif 'audio_bytes' in data:
                # Bytes input (e.g., from streaming)
                return self._load_audio_bytes(data['audio_bytes'], data.get('chunk_id', event.event_id))
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting audio chunk: {e}")
            return None
    
    def _load_audio_file(self, file_path: str, chunk_id: str) -> AudioChunk:
        """Load audio from file path."""
        audio_data, sample_rate = librosa.load(file_path, sr=16000)
        return AudioChunk(
            data=audio_data,
            sample_rate=sample_rate,
            chunk_id=chunk_id,
            timestamp=time.time(),
            duration=len(audio_data) / sample_rate
        )
    
    def _load_audio_bytes(self, audio_bytes: bytes, chunk_id: str) -> AudioChunk:
        """Load audio from bytes."""
        audio_io = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(audio_io)
        
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        return AudioChunk(
            data=audio_data,
            sample_rate=sample_rate,
            chunk_id=chunk_id,
            timestamp=time.time(),
            duration=len(audio_data) / sample_rate
        )
    
    async def _process_transcription(self, audio_chunk: AudioChunk) -> TranscriptionResult:
        """Process transcription for audio chunk."""
        start_time = time.time()
        
        # Prepare temporary audio file
        temp_file = f"temp_chunk_{audio_chunk.chunk_id}.wav"
        sf.write(temp_file, audio_chunk.data, audio_chunk.sample_rate)
        
        try:
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            transcription_data = await loop.run_in_executor(
                None,
                self._transcribe_sync,
                temp_file
            )
            
            processing_time = time.time() - start_time
            
            # Extract results
            text = transcription_data.get('text', '')
            confidence = transcription_data.get('confidence', 0.0)
            
            # Calculate confidence if not provided
            if confidence == 0.0:
                confidence = self._estimate_confidence(text, audio_chunk.duration)
            
            return TranscriptionResult(
                text=text,
                confidence=confidence,
                chunk_id=audio_chunk.chunk_id,
                processing_time=processing_time,
                word_timestamps=transcription_data.get('timestamps'),
                alternatives=transcription_data.get('alternatives')
            )
            
        finally:
            # Cleanup
            if Path(temp_file).exists():
                Path(temp_file).unlink()
    
    def _transcribe_sync(self, audio_file: str) -> Dict[str, Any]:
        """Synchronous transcription processing."""
        try:
            # Get transcription with hypotheses for confidence
            hypotheses = self._model.transcribe(
                audio=[audio_file], 
                return_hypotheses=True
            )
            
            if hypotheses and len(hypotheses) > 0:
                primary_hypothesis = hypotheses[0]
                text = primary_hypothesis.text if hasattr(primary_hypothesis, 'text') else ''
                
                # Extract confidence if available
                confidence = 0.0
                if hasattr(primary_hypothesis, 'score'):
                    confidence = float(primary_hypothesis.score)
                elif hasattr(primary_hypothesis, 'confidence'):
                    confidence = float(primary_hypothesis.confidence)
                
                # Extract alternatives
                alternatives = []
                if len(hypotheses) > 1:
                    for i, hyp in enumerate(hypotheses[1:6]):  # Top 5 alternatives
                        alt_text = hyp.text if hasattr(hyp, 'text') else ''
                        alt_conf = float(hyp.score) if hasattr(hyp, 'score') else 0.0
                        alternatives.append({
                            'text': alt_text,
                            'confidence': alt_conf,
                            'rank': i + 2
                        })
                
                return {
                    'text': text,
                    'confidence': confidence,
                    'alternatives': alternatives if alternatives else None
                }
            else:
                return {'text': '', 'confidence': 0.0}
                
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return {'text': '', 'confidence': 0.0}
    
    def _estimate_confidence(self, text: str, duration: float) -> float:
        """Estimate confidence based on text characteristics."""
        if not text.strip():
            return 0.0
        
        # Simple heuristic based on text length and audio duration
        words = text.split()
        word_rate = len(words) / max(duration, 0.1)
        
        # Typical speaking rate is 2-3 words per second
        if 1.5 <= word_rate <= 4.0:
            confidence = 0.8
        elif 1.0 <= word_rate <= 5.0:
            confidence = 0.6
        else:
            confidence = 0.4
        
        # Adjust based on text quality indicators
        if len(words) >= 3:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _create_transcription_event(self, original_event: Event, result: TranscriptionResult) -> Event:
        """Create transcription completed event."""
        return Event(
            event_type=EventTypes.TRANSCRIPTION_COMPLETED,
            data={
                'chunk_id': result.chunk_id,
                'text': result.text,
                'confidence': result.confidence,
                'processing_time': result.processing_time,
                'word_timestamps': result.word_timestamps,
                'alternatives': result.alternatives,
                'meets_threshold': result.confidence >= self.confidence_threshold
            },
            source='ASRProcessor',
            correlation_id=original_event.correlation_id or original_event.event_id
        )
    
    def _create_error_event(self, original_event: Event, error_message: str) -> Event:
        """Create processing error event."""
        return Event(
            event_type=EventTypes.PROCESSING_ERROR,
            data={
                'error': error_message,
                'original_event_type': original_event.event_type,
                'source_service': 'ASRProcessor'
            },
            source='ASRProcessor',
            correlation_id=original_event.correlation_id or original_event.event_id
        )
    
    def _update_stats(self, result: TranscriptionResult):
        """Update processing statistics."""
        self._processing_stats['total_chunks'] += 1
        self._processing_stats['total_processing_time'] += result.processing_time
        
        if result.confidence >= self.confidence_threshold:
            self._processing_stats['successful_transcriptions'] += 1
        else:
            self._processing_stats['failed_transcriptions'] += 1
        
        # Update average
        total_chunks = self._processing_stats['total_chunks']
        self._processing_stats['average_processing_time'] = (
            self._processing_stats['total_processing_time'] / total_chunks
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self._processing_stats,
            'model_name': self.model_name,
            'device': self.device,
            'model_loaded': self._model_loaded,
            'confidence_threshold': self.confidence_threshold
        }
    
    async def shutdown(self):
        """Clean shutdown of the ASR processor."""
        self.logger.info("Shutting down ASR processor")
        # Clean up any resources if needed
        self._model = None
        self._model_loaded = False