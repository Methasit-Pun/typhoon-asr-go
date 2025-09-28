"""
Typhoon ASR Event System - Main Integration Module

This is the main system orchestrator that combines all components
into a simple, easy-to-use interface for developers.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from pathlib import Path

from .core.event_system import EventBus, Event, EventTypes
from .services.asr_processor import ASRProcessor
from .services.transcription_aggregator import TranscriptionAggregator
from .config.settings import Config, default_config, setup_logging
from .utils.helpers import EventUtils, PerformanceMonitor, graceful_shutdown


class TyphoonASRSystem:
    """
    Main system class that orchestrates the entire ASR pipeline.
    
    This is the primary interface developers will use to integrate
    Typhoon ASR into their applications.
    
    Example:
        system = TyphoonASRSystem()
        result = await system.process_audio_file("audio.wav")
        print(result['full_text'])
        await system.shutdown()
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the Typhoon ASR system.
        
        Args:
            config: Optional configuration. Uses defaults if None.
        """
        self.config = config or default_config
        self.logger = setup_logging(self.config.logging)
        
        # Core components
        self.event_bus = EventBus(self.logger)
        self.asr_processor = None
        self.transcription_aggregator = None
        
        # System state
        self._initialized = False
        self._performance_monitor = PerformanceMonitor("system_operations")
    
    async def _initialize_components(self):
        """Initialize all system components."""
        if self._initialized:
            return
        
        self.logger.info("Initializing Typhoon ASR system components...")
        
        # Initialize ASR processor
        self.asr_processor = ASRProcessor(
            model_name=self.config.asr.model_name,
            device=self.config.asr.device,
            batch_size=self.config.asr.batch_size,
            confidence_threshold=self.config.asr.confidence_threshold,
            logger=self.logger
        )
        
        # Initialize transcription aggregator
        self.transcription_aggregator = TranscriptionAggregator(
            redis_host=self.config.redis.host,
            redis_port=self.config.redis.port,
            redis_db=self.config.redis.db,
            redis_password=self.config.redis.password,
            cache_ttl=self.config.aggregation.cache_ttl,
            max_buffer_size=self.config.aggregation.max_buffer_size,
            sentence_timeout=self.config.aggregation.sentence_timeout,
            logger=self.logger
        )
        
        # Subscribe to events
        self.event_bus.subscribe(EventTypes.ASR_PROCESS_REQUEST, self.asr_processor)
        self.event_bus.subscribe(EventTypes.TRANSCRIPTION_COMPLETED, self.transcription_aggregator)
        
        self._initialized = True
        self.logger.info("System initialization completed")
    
    async def process_audio_file(self, 
                               audio_file: str, 
                               session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single audio file and return transcription results.
        
        Args:
            audio_file: Path to audio file
            session_id: Optional session ID for tracking
            
        Returns:
            Dictionary with transcription results
        """
        await self._initialize_components()
        
        # Generate correlation ID and session ID
        correlation_id = EventUtils.generate_correlation_id("file")
        session_id = session_id or f"file_session_{Path(audio_file).stem}"
        
        with self._performance_monitor.time_operation():
            # Create ASR process request event
            asr_event = Event(
                event_type=EventTypes.ASR_PROCESS_REQUEST,
                data={
                    'audio_file': audio_file,
                    'session_id': session_id
                },
                correlation_id=correlation_id,
                source='TyphoonASRSystem'
            )
            
            # Process the event and collect results
            final_result = None
            
            # Subscribe to text ready events for this correlation
            def collect_result(event: Event):
                nonlocal final_result
                if (event.event_type == EventTypes.TEXT_READY and 
                    event.correlation_id == correlation_id):
                    final_result = event.data
            
            self.event_bus.subscribe(EventTypes.TEXT_READY, collect_result)
            
            try:
                # Publish the event and wait for processing
                generated_events = await self.event_bus.publish(asr_event)
                
                # Wait for final result with timeout
                timeout = self.config.event_bus.event_timeout
                start_time = asyncio.get_event_loop().time()
                
                while final_result is None:
                    if asyncio.get_event_loop().time() - start_time > timeout:
                        raise TimeoutError(f"Processing timeout after {timeout}s")
                    
                    await asyncio.sleep(0.1)
                
                # Add system metadata
                final_result['correlation_id'] = correlation_id
                final_result['processing_time'] = self._performance_monitor.get_stats()['latest']
                final_result['success'] = True
                
                return final_result
                
            except Exception as e:
                self.logger.error(f"Error processing audio file {audio_file}: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'correlation_id': correlation_id,
                    'file': audio_file
                }
    
    async def process_audio_stream(self, 
                                 audio_chunks,
                                 session_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process streaming audio chunks and yield results as they become available.
        
        Args:
            audio_chunks: Async generator yielding audio chunk data
            session_id: Optional session ID for the stream
            
        Yields:
            Dictionary with partial transcription results
        """
        await self._initialize_components()
        
        session_id = session_id or EventUtils.generate_correlation_id("stream")
        
        # Track results for this session
        session_results = {}
        
        def collect_results(event: Event):
            if event.event_type == EventTypes.TEXT_READY:
                event_session_id = event.data.get('session_id')
                if event_session_id == session_id:
                    session_results[event.correlation_id] = event.data
        
        self.event_bus.subscribe(EventTypes.TEXT_READY, collect_results)
        
        try:
            async for chunk_data in audio_chunks:
                correlation_id = EventUtils.generate_correlation_id("chunk")
                
                # Create ASR process request
                asr_event = Event(
                    event_type=EventTypes.ASR_PROCESS_REQUEST,
                    data={
                        **chunk_data,
                        'session_id': session_id
                    },
                    correlation_id=correlation_id,
                    source='TyphoonASRSystem'
                )
                
                # Process the chunk
                await self.event_bus.publish(asr_event)
                
                # Wait a bit for processing and check for results
                await asyncio.sleep(0.1)
                
                # Yield any new results
                for corr_id, result in list(session_results.items()):
                    yield result
                    del session_results[corr_id]
                    
        except Exception as e:
            self.logger.error(f"Error in streaming processing: {e}")
            yield {
                'success': False,
                'error': str(e),
                'session_id': session_id
            }
    
    async def get_session_history(self, 
                                session_id: str, 
                                limit: int = 10) -> List[Any]:
        """
        Get historical transcription results for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of results to return
            
        Returns:
            List of historical transcription results
        """
        await self._initialize_components()
        
        try:
            return await self.transcription_aggregator.get_session_history(session_id, limit)
        except Exception as e:
            self.logger.error(f"Error retrieving session history: {e}")
            return []
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics and performance metrics.
        
        Returns:
            Dictionary with system stats
        """
        stats = {
            'system': {
                'initialized': self._initialized,
                'config': self.config.to_dict(),
                'performance': self._performance_monitor.get_stats()
            },
            'event_bus': self.event_bus.get_stats() if self._initialized else {}
        }
        
        if self._initialized:
            if self.asr_processor:
                stats['asr_processor'] = self.asr_processor.get_stats()
            
            if self.transcription_aggregator:
                stats['transcription_aggregator'] = self.transcription_aggregator.get_stats()
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform system health check.
        
        Returns:
            Health status dictionary
        """
        health_status = {
            'system_initialized': self._initialized,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        if self._initialized:
            try:
                # Check ASR processor
                if self.asr_processor:
                    asr_stats = self.asr_processor.get_stats()
                    health_status['asr_model_loaded'] = asr_stats.get('model_loaded', False)
                
                # Check aggregator (Redis connection)
                if self.transcription_aggregator:
                    try:
                        redis_client = await self.transcription_aggregator._get_redis_client()
                        await redis_client.ping()
                        health_status['redis_connected'] = True
                    except Exception:
                        health_status['redis_connected'] = False
                
                health_status['overall_healthy'] = all([
                    health_status.get('asr_model_loaded', True),
                    health_status.get('redis_connected', True)
                ])
                
            except Exception as e:
                health_status['error'] = str(e)
                health_status['overall_healthy'] = False
        
        return health_status
    
    async def shutdown(self):
        """
        Gracefully shutdown the system and clean up resources.
        """
        self.logger.info("Shutting down Typhoon ASR system...")
        
        if self.asr_processor:
            await self.asr_processor.shutdown()
        
        if self.transcription_aggregator:
            await self.transcription_aggregator.shutdown()
        
        self._initialized = False
        self.logger.info("System shutdown completed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize_components()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()


# Convenience functions for simple usage
async def transcribe_file(audio_file: str, 
                         config: Optional[Config] = None) -> str:
    """
    Quick function to transcribe a single file and return just the text.
    
    Args:
        audio_file: Path to audio file
        config: Optional configuration
        
    Returns:
        Transcribed text
    """
    async with TyphoonASRSystem(config) as system:
        result = await system.process_audio_file(audio_file)
        return result.get('full_text', '')


async def transcribe_files(audio_files: List[str], 
                          config: Optional[Config] = None) -> List[Dict[str, Any]]:
    """
    Quick function to transcribe multiple files.
    
    Args:
        audio_files: List of audio file paths
        config: Optional configuration
        
    Returns:
        List of transcription results
    """
    results = []
    async with TyphoonASRSystem(config) as system:
        for audio_file in audio_files:
            result = await system.process_audio_file(audio_file)
            results.append({
                'file': audio_file,
                'text': result.get('full_text', ''),
                'confidence': result.get('confidence', 0.0),
                'success': result.get('success', False)
            })
    
    return results


# Export main classes and functions
__all__ = [
    'TyphoonASRSystem',
    'transcribe_file', 
    'transcribe_files',
    'Config',
    'EventTypes'
]