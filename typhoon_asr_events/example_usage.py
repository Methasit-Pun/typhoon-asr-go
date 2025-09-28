"""
Typhoon ASR Event-Driven System Example

This example demonstrates how to use the event-driven architecture for
real-time audio transcription with Typhoon ASR, including:

1. Setting up the event bus
2. Configuring ASR processing service  
3. Setting up transcription aggregation
4. Processing audio files and streams
5. Handling events and results

Usage:
    python example_usage.py --audio-file path/to/audio.wav
    python example_usage.py --stream-mode
"""

import asyncio
import argparse
import logging
from pathlib import Path
from typing import Optional

from typhoon_asr_events import EventBus, Event
from typhoon_asr_events.core import EventTypes
from typhoon_asr_events.services import ASRProcessor, TranscriptionAggregator
from typhoon_asr_events.config import Config, default_config, setup_logging
from typhoon_asr_events.utils import AudioUtils, EventUtils, PerformanceMonitor, graceful_shutdown


class TyphoonASRSystem:
    """
    Main system class that orchestrates the event-driven ASR processing.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the ASR system.
        
        Args:
            config: Optional configuration (uses default if not provided)
        """
        self.config = config or default_config
        self.logger = setup_logging(self.config.logging)
        
        # Initialize event bus
        self.event_bus = EventBus(logger=self.logger)
        
        # Initialize services
        self.asr_processor = ASRProcessor(
            model_name=self.config.asr.model_name,
            device=self.config.asr.device,
            confidence_threshold=self.config.asr.confidence_threshold,
            logger=self.logger
        )
        
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
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor("asr_system")
        
        # Setup event subscriptions
        self._setup_event_handlers()
        
        self.logger.info("Typhoon ASR System initialized")
    
    def _setup_event_handlers(self):
        """Set up event handlers and subscriptions."""
        
        # Subscribe ASR processor to handle transcription requests
        self.event_bus.subscribe(EventTypes.ASR_PROCESS_REQUEST, self.asr_processor)
        
        # Subscribe aggregator to handle completed transcriptions
        self.event_bus.subscribe(EventTypes.TRANSCRIPTION_COMPLETED, self.transcription_aggregator)
        
        # Subscribe to final text results
        self.event_bus.subscribe(EventTypes.TEXT_READY, self._handle_text_ready)
        
        # Subscribe to errors
        self.event_bus.subscribe(EventTypes.PROCESSING_ERROR, self._handle_error)
        
        self.logger.info("Event handlers configured")
    
    async def process_audio_file(self, audio_file: str, session_id: Optional[str] = None) -> dict:
        """
        Process a single audio file.
        
        Args:
            audio_file: Path to audio file
            session_id: Optional session ID for tracking
            
        Returns:
            Processing results
        """
        audio_path = Path(audio_file)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        if not AudioUtils.validate_audio_format(str(audio_path)):
            raise ValueError(f"Unsupported audio format: {audio_path.suffix}")
        
        # Generate session ID if not provided
        if not session_id:
            session_id = EventUtils.generate_correlation_id("session")
        
        self.logger.info(f"Processing audio file: {audio_file} (session: {session_id})")
        
        # Get audio info
        audio_info = AudioUtils.get_audio_info(str(audio_path))
        self.logger.info(f"Audio info: {audio_info}")
        
        # Create ASR request event
        asr_event = Event(
            event_type=EventTypes.ASR_PROCESS_REQUEST,
            data={
                'audio_file': str(audio_path),
                'chunk_id': f"file_{audio_path.stem}",
                'session_id': session_id
            },
            source='TyphoonASRSystem',
            correlation_id=session_id
        )
        
        # Process event
        with self.perf_monitor.time_operation():
            generated_events = await self.event_bus.publish(asr_event)
            
            # Process any generated events (transcription.completed -> text.ready)
            for event in generated_events:
                await self.event_bus.publish(event)
        
        # Return processing stats
        return {
            'session_id': session_id,
            'audio_file': str(audio_path),
            'audio_info': audio_info,
            'performance': self.perf_monitor.get_stats(),
            'asr_stats': self.asr_processor.get_stats(),
            'aggregation_stats': self.transcription_aggregator.get_stats()
        }
    
    async def process_audio_stream(self, audio_chunks, session_id: Optional[str] = None):
        """
        Process streaming audio chunks.
        
        Args:
            audio_chunks: Async iterator of audio chunks
            session_id: Optional session ID for tracking
        """
        if not session_id:
            session_id = EventUtils.generate_correlation_id("stream")
        
        self.logger.info(f"Starting stream processing (session: {session_id})")
        
        chunk_count = 0
        
        async for chunk_data in audio_chunks:
            chunk_count += 1
            chunk_id = f"stream_chunk_{chunk_count}"
            
            # Create ASR request event
            asr_event = Event(
                event_type=EventTypes.ASR_PROCESS_REQUEST,
                data={
                    'audio_data': chunk_data.get('audio_data'),
                    'sample_rate': chunk_data.get('sample_rate', 16000),
                    'chunk_id': chunk_id,
                    'session_id': session_id,
                    'sequence_number': chunk_count
                },
                source='TyphoonASRSystem',
                correlation_id=session_id
            )
            
            # Process chunk
            generated_events = await self.event_bus.publish(asr_event)
            
            # Process generated events
            for event in generated_events:
                await self.event_bus.publish(event)
            
            self.logger.debug(f"Processed chunk {chunk_count}")
    
    async def _handle_text_ready(self, event: Event) -> None:
        """Handle text ready events."""
        data = event.data
        
        self.logger.info("🎯 TEXT READY")
        self.logger.info(f"Session: {data['session_id']}")
        self.logger.info(f"Full Text: {data['full_text']}")
        self.logger.info(f"Sentences: {len(data['sentences'])}")
        
        for i, sentence in enumerate(data['sentences'], 1):
            self.logger.info(f"  {i}. {sentence}")
        
        self.logger.info(f"Confidence: {data['confidence']:.3f}")
        self.logger.info(f"Chunks: {data['chunk_count']}")
        self.logger.info(f"Duration: {data['duration']:.2f}s")
        print("-" * 60)
    
    async def _handle_error(self, event: Event) -> None:
        """Handle error events."""
        data = event.data
        self.logger.error(f"Processing error: {data['error']}")
        self.logger.error(f"Source: {data['source_service']}")
    
    async def get_session_history(self, session_id: str, limit: int = 10):
        """
        Get historical results for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum results to return
            
        Returns:
            List of historical results
        """
        return await self.transcription_aggregator.get_session_history(session_id, limit)
    
    async def get_system_stats(self) -> dict:
        """Get comprehensive system statistics."""
        return {
            'event_bus': self.event_bus.get_stats(),
            'asr_processor': self.asr_processor.get_stats(),
            'transcription_aggregator': self.transcription_aggregator.get_stats(),
            'performance': self.perf_monitor.get_stats()
        }
    
    async def shutdown(self):
        """Gracefully shutdown the system."""
        self.logger.info("Shutting down Typhoon ASR System...")
        
        await self.asr_processor.shutdown()
        await self.transcription_aggregator.shutdown()
        
        self.logger.info("Shutdown complete")


# Example usage functions
async def example_single_file(audio_file: str):
    """Example: Process a single audio file."""
    print(f"🎵 Processing single file: {audio_file}")
    
    system = TyphoonASRSystem()
    
    try:
        result = await system.process_audio_file(audio_file)
        print("✅ Processing complete!")
        print(f"📊 Results: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await system.shutdown()


async def example_streaming():
    """Example: Process streaming audio chunks."""
    print("🎵 Processing streaming audio...")
    
    system = TyphoonASRSystem()
    
    # Simulate streaming audio chunks
    async def mock_audio_stream():
        """Mock audio stream for demonstration."""
        import numpy as np
        
        for i in range(5):
            # Generate 2 seconds of mock audio data
            duration = 2.0
            sample_rate = 16000
            samples = int(duration * sample_rate)
            
            # Simple sine wave as mock audio
            t = np.linspace(0, duration, samples)
            frequency = 440 + i * 100  # Different frequencies for each chunk
            audio_data = np.sin(2 * np.pi * frequency * t) * 0.1
            
            yield {
                'audio_data': audio_data.tolist(),
                'sample_rate': sample_rate,
                'chunk_number': i + 1
            }
            
            # Simulate real-time streaming delay
            await asyncio.sleep(0.5)
    
    try:
        await system.process_audio_stream(mock_audio_stream())
        print("✅ Streaming complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await system.shutdown()


async def example_batch_processing(audio_files: list):
    """Example: Process multiple audio files."""
    print(f"🎵 Processing {len(audio_files)} files...")
    
    system = TyphoonASRSystem()
    
    try:
        results = []
        
        for audio_file in audio_files:
            print(f"Processing: {Path(audio_file).name}")
            result = await system.process_audio_file(audio_file)
            results.append(result)
        
        print("✅ Batch processing complete!")
        print(f"📊 Processed {len(results)} files")
        
        # Print summary stats
        stats = await system.get_system_stats()
        print(f"📈 System stats: {stats}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await system.shutdown()


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Typhoon ASR Event System Example")
    parser.add_argument("--audio-file", help="Path to audio file to process")
    parser.add_argument("--audio-files", nargs="+", help="Multiple audio files to process")
    parser.add_argument("--stream-mode", action="store_true", help="Run in streaming mode")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load custom config if provided
    if args.config:
        try:
            config = Config.from_file(args.config)
            print(f"📁 Loaded config from: {args.config}")
        except Exception as e:
            print(f"⚠️  Failed to load config: {e}")
            config = default_config
    else:
        config = default_config
    
    # Run appropriate example
    if args.audio_file:
        asyncio.run(example_single_file(args.audio_file))
    elif args.audio_files:
        asyncio.run(example_batch_processing(args.audio_files))
    elif args.stream_mode:
        asyncio.run(example_streaming())
    else:
        print("❓ Please specify --audio-file, --audio-files, or --stream-mode")
        parser.print_help()


if __name__ == "__main__":
    main()