"""
Typhoon ASR Library - Complete Integration Demo

This file demonstrates how to use the typhoon_asr_events library
as a plug-and-play solution for voice-to-text in any Python project.

Simply copy the typhoon_asr_events folder to your project and import!
"""

import asyncio
import logging
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import time

# Import the library (assumes typhoon_asr_events folder is in your project)
try:
    from typhoon_asr_events import TyphoonASRSystem
    from typhoon_asr_events.config import Config
    from typhoon_asr_events.core import EventBus, Event, EventTypes, EventHandler
    from typhoon_asr_events.services import ASRProcessor, TranscriptionAggregator
    from typhoon_asr_events.utils import AudioUtils, PerformanceMonitor, HealthChecker
    LIBRARY_AVAILABLE = True
    print("✅ Typhoon ASR Events library loaded successfully")
except ImportError as e:
    print(f"❌ Library import failed: {e}")
    print("📁 Make sure typhoon_asr_events folder is in your project directory")
    print("📦 Install dependencies: pip install torch librosa soundfile nemo-toolkit redis pyyaml")
    LIBRARY_AVAILABLE = False


class VoiceToTextLibrary:
    """
    Main class demonstrating how to integrate Typhoon ASR into your pipeline.
    This shows all the key functions you'll need as a developer.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the voice-to-text system.
        
        Args:
            config: Optional configuration. If None, uses defaults.
        """
        self.config = config or self._create_default_config()
        self.system: Optional[TyphoonASRSystem] = None
        self.performance_monitor = PerformanceMonitor("voice_to_text_operations")
        self._is_initialized = False
    
    def _create_default_config(self) -> Config:
        """Create default configuration for the library."""
        config = Config()
        
        # ASR Settings
        config.asr.model_name = "scb10x/typhoon-asr-realtime"
        config.asr.device = "auto"  # Auto-detect GPU/CPU
        config.asr.confidence_threshold = 0.7
        config.asr.batch_size = 1
        
        # Redis Settings (optional - works without Redis)
        config.redis.host = os.getenv("REDIS_HOST", "localhost")
        config.redis.port = int(os.getenv("REDIS_PORT", "6379"))
        config.redis.password = os.getenv("REDIS_PASSWORD")
        
        # Logging
        config.logging.level = "INFO"
        config.logging.enable_console = True
        
        return config
    
    async def initialize(self) -> bool:
        """
        Initialize the ASR system. Call this before processing audio.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._is_initialized:
            return True
        
        try:
            print("🚀 Initializing Typhoon ASR system...")
            self.system = TyphoonASRSystem(self.config)
            self._is_initialized = True
            
            print(f"✅ System ready! Using device: {self.config.asr.device}")
            print(f"🎯 Confidence threshold: {self.config.asr.confidence_threshold}")
            
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    async def transcribe_file(self, audio_file: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert a single audio file to text.
        
        This is the most common function you'll use - just pass an audio file path.
        
        Args:
            audio_file: Path to audio file (.wav, .mp3, .m4a, .flac, etc.)
            session_id: Optional session ID for tracking multiple related files
        
        Returns:
            Dictionary containing transcription and metadata
        """
        if not self._is_initialized:
            await self.initialize()
        
        # Validate file
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        if not AudioUtils.validate_audio_format(audio_file):
            raise ValueError(f"Unsupported audio format: {Path(audio_file).suffix}")
        
        # Get audio information
        audio_info = AudioUtils.get_audio_info(audio_file)
        
        # Process with performance monitoring
        with self.performance_monitor.time_operation():
            result = await self.system.process_audio_file(audio_file, session_id)
        
        # Add performance info
        result['processing_stats'] = {
            **result.get('processing_stats', {}),
            'file_size_mb': Path(audio_file).stat().st_size / (1024 * 1024),
            'audio_duration': audio_info.get('duration', 0),
            'real_time_factor': self.performance_monitor.get_stats().get('latest', 0) / audio_info.get('duration', 1)
        }
        
        return result
    
    async def transcribe_multiple_files(self, 
                                     audio_files: List[str], 
                                     session_id: Optional[str] = None,
                                     max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """
        Convert multiple audio files to text with concurrent processing.
        
        Args:
            audio_files: List of audio file paths
            session_id: Optional session ID for all files
            max_concurrent: Maximum number of files to process simultaneously
        
        Returns:
            List of transcription results
        """
        if not self._is_initialized:
            await self.initialize()
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_file(audio_file: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await self.transcribe_file(audio_file, session_id)
                except Exception as e:
                    return {
                        'file': audio_file,
                        'error': str(e),
                        'success': False
                    }
        
        print(f"📂 Processing {len(audio_files)} files (max {max_concurrent} concurrent)...")
        
        # Process all files
        tasks = [process_single_file(file) for file in audio_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = [r for r in results if isinstance(r, dict) and r.get('success', True)]
        
        print(f"✅ Completed: {len(successful_results)}/{len(audio_files)} files successful")
        
        return results
    
    async def transcribe_stream(self, 
                              audio_chunks,
                              session_id: Optional[str] = None,
                              chunk_callback: Optional[callable] = None):
        """
        Process streaming audio data in real-time.
        
        Args:
            audio_chunks: Async generator yielding audio data
            session_id: Session ID for the stream
            chunk_callback: Optional callback for each processed chunk
        
        Yields:
            Transcription results as they become available
        """
        if not self._is_initialized:
            await self.initialize()
        
        session_id = session_id or f"stream_{int(time.time())}"
        
        async for result in self.system.process_audio_stream(audio_chunks, session_id):
            if chunk_callback:
                await chunk_callback(result)
            
            yield result
    
    async def get_session_transcripts(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get historical transcripts for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of results
        
        Returns:
            List of historical transcription results
        """
        if not self._is_initialized:
            await self.initialize()
        
        history = await self.system.get_session_history(session_id, limit)
        
        # Convert to simple dictionaries
        return [
            {
                'session_id': item.session_id,
                'text': item.full_text,
                'sentences': item.sentences,
                'confidence': item.average_confidence,
                'chunks': item.total_chunks,
                'start_time': item.start_time,
                'end_time': item.end_time
            }
            for item in history
        ]
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get detailed performance and system statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            'library_stats': self.performance_monitor.get_stats(),
            'library_operations': self.performance_monitor.measurements
        }
        
        if self._is_initialized:
            system_stats = await self.system.get_system_stats()
            stats['system_stats'] = system_stats
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform system health check.
        
        Returns:
            Health status dictionary
        """
        health = HealthChecker()
        
        # Add basic checks
        health.add_check("system_initialized", lambda: self._is_initialized)
        
        if self._is_initialized and self.system:
            health.add_check("asr_processor_ready", 
                           lambda: self.system.asr_processor._model_loaded)
        
        # Check Redis if configured
        try:
            import redis
            redis_client = redis.Redis(
                host=self.config.redis.host,
                port=self.config.redis.port,
                password=self.config.redis.password,
                socket_timeout=1
            )
            health.add_check("redis_connection", lambda: redis_client.ping())
        except:
            pass
        
        return await health.run_checks()
    
    def configure_for_production(self):
        """Configure the system for production use."""
        self.config.asr.device = "cuda"  # Use GPU
        self.config.asr.confidence_threshold = 0.8  # Higher quality
        self.config.aggregation.cache_ttl = 7200  # 2 hours
        self.config.logging.level = "INFO"
        self.config.logging.file_path = "logs/typhoon_asr.log"
    
    def configure_for_development(self):
        """Configure the system for development use."""
        self.config.asr.device = "cpu"  # No GPU required
        self.config.asr.confidence_threshold = 0.6  # Lower for testing
        self.config.logging.level = "DEBUG"
        self.config.logging.enable_console = True
    
    async def cleanup(self):
        """Clean up system resources."""
        if self.system:
            await self.system.shutdown()
        self._is_initialized = False
        print("🧹 System cleaned up")


# =============================================================================
# INTEGRATION EXAMPLES FOR DIFFERENT USE CASES
# =============================================================================

class WebAPIIntegration:
    """Example: Integrating with a web API (FastAPI/Flask)"""
    
    def __init__(self):
        self.voice_system = VoiceToTextLibrary()
    
    async def setup(self):
        """Setup for web server"""
        await self.voice_system.initialize()
    
    async def handle_file_upload(self, uploaded_file_path: str) -> Dict[str, Any]:
        """Handle file upload endpoint"""
        try:
            result = await self.voice_system.transcribe_file(uploaded_file_path)
            
            return {
                'success': True,
                'transcription': result['full_text'],
                'confidence': result['confidence'],
                'processing_time': result.get('processing_stats', {}).get('real_time_factor', 0)
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def handle_batch_upload(self, file_paths: List[str]) -> Dict[str, Any]:
        """Handle batch file processing"""
        results = await self.voice_system.transcribe_multiple_files(file_paths)
        
        successful = [r for r in results if r.get('success', True)]
        failed = [r for r in results if not r.get('success', True)]
        
        return {
            'total_files': len(file_paths),
            'successful': len(successful),
            'failed': len(failed),
            'results': successful,
            'errors': failed
        }


class DataPipelineIntegration:
    """Example: Integrating with data processing pipeline"""
    
    def __init__(self, output_format: str = "json"):
        self.voice_system = VoiceToTextLibrary()
        self.output_format = output_format
    
    async def process_audio_folder(self, input_folder: str, output_folder: str):
        """Process all audio files in a folder"""
        await self.voice_system.initialize()
        
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        audio_files = []
        for ext in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']:
            audio_files.extend(input_path.glob(f"**/*{ext}"))
        
        print(f"📂 Found {len(audio_files)} audio files in {input_folder}")
        
        # Process files
        results = []
        for audio_file in audio_files:
            print(f"🎵 Processing: {audio_file.name}")
            
            try:
                result = await self.voice_system.transcribe_file(str(audio_file))
                
                # Prepare output data
                output_data = {
                    'file_name': audio_file.name,
                    'file_path': str(audio_file),
                    'transcription': result['full_text'],
                    'confidence': result['confidence'],
                    'sentences': result['sentences'],
                    'processing_stats': result['processing_stats']
                }
                
                results.append(output_data)
                
                # Save individual result
                output_file = output_path / f"{audio_file.stem}_transcript.{self.output_format}"
                
                if self.output_format == "json":
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                elif self.output_format == "txt":
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result['full_text'])
                
                print(f"✅ Saved: {output_file}")
                
            except Exception as e:
                print(f"❌ Error processing {audio_file}: {e}")
        
        # Save combined results
        combined_file = output_path / f"all_transcriptions.{self.output_format}"
        
        if self.output_format == "json":
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"📊 Processed {len(results)} files, saved to {output_folder}")
        
        await self.voice_system.cleanup()


class RealtimeStreamingIntegration:
    """Example: Real-time streaming audio processing"""
    
    def __init__(self):
        self.voice_system = VoiceToTextLibrary()
        self.is_streaming = False
    
    async def start_streaming(self, audio_source):
        """Start processing streaming audio"""
        await self.voice_system.initialize()
        self.is_streaming = True
        
        print("🎤 Starting real-time transcription...")
        
        async def stream_processor():
            session_id = f"realtime_{int(time.time())}"
            
            async for result in self.voice_system.transcribe_stream(
                audio_source, 
                session_id=session_id,
                chunk_callback=self.on_chunk_processed
            ):
                if not self.is_streaming:
                    break
                
                # Send result to real-time handlers
                await self.on_transcription_ready(result)
        
        await stream_processor()
    
    async def on_chunk_processed(self, chunk_result: Dict[str, Any]):
        """Called when each audio chunk is processed"""
        print(f"📡 Chunk processed: confidence {chunk_result['confidence']:.2f}")
    
    async def on_transcription_ready(self, result: Dict[str, Any]):
        """Called when complete transcription is ready"""
        print(f"📝 Transcription: {result['full_text']}")
        
        # Your custom handling here:
        # - Send to WebSocket clients
        # - Save to database  
        # - Trigger other processes
        # - Send notifications
    
    def stop_streaming(self):
        """Stop the streaming process"""
        self.is_streaming = False
        print("⏹️  Streaming stopped")


# =============================================================================
# COMMAND LINE INTERFACE FOR TESTING
# =============================================================================

async def run_simple_demo():
    """Simple demo showing basic functionality"""
    print("🎯 Simple Voice-to-Text Demo")
    print("="*50)
    
    # Initialize the library
    voice_to_text = VoiceToTextLibrary()
    
    try:
        # Setup
        success = await voice_to_text.initialize()
        if not success:
            print("❌ Failed to initialize system")
            return
        
        # Health check
        health = await voice_to_text.health_check()
        print(f"🏥 System Health: {'✅ Healthy' if health['overall']['healthy'] else '❌ Issues detected'}")
        
        # Look for example audio files
        example_files = [
            "example.wav", "test.mp3", "sample.m4a", "demo.wav",
            "audio.wav", "voice.mp3", "speech.wav"
        ]
        
        audio_file = None
        for file in example_files:
            if Path(file).exists():
                audio_file = file
                break
        
        if not audio_file:
            print("⚠️  No example audio files found.")
            print("   Create an audio file (example.wav, test.mp3, etc.) to test")
            print("   Supported formats: .wav, .mp3, .m4a, .flac, .ogg")
            return
        
        print(f"🎵 Processing: {audio_file}")
        
        # Transcribe
        result = await voice_to_text.transcribe_file(audio_file)
        
        # Display results
        print(f"\n📝 Transcription Results:")
        print(f"   Text: '{result['full_text']}'")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Sentences: {len(result['sentences'])}")
        
        if len(result['sentences']) > 1:
            print(f"   Detected sentences:")
            for i, sentence in enumerate(result['sentences'], 1):
                print(f"     {i}. {sentence}")
        
        # Performance info
        stats = result.get('processing_stats', {})
        if 'real_time_factor' in stats:
            rtf = stats['real_time_factor']
            print(f"   Performance: {rtf:.2f}x real-time ({'⚡ Fast' if rtf < 1 else '🐌 Slow'})")
        
        print(f"\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        
    finally:
        await voice_to_text.cleanup()


async def run_batch_demo(audio_folder: str):
    """Demo batch processing"""
    print("📂 Batch Processing Demo")
    print("="*50)
    
    # Initialize pipeline
    pipeline = DataPipelineIntegration(output_format="json")
    
    output_folder = f"{audio_folder}_transcripts"
    
    print(f"Input folder: {audio_folder}")
    print(f"Output folder: {output_folder}")
    
    await pipeline.process_audio_folder(audio_folder, output_folder)


async def run_api_demo():
    """Demo API integration"""
    print("🌐 API Integration Demo")
    print("="*50)
    
    api = WebAPIIntegration()
    await api.setup()
    
    # Simulate file uploads
    test_files = ["test1.wav", "test2.mp3"]
    existing_files = [f for f in test_files if Path(f).exists()]
    
    if not existing_files:
        print("⚠️  No test files found for API demo")
        return
    
    # Single file
    result = await api.handle_file_upload(existing_files[0])
    print(f"Single file result: {result}")
    
    # Batch files
    if len(existing_files) > 1:
        batch_result = await api.handle_batch_upload(existing_files)
        print(f"Batch result: {batch_result}")


async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Typhoon ASR Library Integration Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python library_demo.py                    # Simple demo
  python library_demo.py --demo batch --audio-dir ./audio_files
  python library_demo.py --demo api
  python library_demo.py --file audio.wav  # Process single file
        """
    )
    
    parser.add_argument("--demo", 
                       choices=["simple", "batch", "api", "performance"],
                       default="simple",
                       help="Demo type to run")
    
    parser.add_argument("--file", 
                       help="Single audio file to process")
    
    parser.add_argument("--audio-dir",
                       help="Directory containing audio files (for batch demo)")
    
    parser.add_argument("--config",
                       choices=["dev", "prod"],
                       default="dev", 
                       help="Configuration preset")
    
    args = parser.parse_args()
    
    if not LIBRARY_AVAILABLE:
        print("❌ Cannot run demo - library not available")
        return 1
    
    try:
        if args.file:
            # Single file processing
            voice_system = VoiceToTextLibrary()
            
            if args.config == "prod":
                voice_system.configure_for_production()
            else:
                voice_system.configure_for_development()
            
            await voice_system.initialize()
            result = await voice_system.transcribe_file(args.file)
            
            print(f"File: {args.file}")
            print(f"Transcription: {result['full_text']}")
            print(f"Confidence: {result['confidence']:.2f}")
            
            await voice_system.cleanup()
        
        elif args.demo == "simple":
            await run_simple_demo()
        
        elif args.demo == "batch":
            if not args.audio_dir:
                print("❌ --audio-dir required for batch demo")
                return 1
            await run_batch_demo(args.audio_dir)
        
        elif args.demo == "api":
            await run_api_demo()
        
        elif args.demo == "performance":
            # Performance test
            voice_system = VoiceToTextLibrary()
            await voice_system.initialize()
            
            stats = await voice_system.get_performance_stats()
            health = await voice_system.health_check()
            
            print("📊 Performance Stats:")
            print(json.dumps(stats, indent=2, default=str))
            
            print("\n🏥 Health Check:")
            print(json.dumps(health, indent=2, default=str))
            
            await voice_system.cleanup()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 0
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    if LIBRARY_AVAILABLE:
        exit_code = asyncio.run(main())
        exit(exit_code)
    else:
        print("\n📋 Library Integration Instructions:")
        print("1. Copy typhoon_asr_events/ folder to your project")
        print("2. Install: pip install torch librosa soundfile nemo-toolkit redis pyyaml")
        print("3. Import: from typhoon_asr_events import TyphoonASRSystem")
        print("4. Use: result = await TyphoonASRSystem().process_audio_file('audio.wav')")
        exit(1)