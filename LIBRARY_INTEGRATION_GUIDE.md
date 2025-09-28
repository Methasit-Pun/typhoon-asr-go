# Typhoon ASR Events Library - Integration Guide

A plug-and-play library for integrating SCB10X Typhoon ASR into any Python application or pipeline. Simply drop the `typhoon_asr_events` folder into your project and start converting voice to text with powerful event-driven processing.

## 📦 Library Integration

### Quick Setup

1. **Copy the library folder** into your project:
```bash
cp -r typhoon_asr_events/ /path/to/your/project/
```

2. **Install dependencies**:
```bash
pip install -r typhoon_asr_events/requirements.txt
# Or install specific packages:
pip install torch librosa soundfile nemo-toolkit redis pyyaml numpy
```

3. **Start using** immediately:
```python
from typhoon_asr_events import TyphoonASRSystem
```

## 🚀 Function Reference Guide

### 1. TyphoonASRSystem - Main Library Class

The primary interface for all voice-to-text operations.

#### `__init__(config=None)`
Initialize the ASR system with optional configuration.

```python
from typhoon_asr_events import TyphoonASRSystem
from typhoon_asr_events.config import Config

# Default configuration
system = TyphoonASRSystem()

# Custom configuration
config = Config()
config.asr.model_name = "scb10x/typhoon-asr-realtime"
config.asr.device = "cuda"  # Use GPU if available
config.asr.confidence_threshold = 0.8
system = TyphoonASRSystem(config)
```

#### `process_audio_file(audio_file, session_id=None)`
Convert a single audio file to text.

```python
async def transcribe_file():
    system = TyphoonASRSystem()
    
    # Basic usage
    result = await system.process_audio_file("meeting_audio.wav")
    print(f"Transcription: {result['full_text']}")
    
    # With session tracking
    result = await system.process_audio_file(
        "customer_call.mp3", 
        session_id="call_12345"
    )
    
    await system.shutdown()

# Run it
import asyncio
asyncio.run(transcribe_file())
```

**Return Format:**
```python
{
    'session_id': 'session_123',
    'full_text': 'สวัสดีครับ ผมชื่อจอห์น',
    'sentences': ['สวัสดีครับ', 'ผมชื่อจอห์น'],
    'confidence': 0.92,
    'chunk_count': 3,
    'duration': 5.2,
    'processing_stats': {...}
}
```

#### `process_audio_stream(audio_chunks, session_id=None)`
Process streaming audio data in real-time.

```python
async def stream_transcription():
    system = TyphoonASRSystem()
    
    # Your audio stream generator
    async def audio_generator():
        # Example: reading from microphone, socket, etc.
        for chunk_file in ["chunk1.wav", "chunk2.wav", "chunk3.wav"]:
            yield {
                'audio_file': chunk_file,
                'chunk_id': f"chunk_{chunk_file}"
            }
    
    # Process the stream
    async for result in system.process_audio_stream(
        audio_generator(), 
        session_id="livestream_001"
    ):
        print(f"Partial: {result['full_text']}")
    
    await system.shutdown()
```

#### `get_session_history(session_id, limit=10)`
Retrieve historical transcriptions for a session.

```python
async def get_history():
    system = TyphoonASRSystem()
    
    # Get last 10 transcriptions for a session
    history = await system.get_session_history("call_12345", limit=10)
    
    for item in history:
        print(f"Time: {item.start_time}")
        print(f"Text: {item.full_text}")
        print(f"Confidence: {item.average_confidence}")
    
    await system.shutdown()
```

#### `get_system_stats()`
Get performance statistics and system health.

```python
async def check_performance():
    system = TyphoonASRSystem()
    
    stats = await system.get_system_stats()
    
    print(f"Processed chunks: {stats['asr_processor']['total_chunks']}")
    print(f"Average processing time: {stats['asr_processor']['average_processing_time']:.2f}s")
    print(f"Active sessions: {stats['transcription_aggregator']['active_sessions']}")
    
    await system.shutdown()
```

### 2. Event-Driven Processing

For advanced integrations, use the event system directly.

#### EventBus - Message Router

```python
from typhoon_asr_events.core import EventBus, Event, EventTypes

# Create event bus
event_bus = EventBus()

# Subscribe to events
def handle_transcription(event):
    if event.event_type == EventTypes.TEXT_READY:
        text = event.data['full_text']
        print(f"New transcription: {text}")
        
        # Your custom processing
        save_to_database(text)
        send_to_api(text)

event_bus.subscribe(EventTypes.TEXT_READY, handle_transcription)

# Publish events
event = Event(
    event_type=EventTypes.ASR_PROCESS_REQUEST,
    data={'audio_file': 'input.wav'}
)
await event_bus.publish(event)
```

#### ASRProcessor - Core Transcription Engine

```python
from typhoon_asr_events.services import ASRProcessor
from typhoon_asr_events.core import EventBus, Event, EventTypes

# Initialize processor
asr_processor = ASRProcessor(
    model_name="scb10x/typhoon-asr-realtime",
    device="cuda",
    confidence_threshold=0.8
)

# Set up event system
event_bus = EventBus()
event_bus.subscribe(EventTypes.ASR_PROCESS_REQUEST, asr_processor)

# Process audio
event = Event(
    event_type=EventTypes.ASR_PROCESS_REQUEST,
    data={
        'audio_file': 'speech.wav',
        'chunk_id': 'chunk_001'
    }
)

results = await event_bus.publish(event)
# Results contain transcription.completed events
```

#### TranscriptionAggregator - Text Assembly

```python
from typhoon_asr_events.services import TranscriptionAggregator

# Initialize aggregator with Redis
aggregator = TranscriptionAggregator(
    redis_host="localhost",
    redis_port=6379,
    sentence_timeout=5.0
)

# Subscribe to transcription events
event_bus.subscribe(EventTypes.TRANSCRIPTION_COMPLETED, aggregator)

# The aggregator automatically combines partial results
# and emits TEXT_READY events when complete
```

### 3. Configuration System

Flexible configuration for different environments.

#### Using Environment Variables

```python
import os

# Set configuration via environment
os.environ['TYPHOON_ASR_ASR_DEVICE'] = 'cuda'
os.environ['TYPHOON_ASR_REDIS_HOST'] = 'redis-server.com'
os.environ['TYPHOON_ASR_LOGGING_LEVEL'] = 'DEBUG'

# Load automatically
from typhoon_asr_events.config import Config
config = Config.from_env()

system = TyphoonASRSystem(config)
```

#### Using Configuration Files

```python
# config.yaml
"""
asr:
  model_name: "scb10x/typhoon-asr-realtime"
  device: "cuda"
  confidence_threshold: 0.9

redis:
  host: "prod-redis.example.com"
  port: 6379
  password: "secret"

logging:
  level: "INFO"
  file_path: "logs/asr.log"
"""

# Load config
config = Config.from_file("config.yaml")
system = TyphoonASRSystem(config)
```

### 4. Audio Utilities

Helper functions for audio processing.

#### AudioUtils

```python
from typhoon_asr_events.utils import AudioUtils

# Validate audio format
is_valid = AudioUtils.validate_audio_format("audio.mp3")  # True

# Get audio information
info = AudioUtils.get_audio_info("meeting.wav")
print(f"Duration: {info['duration']} seconds")
print(f"Sample rate: {info['sample_rate']} Hz")

# Normalize audio
import numpy as np
normalized = AudioUtils.normalize_audio(audio_array, target_db=-20.0)

# Split into chunks
chunks = AudioUtils.split_audio_chunks(
    audio_array, 
    sample_rate=16000,
    chunk_duration=10.0,
    overlap_duration=1.0
)
```

### 5. Performance Monitoring

Track system performance and health.

#### PerformanceMonitor

```python
from typhoon_asr_events.utils import PerformanceMonitor

monitor = PerformanceMonitor("transcription_pipeline")

# Time operations
with monitor.time_operation():
    result = await system.process_audio_file("large_file.wav")

# Get statistics
stats = monitor.get_stats()
print(f"Average time: {stats['average']:.3f}s")
print(f"Total operations: {stats['count']}")
```

#### HealthChecker

```python
from typhoon_asr_events.utils import HealthChecker

health = HealthChecker()

# Add custom health checks
health.add_check("redis", lambda: redis_client.ping())
health.add_check("model", lambda: asr_processor.is_model_loaded())
health.add_check("disk_space", lambda: get_disk_usage() < 90)

# Run checks
results = await health.run_checks()
if results['overall']['healthy']:
    print("System is healthy")
```

## 🔧 Integration Patterns

### 1. Web API Integration

```python
from fastapi import FastAPI, UploadFile
from typhoon_asr_events import TyphoonASRSystem

app = FastAPI()
asr_system = TyphoonASRSystem()

@app.post("/transcribe")
async def transcribe_endpoint(audio: UploadFile):
    # Save uploaded file
    with open(f"temp_{audio.filename}", "wb") as f:
        f.write(await audio.read())
    
    # Transcribe
    result = await asr_system.process_audio_file(f"temp_{audio.filename}")
    
    return {
        "transcription": result['full_text'],
        "confidence": result['confidence'],
        "duration": result['duration']
    }

@app.on_event("shutdown")
async def shutdown():
    await asr_system.shutdown()
```

### 2. Message Queue Integration

```python
import asyncio
from celery import Celery
from typhoon_asr_events import TyphoonASRSystem

# Celery task
celery_app = Celery('transcription')

@celery_app.task
def transcribe_async(audio_file_path, session_id=None):
    async def process():
        system = TyphoonASRSystem()
        result = await system.process_audio_file(audio_file_path, session_id)
        await system.shutdown()
        return result
    
    return asyncio.run(process())

# Usage
result = transcribe_async.delay("audio.wav", "session_123")
transcription = result.get()
```

### 3. Streaming Pipeline Integration

```python
import asyncio
from typhoon_asr_events import TyphoonASRSystem

class AudioPipeline:
    def __init__(self):
        self.asr_system = TyphoonASRSystem()
    
    async def process_stream(self, audio_stream):
        """Process continuous audio stream"""
        
        async def chunk_generator():
            async for audio_chunk in audio_stream:
                yield {
                    'audio_data': audio_chunk.data,
                    'sample_rate': audio_chunk.sample_rate,
                    'chunk_id': audio_chunk.id
                }
        
        # Process stream and get real-time results
        async for result in self.asr_system.process_audio_stream(
            chunk_generator(), 
            session_id="live_stream"
        ):
            # Send to downstream processors
            await self.send_to_nlp(result['full_text'])
            await self.save_to_database(result)
            await self.notify_clients(result)
    
    async def send_to_nlp(self, text):
        # Your NLP processing
        pass
    
    async def save_to_database(self, result):
        # Your database logic
        pass
    
    async def notify_clients(self, result):
        # WebSocket notifications, etc.
        pass

# Usage
pipeline = AudioPipeline()
await pipeline.process_stream(microphone_stream)
```

### 4. Batch Processing Integration

```python
import asyncio
from pathlib import Path
from typhoon_asr_events import TyphoonASRSystem

async def batch_transcribe_folder(folder_path, output_file="transcriptions.json"):
    """Transcribe all audio files in a folder"""
    
    system = TyphoonASRSystem()
    results = []
    
    # Find all audio files
    audio_files = []
    for ext in ['.wav', '.mp3', '.m4a', '.flac']:
        audio_files.extend(Path(folder_path).glob(f"**/*{ext}"))
    
    # Process files concurrently
    semaphore = asyncio.Semaphore(5)  # Limit concurrent processing
    
    async def process_file(file_path):
        async with semaphore:
            result = await system.process_audio_file(str(file_path))
            return {
                'file': str(file_path),
                'transcription': result['full_text'],
                'confidence': result['confidence'],
                'duration': result['duration']
            }
    
    # Process all files
    tasks = [process_file(f) for f in audio_files]
    results = await asyncio.gather(*tasks)
    
    # Save results
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    await system.shutdown()
    return results

# Usage
results = await batch_transcribe_folder("audio_files/", "results.json")
```

### 5. Custom Event Handler Integration

```python
from typhoon_asr_events.core import EventHandler, Event, EventTypes
from typhoon_asr_events import TyphoonASRSystem

class CustomTranscriptionProcessor(EventHandler):
    """Custom processor for transcription results"""
    
    @property
    def handled_events(self):
        return [EventTypes.TEXT_READY]
    
    async def handle(self, event: Event):
        # Extract transcription
        text = event.data['full_text']
        confidence = event.data['confidence']
        
        # Your custom processing
        if confidence > 0.9:
            # High confidence - process immediately
            await self.process_high_confidence(text)
        elif confidence > 0.7:
            # Medium confidence - flag for review
            await self.flag_for_review(text)
        else:
            # Low confidence - request human transcription
            await self.request_human_transcription(event.data)
        
        # Return processed event
        return Event(
            event_type="custom.transcription.processed",
            data={
                'original_text': text,
                'processed': True,
                'confidence_category': self.get_confidence_category(confidence)
            }
        )
    
    async def process_high_confidence(self, text):
        # Your processing logic
        pass
    
    async def flag_for_review(self, text):
        # Your review logic
        pass
    
    async def request_human_transcription(self, data):
        # Your human transcription request logic
        pass
    
    def get_confidence_category(self, confidence):
        if confidence > 0.9:
            return "high"
        elif confidence > 0.7:
            return "medium"
        else:
            return "low"

# Integration
system = TyphoonASRSystem()
custom_processor = CustomTranscriptionProcessor()

# Subscribe to events
system.event_bus.subscribe(EventTypes.TEXT_READY, custom_processor)

# Process as normal - your custom handler will be called automatically
result = await system.process_audio_file("audio.wav")
```

## 🛠️ Configuration Examples

### Development Configuration

```python
from typhoon_asr_events.config import Config

# Development setup
config = Config()
config.asr.device = "cpu"  # No GPU required
config.asr.confidence_threshold = 0.5  # Lower threshold for testing
config.redis.host = "localhost"
config.logging.level = "DEBUG"
config.logging.enable_console = True

system = TyphoonASRSystem(config)
```

### Production Configuration

```python
# Production setup
config = Config()
config.asr.device = "cuda"  # Use GPU
config.asr.confidence_threshold = 0.8  # Higher quality
config.asr.batch_size = 4  # Better throughput
config.redis.host = "redis-cluster.prod.com"
config.redis.password = "secure_password"
config.aggregation.cache_ttl = 7200  # 2 hours
config.logging.level = "INFO"
config.logging.file_path = "/var/log/typhoon_asr.log"

system = TyphoonASRSystem(config)
```

### Docker Environment

```yaml
# docker-compose.yml
version: '3.8'
services:
  your-app:
    build: .
    environment:
      - TYPHOON_ASR_REDIS_HOST=redis
      - TYPHOON_ASR_ASR_DEVICE=cuda
      - TYPHOON_ASR_LOGGING_LEVEL=INFO
    volumes:
      - ./audio_files:/app/audio
    depends_on:
      - redis
  
  redis:
    image: redis:7-alpine
```

## 🎯 Quick Start Examples

### 1. Simple Voice-to-Text

```python
import asyncio
from typhoon_asr_events import TyphoonASRSystem

async def simple_transcribe():
    system = TyphoonASRSystem()
    result = await system.process_audio_file("voice_note.wav")
    print(result['full_text'])
    await system.shutdown()

asyncio.run(simple_transcribe())
```

### 2. Batch Processing

```python
import asyncio
from typhoon_asr_events import TyphoonASRSystem

async def batch_process():
    system = TyphoonASRSystem()
    
    files = ["meeting1.wav", "meeting2.mp3", "call.m4a"]
    
    for file in files:
        result = await system.process_audio_file(file)
        print(f"{file}: {result['full_text']}")
    
    await system.shutdown()

asyncio.run(batch_process())
```

### 3. With Custom Configuration

```python
import asyncio
from typhoon_asr_events import TyphoonASRSystem
from typhoon_asr_events.config import Config

async def custom_config_example():
    # Custom configuration
    config = Config()
    config.asr.confidence_threshold = 0.9  # Higher quality
    config.asr.device = "cuda"  # Use GPU
    
    system = TyphoonASRSystem(config)
    result = await system.process_audio_file("important_audio.wav")
    
    print(f"Text: {result['full_text']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    await system.shutdown()

asyncio.run(custom_config_example())
```

## 🚀 Getting Started Checklist

- [ ] Copy `typhoon_asr_events/` folder to your project
- [ ] Install requirements: `pip install -r typhoon_asr_events/requirements.txt`
- [ ] (Optional) Install Redis: `pip install redis` + Redis server
- [ ] Import the library: `from typhoon_asr_events import TyphoonASRSystem`
- [ ] Create system instance: `system = TyphoonASRSystem()`
- [ ] Process audio: `result = await system.process_audio_file("audio.wav")`
- [ ] Clean up: `await system.shutdown()`

## 📞 Support

- **Issues**: Check function documentation and error messages
- **Performance**: Use GPU with `config.asr.device = "cuda"`
- **Memory**: Adjust `batch_size` and `cache_ttl` settings
- **Scaling**: Use Redis cluster and multiple worker processes

Ready to integrate SCB10X Typhoon ASR into your pipeline! 🎉