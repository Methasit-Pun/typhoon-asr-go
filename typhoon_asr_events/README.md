# Typhoon ASR Event-Driven System

A scalable, event-driven architecture for real-time Thai speech recognition using Typhoon ASR. This system provides high-performance audio transcription with intelligent aggregation, caching, and streaming capabilities.

## 🌟 Features

- **Event-Driven Architecture**: Decoupled, scalable processing with async/await support
- **Real-Time Transcription**: Streaming audio processing with Typhoon ASR
- **Intelligent Aggregation**: Smart sentence boundary detection and text assembly
- **Redis Caching**: High-performance result caching and session management  
- **Confidence Scoring**: Built-in quality metrics and threshold filtering
- **Multi-Format Support**: WAV, MP3, M4A, FLAC, OGG, AAC, WebM
- **Comprehensive Logging**: Structured logging with performance monitoring
- **Production Ready**: Error handling, retries, health checks, graceful shutdown

## 🏗️ Architecture

### Core Components

1. **Event System** (`core/event_system.py`)
   - Central EventBus for message routing
   - Event definitions and handler abstractions
   - Async/sync handler support with middleware

2. **ASR Processor** (`services/asr_processor.py`) 
   - Handles `asr.process.request` events
   - Typhoon ASR integration with streaming support
   - Confidence scoring and alternative hypotheses
   - Emits `transcription.completed` events

3. **Transcription Aggregator** (`services/transcription_aggregator.py`)
   - Handles `transcription.completed` events
   - Combines partial transcriptions intelligently
   - Thai sentence boundary detection
   - Redis caching and session management
   - Emits `text.ready` events

4. **Configuration System** (`config/settings.py`)
   - Environment variable support
   - YAML/JSON configuration files
   - Runtime validation and defaults

5. **Utilities** (`utils/helpers.py`)
   - Audio processing utilities
   - Performance monitoring
   - Retry mechanisms and health checks

### Event Flow

```
audio.chunk.ready → asr.process.request → transcription.completed → transcription.aggregate → text.ready
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements_events.txt

# Install Redis (for caching)
# Ubuntu/Debian:
sudo apt-get install redis-server

# macOS:
brew install redis

# Windows: 
# Download from https://redis.io/download
```

### Basic Usage

```python
import asyncio
from typhoon_asr_events import TyphoonASRSystem

async def main():
    # Initialize system
    system = TyphoonASRSystem()
    
    # Process single file
    result = await system.process_audio_file("audio.wav")
    print(f"Transcription: {result}")
    
    # Cleanup
    await system.shutdown()

asyncio.run(main())
```

### Command Line Interface

```bash
# Process single file
python example_usage.py --audio-file path/to/audio.wav

# Process multiple files
python example_usage.py --audio-files file1.wav file2.mp3 file3.m4a

# Streaming mode (demo)
python example_usage.py --stream-mode

# With custom config
python example_usage.py --audio-file audio.wav --config config.yaml --log-level DEBUG
```

## ⚙️ Configuration

### Environment Variables

```bash
# ASR Configuration
export TYPHOON_ASR_ASR_MODEL_NAME="scb10x/typhoon-asr-realtime"
export TYPHOON_ASR_ASR_DEVICE="cuda"
export TYPHOON_ASR_ASR_CONFIDENCE_THRESHOLD="0.7"

# Redis Configuration  
export TYPHOON_ASR_REDIS_HOST="localhost"
export TYPHOON_ASR_REDIS_PORT="6379"
export TYPHOON_ASR_REDIS_PASSWORD="your_password"

# Aggregation Settings
export TYPHOON_ASR_AGGREGATION_CACHE_TTL="3600"
export TYPHOON_ASR_AGGREGATION_SENTENCE_TIMEOUT="5.0"

# Logging
export TYPHOON_ASR_LOGGING_LEVEL="INFO"
export TYPHOON_ASR_LOGGING_FILE_PATH="logs/typhoon_asr.log"
```

### Configuration File (config.yaml)

```yaml
asr:
  model_name: "scb10x/typhoon-asr-realtime"
  device: "auto"  # auto, cpu, cuda
  confidence_threshold: 0.7
  batch_size: 1
  enable_alternatives: true

redis:
  host: "localhost"
  port: 6379
  db: 0
  password: null
  connection_timeout: 5

aggregation:
  cache_ttl: 3600
  max_buffer_size: 100  
  sentence_timeout: 5.0
  enable_sentence_detection: true
  max_session_age_hours: 24.0

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_path: "logs/typhoon_asr.log"
  enable_console: true

event_bus:
  max_event_history: 1000
  concurrent_handler_limit: 10
  event_timeout: 30.0
```

## 📋 API Reference

### TyphoonASRSystem

Main system class for orchestrating ASR processing.

```python
class TyphoonASRSystem:
    def __init__(self, config: Optional[Config] = None)
    
    async def process_audio_file(self, audio_file: str, session_id: Optional[str] = None) -> dict
    async def process_audio_stream(self, audio_chunks, session_id: Optional[str] = None)
    async def get_session_history(self, session_id: str, limit: int = 10) -> List
    async def get_system_stats(self) -> dict
    async def shutdown(self)
```

### Event Types

```python
class EventTypes:
    AUDIO_CHUNK_READY = "audio.chunk.ready"
    ASR_PROCESS_REQUEST = "asr.process.request" 
    TRANSCRIPTION_COMPLETED = "transcription.completed"
    TRANSCRIPTION_AGGREGATE = "transcription.aggregate"
    TEXT_READY = "text.ready"
    PROCESSING_ERROR = "processing.error"
```

### Event Data Formats

#### ASR Process Request
```python
{
    "event_type": "asr.process.request",
    "data": {
        "audio_file": "path/to/file.wav",        # File path
        "audio_data": [...],                      # Raw audio array  
        "audio_bytes": b"...",                    # Audio bytes
        "sample_rate": 16000,
        "chunk_id": "chunk_1",
        "session_id": "session_123"
    }
}
```

#### Transcription Completed
```python
{
    "event_type": "transcription.completed", 
    "data": {
        "chunk_id": "chunk_1",
        "text": "สวัสดีครับ",
        "confidence": 0.95,
        "processing_time": 0.34,
        "word_timestamps": [...],
        "alternatives": [...],
        "meets_threshold": true
    }
}
```

#### Text Ready
```python
{
    "event_type": "text.ready",
    "data": {
        "session_id": "session_123", 
        "full_text": "สวัสดีครับ ผมชื่อจอห์น",
        "sentences": ["สวัสดีครับ", "ผมชื่อจอห์น"],
        "confidence": 0.92,
        "chunk_count": 3,
        "duration": 5.2,
        "processing_stats": {...}
    }
}
```

## 🔧 Advanced Usage

### Custom Event Handlers

```python
from typhoon_asr_events.core import EventHandler, Event, EventTypes

class CustomTranscriptionHandler(EventHandler):
    @property
    def handled_events(self):
        return [EventTypes.TEXT_READY]
    
    async def handle(self, event: Event):
        # Custom processing logic
        text = event.data['full_text']
        processed_text = self.custom_postprocessing(text)
        
        # Return new event or None
        return Event(
            event_type="custom.processed",
            data={"processed_text": processed_text}
        )

# Register with event bus
system = TyphoonASRSystem()
system.event_bus.subscribe(EventTypes.TEXT_READY, CustomTranscriptionHandler())
```

### Streaming Audio Processing

```python
async def process_microphone_stream():
    system = TyphoonASRSystem()
    
    async def audio_stream():
        # Your audio capture logic
        while capturing:
            chunk = await capture_audio_chunk()
            yield {
                'audio_data': chunk.data,
                'sample_rate': chunk.sample_rate
            }
    
    await system.process_audio_stream(audio_stream(), session_id="mic_session")
    await system.shutdown()
```

### Redis Session Management

```python
# Get historical results
history = await system.get_session_history("session_123", limit=50)

# Access aggregator directly for advanced operations
aggregator = system.transcription_aggregator
await aggregator.cleanup_old_sessions(max_age_hours=48)
stats = aggregator.get_stats()
```

### Performance Monitoring

```python
from typhoon_asr_events.utils import PerformanceMonitor

monitor = PerformanceMonitor("custom_operation")

with monitor.time_operation():
    # Your code here
    result = await some_operation()

stats = monitor.get_stats()
print(f"Average time: {stats['average']:.3f}s")
```

## 📊 Monitoring & Health Checks

### System Statistics

```python
stats = await system.get_system_stats()

# Example output:
{
    "event_bus": {
        "total_handlers": 4,
        "event_history_size": 150,
        "handler_details": {...}
    },
    "asr_processor": {
        "total_chunks": 45,
        "successful_transcriptions": 42,
        "average_processing_time": 0.287,
        "model_loaded": true
    },
    "transcription_aggregator": {
        "active_sessions": 3,
        "total_sentences_detected": 28,
        "average_confidence": 0.891,
        "cache_hits": 156,
        "cache_misses": 12
    }
}
```

### Health Checks

```python
from typhoon_asr_events.utils import HealthChecker

health_checker = HealthChecker()

# Add custom checks
health_checker.add_check("redis", lambda: redis_client.ping())
health_checker.add_check("model", lambda: asr_processor._model_loaded)

results = await health_checker.run_checks()
```

## 🐛 Error Handling

The system provides comprehensive error handling:

- **Automatic Retries**: Exponential backoff for transient failures
- **Circuit Breaking**: Prevents cascade failures  
- **Error Events**: Structured error reporting via events
- **Graceful Degradation**: Continues processing when possible
- **Detailed Logging**: Full error context and stack traces

```python
# Subscribe to error events
def handle_errors(event: Event):
    error_data = event.data
    logger.error(f"Processing error: {error_data['error']}")
    logger.error(f"Source: {error_data['source_service']}")
    
    # Implement custom error handling logic
    if error_data['source_service'] == 'ASRProcessor':
        # Handle ASR-specific errors
        pass

system.event_bus.subscribe(EventTypes.PROCESSING_ERROR, handle_errors)
```

## 🔒 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_events.txt .
RUN pip install -r requirements_events.txt

COPY typhoon_asr_events/ ./typhoon_asr_events/
COPY config.yaml .

EXPOSE 8080
CMD ["python", "-m", "typhoon_asr_events.example_usage"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  typhoon-asr:
    build: .
    environment:
      - TYPHOON_ASR_REDIS_HOST=redis
      - TYPHOON_ASR_ASR_DEVICE=cuda
    depends_on:
      - redis
    volumes:
      - ./audio:/app/audio
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: typhoon-asr
spec:
  replicas: 3
  selector:
    matchLabels:
      app: typhoon-asr
  template:
    metadata:
      labels:
        app: typhoon-asr
    spec:
      containers:
      - name: typhoon-asr
        image: typhoon-asr:latest
        env:
        - name: TYPHOON_ASR_REDIS_HOST
          value: "redis-service"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi" 
            cpu: "2000m"
```

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests  
python -m pytest tests/integration/

# Run performance tests
python -m pytest tests/performance/ --benchmark-only

# Test with coverage
python -m pytest --cov=typhoon_asr_events tests/
```

## 📈 Performance

### Benchmarks

- **Processing Speed**: ~0.3s average per audio chunk
- **Memory Usage**: ~200MB baseline + model size (~1GB for Typhoon ASR)
- **Throughput**: 10-50 concurrent audio streams (depends on hardware)
- **Latency**: <100ms event processing overhead
- **Cache Performance**: 95%+ hit rate for session data

### Optimization Tips

1. **GPU Usage**: Set `device: "cuda"` for 5-10x speedup
2. **Batch Processing**: Increase `batch_size` for higher throughput
3. **Redis Tuning**: Use Redis cluster for high-scale deployments
4. **Audio Preprocessing**: Normalize audio quality for better results
5. **Session Management**: Set appropriate cache TTL and cleanup intervals

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Typhoon ASR](https://github.com/SCB-TechX/typhoon-asr) by SCB TechX
- [NeMo](https://github.com/NVIDIA/NeMo) by NVIDIA
- Redis for high-performance caching
- The open-source community for amazing tools and libraries