# Quick Setup Guide - Typhoon ASR Events Library

## 🚀 1-Minute Setup

### Copy Library to Your Project
```bash
# Copy the library folder to your project
cp -r typhoon_asr_events/ /path/to/your/project/

# Or download/clone and copy manually
```

### Install Dependencies  
```bash
pip install torch librosa soundfile nemo-toolkit redis pyyaml numpy
```

### Basic Usage
```python
import asyncio
from typhoon_asr_events import TyphoonASRSystem

async def main():
    # Initialize system
    system = TyphoonASRSystem()
    
    # Convert voice to text
    result = await system.process_audio_file("your_audio.wav")
    print(f"Transcription: {result['full_text']}")
    
    # Clean up
    await system.shutdown()

# Run it
asyncio.run(main())
```

## 🎯 Quick Examples

### Single File Processing
```python
from typhoon_asr_events import transcribe_file

# Simple one-liner
text = await transcribe_file("meeting.wav")
print(text)
```

### Batch Processing
```python
from typhoon_asr_events import transcribe_files

files = ["audio1.wav", "audio2.mp3", "audio3.m4a"]
results = await transcribe_files(files)

for result in results:
    print(f"{result['file']}: {result['text']}")
```

### With Custom Configuration
```python
from typhoon_asr_events import TyphoonASRSystem, Config

# Configure for production
config = Config()
config.asr.device = "cuda"  # Use GPU
config.asr.confidence_threshold = 0.8  # Higher quality

system = TyphoonASRSystem(config)
result = await system.process_audio_file("important_audio.wav")
```

### Streaming Processing
```python
async def process_stream():
    system = TyphoonASRSystem()
    
    # Your audio stream source
    async def audio_chunks():
        for file in ["chunk1.wav", "chunk2.wav", "chunk3.wav"]:
            yield {'audio_file': file}
    
    # Process stream
    async for result in system.process_audio_stream(audio_chunks()):
        print(f"Partial result: {result['full_text']}")
    
    await system.shutdown()
```

## 🔧 Configuration Options

### Environment Variables
```bash
export TYPHOON_ASR_ASR_DEVICE="cuda"
export TYPHOON_ASR_REDIS_HOST="localhost"
export TYPHOON_ASR_LOGGING_LEVEL="INFO"
```

### Configuration File (config.yaml)
```yaml
asr:
  device: "cuda"
  confidence_threshold: 0.8
  
redis:
  host: "localhost" 
  port: 6379

logging:
  level: "INFO"
```

### Programmatic Configuration
```python
from typhoon_asr_events import Config, TyphoonASRSystem

config = Config()
config.asr.device = "cuda"
config.asr.confidence_threshold = 0.9

system = TyphoonASRSystem(config)
```

## 📁 Supported Audio Formats

✅ **Supported**: WAV, MP3, M4A, FLAC, OGG, AAC, WebM
❌ **Not supported**: Video files, raw PCM without headers

## 🐛 Troubleshooting

### Common Issues

1. **"Module not found"**
   - Make sure `typhoon_asr_events` folder is in your project directory
   - Check Python path: `sys.path.append('/path/to/folder')`

2. **"Model loading failed"**
   - Install NeMo toolkit: `pip install nemo-toolkit[asr]`
   - Check internet connection (downloads model on first use)

3. **"CUDA not available"** 
   - Set `config.asr.device = "cpu"` for CPU-only processing
   - Install PyTorch with CUDA support for GPU acceleration

4. **"Redis connection failed"**
   - Install Redis: `pip install redis` + Redis server
   - Or disable Redis by not setting redis configuration

### Performance Tips

- **Use GPU**: Set `device="cuda"` for 5-10x speedup
- **Batch processing**: Process multiple files concurrently
- **Audio quality**: Better audio = better transcription accuracy
- **Confidence threshold**: Adjust based on your quality needs

## 🚀 Integration Examples

### Web API (FastAPI)
```python
from fastapi import FastAPI, UploadFile
from typhoon_asr_events import TyphoonASRSystem

app = FastAPI()
asr_system = TyphoonASRSystem()

@app.post("/transcribe")
async def transcribe(audio: UploadFile):
    # Save file temporarily
    with open(audio.filename, "wb") as f:
        f.write(await audio.read())
    
    # Transcribe
    result = await asr_system.process_audio_file(audio.filename)
    
    return {"transcription": result['full_text']}
```

### Data Pipeline
```python
from pathlib import Path
from typhoon_asr_events import TyphoonASRSystem

async def process_audio_folder(input_folder, output_folder):
    system = TyphoonASRSystem()
    
    # Find audio files
    audio_files = []
    for ext in ['.wav', '.mp3', '.m4a']:
        audio_files.extend(Path(input_folder).glob(f"**/*{ext}"))
    
    # Process each file
    results = []
    for audio_file in audio_files:
        result = await system.process_audio_file(str(audio_file))
        results.append(result)
    
    # Save results
    import json
    with open(f"{output_folder}/transcriptions.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    await system.shutdown()
```

### Message Queue Integration
```python
# Celery task
from celery import Celery
from typhoon_asr_events import transcribe_file

app = Celery('transcription')

@app.task
def transcribe_async(audio_file_path):
    import asyncio
    return asyncio.run(transcribe_file(audio_file_path))

# Usage
result = transcribe_async.delay("audio.wav")
transcription = result.get()
```

## 📞 Need Help?

1. **Check the logs** - Enable debug logging: `config.logging.level = "DEBUG"`
2. **Test with simple files** - Try with a short, clear WAV file first  
3. **Check system resources** - Ensure enough RAM/GPU memory
4. **Validate audio files** - Use `AudioUtils.validate_audio_format()`

## 🎉 You're Ready!

The library is designed to be drop-in ready. Just copy, install dependencies, and start transcribing!

```python
# Minimal example - this should work out of the box
import asyncio
from typhoon_asr_events import transcribe_file

async def test():
    text = await transcribe_file("test_audio.wav")
    print(f"Got: {text}")

asyncio.run(test())
```