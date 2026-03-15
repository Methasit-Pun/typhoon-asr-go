#!/usr/bin/env python3
"""
Simple script to transcribe sample_voice.wav using typhoon_asr_events
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from typhoon_asr_events import TyphoonASREventSystem, MinimalConfig


async def transcribe_sample_voice():
    """Transcribe the sample_voice.wav file and print results."""
    
    # Path to the sample voice file
    audio_file = "data/sample_voice.wav"
    
    if not Path(audio_file).exists():
        print(f" Error: Audio file '{audio_file}' not found!")
        return
    
    print(f" Starting transcription of: {audio_file}")
    print("=" * 60)
    
    # Initialize the ASR system with minimal config
    config = MinimalConfig()
    system = TyphoonASREventSystem(config)
    
    try:
        # Process the audio file
        print(" Processing audio file...")
        result = await system.process_audio_file(audio_file, session_id="sample_session")
        
        print("\n Transcription completed successfully!")
        print("=" * 60)
        
        # Print the transcribed text
        print("� Transcribed Text:")
        print(f"   {result.get('text', result.get('full_text', 'No text found'))}")
        
        # Print additional information if available
        if 'sentences' in result and result['sentences']:
            print(f"\n Sentences ({len(result['sentences'])}):")
            for i, sentence in enumerate(result['sentences'], 1):
                print(f"   {i}. {sentence}")
        
        if 'confidence' in result:
            print(f"\n Confidence Score: {result['confidence']:.2f}")
        
        if 'processing_time' in result:
            print(f" Processing Time: {result['processing_time']:.2f} seconds")
        
        if 'aggregated' in result and result['aggregated']:
            print(" Result includes aggregated data from session")
    
    except Exception as e:
        print(f" Error during transcription: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean shutdown
        await system.shutdown()
        print("\n System shutdown completed")


if __name__ == "__main__":
    print(" Typhoon ASR Transcription Tool")
    print("=" * 60)
    
    # Run the transcription
    asyncio.run(transcribe_sample_voice())