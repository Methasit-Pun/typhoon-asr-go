#!/usr/bin/env python3
"""
Simple wrapper to use typhoon_asr_events functionality for transcribing sample_voice.wav
"""

import asyncio
import sys
from pathlib import Path
import time

# Import the main inference functionality
try:
    import nemo.collections.asr as nemo_asr
    import torch
    import librosa
    import soundfile as sf
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f" Required dependencies not available: {e}")
    print("Please install: pip install nemo-toolkit[asr] torch librosa soundfile")
    DEPENDENCIES_AVAILABLE = False


class SimpleTyphoonASR:
    """Simple wrapper around Typhoon ASR for easy transcription"""
    
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    async def load_model(self):
        """Load the Typhoon ASR model"""
        if self.model is not None:
            return
        
        print(f" Loading Typhoon ASR model on {self.device}...")
        try:
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name="scb10x/typhoon-asr-realtime",
                map_location=self.device
            )
            print(" Model loaded successfully!")
        except Exception as e:
            print(f" Failed to load model: {e}")
            raise
    
    def prepare_audio(self, input_path, target_sr=16000):
        """Prepare audio for processing"""
        print(f" Preparing audio: {input_path}")
        
        # Load and resample audio
        audio, sr = librosa.load(input_path, sr=target_sr)
        
        # Save processed audio
        output_path = f"processed_{Path(input_path).name}"
        sf.write(output_path, audio, target_sr)
        
        # Get audio info
        audio_info = sf.info(output_path)
        
        print(f"   Duration: {audio_info.duration:.1f}s")
        print(f"   Sample rate: {audio_info.samplerate} Hz")
        
        return output_path, audio_info
    
    async def transcribe(self, audio_file):
        """Transcribe an audio file"""
        await self.load_model()
        
        # Prepare audio
        processed_file, audio_info = self.prepare_audio(audio_file)
        
        try:
            print(" Running transcription...")
            start_time = time.time()
            
            # Run transcription
            transcriptions = self.model.transcribe(audio=[processed_file])
            processing_time = time.time() - start_time
            
            transcription = transcriptions[0] if transcriptions else ""
            
            # Calculate real-time factor
            rtf = processing_time / audio_info.duration if audio_info.duration > 0 else 0
            
            result = {
                'text': transcription,
                'full_text': transcription,
                'sentences': [transcription] if transcription.strip() else [],
                'processing_time': processing_time,
                'audio_duration': audio_info.duration,
                'rtf': rtf,
                'device': self.device,
                'model': 'scb10x/typhoon-asr-realtime'
            }
            
            return result
            
        finally:
            # Cleanup processed file
            if Path(processed_file).exists():
                Path(processed_file).unlink()
                print(f" Cleaned up: {processed_file}")


async def transcribe_sample_voice():
    """Main function to transcribe sample_voice.wav"""
    
    if not DEPENDENCIES_AVAILABLE:
        return
    
    # Check if sample file exists
    audio_file = "data/sample_voice.wav"
    if not Path(audio_file).exists():
        print(f" Audio file '{audio_file}' not found!")
        return
    
    print(" Typhoon ASR Events - Simple Transcription")
    print("=" * 60)
    print(f" File: {audio_file}")
    
    # Initialize ASR system
    asr = SimpleTyphoonASR()
    
    try:
        # Transcribe the audio
        result = await asr.transcribe(audio_file)
        
        # Display results
        print("\n" + "=" * 60)
        print(" TRANSCRIPTION RESULTS")
        print("=" * 60)
        
        print(f" Transcribed Text:")
        print(f"   '{result['text']}'")
        
        print(f"\n Processing Details:")
        print(f"   Audio Duration: {result['audio_duration']:.1f}s")
        print(f"   Processing Time: {result['processing_time']:.2f}s")
        print(f"   RTF: {result['rtf']:.3f}x", end="")
        
        if result['rtf'] < 1.0:
            print("  (Real-time capable!)")
        else:
            print("  (Batch processing)")
        
        print(f"   Device: {result['device']}")
        print(f"   Model: {result['model']}")
        
        if result['sentences']:
            print(f"\n Sentences ({len(result['sentences'])}):")
            for i, sentence in enumerate(result['sentences'], 1):
                print(f"   {i}. {sentence}")
        
        print("\n Transcription completed successfully!")
        
    except Exception as e:
        print(f" Error during transcription: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print(" Typhoon ASR Events Transcription Tool")
    print("=" * 60)
    
    # Run the transcription
    asyncio.run(transcribe_sample_voice())