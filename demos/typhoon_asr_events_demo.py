#!/usr/bin/env python3
"""
Simple events-based wrapper for Typhoon ASR transcription using the existing infrastructure
"""

import subprocess
import sys
from pathlib import Path
import json
import re


class TyphoonASREvents:
    """
    Simple event-driven wrapper around the existing typhoon_asr_inference.py
    This demonstrates how typhoon_asr_events can work by wrapping existing functionality
    """
    
    def __init__(self):
        self.sessions = {}
        self.results_history = []
    
    async def process_audio_file(self, audio_file, session_id=None):
        """
        Process an audio file using the existing typhoon_asr_inference.py
        but wrap it in an event-like structure
        """
        print(f" Processing audio file with events: {audio_file}")
        
        if session_id:
            print(f" Session ID: {session_id}")
        
        # Check if file exists
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        try:
            # Run the existing typhoon_asr_inference.py script
            result = subprocess.run([
                sys.executable, "src/typhoon_asr_inference.py", audio_file
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode != 0:
                raise RuntimeError(f"Transcription failed: {result.stderr}")
            
            # Parse the output to extract transcription and metadata
            output_lines = result.stdout.split('\n')
            transcription = self._extract_transcription(output_lines)
            metadata = self._extract_metadata(output_lines)
            
            # Create event-style result
            event_result = {
                'text': transcription,
                'full_text': transcription,
                'sentences': [transcription] if transcription.strip() else [],
                'session_id': session_id,
                'audio_file': str(Path(audio_file).resolve()),
                'processing_time': metadata.get('processing_time', 0.0),
                'audio_duration': metadata.get('duration', 0.0),
                'rtf': metadata.get('rtf', 0.0),
                'mode': metadata.get('mode', 'basic'),
                'confidence': 1.0,  # Assume high confidence
                'timestamp': metadata.get('timestamp'),
                'event_type': 'transcription.completed',
                'source': 'typhoon_asr_events'
            }
            
            # Store in session if provided
            if session_id:
                if session_id not in self.sessions:
                    self.sessions[session_id] = []
                self.sessions[session_id].append(event_result)
            
            # Add to history
            self.results_history.append(event_result)
            
            return event_result
            
        except Exception as e:
            # Create error event
            error_event = {
                'event_type': 'processing.error',
                'error': str(e),
                'audio_file': str(Path(audio_file).resolve()),
                'session_id': session_id,
                'source': 'typhoon_asr_events'
            }
            raise RuntimeError(f"Transcription failed: {e}") from e
    
    def _extract_transcription(self, output_lines):
        """Extract transcription from output"""
        for line in output_lines:
            if line.strip().startswith("'") and line.strip().endswith("'"):
                # This is likely the transcription line
                return line.strip()[1:-1]  # Remove quotes
        return ""
    
    def _extract_metadata(self, output_lines):
        """Extract metadata from output"""
        metadata = {}
        
        for line in output_lines:
            line = line.strip()
            
            # Extract duration
            if "Duration:" in line:
                duration_match = re.search(r"Duration:\s*([\d.]+)s", line)
                if duration_match:
                    metadata['duration'] = float(duration_match.group(1))
            
            # Extract processing time
            if "Processing:" in line:
                proc_match = re.search(r"Processing:\s*([\d.]+)s", line)
                if proc_match:
                    metadata['processing_time'] = float(proc_match.group(1))
            
            # Extract RTF
            if "RTF:" in line:
                rtf_match = re.search(r"RTF:\s*([\d.]+)x", line)
                if rtf_match:
                    metadata['rtf'] = float(rtf_match.group(1))
            
            # Extract mode
            if "Mode:" in line:
                mode_match = re.search(r"Mode:\s*(.+)", line)
                if mode_match:
                    metadata['mode'] = mode_match.group(1).strip()
        
        return metadata
    
    def get_session_results(self, session_id):
        """Get all results for a session"""
        return self.sessions.get(session_id, [])
    
    def get_all_results(self):
        """Get all processing results"""
        return self.results_history
    
    async def shutdown(self):
        """Shutdown the system (placeholder for compatibility)"""
        print(" Typhoon ASR Events system shutdown")


async def main():
    """Main function to demonstrate typhoon_asr_events usage"""
    
    # Path to the sample voice file
    audio_file = "data/sample_voice.wav"
    
    if not Path(audio_file).exists():
        print(f" Audio file '{audio_file}' not found!")
        return
    
    print(" Typhoon ASR Events - Event-Driven Transcription")
    print("=" * 70)
    
    # Initialize the event system
    asr_events = TyphoonASREvents()
    
    try:
        # Process the audio file with session tracking
        session_id = "demo_session_001"
        
        print(f" Processing: {audio_file}")
        print(f" Session: {session_id}")
        print("=" * 70)
        
        result = await asr_events.process_audio_file(audio_file, session_id=session_id)
        
        # Display event-style results
        print("\n EVENT RESULTS")
        print("=" * 70)
        print(f" Event Type: {result['event_type']}")
        print(f" Transcribed Text: '{result['text']}'")
        
        print(f"\n Event Metadata:")
        print(f"   Session ID: {result['session_id']}")
        print(f"   Audio File: {Path(result['audio_file']).name}")
        print(f"   Duration: {result['audio_duration']:.1f}s")
        print(f"   Processing: {result['processing_time']:.2f}s")
        print(f"   RTF: {result['rtf']:.3f}x")
        print(f"   Mode: {result['mode']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Source: {result['source']}")
        
        if result['sentences']:
            print(f"\n Sentences ({len(result['sentences'])}):")
            for i, sentence in enumerate(result['sentences'], 1):
                print(f"   {i}. {sentence}")
        
        # Show session information
        session_results = asr_events.get_session_results(session_id)
        print(f"\n Session Summary:")
        print(f"   Total results in session: {len(session_results)}")
        
        all_results = asr_events.get_all_results()
        print(f"   Total system results: {len(all_results)}")
        
        print("\n Event-driven transcription completed!")
        
    except Exception as e:
        print(f" Event processing error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await asr_events.shutdown()


if __name__ == "__main__":
    import asyncio
    
    print(" Typhoon ASR Events Demo")
    print("=" * 70)
    asyncio.run(main())