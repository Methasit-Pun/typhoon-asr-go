"""
Simple Demo for Typhoon ASR Events

Minimal example showing how to use the event wrapper with the existing typhoon-asr package.
"""

import asyncio
from pathlib import Path

# Import the minimal event wrapper
try:
    from typhoon_asr_events import TyphoonASREventSystem, transcribe_with_events, MinimalConfig
    from typhoon_asr_events import transcribe as base_transcribe  # Re-exported from typhoon_asr
    EVENTS_AVAILABLE = True
except ImportError as e:
    print(f"Events wrapper not available: {e}")
    EVENTS_AVAILABLE = False

# Also try direct typhoon_asr import
try:
    from typhoon_asr import transcribe as direct_transcribe
    TYPHOON_DIRECT_AVAILABLE = True
except ImportError:
    print("typhoon_asr not available. Please install: pip install typhoon-asr")
    TYPHOON_DIRECT_AVAILABLE = False


async def demo_simple_usage():
    """Demo 1: Simple usage with events (minimal overhead)"""
    print(" Demo 1: Simple Event-Enhanced Transcription")
    print("=" * 50)
    
    # Find an audio file to test with
    test_files = ["example.wav", "test.mp3", "audio.wav", "sample.m4a"]
    audio_file = None
    
    for file in test_files:
        if Path(file).exists():
            audio_file = file
            break
    
    if not audio_file:
        print("  No test audio files found. Create example.wav or test.mp3 to test.")
        return
    
    # Method 1: Direct function call
    print(f"Processing: {audio_file}")
    result = await transcribe_with_events(audio_file)
    
    print(f" Text: {result['text']}")
    print(f"⏱  Processing time: {result['processing_time']:.2f}s")
    print(f" Audio duration: {result['audio_duration']:.1f}s")


async def demo_session_management():
    """Demo 2: Session management with aggregation"""
    print("\n  Demo 2: Session Management")
    print("=" * 50)
    
    # Initialize with aggregation enabled
    config = MinimalConfig()
    config.enable_aggregation = True
    config.sentence_timeout = 2.0  # Quick aggregation for demo
    
    system = TyphoonASREventSystem(config)
    
    # Find multiple files or use the same one multiple times
    test_files = [f for f in ["example.wav", "test.mp3", "audio.wav"] if Path(f).exists()]
    
    if not test_files:
        print("  No test files for session demo")
        return
    
    session_id = "demo_session"
    
    # Process files in the same session
    for i, audio_file in enumerate(test_files[:3], 1):  # Max 3 files
        print(f"\nProcessing file {i}: {Path(audio_file).name}")
        
        result = await system.process_audio_file(audio_file, session_id)
        
        print(f"   Result: {result['text'][:50]}...")
        print(f"   Aggregated: {result['aggregated']}")
    
    # Wait a bit for aggregation
    await asyncio.sleep(3)
    
    # Try one more to trigger aggregation
    if test_files:
        print(f"\nProcessing final file to trigger aggregation...")
        result = await system.process_audio_file(test_files[0], session_id)
        
        if result['aggregated']:
            print(f" Aggregated result: {result['full_text']}")
            print(f"   Sentences: {len(result['sentences'])}")
        
    # Get session history
    history = await system.get_session_history(session_id)
    print(f"\n Session history: {len(history)} entries")
    
    await system.shutdown()


def demo_direct_comparison():
    """Demo 3: Compare direct typhoon_asr vs event wrapper"""
    print("\n  Demo 3: Direct vs Event Wrapper Comparison")
    print("=" * 50)
    
    if not TYPHOON_DIRECT_AVAILABLE:
        print("typhoon_asr not available for comparison")
        return
    
    # Find test file
    test_files = ["example.wav", "test.mp3", "audio.wav"]
    audio_file = None
    
    for file in test_files:
        if Path(file).exists():
            audio_file = file
            break
    
    if not audio_file:
        print("  No test files for comparison")
        return
    
    print(f"Testing with: {audio_file}")
    
    # Method 1: Direct typhoon_asr
    print("\n1. Direct typhoon_asr.transcribe():")
    direct_result = direct_transcribe(audio_file)
    print(f"   Text: {direct_result['text']}")
    print(f"   Keys: {list(direct_result.keys())}")
    
    # Method 2: Event wrapper (synchronous-style)
    print("\n2. Event wrapper (re-exported transcribe):")
    if EVENTS_AVAILABLE:
        wrapper_result = base_transcribe(audio_file)  # Re-exported function
        print(f"   Text: {wrapper_result['text']}")
        print(f"   Same result: {direct_result == wrapper_result}")


async def demo_streaming_simulation():
    """Demo 4: Simple streaming with multiple files"""
    print("\n Demo 4: Streaming Simulation")
    print("=" * 50)
    
    if not EVENTS_AVAILABLE:
        print("Events wrapper not available")
        return
    
    # Find test files
    test_files = [f for f in ["example.wav", "test.mp3", "audio.wav"] if Path(f).exists()]
    
    if len(test_files) < 2:
        print("  Need at least 2 audio files for streaming demo")
        # Duplicate the file if we only have one
        if test_files:
            test_files = test_files * 3  # Use same file 3 times
        else:
            return
    
    system = TyphoonASREventSystem()
    
    print(f"Streaming {len(test_files)} files...")
    
    async for result in system.process_audio_stream(test_files, "streaming_session"):
        if result.get('error'):
            print(f" Error: {result['error']}")
        else:
            print(f" Stream result: {result['text'][:50]}...")
    
    await system.shutdown()


async def main():
    """Run all demos"""
    print("  Typhoon ASR Events - Minimal Demo")
    print("=" * 60)
    
    if not EVENTS_AVAILABLE:
        print(" Typhoon ASR Events not available")
        print("Make sure typhoon_asr_events folder is in your Python path")
        return
    
    # Run demos
    try:
        await demo_simple_usage()
        await demo_session_management()
        demo_direct_comparison()
        await demo_streaming_simulation()
        
        print("\n All demos completed!")
        
    except Exception as e:
        print(f" Demo failed: {e}")
        import traceback
        traceback.print_exc()


def show_simple_usage_examples():
    """Show code examples for documentation"""
    print("\n Simple Usage Examples:")
    print("=" * 50)
    
    examples = [
        ("Direct typhoon_asr usage:", '''
from typhoon_asr import transcribe

result = transcribe("audio.wav")
print(result['text'])
        '''),
        
        ("With event wrapper:", '''
import asyncio
from typhoon_asr_events import transcribe_with_events

async def main():
    result = await transcribe_with_events("audio.wav")
    print(result['text'])

asyncio.run(main())
        '''),
        
        ("With session management:", '''
from typhoon_asr_events import TyphoonASREventSystem

async def main():
    system = TyphoonASREventSystem()
    
    # Process multiple files in same session
    result1 = await system.process_audio_file("audio1.wav", "session1")
    result2 = await system.process_audio_file("audio2.wav", "session1")
    
    # Get aggregated history
    history = await system.get_session_history("session1")
    
    await system.shutdown()
        '''),
        
        ("Configuration:", '''
from typhoon_asr_events import MinimalConfig, TyphoonASREventSystem

config = MinimalConfig()
config.model_name = "scb10x/typhoon-asr-realtime"
config.device = "cuda"
config.enable_aggregation = True

system = TyphoonASREventSystem(config)
        ''')
    ]
    
    for title, code in examples:
        print(f"\n{title}")
        print(code)


if __name__ == "__main__":
    if EVENTS_AVAILABLE or TYPHOON_DIRECT_AVAILABLE:
        asyncio.run(main())
    else:
        print("\n Neither typhoon_asr nor typhoon_asr_events available")
        print("\nInstallation instructions:")
        print("1. Install typhoon-asr: pip install typhoon-asr")
        print("2. Copy typhoon_asr_events folder to your project")
        print("3. Install optional dependencies: pip install redis")
        
        show_simple_usage_examples()