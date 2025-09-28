"""
Utility Functions

Common utilities for audio processing, event handling, and system operations.
"""

import asyncio
import time
import hashlib
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path
from contextlib import asynccontextmanager, contextmanager

import numpy as np
import librosa
import soundfile as sf


class AudioUtils:
    """Utilities for audio processing."""
    
    @staticmethod
    def validate_audio_format(file_path: str) -> bool:
        """
        Validate if file is in supported audio format.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            True if format is supported
        """
        supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.webm']
        return Path(file_path).suffix.lower() in supported_formats
    
    @staticmethod
    def get_audio_info(file_path: str) -> Dict[str, Any]:
        """
        Get audio file information.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        try:
            # Use soundfile for metadata
            info = sf.info(file_path)
            
            return {
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'format': info.format,
                'subtype': info.subtype,
                'frames': info.frames,
                'file_size': Path(file_path).stat().st_size
            }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def normalize_audio(audio_data: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        Normalize audio to target dB level.
        
        Args:
            audio_data: Input audio array
            target_db: Target dB level
            
        Returns:
            Normalized audio array
        """
        # Calculate RMS
        rms = np.sqrt(np.mean(audio_data**2))
        
        if rms == 0:
            return audio_data
            
        # Calculate target amplitude
        target_amplitude = 10**(target_db / 20)
        
        # Normalize
        normalized = audio_data * (target_amplitude / rms)
        
        # Clip to prevent overflow
        return np.clip(normalized, -1.0, 1.0)
    
    @staticmethod
    def split_audio_chunks(audio_data: np.ndarray, 
                          sample_rate: int,
                          chunk_duration: float = 10.0,
                          overlap_duration: float = 1.0) -> List[np.ndarray]:
        """
        Split audio into overlapping chunks.
        
        Args:
            audio_data: Input audio array
            sample_rate: Audio sample rate
            chunk_duration: Duration of each chunk in seconds
            overlap_duration: Overlap between chunks in seconds
            
        Returns:
            List of audio chunks
        """
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap_duration * sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        chunks = []
        start = 0
        
        while start < len(audio_data):
            end = min(start + chunk_samples, len(audio_data))
            chunk = audio_data[start:end]
            
            # Pad if too short
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
            
            chunks.append(chunk)
            
            if end >= len(audio_data):
                break
                
            start += step_samples
        
        return chunks


class EventUtils:
    """Utilities for event handling."""
    
    @staticmethod
    def generate_correlation_id(prefix: str = "corr") -> str:
        """
        Generate a correlation ID for event tracking.
        
        Args:
            prefix: Prefix for the correlation ID
            
        Returns:
            Unique correlation ID
        """
        timestamp = str(int(time.time() * 1000))  # Millisecond timestamp
        random_part = hashlib.md5(f"{time.time()}{np.random.random()}".encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{random_part}"
    
    @staticmethod
    def calculate_event_latency(event_timestamp: float) -> float:
        """
        Calculate latency since event creation.
        
        Args:
            event_timestamp: Original event timestamp
            
        Returns:
            Latency in seconds
        """
        return time.time() - event_timestamp
    
    @staticmethod
    def create_event_metadata(session_id: Optional[str] = None,
                            user_id: Optional[str] = None,
                            request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create standard event metadata.
        
        Args:
            session_id: Optional session identifier
            user_id: Optional user identifier  
            request_id: Optional request identifier
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'timestamp': time.time(),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        }
        
        if session_id:
            metadata['session_id'] = session_id
        if user_id:
            metadata['user_id'] = user_id
        if request_id:
            metadata['request_id'] = request_id
            
        return metadata


class PerformanceMonitor:
    """Simple performance monitoring utilities."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.measurements: List[float] = []
    
    def start(self):
        """Start timing."""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """
        Stop timing and return duration.
        
        Returns:
            Duration in seconds
        """
        self.end_time = time.time()
        if self.start_time is not None:
            duration = self.end_time - self.start_time
            self.measurements.append(duration)
            return duration
        return 0.0
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.measurements:
            return {}
        
        measurements = self.measurements
        return {
            'count': len(measurements),
            'total': sum(measurements),
            'average': sum(measurements) / len(measurements),
            'min': min(measurements),
            'max': max(measurements),
            'latest': measurements[-1] if measurements else 0.0
        }
    
    def reset(self):
        """Reset all measurements."""
        self.measurements.clear()
        self.start_time = None
        self.end_time = None
    
    @contextmanager
    def time_operation(self):
        """Context manager for timing operations."""
        self.start()
        try:
            yield self
        finally:
            self.stop()


class RetryHandler:
    """Utilities for handling retries with exponential backoff."""
    
    @staticmethod
    async def retry_async(func: Callable,
                         max_retries: int = 3,
                         base_delay: float = 1.0,
                         max_delay: float = 60.0,
                         exponential_base: float = 2.0,
                         exceptions: tuple = (Exception,)) -> Any:
        """
        Retry async function with exponential backoff.
        
        Args:
            func: Async function to retry
            max_retries: Maximum number of retries
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            exceptions: Tuple of exceptions to catch
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries failed
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await func()
            except exceptions as e:
                last_exception = e
                
                if attempt == max_retries:
                    break
                
                delay = min(base_delay * (exponential_base ** attempt), max_delay)
                logging.debug(f"Retry {attempt + 1}/{max_retries} after {delay:.2f}s: {e}")
                await asyncio.sleep(delay)
        
        raise last_exception
    
    @staticmethod
    def retry_sync(func: Callable,
                  max_retries: int = 3,
                  base_delay: float = 1.0,
                  max_delay: float = 60.0,
                  exponential_base: float = 2.0,
                  exceptions: tuple = (Exception,)) -> Any:
        """
        Retry sync function with exponential backoff.
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retries
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            exceptions: Tuple of exceptions to catch
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries failed
        """
        import time as sync_time
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except exceptions as e:
                last_exception = e
                
                if attempt == max_retries:
                    break
                
                delay = min(base_delay * (exponential_base ** attempt), max_delay)
                logging.debug(f"Retry {attempt + 1}/{max_retries} after {delay:.2f}s: {e}")
                sync_time.sleep(delay)
        
        raise last_exception


class HealthChecker:
    """Health checking utilities for system components."""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
    
    def add_check(self, name: str, check_func: Callable[[], bool]):
        """
        Add a health check.
        
        Args:
            name: Name of the health check
            check_func: Function that returns True if healthy
        """
        self.checks[name] = check_func
    
    async def run_checks(self) -> Dict[str, Any]:
        """
        Run all health checks.
        
        Returns:
            Dictionary with check results
        """
        results = {}
        overall_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                # Handle both sync and async check functions
                if asyncio.iscoroutinefunction(check_func):
                    is_healthy = await check_func()
                else:
                    is_healthy = check_func()
                
                results[name] = {
                    'healthy': is_healthy,
                    'status': 'OK' if is_healthy else 'FAILED'
                }
                
                if not is_healthy:
                    overall_healthy = False
                    
            except Exception as e:
                results[name] = {
                    'healthy': False,
                    'status': 'ERROR',
                    'error': str(e)
                }
                overall_healthy = False
        
        results['overall'] = {
            'healthy': overall_healthy,
            'status': 'OK' if overall_healthy else 'FAILED',
            'timestamp': time.time()
        }
        
        return results


@asynccontextmanager
async def graceful_shutdown(shutdown_timeout: float = 30.0):
    """
    Context manager for graceful shutdown handling.
    
    Args:
        shutdown_timeout: Maximum time to wait for shutdown
    """
    shutdown_event = asyncio.Event()
    
    def signal_handler():
        shutdown_event.set()
    
    # Register signal handlers
    import signal
    
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda s, f: signal_handler())
    
    try:
        yield shutdown_event
    finally:
        # Wait for graceful shutdown
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=shutdown_timeout)
        except asyncio.TimeoutError:
            logging.warning(f"Graceful shutdown timed out after {shutdown_timeout}s")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0B"
        
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes:.0f}m {remaining_seconds:.0f}s"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {remaining_minutes:.0f}m"