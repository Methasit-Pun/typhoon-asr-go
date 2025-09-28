"""
Configuration Management

Centralized configuration for the Typhoon ASR Event System.
Supports environment variables, YAML/JSON config files, and runtime overrides.
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path

import yaml


@dataclass
class ASRConfig:
    """Configuration for ASR processing."""
    model_name: str = "scb10x/typhoon-asr-realtime"
    device: str = "auto"
    batch_size: int = 1
    confidence_threshold: float = 0.7
    enable_alternatives: bool = True
    max_alternatives: int = 5


@dataclass  
class RedisConfig:
    """Configuration for Redis caching."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    connection_timeout: int = 5
    socket_timeout: int = 5
    max_connections: int = 20


@dataclass
class AggregationConfig:
    """Configuration for transcription aggregation."""
    cache_ttl: int = 3600  # 1 hour
    max_buffer_size: int = 100
    sentence_timeout: float = 5.0
    cleanup_interval: int = 1800  # 30 minutes
    max_session_age_hours: float = 24.0
    enable_sentence_detection: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True


@dataclass
class EventBusConfig:
    """Configuration for the event bus."""
    max_event_history: int = 1000
    enable_middleware: bool = True
    concurrent_handler_limit: int = 10
    event_timeout: float = 30.0


@dataclass
class SystemConfig:
    """Configuration for system-level settings."""
    temp_dir: str = "/tmp"
    cleanup_temp_files: bool = True
    enable_metrics: bool = True
    health_check_interval: int = 60
    graceful_shutdown_timeout: int = 30


@dataclass
class Config:
    """Main configuration class combining all subsystem configs."""
    asr: ASRConfig = field(default_factory=ASRConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    event_bus: EventBusConfig = field(default_factory=EventBusConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """
        Load configuration from file.
        
        Args:
            config_path: Path to YAML or JSON config file
            
        Returns:
            Config instance
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls._from_dict(data)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """
        Load configuration from environment variables.
        
        Environment variable names follow the pattern:
        TYPHOON_ASR_<SECTION>_<SETTING>
        
        Example: TYPHOON_ASR_ASR_MODEL_NAME
        
        Returns:
            Config instance with environment overrides
        """
        config = cls()
        
        # Environment variable mappings
        env_mappings = {
            # ASR config
            'TYPHOON_ASR_ASR_MODEL_NAME': ('asr', 'model_name'),
            'TYPHOON_ASR_ASR_DEVICE': ('asr', 'device'),
            'TYPHOON_ASR_ASR_BATCH_SIZE': ('asr', 'batch_size', int),
            'TYPHOON_ASR_ASR_CONFIDENCE_THRESHOLD': ('asr', 'confidence_threshold', float),
            
            # Redis config
            'TYPHOON_ASR_REDIS_HOST': ('redis', 'host'),
            'TYPHOON_ASR_REDIS_PORT': ('redis', 'port', int),
            'TYPHOON_ASR_REDIS_DB': ('redis', 'db', int),
            'TYPHOON_ASR_REDIS_PASSWORD': ('redis', 'password'),
            
            # Aggregation config
            'TYPHOON_ASR_AGGREGATION_CACHE_TTL': ('aggregation', 'cache_ttl', int),
            'TYPHOON_ASR_AGGREGATION_MAX_BUFFER_SIZE': ('aggregation', 'max_buffer_size', int),
            'TYPHOON_ASR_AGGREGATION_SENTENCE_TIMEOUT': ('aggregation', 'sentence_timeout', float),
            
            # Logging config
            'TYPHOON_ASR_LOGGING_LEVEL': ('logging', 'level'),
            'TYPHOON_ASR_LOGGING_FILE_PATH': ('logging', 'file_path'),
            
            # System config
            'TYPHOON_ASR_SYSTEM_TEMP_DIR': ('system', 'temp_dir'),
        }
        
        for env_var, mapping in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                section_name = mapping[0]
                setting_name = mapping[1]
                converter = mapping[2] if len(mapping) > 2 else str
                
                # Convert value
                try:
                    converted_value = converter(value) if converter != str else value
                except (ValueError, TypeError) as e:
                    logging.warning(f"Invalid value for {env_var}: {value} ({e})")
                    continue
                
                # Set the value
                section = getattr(config, section_name)
                setattr(section, setting_name, converted_value)
        
        return config
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        config = cls()
        
        for section_name, section_data in data.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section = getattr(config, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_file(self, config_path: str, format_type: str = 'yaml'):
        """
        Save configuration to file.
        
        Args:
            config_path: Output file path
            format_type: Format ('yaml' or 'json')
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if format_type.lower() == 'yaml':
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            elif format_type.lower() == 'json':
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
    
    def merge(self, other: 'Config') -> 'Config':
        """
        Merge another config into this one.
        
        Args:
            other: Config to merge
            
        Returns:
            New merged config
        """
        merged_dict = self.to_dict()
        other_dict = other.to_dict()
        
        # Deep merge
        def deep_merge(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged = deep_merge(merged_dict, other_dict)
        return self._from_dict(merged)
    
    def validate(self) -> List[str]:
        """
        Validate configuration values.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate ASR config
        if not self.asr.model_name:
            errors.append("ASR model name cannot be empty")
        
        if self.asr.confidence_threshold < 0.0 or self.asr.confidence_threshold > 1.0:
            errors.append("ASR confidence threshold must be between 0.0 and 1.0")
        
        if self.asr.batch_size < 1:
            errors.append("ASR batch size must be >= 1")
        
        # Validate Redis config
        if self.redis.port < 1 or self.redis.port > 65535:
            errors.append("Redis port must be between 1 and 65535")
        
        if self.redis.db < 0:
            errors.append("Redis database number must be >= 0")
        
        # Validate aggregation config
        if self.aggregation.cache_ttl < 0:
            errors.append("Cache TTL must be >= 0")
        
        if self.aggregation.sentence_timeout < 0:
            errors.append("Sentence timeout must be >= 0")
        
        # Validate logging config
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.logging.level.upper() not in valid_log_levels:
            errors.append(f"Invalid log level: {self.logging.level}")
        
        return errors


def setup_logging(config: LoggingConfig) -> logging.Logger:
    """
    Set up logging based on configuration.
    
    Args:
        config: Logging configuration
        
    Returns:
        Configured logger
    """
    import logging.handlers
    
    # Create logger
    logger = logging.getLogger('typhoon_asr_events')
    logger.setLevel(getattr(logging, config.level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(config.format)
    
    # Console handler
    if config.enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if config.file_path:
        file_path = Path(config.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Default configuration instance
default_config = Config()

# Try to load from environment
try:
    env_config = Config.from_env()
    default_config = default_config.merge(env_config)
except Exception as e:
    logging.warning(f"Failed to load environment config: {e}")

# Try to load from default config file locations
default_config_paths = [
    'config/typhoon_asr.yaml',
    'typhoon_asr.yaml',
    'config.yaml'
]

for config_path in default_config_paths:
    try:
        if Path(config_path).exists():
            file_config = Config.from_file(config_path)
            default_config = default_config.merge(file_config)
            break
    except Exception as e:
        logging.debug(f"Could not load config from {config_path}: {e}")

# Validate default config
config_errors = default_config.validate()
if config_errors:
    logging.warning(f"Configuration validation errors: {config_errors}")

# Set up default logger
default_logger = setup_logging(default_config.logging)