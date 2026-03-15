// Package config centralises all server configuration.
//
// Precedence (highest → lowest): CLI flag → environment variable → default.
// This makes the binary friendly for both local development and container
// deployments (docker, k8s) without changing the binary.
//
// Environment variables:
//
//	TYPHOON_ADDR                 gRPC listen address               (default :50051)
//	TYPHOON_MODEL                NeMo model name
//	TYPHOON_DEVICE               auto | cpu | cuda
//	TYPHOON_PYTHON               Python interpreter path
//	TYPHOON_SCRIPT               Path to python/bridge_server.py
//	TYPHOON_MAX_INFLIGHT         max concurrent bridge requests    (default 64)
//
//	TYPHOON_AUTH_TOKEN           if set, require Bearer token on all RPCs
//	TYPHOON_TLS_CERT             PEM certificate file for TLS / mTLS
//	TYPHOON_TLS_KEY              PEM private-key file for TLS / mTLS
//	TYPHOON_TLS_CLIENT_CA        PEM CA file; enables mutual TLS when set
//
//	TYPHOON_MAX_AUDIO_BYTES      max raw audio bytes accepted      (default 104857600)
//	TYPHOON_MAX_AUDIO_DURATION_S max audio duration in seconds     (default 0 = no cap)
//	TYPHOON_AUDIO_DIR            directory allowed for file_path   (default "" = disabled)
//
//	TYPHOON_LOG_LEVEL            debug | info | warn | error       (default info)
//	TYPHOON_LOG_FORMAT           text | json                       (default text)
package config

import (
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"
)

// Config holds every knob the server exposes.
// Adding a new setting requires only a field here plus a line in Load().
type Config struct {
	// Network
	Addr string // gRPC listen address, e.g. ":50051"

	// Python bridge
	ModelName   string // NeMo model identifier
	Device      string // "auto", "cpu", or "cuda"
	PythonBin   string // path to python3 interpreter
	ScriptPath  string // path to python/bridge_server.py (empty = auto-resolve)
	MaxInFlight int    // max concurrent in-flight bridge requests (0 = use default)

	// Authentication
	AuthToken   string // if non-empty, require "Authorization: Bearer <token>"
	TLSCertFile string // PEM server certificate; enables TLS when set
	TLSKeyFile  string // PEM server private key; required when TLSCertFile is set
	TLSClientCA string // PEM CA for client certs; enables mTLS when set

	// Input validation
	MaxAudioBytes       int64   // max raw audio bytes per request (0 → 100 MiB default)
	MaxAudioDurationSecs float64 // max audio duration in seconds (0 → no cap)
	AllowedAudioDir     string  // directory for file_path requests (empty → disabled)

	// Observability
	LogLevel  slog.Level
	LogFormat string // "text" or "json"
}

// Load parses CLI flags and falls back to environment variables, then defaults.
// Call this once at startup; the returned *Config is read-only after Load returns.
func Load() *Config {
	cfg := &Config{}

	flag.StringVar(&cfg.Addr,        "addr",         envOr("TYPHOON_ADDR",   ":50051"),                      "gRPC listen address")
	flag.StringVar(&cfg.ModelName,   "model",        envOr("TYPHOON_MODEL",  "scb10x/typhoon-asr-realtime"), "NeMo ASR model name")
	flag.StringVar(&cfg.Device,      "device",       envOr("TYPHOON_DEVICE", "auto"),                        "inference device: auto|cpu|cuda")
	flag.StringVar(&cfg.PythonBin,   "python",       envOr("TYPHOON_PYTHON", "python3"),                     "Python interpreter path")
	flag.StringVar(&cfg.ScriptPath,  "script",       envOr("TYPHOON_SCRIPT", ""),                            "path to bridge_server.py (auto-resolved if empty)")
	flag.IntVar(&cfg.MaxInFlight,    "max-inflight", envInt("TYPHOON_MAX_INFLIGHT", 64),                     "max concurrent in-flight bridge requests")

	// Auth / TLS
	flag.StringVar(&cfg.AuthToken,   "auth-token",   envOr("TYPHOON_AUTH_TOKEN",   ""), "require Bearer token on all RPCs (empty = no auth)")
	flag.StringVar(&cfg.TLSCertFile, "tls-cert",     envOr("TYPHOON_TLS_CERT",     ""), "PEM server certificate for TLS/mTLS")
	flag.StringVar(&cfg.TLSKeyFile,  "tls-key",      envOr("TYPHOON_TLS_KEY",      ""), "PEM server private key for TLS/mTLS")
	flag.StringVar(&cfg.TLSClientCA, "tls-client-ca",envOr("TYPHOON_TLS_CLIENT_CA",""), "PEM CA for client certificates (enables mTLS)")

	// Input validation
	flag.Int64Var(&cfg.MaxAudioBytes,       "max-audio-bytes",    envInt64("TYPHOON_MAX_AUDIO_BYTES", 0),    "max raw audio bytes per request (0 = 100 MiB default)")
	flag.Float64Var(&cfg.MaxAudioDurationSecs,"max-audio-duration",envFloat64("TYPHOON_MAX_AUDIO_DURATION_S",0), "max audio duration in seconds (0 = no cap)")
	flag.StringVar(&cfg.AllowedAudioDir,    "audio-dir",          envOr("TYPHOON_AUDIO_DIR", ""),            "directory for file_path requests (empty = disabled)")

	logLevelStr  := flag.String("log-level",  envOr("TYPHOON_LOG_LEVEL",  "info"), "log level: debug|info|warn|error")
	logFormatStr := flag.String("log-format", envOr("TYPHOON_LOG_FORMAT", "text"), "log format: text|json")

	flag.Parse()

	cfg.LogLevel  = parseLevel(*logLevelStr)
	cfg.LogFormat = *logFormatStr
	return cfg
}

// Validate returns an error if any field holds an invalid value.
func (c *Config) Validate() error {
	var errs []string

	if c.Addr == "" {
		errs = append(errs, "addr must not be empty")
	}
	if c.ModelName == "" {
		errs = append(errs, "model must not be empty")
	}
	switch c.Device {
	case "auto", "cpu", "cuda":
	default:
		errs = append(errs, fmt.Sprintf("device %q is invalid; choose auto, cpu, or cuda", c.Device))
	}
	switch c.LogFormat {
	case "text", "json":
	default:
		errs = append(errs, fmt.Sprintf("log-format %q is invalid; choose text or json", c.LogFormat))
	}

	// TLS: cert and key must be specified together.
	if (c.TLSCertFile == "") != (c.TLSKeyFile == "") {
		errs = append(errs, "tls-cert and tls-key must both be set or both be empty")
	}
	if c.TLSClientCA != "" && c.TLSCertFile == "" {
		errs = append(errs, "tls-client-ca requires tls-cert and tls-key to also be set")
	}

	if len(errs) > 0 {
		return errors.New(strings.Join(errs, "; "))
	}
	return nil
}

// NewLogger builds a *slog.Logger from the config.
func (c *Config) NewLogger() *slog.Logger {
	opts := &slog.HandlerOptions{Level: c.LogLevel}
	if c.LogFormat == "json" {
		return slog.New(slog.NewJSONHandler(os.Stdout, opts))
	}
	return slog.New(slog.NewTextHandler(os.Stdout, opts))
}

// envOr returns the value of the named environment variable, or fallback if unset.
func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// envInt returns the named environment variable parsed as int, or fallback if unset/invalid.
func envInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}

// envInt64 returns the named environment variable parsed as int64, or fallback if unset/invalid.
func envInt64(key string, fallback int64) int64 {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			return n
		}
	}
	return fallback
}

// envFloat64 returns the named environment variable parsed as float64, or fallback if unset/invalid.
func envFloat64(key string, fallback float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return fallback
}

// parseLevel converts a level name string to slog.Level, defaulting to Info.
func parseLevel(s string) slog.Level {
	switch strings.ToLower(s) {
	case "debug":
		return slog.LevelDebug
	case "warn", "warning":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}
