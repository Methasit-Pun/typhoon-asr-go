package server

import (
	"path/filepath"
	"strings"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	pb "github.com/scb10x/typhoon-asr/go/pkg/pb"
)

// ValidatorConfig holds the limits applied to every incoming request before it
// reaches the Python bridge. Zero values use built-in defaults where noted.
type ValidatorConfig struct {
	// MaxAudioBytes is the maximum size of raw audio_data bytes accepted.
	// Default: 100 MiB.  Set to -1 to disable the check (not recommended).
	MaxAudioBytes int64

	// MaxAudioDurationSecs caps inference time on CPU by rejecting audio
	// whose byte count implies a duration longer than this many seconds.
	// Only a rough guard (assumes ~16 kHz mono 16-bit PCM ≈ 32 kB/s).
	// 0 means no cap.
	MaxAudioDurationSecs float64

	// AllowedAudioDir, if non-empty, is the only directory from which
	// file_path references are accepted.  If empty, file_path is rejected
	// entirely — callers must send audio_data bytes instead.
	AllowedAudioDir string
}

const defaultMaxAudioBytes = 100 * 1024 * 1024 // 100 MiB

var (
	validLanguageHints = map[string]bool{"": true, "th": true, "en": true, "auto": true}
	validDevices       = map[string]bool{"": true, "auto": true, "cpu": true, "cuda": true}
)

// ValidateTranscribe checks a unary TranscribeRequest and returns a gRPC
// status error if any field is invalid.
func (v ValidatorConfig) ValidateTranscribe(req *pb.TranscribeRequest) error {
	switch src := req.AudioSource.(type) {
	case *pb.TranscribeRequest_FilePath:
		if err := v.checkFilePath(src.FilePath); err != nil {
			return err
		}
	case *pb.TranscribeRequest_AudioData:
		if err := v.checkAudioBytes(src.AudioData); err != nil {
			return err
		}
	default:
		return status.Error(codes.InvalidArgument, "audio_source must be file_path or audio_data")
	}

	return validateOptions(req.Options)
}

// ValidateChunk checks the size of an accumulated streaming audio buffer.
func (v ValidatorConfig) ValidateChunk(data []byte) error {
	return v.checkAudioBytes(data)
}

func (v ValidatorConfig) checkAudioBytes(data []byte) error {
	limit := v.MaxAudioBytes
	if limit == 0 {
		limit = defaultMaxAudioBytes
	}
	if limit > 0 && int64(len(data)) > limit {
		return status.Errorf(codes.InvalidArgument,
			"audio_data size %d bytes exceeds the server limit of %d bytes",
			len(data), limit)
	}

	// Rough duration guard: 16 kHz mono 16-bit PCM = 32 000 bytes/second.
	if v.MaxAudioDurationSecs > 0 {
		const bytesPerSec = 32_000
		estimatedSecs := float64(len(data)) / bytesPerSec
		if estimatedSecs > v.MaxAudioDurationSecs {
			return status.Errorf(codes.InvalidArgument,
				"audio_data implies %.0f s of audio which exceeds the server cap of %.0f s",
				estimatedSecs, v.MaxAudioDurationSecs)
		}
	}
	return nil
}

func (v ValidatorConfig) checkFilePath(path string) error {
	if v.AllowedAudioDir == "" {
		return status.Error(codes.PermissionDenied,
			"file_path is disabled on this server; send audio_data bytes instead")
	}
	if path == "" {
		return status.Error(codes.InvalidArgument, "file_path must not be empty")
	}
	// Reject traversal sequences before any filesystem access.
	if strings.Contains(path, "..") {
		return status.Error(codes.InvalidArgument, "file_path must not contain '..'")
	}

	abs, err := filepath.Abs(path)
	if err != nil {
		return status.Errorf(codes.InvalidArgument, "invalid file_path: %v", err)
	}
	allowed := filepath.Clean(v.AllowedAudioDir)
	// Require the path to be strictly inside the allowed directory.
	if abs != allowed && !strings.HasPrefix(abs, allowed+string(filepath.Separator)) {
		return status.Error(codes.PermissionDenied,
			"file_path is outside the server's allowed audio directory")
	}
	return nil
}

func validateOptions(opts *pb.TranscribeOptions) error {
	if opts == nil {
		return nil
	}
	if !validLanguageHints[opts.LanguageHint] {
		return status.Errorf(codes.InvalidArgument,
			"language_hint %q is not supported; choose: th, en, auto", opts.LanguageHint)
	}
	if !validDevices[opts.Device] {
		return status.Errorf(codes.InvalidArgument,
			"device %q is not supported; choose: auto, cpu, cuda", opts.Device)
	}
	if opts.ConfidenceThreshold < 0 || opts.ConfidenceThreshold > 1 {
		return status.Error(codes.InvalidArgument,
			"confidence_threshold must be in [0.0, 1.0]")
	}
	return nil
}
