// Package pb contains Go type definitions mirroring proto/asr.proto.
//
// These stubs are hand-written to match the structure that protoc-gen-go would
// generate, so that the codebase compiles and tests run without requiring protoc
// or a generated go.sum entry for the protoc-gen plugins.
//
// Regenerate with protoc once tooling is available:
//
//	make proto   (from the go/ directory)
package pb

import (
	"context"

	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/durationpb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// ─── Enumerations ─────────────────────────────────────────────────────────────

// TranscribeStatus enumerates possible transcription outcomes.
type TranscribeStatus int32

const (
	TranscribeStatus_TRANSCRIBE_STATUS_UNSPECIFIED    TranscribeStatus = 0
	TranscribeStatus_TRANSCRIBE_STATUS_OK             TranscribeStatus = 1
	TranscribeStatus_TRANSCRIBE_STATUS_ERROR          TranscribeStatus = 2
	TranscribeStatus_TRANSCRIBE_STATUS_PARTIAL        TranscribeStatus = 3
	TranscribeStatus_TRANSCRIBE_STATUS_NO_SPEECH      TranscribeStatus = 4
	TranscribeStatus_TRANSCRIBE_STATUS_LOW_CONFIDENCE TranscribeStatus = 5
)

// ServiceStatus enumerates service health states.
type ServiceStatus int32

const (
	ServiceStatus_SERVICE_STATUS_UNSPECIFIED ServiceStatus = 0
	ServiceStatus_SERVICE_STATUS_SERVING     ServiceStatus = 1
	ServiceStatus_SERVICE_STATUS_NOT_SERVING ServiceStatus = 2
)

// AudioEncoding lists supported PCM encoding formats.
type AudioEncoding int32

const (
	AudioEncoding_AUDIO_ENCODING_UNSPECIFIED AudioEncoding = 0
	AudioEncoding_AUDIO_ENCODING_LINEAR16    AudioEncoding = 1
	AudioEncoding_AUDIO_ENCODING_FLOAT32     AudioEncoding = 2
)

// ─── Unary transcription messages ─────────────────────────────────────────────

// TranscribeRequest is the input for the unary Transcribe RPC.
type TranscribeRequest struct {
	// oneof audio_source
	AudioSource isTranscribeRequest_AudioSource
	Options     *TranscribeOptions
	SessionId   string
	RequestId   string
}

type isTranscribeRequest_AudioSource interface{ isTranscribeRequest_AudioSource() }

// TranscribeRequest_AudioData carries raw audio bytes.
type TranscribeRequest_AudioData struct{ AudioData []byte }

// TranscribeRequest_FilePath carries a server-local file path.
type TranscribeRequest_FilePath struct{ FilePath string }

func (*TranscribeRequest_AudioData) isTranscribeRequest_AudioSource() {}
func (*TranscribeRequest_FilePath) isTranscribeRequest_AudioSource()  {}

// TranscribeOptions holds per-request inference parameters.
type TranscribeOptions struct {
	ModelName           string
	WithTimestamps      bool
	LanguageHint        string
	ConfidenceThreshold float32
	Device              string
}

func (o *TranscribeOptions) GetWithTimestamps() bool {
	if o == nil {
		return false
	}
	return o.WithTimestamps
}

func (o *TranscribeOptions) GetDevice() string {
	if o == nil {
		return ""
	}
	return o.Device
}

func (o *TranscribeOptions) GetModelName() string {
	if o == nil {
		return ""
	}
	return o.ModelName
}

// TranscribeResponse is the output of the unary Transcribe RPC.
type TranscribeResponse struct {
	Text           string
	WordTimestamps []*WordTimestamp
	Confidence     float32
	AudioDuration  *durationpb.Duration
	ProcessingTime *durationpb.Duration
	RealTimeFactor float32
	SessionId      string
	RequestId      string
	Status         TranscribeStatus
	ErrorMessage   string
}

// WordTimestamp holds word-level timing for one token.
type WordTimestamp struct {
	Word       string
	StartTime  *durationpb.Duration
	EndTime    *durationpb.Duration
	Confidence float32
}

// ─── Streaming transcription messages ────────────────────────────────────────

// StreamTranscribeRequest wraps either a StreamConfig (first message) or an
// AudioChunk (subsequent messages).
type StreamTranscribeRequest struct {
	Payload isStreamTranscribeRequest_Payload
}

type isStreamTranscribeRequest_Payload interface{ isStreamTranscribeRequest_Payload() }

// StreamTranscribeRequest_Config is the first message in a streaming RPC.
type StreamTranscribeRequest_Config struct{ Config *StreamConfig }

// StreamTranscribeRequest_Chunk carries an audio chunk.
type StreamTranscribeRequest_Chunk struct{ Chunk *AudioChunk }

func (*StreamTranscribeRequest_Config) isStreamTranscribeRequest_Payload() {}
func (*StreamTranscribeRequest_Chunk) isStreamTranscribeRequest_Payload()  {}

// StreamConfig is the session initialisation payload.
type StreamConfig struct {
	SampleRate int32
	Channels   int32
	Encoding   AudioEncoding
	Options    *TranscribeOptions
	SessionId  string
}

// AudioChunk carries raw audio bytes for one streaming segment.
type AudioChunk struct {
	Data           []byte
	SequenceNumber int64
	CapturedAt     *timestamppb.Timestamp
	IsFinal        bool
}

// StreamTranscribeResponse is a single response message in the streaming RPC.
type StreamTranscribeResponse struct {
	Text           string
	WordTimestamps []*WordTimestamp
	Confidence     float32
	IsFinal        bool
	SequenceNumber int64
	SessionId      string
	Status         TranscribeStatus
	ErrorMessage   string
}

// ─── Health check messages ────────────────────────────────────────────────────

// HealthCheckRequest is the input for the HealthCheck RPC.
type HealthCheckRequest struct {
	IncludeModelStatus bool
}

// HealthCheckResponse is the output of the HealthCheck RPC.
type HealthCheckResponse struct {
	Status       ServiceStatus
	Version      string
	Model        *ModelStatus
	PythonBridge *PythonBridgeStatus
	CheckedAt    *timestamppb.Timestamp
}

// ModelStatus describes the currently loaded ASR model.
type ModelStatus struct {
	Loaded    bool
	ModelName string
	Device    string
}

// PythonBridgeStatus describes the Python subprocess managed by the bridge.
type PythonBridgeStatus struct {
	Alive           bool
	Pid             int32
	RequestsHandled int64
	Errors          int64
}

// ─── gRPC service interface and registration ──────────────────────────────────

// ASRServiceServer is the server-side interface for the ASRService.
type ASRServiceServer interface {
	Transcribe(context.Context, *TranscribeRequest) (*TranscribeResponse, error)
	StreamTranscribe(ASRService_StreamTranscribeServer) error
	HealthCheck(context.Context, *HealthCheckRequest) (*HealthCheckResponse, error)
	mustEmbedUnimplementedASRServiceServer()
}

// UnimplementedASRServiceServer must be embedded to maintain forward compatibility.
type UnimplementedASRServiceServer struct{}

func (UnimplementedASRServiceServer) Transcribe(_ context.Context, _ *TranscribeRequest) (*TranscribeResponse, error) {
	return nil, nil
}
func (UnimplementedASRServiceServer) StreamTranscribe(_ ASRService_StreamTranscribeServer) error {
	return nil
}
func (UnimplementedASRServiceServer) HealthCheck(_ context.Context, _ *HealthCheckRequest) (*HealthCheckResponse, error) {
	return nil, nil
}
func (UnimplementedASRServiceServer) mustEmbedUnimplementedASRServiceServer() {}

// ASRService_StreamTranscribeServer is the server-side stream handle for StreamTranscribe.
type ASRService_StreamTranscribeServer interface {
	Send(*StreamTranscribeResponse) error
	Recv() (*StreamTranscribeRequest, error)
	grpc.ServerStream
}

// ASRService_ServiceDesc is the grpc.ServiceDesc for ASRService.
// Used by RegisterASRServiceServer.
var ASRService_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "typhoon.asr.v1.ASRService",
	HandlerType: (*ASRServiceServer)(nil),
	Methods:     []grpc.MethodDesc{},
	Streams:     []grpc.StreamDesc{},
}

// RegisterASRServiceServer registers the service implementation with a gRPC server.
func RegisterASRServiceServer(s grpc.ServiceRegistrar, srv ASRServiceServer) {
	s.RegisterService(&ASRService_ServiceDesc, srv)
}
