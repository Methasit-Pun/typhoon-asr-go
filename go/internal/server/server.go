// Package server implements the ASRService gRPC server defined in proto/asr.proto.
package server

import (
	"context"
	"encoding/base64"
	"io"
	"log/slog"
	"time"

	"github.com/google/uuid"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/scb10x/typhoon-asr/go/internal/python"
	pb "github.com/scb10x/typhoon-asr/go/pkg/pb"
)

const version = "0.1.0"

// rtfWarnThreshold is the real-time factor above which a warning is logged.
// RTF > 1.0 means inference is slower than real-time; on CPU this is common
// for long audio and signals that GPU deployment should be considered.
const rtfWarnThreshold = 0.8

// ASRServer satisfies the pb.ASRServiceServer interface.
type ASRServer struct {
	pb.UnimplementedASRServiceServer
	bridge    *python.Bridge
	validator ValidatorConfig
	log       *slog.Logger
}

// New creates an ASRServer backed by the given Python bridge.
func New(bridge *python.Bridge, validator ValidatorConfig, log *slog.Logger) *ASRServer {
	return &ASRServer{bridge: bridge, validator: validator, log: log}
}

// ---------------------------------------------------------------------------
// Transcribe — unary RPC
// ---------------------------------------------------------------------------

func (s *ASRServer) Transcribe(ctx context.Context, req *pb.TranscribeRequest) (*pb.TranscribeResponse, error) {
	if err := s.validator.ValidateTranscribe(req); err != nil {
		return nil, err
	}

	reqID := req.RequestId
	if reqID == "" {
		reqID = uuid.NewString()
	}

	bridgeReq := python.BridgeRequest{RequestID: reqID}

	if opts := req.Options; opts != nil {
		bridgeReq.ModelName      = opts.ModelName
		bridgeReq.WithTimestamps = opts.WithTimestamps
		bridgeReq.Device         = opts.Device
		bridgeReq.LanguageHint   = opts.LanguageHint
	}

	switch src := req.AudioSource.(type) {
	case *pb.TranscribeRequest_FilePath:
		bridgeReq.AudioPath = src.FilePath
	case *pb.TranscribeRequest_AudioData:
		bridgeReq.AudioB64 = base64.StdEncoding.EncodeToString(src.AudioData)
	default:
		return nil, status.Error(codes.InvalidArgument, "audio_source must be file_path or audio_data")
	}

	resp, err := s.bridge.Call(ctx, bridgeReq)
	if err != nil {
		s.log.Error("bridge call failed", "request_id", reqID, "err", err)
		return &pb.TranscribeResponse{
			RequestId:    reqID,
			SessionId:    req.SessionId,
			Status:       pb.TranscribeStatus_TRANSCRIBE_STATUS_ERROR,
			ErrorMessage: err.Error(),
		}, nil
	}

	pbResp := mapResponse(resp, req.SessionId, reqID)

	if opts := req.Options; opts != nil && opts.ConfidenceThreshold > 0 {
		if resp.Confidence < float64(opts.ConfidenceThreshold) &&
			pbResp.Status == pb.TranscribeStatus_TRANSCRIBE_STATUS_OK {
			pbResp.Status = pb.TranscribeStatus_TRANSCRIBE_STATUS_LOW_CONFIDENCE
		}
	}

	s.warnHighRTF(reqID, pbResp.RealTimeFactor)

	return pbResp, nil
}

// ---------------------------------------------------------------------------
// StreamTranscribe — bidirectional streaming RPC
// ---------------------------------------------------------------------------

func (s *ASRServer) StreamTranscribe(stream pb.ASRService_StreamTranscribeServer) error {
	ctx := stream.Context()

	// First message must be a StreamConfig.
	firstMsg, err := stream.Recv()
	if err != nil {
		return status.Errorf(codes.InvalidArgument, "expected StreamConfig as first message: %v", err)
	}
	cfgMsg, ok := firstMsg.Payload.(*pb.StreamTranscribeRequest_Config)
	if !ok {
		return status.Error(codes.InvalidArgument, "first message must be a StreamConfig")
	}
	cfg := cfgMsg.Config

	if err := validateOptions(cfg.Options); err != nil {
		return err
	}

	sessionID := cfg.SessionId
	if sessionID == "" {
		sessionID = uuid.NewString()
	}
	s.log.Info("streaming session started", "session_id", sessionID)

	var (
		chunkBuf []byte
		seqNum   int64
	)

	for {
		msg, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return status.Errorf(codes.Internal, "recv stream chunk: %v", err)
		}

		chunkMsg, ok := msg.Payload.(*pb.StreamTranscribeRequest_Chunk)
		if !ok {
			continue
		}
		chunk := chunkMsg.Chunk
		seqNum    = chunk.SequenceNumber
		chunkBuf  = append(chunkBuf, chunk.Data...)

		// Reject accumulation that exceeds the configured byte limit.
		if err := s.validator.ValidateChunk(chunkBuf); err != nil {
			return err
		}

		if !chunk.IsFinal {
			continue
		}

		if err := s.flushChunk(ctx, stream, sessionID, seqNum, chunkBuf, cfg.Options); err != nil {
			return err
		}
		chunkBuf = chunkBuf[:0]
	}

	// Flush any remaining audio after the client closed the stream.
	if len(chunkBuf) > 0 {
		if err := s.flushChunk(ctx, stream, sessionID, seqNum, chunkBuf, cfg.Options); err != nil {
			return err
		}
	}

	s.log.Info("streaming session ended", "session_id", sessionID)
	return nil
}

// flushChunk sends accumulated audio bytes to Python and forwards the result to the stream.
func (s *ASRServer) flushChunk(
	ctx context.Context,
	stream pb.ASRService_StreamTranscribeServer,
	sessionID string,
	seqNum int64,
	data []byte,
	opts *pb.TranscribeOptions,
) error {
	bridgeReq := python.BridgeRequest{
		RequestID:      uuid.NewString(),
		AudioB64:       base64.StdEncoding.EncodeToString(data),
		WithTimestamps: opts.GetWithTimestamps(),
		Device:         opts.GetDevice(),
		ModelName:      opts.GetModelName(),
	}

	resp, callErr := s.bridge.Call(ctx, bridgeReq)
	pbResp := &pb.StreamTranscribeResponse{
		SessionId:      sessionID,
		SequenceNumber: seqNum,
		IsFinal:        true,
	}

	if callErr != nil {
		pbResp.Status       = pb.TranscribeStatus_TRANSCRIBE_STATUS_ERROR
		pbResp.ErrorMessage = callErr.Error()
	} else {
		pbResp.Text           = resp.Text
		pbResp.Confidence     = float32(resp.Confidence)
		pbResp.Status         = pb.TranscribeStatus_TRANSCRIBE_STATUS_OK
		pbResp.WordTimestamps = mapTimestamps(resp.Timestamps)

		if resp.AudioDuration > 0 {
			s.warnHighRTF(bridgeReq.RequestID, float32(resp.ProcessingTime/resp.AudioDuration))
		}
	}

	if err := stream.Send(pbResp); err != nil {
		return status.Errorf(codes.Internal, "send stream response: %v", err)
	}
	return nil
}

// ---------------------------------------------------------------------------
// HealthCheck — unary RPC
// ---------------------------------------------------------------------------

func (s *ASRServer) HealthCheck(ctx context.Context, req *pb.HealthCheckRequest) (*pb.HealthCheckResponse, error) {
	resp := &pb.HealthCheckResponse{
		Version:   version,
		Status:    pb.ServiceStatus_SERVICE_STATUS_SERVING,
		CheckedAt: timestamppb.Now(),
	}

	if req.IncludeModelStatus {
		pingCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()

		health, err := s.bridge.Ping(pingCtx)
		if err != nil || !health.Alive {
			resp.Status = pb.ServiceStatus_SERVICE_STATUS_NOT_SERVING
			s.log.Warn("Python bridge unhealthy", "err", err)
		}

		handled, errs := s.bridge.Stats()
		resp.PythonBridge = &pb.PythonBridgeStatus{
			Alive:           health.Alive,
			Pid:             int32(s.bridge.PID()),
			RequestsHandled: handled,
			Errors:          errs,
		}
		resp.Model = &pb.ModelStatus{
			Loaded:    health.ModelLoaded,
			ModelName: health.ModelName,
			Device:    health.Device,
		}
	}

	return resp, nil
}

// warnHighRTF logs a warning when the real-time factor approaches or exceeds 1.0.
// An RTF ≥ 0.8 on CPU means inference is close to real-time and will not scale;
// GPU deployment should be considered.
func (s *ASRServer) warnHighRTF(requestID string, rtf float32) {
	if rtf >= rtfWarnThreshold {
		s.log.Warn("high real-time factor — consider GPU deployment",
			"request_id", requestID,
			"rtf", rtf,
		)
	}
}

// Compile-time interface check.
var _ pb.ASRServiceServer = (*ASRServer)(nil)
