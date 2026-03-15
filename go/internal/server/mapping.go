package server

import (
	"time"

	"google.golang.org/protobuf/types/known/durationpb"

	"github.com/scb10x/typhoon-asr/go/internal/python"
	pb "github.com/scb10x/typhoon-asr/go/pkg/pb"
)

// mapResponse converts a python.BridgeResponse into a pb.TranscribeResponse.
func mapResponse(resp python.BridgeResponse, sessionID, reqID string) *pb.TranscribeResponse {
	pbResp := &pb.TranscribeResponse{
		Text:      resp.Text,
		Confidence: float32(resp.Confidence),
		SessionId: sessionID,
		RequestId: reqID,
	}

	if resp.AudioDuration > 0 {
		pbResp.AudioDuration  = durationpb.New(floatToDuration(resp.AudioDuration))
		pbResp.ProcessingTime = durationpb.New(floatToDuration(resp.ProcessingTime))
		pbResp.RealTimeFactor = float32(resp.ProcessingTime / resp.AudioDuration)
	}

	if resp.Text == "" {
		pbResp.Status = pb.TranscribeStatus_TRANSCRIBE_STATUS_NO_SPEECH
	} else {
		pbResp.Status = pb.TranscribeStatus_TRANSCRIBE_STATUS_OK
	}

	pbResp.WordTimestamps = mapTimestamps(resp.Timestamps)
	return pbResp
}

// mapTimestamps converts a slice of python.WordTimestamp to proto equivalents.
func mapTimestamps(ts []python.WordTimestamp) []*pb.WordTimestamp {
	out := make([]*pb.WordTimestamp, 0, len(ts))
	for _, t := range ts {
		out = append(out, &pb.WordTimestamp{
			Word:       t.Word,
			StartTime:  durationpb.New(floatToDuration(t.Start)),
			EndTime:    durationpb.New(floatToDuration(t.End)),
			Confidence: float32(t.Confidence),
		})
	}
	return out
}

// floatToDuration converts a float64 number of seconds to a time.Duration.
func floatToDuration(secs float64) time.Duration {
	return time.Duration(secs * float64(time.Second))
}
