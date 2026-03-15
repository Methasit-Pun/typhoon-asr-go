// Package middleware provides reusable gRPC server interceptors.
package middleware

import (
	"context"
	"log/slog"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/status"
)

// UnaryLogging returns a unary server interceptor that logs method, latency,
// and gRPC status code for every call.
func UnaryLogging(log *slog.Logger) grpc.UnaryServerInterceptor {
	return func(
		ctx context.Context,
		req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (interface{}, error) {
		start := time.Now()
		resp, err := handler(ctx, req)
		log.Info("unary rpc",
			"method",     info.FullMethod,
			"latency_ms", time.Since(start).Milliseconds(),
			"code",       grpcCode(err),
		)
		return resp, err
	}
}

// StreamLogging returns a stream server interceptor that logs method, latency,
// and gRPC status code for every streaming call.
func StreamLogging(log *slog.Logger) grpc.StreamServerInterceptor {
	return func(
		srv interface{},
		ss grpc.ServerStream,
		info *grpc.StreamServerInfo,
		handler grpc.StreamHandler,
	) error {
		start := time.Now()
		err := handler(srv, ss)
		log.Info("stream rpc",
			"method",     info.FullMethod,
			"latency_ms", time.Since(start).Milliseconds(),
			"code",       grpcCode(err),
		)
		return err
	}
}

// grpcCode extracts the gRPC status code string from an error (or "OK").
func grpcCode(err error) string {
	if err == nil {
		return "OK"
	}
	return status.Code(err).String()
}
