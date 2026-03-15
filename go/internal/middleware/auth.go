package middleware

import (
	"context"
	"strings"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// BearerToken returns a unary interceptor that requires
// "Authorization: Bearer <token>" metadata on every call.
// If token is empty the interceptor is a pass-through (auth disabled).
func BearerToken(token string) grpc.UnaryServerInterceptor {
	if token == "" {
		return func(ctx context.Context, req interface{}, _ *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
			return handler(ctx, req)
		}
	}
	return func(ctx context.Context, req interface{}, _ *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		if err := checkBearer(ctx, token); err != nil {
			return nil, err
		}
		return handler(ctx, req)
	}
}

// BearerTokenStream returns a stream interceptor with the same semantics as BearerToken.
func BearerTokenStream(token string) grpc.StreamServerInterceptor {
	if token == "" {
		return func(srv interface{}, ss grpc.ServerStream, _ *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
			return handler(srv, ss)
		}
	}
	return func(srv interface{}, ss grpc.ServerStream, _ *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		if err := checkBearer(ss.Context(), token); err != nil {
			return err
		}
		return handler(srv, ss)
	}
}

func checkBearer(ctx context.Context, want string) error {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return status.Error(codes.Unauthenticated, "missing request metadata")
	}
	vals := md.Get("authorization")
	if len(vals) == 0 {
		return status.Error(codes.Unauthenticated, "missing Authorization header")
	}
	got := vals[0]
	const prefix = "Bearer "
	if !strings.HasPrefix(got, prefix) {
		return status.Error(codes.Unauthenticated, "Authorization must use Bearer scheme")
	}
	if got[len(prefix):] != want {
		return status.Error(codes.Unauthenticated, "invalid token")
	}
	return nil
}
