package main

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"syscall"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/reflection"

	"github.com/scb10x/typhoon-asr/go/internal/config"
	"github.com/scb10x/typhoon-asr/go/internal/middleware"
	"github.com/scb10x/typhoon-asr/go/internal/python"
	"github.com/scb10x/typhoon-asr/go/internal/server"
	pb "github.com/scb10x/typhoon-asr/go/pkg/pb"
)

// App owns every long-lived resource created at startup.
// It is the single place responsible for wiring components together
// and coordinating graceful shutdown.
type App struct {
	cfg        *config.Config
	log        *slog.Logger
	bridge     *python.Bridge
	grpcServer *grpc.Server
	listener   net.Listener
}

// NewApp validates configuration, wires all components, and returns a ready-to-run App.
// No goroutines are started here; call Run() or RunContext() to begin serving.
func NewApp(cfg *config.Config) (*App, error) {
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	log := cfg.NewLogger()
	slog.SetDefault(log)

	bridge, err := python.NewBridge(python.BridgeConfig{
		PythonBin:   cfg.PythonBin,
		ScriptPath:  cfg.ScriptPath,
		ModelName:   cfg.ModelName,
		Device:      cfg.Device,
		MaxInFlight: cfg.MaxInFlight,
	})
	if err != nil {
		return nil, fmt.Errorf("start Python bridge: %w", err)
	}

	validator := server.ValidatorConfig{
		MaxAudioBytes:        cfg.MaxAudioBytes,
		MaxAudioDurationSecs: cfg.MaxAudioDurationSecs,
		AllowedAudioDir:      cfg.AllowedAudioDir,
	}

	grpcOpts := []grpc.ServerOption{
		grpc.ChainUnaryInterceptor(
			middleware.BearerToken(cfg.AuthToken),
			middleware.UnaryLogging(log),
		),
		grpc.ChainStreamInterceptor(
			middleware.BearerTokenStream(cfg.AuthToken),
			middleware.StreamLogging(log),
		),
	}

	if creds, err := buildTLSCredentials(cfg); err != nil {
		bridge.Shutdown()
		return nil, fmt.Errorf("build TLS credentials: %w", err)
	} else if creds != nil {
		grpcOpts = append(grpcOpts, grpc.Creds(creds))
		log.Info("TLS enabled", "cert", cfg.TLSCertFile, "mtls", cfg.TLSClientCA != "")
	}

	grpcSrv := grpc.NewServer(grpcOpts...)
	pb.RegisterASRServiceServer(grpcSrv, server.New(bridge, validator, log))
	reflection.Register(grpcSrv)

	lis, err := net.Listen("tcp", cfg.Addr)
	if err != nil {
		bridge.Shutdown()
		return nil, fmt.Errorf("listen on %s: %w", cfg.Addr, err)
	}

	return &App{
		cfg:        cfg,
		log:        log,
		bridge:     bridge,
		grpcServer: grpcSrv,
		listener:   lis,
	}, nil
}

// Run blocks until SIGINT or SIGTERM, then shuts down gracefully.
func (a *App) Run() error {
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(sig)

	return a.serveUntil(func() {
		s := <-sig
		a.log.Info("received signal, shutting down", "signal", s)
	})
}

// RunContext blocks until ctx is cancelled, then shuts down gracefully.
// Intended for tests and embedded use where the caller controls the lifetime.
func (a *App) RunContext(ctx context.Context) error {
	return a.serveUntil(func() { <-ctx.Done() })
}

// serveUntil starts the gRPC server once, calls wait() to block, then shuts down.
// Both Run and RunContext share this so the listener is consumed exactly once.
func (a *App) serveUntil(wait func()) error {
	a.log.Info("Typhoon ASR gRPC server listening",
		"addr",   a.cfg.Addr,
		"model",  a.cfg.ModelName,
		"device", a.cfg.Device,
	)

	errCh := make(chan error, 1)
	go func() {
		errCh <- a.grpcServer.Serve(a.listener)
	}()

	done := make(chan struct{})
	go func() { wait(); close(done) }()

	select {
	case err := <-errCh:
		a.shutdown()
		return fmt.Errorf("gRPC server stopped unexpectedly: %w", err)
	case <-done:
		a.shutdown()
		return nil
	}
}

// shutdown stops the gRPC server gracefully, then terminates the Python bridge.
func (a *App) shutdown() {
	a.grpcServer.GracefulStop()
	a.bridge.Shutdown()
	a.log.Info("shutdown complete")
}

// buildTLSCredentials constructs gRPC transport credentials from the config.
// Returns (nil, nil) when TLS is not configured.
func buildTLSCredentials(cfg *config.Config) (credentials.TransportCredentials, error) {
	if cfg.TLSCertFile == "" {
		return nil, nil
	}

	cert, err := tls.LoadX509KeyPair(cfg.TLSCertFile, cfg.TLSKeyFile)
	if err != nil {
		return nil, fmt.Errorf("load TLS cert/key: %w", err)
	}

	tlsCfg := &tls.Config{
		Certificates: []tls.Certificate{cert},
		MinVersion:   tls.VersionTLS12,
	}

	if cfg.TLSClientCA != "" {
		pem, err := os.ReadFile(cfg.TLSClientCA)
		if err != nil {
			return nil, fmt.Errorf("read TLS client CA: %w", err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(pem) {
			return nil, fmt.Errorf("parse TLS client CA PEM: no valid certificates found")
		}
		tlsCfg.ClientCAs  = pool
		tlsCfg.ClientAuth = tls.RequireAndVerifyClientCert
	}

	return credentials.NewTLS(tlsCfg), nil
}
