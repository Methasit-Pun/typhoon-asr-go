// Command server starts the Typhoon ASR gRPC server.
//
// Configuration is loaded from CLI flags with environment variable fallbacks
// (see internal/config for the full list). Example:
//
//	server --addr :50051 --model scb10x/typhoon-asr-realtime --device cuda
//	TYPHOON_DEVICE=cuda server
package main

import (
	"fmt"
	"os"

	"github.com/scb10x/typhoon-asr/go/internal/config"
)

func main() {
	cfg := config.Load()

	app, err := NewApp(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "startup error: %v\n", err)
		os.Exit(1)
	}

	if err := app.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}
}
