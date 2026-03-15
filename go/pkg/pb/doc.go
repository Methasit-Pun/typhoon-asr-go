// Package pb contains the Go bindings generated from proto/asr.proto.
//
// Regenerate after editing the proto file:
//
//	make proto        (uses the Makefile in the repo root go/ directory)
//
// or manually:
//
//	protoc \
//	  --go_out=.      --go_opt=paths=source_relative \
//	  --go-grpc_out=. --go-grpc_opt=paths=source_relative \
//	  -I ../../proto ../../proto/asr.proto
//
// Run from: go/pkg/pb/
//
//go:generate protoc --go_out=. --go_opt=paths=source_relative --go-grpc_out=. --go-grpc_opt=paths=source_relative -I ../../proto ../../proto/asr.proto
package pb
