.PHONY: proto build run tidy

PROTO_DIR  := ../proto
PB_OUT     := ./pkg/pb
BINARY     := ./bin/server

# Generate Go bindings from proto/asr.proto.
# Requires: protoc, protoc-gen-go, protoc-gen-go-grpc
#   go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
#   go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
proto:
	protoc \
	  --go_out=$(PB_OUT)      --go_opt=paths=source_relative \
	  --go-grpc_out=$(PB_OUT) --go-grpc_opt=paths=source_relative \
	  -I $(PROTO_DIR) $(PROTO_DIR)/asr.proto

# Fetch/update dependencies and generate go.sum.
tidy:
	go mod tidy

# Build the server binary.
build: proto
	go build -o $(BINARY) ./cmd/server

# Run directly (no binary; useful during development).
run:
	go run ./cmd/server $(ARGS)
