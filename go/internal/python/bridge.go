// Package python provides a subprocess bridge to the Typhoon ASR Python inference layer.
//
// The bridge spawns python/bridge_server.py as a child process and communicates
// via newline-delimited JSON over stdin/stdout. A single Python process is reused
// across requests so the NeMo model stays in memory (model loading takes ~30 s).
//
// Crash recovery: when the Python process exits unexpectedly, all in-flight
// requests are drained with an error and the process is restarted with
// exponential back-off (1 s → 2 s → 4 s … capped at 30 s).
//
// Backpressure: a semaphore (configured via MaxInFlight) limits the number of
// concurrent requests. Callers that exceed the limit receive an immediate error
// instead of blocking indefinitely.
package python

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
)

// BridgeRequest is the JSON envelope sent to the Python process via stdin.
type BridgeRequest struct {
	RequestID      string  `json:"request_id"`
	Action         string  `json:"action,omitempty"` // "transcribe" (default) or "ping"
	AudioPath      string  `json:"audio_path,omitempty"`
	AudioB64       string  `json:"audio_b64,omitempty"` // base64-encoded bytes
	ModelName      string  `json:"model_name,omitempty"`
	WithTimestamps bool    `json:"with_timestamps"`
	Device         string  `json:"device,omitempty"`
	LanguageHint   string  `json:"language_hint,omitempty"`
}

// WordTimestamp mirrors the dict produced by the Python inference layer.
type WordTimestamp struct {
	Word       string  `json:"word"`
	Start      float64 `json:"start"`
	End        float64 `json:"end"`
	Confidence float64 `json:"confidence,omitempty"`
}

// BridgeResponse is the JSON envelope read back from the Python process stdout.
type BridgeResponse struct {
	RequestID      string          `json:"request_id"`
	Text           string          `json:"text"`
	Timestamps     []WordTimestamp `json:"timestamps,omitempty"`
	Confidence     float64         `json:"confidence"`
	AudioDuration  float64         `json:"audio_duration"`
	ProcessingTime float64         `json:"processing_time"`
	Error          string          `json:"error,omitempty"`
	ModelLoaded    bool            `json:"model_loaded"`
	Device         string          `json:"device"`
}

// HealthResponse is returned by the Python process in response to a ping.
type HealthResponse struct {
	Alive       bool   `json:"alive"`
	ModelName   string `json:"model_name"`
	Device      string `json:"device"`
	ModelLoaded bool   `json:"model_loaded"`
}

// inflightReq holds an in-flight request's reply channel.
type inflightReq struct {
	ch chan BridgeResponse
}

// Bridge manages the Python subprocess and multiplexes concurrent requests.
type Bridge struct {
	mu         sync.Mutex
	cmd        *exec.Cmd
	stdin      io.WriteCloser
	inflight   map[string]*inflightReq
	inflightMu sync.Mutex

	requestsHandled atomic.Int64
	errors          atomic.Int64

	// closed is set by Shutdown to suppress automatic restarts.
	closed atomic.Bool

	// sem is a counting semaphore; nil means unlimited.
	sem chan struct{}

	scriptPath  string
	pythonBin   string
	modelName   string
	device      string
	maxInFlight int

	log *slog.Logger
}

// BridgeConfig configures the Python subprocess bridge.
type BridgeConfig struct {
	// Path to the Python interpreter (default: "python3").
	PythonBin string
	// Path to python/bridge_server.py (default: resolved relative to the binary).
	ScriptPath string
	// Default model name forwarded to Python if not overridden per-request.
	ModelName string
	// Default device ("auto", "cpu", "cuda").
	Device string
	// MaxInFlight limits how many requests can be in-flight simultaneously.
	// Requests that exceed this limit are rejected immediately.
	// 0 means unlimited (not recommended for production).
	MaxInFlight int
}

// NewBridge creates and starts the Python subprocess.
func NewBridge(cfg BridgeConfig) (*Bridge, error) {
	if cfg.PythonBin == "" {
		cfg.PythonBin = "python3"
	}
	if cfg.ScriptPath == "" {
		exe, err := os.Executable()
		if err != nil {
			return nil, fmt.Errorf("resolve executable path: %w", err)
		}
		// Assume the repo root is two levels above cmd/server/
		cfg.ScriptPath = filepath.Join(filepath.Dir(exe), "..", "..", "python", "bridge_server.py")
	}
	if cfg.ModelName == "" {
		cfg.ModelName = "scb10x/typhoon-asr-realtime"
	}
	if cfg.Device == "" {
		cfg.Device = "auto"
	}
	if cfg.MaxInFlight <= 0 {
		cfg.MaxInFlight = 64
	}

	b := &Bridge{
		scriptPath:  cfg.ScriptPath,
		pythonBin:   cfg.PythonBin,
		modelName:   cfg.ModelName,
		device:      cfg.Device,
		maxInFlight: cfg.MaxInFlight,
		inflight:    make(map[string]*inflightReq),
		sem:         make(chan struct{}, cfg.MaxInFlight),
		log:         slog.Default(),
	}

	if err := b.start(); err != nil {
		return nil, err
	}
	return b, nil
}

// start launches (or re-launches) the Python subprocess.
// Callers must not hold b.mu.
func (b *Bridge) start() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.log.Info("starting Python bridge", "script", b.scriptPath, "python", b.pythonBin)

	cmd := exec.Command(b.pythonBin, b.scriptPath,
		"--model", b.modelName,
		"--device", b.device,
	)
	cmd.Stderr = os.Stderr // forward Python logs to our stderr

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("get stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("get stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start Python bridge: %w", err)
	}

	b.cmd   = cmd
	b.stdin = stdin

	go b.readLoop(stdout)
	return nil
}

// readLoop reads newline-delimited JSON from the Python process stdout and
// dispatches responses to waiting goroutines.
// When the loop exits (process crash or EOF), it calls onCrash.
func (b *Bridge) readLoop(r io.Reader) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 4*1024*1024), 4*1024*1024) // 4 MB per line

	for scanner.Scan() {
		line := scanner.Bytes()
		var resp BridgeResponse
		if err := json.Unmarshal(line, &resp); err != nil {
			b.log.Warn("malformed response from Python", "line", string(line), "err", err)
			b.errors.Add(1)
			continue
		}

		b.inflightMu.Lock()
		p, ok := b.inflight[resp.RequestID]
		if ok {
			delete(b.inflight, resp.RequestID)
		}
		b.inflightMu.Unlock()

		if ok {
			p.ch <- resp
		} else {
			b.log.Warn("received response for unknown request_id", "id", resp.RequestID)
		}
	}

	if err := scanner.Err(); err != nil {
		b.log.Error("Python bridge stdout closed with error", "err", err)
	} else {
		b.log.Warn("Python bridge stdout closed (process exited?)")
	}

	b.onCrash()
}

// onCrash is called by readLoop when the Python process exits unexpectedly.
// It drains all in-flight requests with an error response, then triggers a
// restart unless Shutdown has been called.
func (b *Bridge) onCrash() {
	// Invalidate stdin so that concurrent writers fail fast.
	b.mu.Lock()
	b.stdin = nil
	b.mu.Unlock()

	// Drain all in-flight requests so their callers return an error immediately
	// rather than hanging until their context deadline.
	b.inflightMu.Lock()
	for id, p := range b.inflight {
		p.ch <- BridgeResponse{
			RequestID: id,
			Error:     "Python bridge process exited unexpectedly",
		}
		delete(b.inflight, id)
	}
	b.inflightMu.Unlock()

	if b.closed.Load() {
		return // intentional shutdown — do not restart
	}

	go b.restartWithBackoff()
}

// restartWithBackoff retries start() with exponential back-off until it
// succeeds or Shutdown is called.
func (b *Bridge) restartWithBackoff() {
	const maxBackoff = 30 * time.Second
	backoff := time.Second

	for {
		if b.closed.Load() {
			return
		}

		b.log.Info("restarting Python bridge", "backoff", backoff)
		time.Sleep(backoff)

		if b.closed.Load() {
			return
		}

		if err := b.start(); err != nil {
			b.log.Error("Python bridge restart failed", "err", err)
			backoff *= 2
			if backoff > maxBackoff {
				backoff = maxBackoff
			}
			continue
		}

		b.log.Info("Python bridge restarted successfully")
		return
	}
}

// Call sends a BridgeRequest to Python and waits for the response.
//
// It returns immediately with an error if:
//   - the number of in-flight requests has reached MaxInFlight (load-shedding), or
//   - the Python process is currently restarting (write to stdin fails), or
//   - ctx is cancelled before a response arrives.
func (b *Bridge) Call(ctx context.Context, req BridgeRequest) (BridgeResponse, error) {
	// --- backpressure: non-blocking semaphore acquire ---
	select {
	case b.sem <- struct{}{}:
		// acquired
	default:
		b.errors.Add(1)
		return BridgeResponse{}, fmt.Errorf(
			"bridge: too many concurrent requests (max %d); try again later",
			b.maxInFlight,
		)
	}
	defer func() { <-b.sem }()

	if req.ModelName == "" {
		req.ModelName = b.modelName
	}
	if req.Device == "" {
		req.Device = b.device
	}

	inf := &inflightReq{ch: make(chan BridgeResponse, 1)}

	b.inflightMu.Lock()
	b.inflight[req.RequestID] = inf
	b.inflightMu.Unlock()

	payload, err := json.Marshal(req)
	if err != nil {
		b.inflightMu.Lock()
		delete(b.inflight, req.RequestID)
		b.inflightMu.Unlock()
		return BridgeResponse{}, fmt.Errorf("marshal request: %w", err)
	}
	payload = append(payload, '\n')

	b.mu.Lock()
	stdin := b.stdin
	b.mu.Unlock()

	if stdin == nil {
		b.inflightMu.Lock()
		delete(b.inflight, req.RequestID)
		b.inflightMu.Unlock()
		b.errors.Add(1)
		return BridgeResponse{}, fmt.Errorf("bridge: Python process is not running (restarting)")
	}

	b.mu.Lock()
	_, werr := stdin.Write(payload)
	b.mu.Unlock()

	if werr != nil {
		b.inflightMu.Lock()
		delete(b.inflight, req.RequestID)
		b.inflightMu.Unlock()
		b.errors.Add(1)
		return BridgeResponse{}, fmt.Errorf("write to Python stdin: %w", werr)
	}

	select {
	case resp := <-inf.ch:
		b.requestsHandled.Add(1)
		if resp.Error != "" {
			b.errors.Add(1)
			return resp, fmt.Errorf("Python inference error: %s", resp.Error)
		}
		return resp, nil
	case <-ctx.Done():
		b.inflightMu.Lock()
		delete(b.inflight, req.RequestID)
		b.inflightMu.Unlock()
		return BridgeResponse{}, ctx.Err()
	}
}

// Ping sends a health-check to the Python process and returns whether it is alive.
// It reuses Call so error handling and request multiplexing are identical to inference calls.
func (b *Bridge) Ping(ctx context.Context) (HealthResponse, error) {
	resp, err := b.Call(ctx, BridgeRequest{
		RequestID: uuid.NewString(),
		Action:    "ping",
	})
	if err != nil {
		return HealthResponse{}, err
	}
	return HealthResponse{
		Alive:       true,
		ModelName:   b.modelName,
		Device:      resp.Device,
		ModelLoaded: resp.ModelLoaded,
	}, nil
}

// Stats returns cumulative counters for monitoring.
func (b *Bridge) Stats() (requestsHandled, errors int64) {
	return b.requestsHandled.Load(), b.errors.Load()
}

// PID returns the child process PID (0 if not running).
func (b *Bridge) PID() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.cmd != nil && b.cmd.Process != nil {
		return b.cmd.Process.Pid
	}
	return 0
}

// Shutdown terminates the Python subprocess gracefully.
// It sets the closed flag first so that onCrash does not trigger a restart.
func (b *Bridge) Shutdown() {
	b.closed.Store(true)

	b.mu.Lock()
	stdin := b.stdin
	cmd   := b.cmd
	b.stdin = nil
	b.mu.Unlock()

	if stdin != nil {
		_ = stdin.Close()
	}
	if cmd != nil && cmd.Process != nil {
		_ = cmd.Process.Signal(os.Interrupt)
		done := make(chan struct{})
		go func() {
			_ = cmd.Wait()
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			_ = cmd.Process.Kill()
		}
	}
}
