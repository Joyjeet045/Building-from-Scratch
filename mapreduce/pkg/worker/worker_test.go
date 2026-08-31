/*
worker_test.go - worker shutdown behaviour over a real RPC connection

Covers the difference between the two ways a worker's loop can end: the
coordinator handing out an exit task, and the coordinator simply vanishing.
Both must leave the process with a zero exit status, because a worker that
reports failure on a normal job completion turns every pod in a cluster into a
restart loop.
*/
package worker

import (
	"errors"
	"io"
	"net/rpc"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/config"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/coordinator"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/interfaces"
)

type noopMapper struct{}

func (noopMapper) Map(interfaces.MapInput, func(string, string)) {}

type noopReducer struct{}

func (noopReducer) Reduce(interfaces.ReduceInput, func(string)) {}

func serveCoordinator(t *testing.T, mappers, reducers int) (*coordinator.Coordinator, *config.Config, string) {
	t.Helper()

	cfg := config.Default()
	cfg.JobID = "worker-test"
	cfg.Root = t.TempDir()
	cfg.InputDir = filepath.Join(cfg.Root, "in")
	cfg.NumMappers = mappers
	cfg.NumReducers = reducers
	cfg.Address = "127.0.0.1:0"
	cfg.HeartbeatInterval = 10 * time.Millisecond
	cfg.WorkerTimeout = 200 * time.Millisecond
	cfg.Verbose = false
	cfg.Mapper = noopMapper{}
	cfg.Reducer = noopReducer{}

	if err := os.MkdirAll(cfg.InputDir, 0o755); err != nil {
		t.Fatalf("mkdir input: %v", err)
	}

	groups := make([][]string, mappers)
	for i := range groups {
		groups[i] = nil
	}

	c := coordinator.New(cfg, groups)
	if err := c.Layout().Prepare(mappers); err != nil {
		t.Fatalf("prepare layout: %v", err)
	}
	address, err := c.Serve()
	if err != nil {
		t.Fatalf("serve: %v", err)
	}
	t.Cleanup(func() { c.Close() })

	return c, cfg, address
}

func TestWorkerExitsCleanlyWhenTheCoordinatorDisappears(t *testing.T) {
	c, cfg, address := serveCoordinator(t, 1, 1)

	w, err := Dial(cfg, address)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer w.Close()

	done := make(chan error, 1)
	go func() { done <- w.Run() }()

	time.Sleep(50 * time.Millisecond)
	c.Close()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("a worker whose coordinator vanished must exit cleanly, got %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("worker did not notice the coordinator was gone")
	}
}

func TestWorkerExitsCleanlyOnExitTask(t *testing.T) {
	_, cfg, address := serveCoordinator(t, 0, 0)

	w, err := Dial(cfg, address)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer w.Close()

	done := make(chan error, 1)
	go func() { done <- w.Run() }()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("expected a clean exit on TaskExit, got %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("worker did not exit after the job finished")
	}
}

func TestDrainWaitsForWorkersToCollectTheirExit(t *testing.T) {
	c, cfg, address := serveCoordinator(t, 0, 0)

	w, err := Dial(cfg, address)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer w.Close()

	if pending := c.DrainWorkers(100 * time.Millisecond); pending != 1 {
		t.Fatalf("a registered worker that has not exited should be pending, got %d", pending)
	}

	if err := w.Run(); err != nil {
		t.Fatalf("run: %v", err)
	}

	if pending := c.DrainWorkers(time.Second); pending != 0 {
		t.Fatalf("drain should report nothing pending after the worker exits, got %d", pending)
	}
}

func TestDisconnectErrorsAreRecognised(t *testing.T) {
	for _, err := range []error{rpc.ErrShutdown, io.EOF, io.ErrUnexpectedEOF} {
		if !isDisconnect(err) {
			t.Errorf("%v should count as a disconnect", err)
		}
	}
	if isDisconnect(errors.New("task failed: bad record")) {
		t.Error("a task error must not be mistaken for a disconnect")
	}
}
