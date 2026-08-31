/*
mapreduce.go - the single entry point a user program calls

Execute - run the job described by cfg, dispatching on cfg.Mode
Result  - what the job produced and how long it took

Mirrors the original's mapreduce.Execute(cfg). Mode picks the role:

	local       coordinator plus in-process workers, one binary, no cluster
	cluster     coordinator plus worker subprocesses on this host
	coordinator schedule only, wait for workers to dial in
	worker      execute tasks handed out by a remote coordinator
*/
package mapreduce

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"sync"
	"time"

	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/config"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/coordinator"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/storage"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/worker"
)

type Result struct {
	JobID           string
	OutputDir       string
	OutputFiles     []string
	Elapsed         time.Duration
	MapElapsed      time.Duration
	ReduceElapsed   time.Duration
	Counters        map[string]int64
	WorkersLost     int
	Reassignments   int
	BackupsLaunched int
}

func Execute(cfg *config.Config) (*Result, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	switch cfg.Mode {
	case config.ModeWorker:
		return nil, runWorker(cfg)
	case config.ModeCoordinator:
		return runCoordinatorOnly(cfg)
	case config.ModeCluster:
		return runCluster(cfg)
	default:
		return runLocal(cfg)
	}
}

func prepare(cfg *config.Config) (*coordinator.Coordinator, string, error) {
	files, err := storage.ListInputFiles(cfg.InputDir)
	if err != nil {
		return nil, "", err
	}
	groups := storage.SplitFiles(files, cfg.NumMappers)

	c := coordinator.New(cfg, groups)
	if err := c.Layout().Prepare(len(groups)); err != nil {
		return nil, "", err
	}

	address, err := c.Serve()
	if err != nil {
		return nil, "", err
	}

	if cfg.Verbose {
		log.Printf("[job] %s: %d input files, %d map tasks, %d reduce tasks, listening on %s",
			c.Layout().JobID, len(files), len(groups), cfg.NumReducers, address)
	}
	return c, address, nil
}

func collect(cfg *config.Config, c *coordinator.Coordinator, elapsed time.Duration) (*Result, error) {
	lost, reassigned, backups := c.Stats()

	outputs := make([]string, cfg.NumReducers)
	for i := range outputs {
		outputs[i] = c.Layout().OutputPath(i)
	}

	return &Result{
		JobID:           c.Layout().JobID,
		OutputDir:       c.Layout().OutputDir(),
		OutputFiles:     outputs,
		Elapsed:         elapsed,
		Counters:        c.Counters().Snapshot(),
		WorkersLost:     lost,
		Reassignments:   reassigned,
		BackupsLaunched: backups,
	}, nil
}

func runLocal(cfg *config.Config) (*Result, error) {
	c, address, err := prepare(cfg)
	if err != nil {
		return nil, err
	}
	defer c.Close()

	start := time.Now()

	var wg sync.WaitGroup
	errs := make(chan error, cfg.Workers)

	for i := 0; i < cfg.Workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			w, err := worker.Dial(cfg, address)
			if err != nil {
				errs <- err
				return
			}
			defer w.Close()
			if err := w.Run(); err != nil {
				errs <- err
			}
		}()
	}

	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			return nil, err
		}
	}

	select {
	case <-c.Done():
	default:
		return nil, fmt.Errorf("mapreduce: workers exited before the job completed")
	}

	if err := c.Err(); err != nil {
		return nil, err
	}

	return collect(cfg, c, time.Since(start))
}

func runCluster(cfg *config.Config) (*Result, error) {
	c, address, err := prepare(cfg)
	if err != nil {
		return nil, err
	}
	defer c.Close()

	self, err := os.Executable()
	if err != nil {
		return nil, err
	}

	start := time.Now()
	procs := make([]*exec.Cmd, 0, cfg.Workers)

	for i := 0; i < cfg.Workers; i++ {
		cmd := exec.Command(self, "-mode=worker", "-addr="+address)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Start(); err != nil {
			return nil, err
		}
		procs = append(procs, cmd)
	}

	waitErr := c.Wait(cfg.TaskTimeout * time.Duration(len(cfg.InputDir)+10))
	for _, cmd := range procs {
		cmd.Wait()
	}
	if waitErr != nil {
		return nil, waitErr
	}
	if err := c.Err(); err != nil {
		return nil, err
	}

	return collect(cfg, c, time.Since(start))
}

func runCoordinatorOnly(cfg *config.Config) (*Result, error) {
	c, address, err := prepare(cfg)
	if err != nil {
		return nil, err
	}
	defer c.Close()

	log.Printf("[job] coordinator ready on %s, waiting for workers", address)
	start := time.Now()

	<-c.Done()
	elapsed := time.Since(start)

	if pending := c.DrainWorkers(cfg.WorkerTimeout); pending > 0 {
		log.Printf("[job] %d worker(s) did not collect an exit signal before shutdown", pending)
	}

	if err := c.Err(); err != nil {
		return nil, err
	}

	return collect(cfg, c, elapsed)
}

func runWorker(cfg *config.Config) error {
	w, err := worker.Dial(cfg, cfg.Address)
	if err != nil {
		return err
	}
	defer w.Close()

	if cfg.Verbose {
		log.Printf("[%s] connected to %s", w.ID(), cfg.Address)
	}
	return w.Run()
}
