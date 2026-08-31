/*
worker.go - pulls tasks from the coordinator and runs them

Worker - one task-executing client
Run    - register, then loop until the coordinator says exit

The loop is: ask for a task, run it, report the outcome. A background
heartbeat runs the whole time so the coordinator can tell a slow task from a
dead worker. Task failures are reported rather than fatal, letting the
coordinator decide whether to retry elsewhere.
*/
package worker

import (
	"errors"
	"fmt"
	"io"
	"log"
	"net/rpc"
	"os"
	"time"

	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/config"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/counters"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/mapper"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/protocol"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/reducer"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/storage"
)

type Worker struct {
	cfg    *config.Config
	client *rpc.Client
	id     string
	layout storage.Layout
	stop   chan struct{}
}

func Dial(cfg *config.Config, address string) (*Worker, error) {
	var client *rpc.Client
	var err error

	deadline := time.Now().Add(10 * time.Second)
	for {
		client, err = rpc.Dial("tcp", address)
		if err == nil {
			break
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("worker: cannot reach coordinator at %s: %w", address, err)
		}
		time.Sleep(50 * time.Millisecond)
	}

	host, _ := os.Hostname()
	var reply protocol.RegisterReply
	if err := client.Call(protocol.RegisterMethod, &protocol.RegisterArgs{Host: host}, &reply); err != nil {
		client.Close()
		return nil, err
	}

	return &Worker{
		cfg:    cfg,
		client: client,
		id:     reply.WorkerID,
		layout: storage.New(reply.Root, reply.JobID),
		stop:   make(chan struct{}),
	}, nil
}

func (w *Worker) ID() string { return w.id }

// A worker cannot make progress without a coordinator, and the coordinator
// only goes away once the job is over, so a dropped connection ends the run
// normally rather than reporting a task failure.
func isDisconnect(err error) bool {
	return errors.Is(err, rpc.ErrShutdown) ||
		errors.Is(err, io.EOF) ||
		errors.Is(err, io.ErrUnexpectedEOF)
}

func (w *Worker) Close() error {
	close(w.stop)
	return w.client.Close()
}

func (w *Worker) Run() error {
	go w.heartbeat()

	for {
		var reply protocol.RequestTaskReply
		args := protocol.RequestTaskArgs{WorkerID: w.id}
		if err := w.client.Call(protocol.RequestMethod, &args, &reply); err != nil {
			if isDisconnect(err) {
				log.Printf("[%s] coordinator closed the connection, shutting down", w.id)
				return nil
			}
			return err
		}

		switch reply.Kind {
		case protocol.TaskExit:
			return nil
		case protocol.TaskNone:
			wait := reply.RetryAfter
			if wait <= 0 {
				wait = 20 * time.Millisecond
			}
			select {
			case <-w.stop:
				return nil
			case <-time.After(wait):
			}
		case protocol.TaskMap, protocol.TaskReduce:
			w.execute(reply)
		default:
			return fmt.Errorf("worker: unexpected task kind %v", reply.Kind)
		}
	}
}

func (w *Worker) execute(t protocol.RequestTaskReply) {
	set := counters.New()
	start := time.Now()

	var err error
	switch t.Kind {
	case protocol.TaskMap:
		err = mapper.Run(mapper.Options{
			Layout:      w.layout,
			TaskID:      t.TaskID,
			InputFiles:  t.InputFiles,
			NumReducers: t.NumReducers,
			SpillBytes:  w.cfg.SpillBytes,
			Mapper:      w.cfg.Mapper,
			Combiner:    w.cfg.Combiner,
			Counters:    set,
		})
	case protocol.TaskReduce:
		err = reducer.Run(reducer.Options{
			Layout:     w.layout,
			TaskID:     t.TaskID,
			NumMappers: t.NumMappers,
			Reducer:    w.cfg.Reducer,
			Counters:   set,
		})
	}

	report := protocol.ReportTaskArgs{
		WorkerID: w.id,
		Kind:     t.Kind,
		TaskID:   t.TaskID,
		Attempt:  t.Attempt,
		Success:  err == nil,
		Counters: set.Snapshot(),
		Elapsed:  time.Since(start),
	}
	if err != nil {
		report.Err = err.Error()
		if w.cfg.Verbose {
			log.Printf("[%s] %s-%d failed: %v", w.id, t.Kind, t.TaskID, err)
		}
	}

	var reply protocol.ReportTaskReply
	if callErr := w.client.Call(protocol.ReportMethod, &report, &reply); callErr != nil && w.cfg.Verbose {
		log.Printf("[%s] could not report %s-%d: %v", w.id, t.Kind, t.TaskID, callErr)
	}
}

func (w *Worker) heartbeat() {
	interval := w.cfg.HeartbeatInterval
	if interval <= 0 {
		interval = time.Second
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-w.stop:
			return
		case <-ticker.C:
			var reply protocol.HeartbeatReply
			args := protocol.HeartbeatArgs{WorkerID: w.id}
			if err := w.client.Call(protocol.HeartbeatMethod, &args, &reply); err != nil {
				return
			}
		}
	}
}
