/*
coordinator.go - schedules a job's tasks and tracks worker liveness

Coordinator  - the RPC service workers talk to
New          - builds one from a config and its input split
Serve        - listens for workers
Wait         - blocks until every phase completes

Holds the authoritative task table. A task is idle, running, or done; workers
pull idle tasks and report outcomes. Three recovery mechanisms the blog does
not have:

  - a task whose worker stops heartbeating, or which exceeds TaskTimeout, goes
    back to idle and is handed to someone else
  - a straggler past BackupThreshold times the median duration is duplicated,
    and whichever attempt reports first wins
  - a late attempt for an already-finished task is rejected, so duplicate work
    can never corrupt output

Reduce tasks are withheld until every map task is done, because a reducer must
see all mappers' partitions.
*/
package coordinator

import (
	"fmt"
	"log"
	"net"
	"net/rpc"
	"sort"
	"sync"
	"time"

	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/config"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/counters"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/protocol"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/storage"
)

type taskState int

const (
	taskIdle taskState = iota
	taskRunning
	taskDone
)

type task struct {
	id         int
	kind       protocol.TaskKind
	inputFiles []string
	state      taskState
	attempt    int
	started    time.Time
	finished   time.Time
	workers    map[string]int
	failures   int
	backup     bool
}

type worker struct {
	id       string
	host     string
	lastSeen time.Time
	alive    bool
	exited   bool
}

type Coordinator struct {
	mu sync.Mutex

	cfg    *config.Config
	layout storage.Layout

	mapTasks    []*task
	reduceTasks []*task

	workers  map[string]*worker
	nextID   int
	counters *counters.Set

	phase      string
	startedAt  time.Time
	finishedAt time.Time
	jobErr     error
	done       chan struct{}
	closeOnce  sync.Once

	durations       []time.Duration
	lostWorkers     int
	reassignments   int
	backupsLaunched int

	listener net.Listener
	server   *rpc.Server
}

func New(cfg *config.Config, inputGroups [][]string) *Coordinator {
	layout := storage.New(cfg.Root, cfg.ResolveJobID())

	c := &Coordinator{
		cfg:       cfg,
		layout:    layout,
		workers:   make(map[string]*worker),
		counters:  counters.New(),
		phase:     "map",
		startedAt: time.Now(),
		done:      make(chan struct{}),
	}

	for i, files := range inputGroups {
		c.mapTasks = append(c.mapTasks, &task{
			id:         i,
			kind:       protocol.TaskMap,
			inputFiles: files,
			workers:    make(map[string]int),
		})
	}
	for i := 0; i < cfg.NumReducers; i++ {
		c.reduceTasks = append(c.reduceTasks, &task{
			id:      i,
			kind:    protocol.TaskReduce,
			workers: make(map[string]int),
		})
	}
	return c
}

func (c *Coordinator) Layout() storage.Layout { return c.layout }

func (c *Coordinator) NumMapTasks() int { return len(c.mapTasks) }

func (c *Coordinator) Serve() (string, error) {
	server := rpc.NewServer()
	if err := server.RegisterName("Coordinator", c); err != nil {
		return "", err
	}
	listener, err := net.Listen("tcp", c.cfg.Address)
	if err != nil {
		return "", err
	}

	c.mu.Lock()
	c.server = server
	c.listener = listener
	c.mu.Unlock()

	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			go server.ServeConn(conn)
		}
	}()

	go c.monitor()
	return listener.Addr().String(), nil
}

func (c *Coordinator) Close() error {
	c.mu.Lock()
	listener := c.listener
	c.listener = nil
	c.mu.Unlock()
	if listener != nil {
		return listener.Close()
	}
	return nil
}

// Keeps serving until every live worker has collected its exit signal, so a
// normal shutdown never looks like a dropped connection to a polling worker.
func (c *Coordinator) DrainWorkers(timeout time.Duration) int {
	deadline := time.Now().Add(timeout)
	for {
		c.mu.Lock()
		pending := 0
		for _, w := range c.workers {
			if w.exited {
				continue
			}
			if time.Since(w.lastSeen) <= c.cfg.WorkerTimeout {
				pending++
			}
		}
		c.mu.Unlock()

		if pending == 0 || time.Now().After(deadline) {
			return pending
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func (c *Coordinator) Wait(timeout time.Duration) error {
	select {
	case <-c.done:
		return nil
	case <-time.After(timeout):
		return fmt.Errorf("coordinator: job did not finish within %s", timeout)
	}
}

func (c *Coordinator) Done() <-chan struct{} { return c.done }

func (c *Coordinator) Register(args *protocol.RegisterArgs, reply *protocol.RegisterReply) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.nextID++
	id := fmt.Sprintf("worker-%d", c.nextID)
	c.workers[id] = &worker{id: id, host: args.Host, lastSeen: time.Now(), alive: true}

	reply.WorkerID = id
	reply.JobID = c.layout.JobID
	reply.Root = c.layout.Root
	reply.NumMappers = len(c.mapTasks)
	reply.NumReducers = len(c.reduceTasks)
	reply.HeartbeatInterval = c.cfg.HeartbeatInterval

	c.logf("registered %s from %s", id, args.Host)
	return nil
}

func (c *Coordinator) Heartbeat(args *protocol.HeartbeatArgs, reply *protocol.HeartbeatReply) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	w, ok := c.workers[args.WorkerID]
	if ok {
		w.lastSeen = time.Now()
		w.alive = true
	}
	reply.Known = ok
	return nil
}

func (c *Coordinator) RequestTask(args *protocol.RequestTaskArgs, reply *protocol.RequestTaskReply) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if w, ok := c.workers[args.WorkerID]; ok {
		w.lastSeen = time.Now()
		w.alive = true
	}

	if c.jobErr != nil {
		if w, ok := c.workers[args.WorkerID]; ok {
			w.exited = true
		}
		reply.Kind = protocol.TaskExit
		return nil
	}

	c.reclaimLocked()

	if t := c.pickLocked(c.mapTasks, args.WorkerID); t != nil {
		c.assignLocked(t, args.WorkerID, reply)
		return nil
	}

	if !c.allDoneLocked(c.mapTasks) {
		reply.Kind = protocol.TaskNone
		reply.RetryAfter = c.cfg.HeartbeatInterval
		return nil
	}

	if c.phase == "map" {
		c.phase = "reduce"
		c.durations = c.durations[:0]
		c.logf("map phase complete, starting reduce")
	}

	if t := c.pickLocked(c.reduceTasks, args.WorkerID); t != nil {
		c.assignLocked(t, args.WorkerID, reply)
		return nil
	}

	if !c.allDoneLocked(c.reduceTasks) {
		reply.Kind = protocol.TaskNone
		reply.RetryAfter = c.cfg.HeartbeatInterval
		return nil
	}

	c.finishLocked()
	if w, ok := c.workers[args.WorkerID]; ok {
		w.exited = true
	}
	reply.Kind = protocol.TaskExit
	return nil
}

func (c *Coordinator) ReportTask(args *protocol.ReportTaskArgs, reply *protocol.ReportTaskReply) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if w, ok := c.workers[args.WorkerID]; ok {
		w.lastSeen = time.Now()
		w.alive = true
	}

	t := c.lookupLocked(args.Kind, args.TaskID)
	if t == nil {
		reply.Accepted = false
		return nil
	}

	if t.state == taskDone {
		reply.Accepted = false
		c.logf("discarding late %s-%d from %s, already complete", args.Kind, args.TaskID, args.WorkerID)
		return nil
	}

	if !args.Success {
		t.failures++
		t.state = taskIdle
		delete(t.workers, args.WorkerID)
		reply.Accepted = false
		c.logf("%s-%d failed on %s: %s", args.Kind, args.TaskID, args.WorkerID, args.Err)

		if max := c.cfg.MaxTaskAttempts; max > 0 && t.failures >= max {
			c.failLocked(fmt.Errorf("mapreduce: %s-%d failed %d times, last error: %s",
				args.Kind, args.TaskID, t.failures, args.Err))
		}
		return nil
	}

	t.state = taskDone
	t.finished = time.Now()
	c.durations = append(c.durations, args.Elapsed)
	c.counters.Merge(args.Counters)
	reply.Accepted = true

	c.logf("%s-%d done by %s in %s", args.Kind, args.TaskID, args.WorkerID, args.Elapsed.Round(time.Millisecond))

	if c.allDoneLocked(c.mapTasks) && c.allDoneLocked(c.reduceTasks) {
		c.finishLocked()
	}
	return nil
}

func (c *Coordinator) Status(args *protocol.StatusArgs, reply *protocol.StatusReply) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	reply.Phase = c.phase
	reply.MapTotal = len(c.mapTasks)
	reply.ReduceTotal = len(c.reduceTasks)
	reply.MapDone = countDone(c.mapTasks)
	reply.ReduceDone = countDone(c.reduceTasks)
	reply.WorkersLost = c.lostWorkers
	reply.BackupsLaunched = c.backupsLaunched
	reply.Reassignments = c.reassignments
	reply.Counters = c.counters.Snapshot()

	for _, w := range c.workers {
		if w.alive {
			reply.WorkersAlive++
		}
	}

	select {
	case <-c.done:
		reply.Done = true
		reply.Elapsed = c.finishedAt.Sub(c.startedAt)
	default:
		reply.Elapsed = time.Since(c.startedAt)
	}
	return nil
}

func (c *Coordinator) Counters() *counters.Set { return c.counters }

func (c *Coordinator) Stats() (lost, reassigned, backups int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.lostWorkers, c.reassignments, c.backupsLaunched
}

func (c *Coordinator) lookupLocked(kind protocol.TaskKind, id int) *task {
	var list []*task
	switch kind {
	case protocol.TaskMap:
		list = c.mapTasks
	case protocol.TaskReduce:
		list = c.reduceTasks
	default:
		return nil
	}
	if id < 0 || id >= len(list) {
		return nil
	}
	return list[id]
}

func (c *Coordinator) pickLocked(list []*task, workerID string) *task {
	for _, t := range list {
		if t.state == taskIdle {
			return t
		}
	}
	if !c.cfg.EnableBackupTasks {
		return nil
	}
	return c.pickBackupLocked(list, workerID)
}

func (c *Coordinator) pickBackupLocked(list []*task, workerID string) *task {
	threshold := c.medianDurationLocked()
	if threshold <= 0 {
		return nil
	}
	limit := time.Duration(float64(threshold) * c.cfg.BackupThreshold)

	var candidate *task
	var longest time.Duration
	for _, t := range list {
		if t.state != taskRunning || t.backup {
			continue
		}
		if _, alreadyRunning := t.workers[workerID]; alreadyRunning {
			continue
		}
		elapsed := time.Since(t.started)
		if elapsed > limit && elapsed > longest {
			candidate, longest = t, elapsed
		}
	}

	if candidate != nil {
		candidate.backup = true
		c.backupsLaunched++
		c.logf("launching backup for %s-%d, running %s vs median %s",
			candidate.kind, candidate.id, longest.Round(time.Millisecond), threshold.Round(time.Millisecond))
	}
	return candidate
}

func (c *Coordinator) medianDurationLocked() time.Duration {
	if len(c.durations) < 2 {
		return 0
	}
	sorted := make([]time.Duration, len(c.durations))
	copy(sorted, c.durations)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	return sorted[len(sorted)/2]
}

func (c *Coordinator) assignLocked(t *task, workerID string, reply *protocol.RequestTaskReply) {
	if t.state != taskRunning {
		t.started = time.Now()
	}
	t.state = taskRunning
	t.attempt++
	t.workers[workerID] = t.attempt

	reply.Kind = t.kind
	reply.TaskID = t.id
	reply.Attempt = t.attempt
	reply.InputFiles = t.inputFiles
	reply.NumMappers = len(c.mapTasks)
	reply.NumReducers = len(c.reduceTasks)
}

func (c *Coordinator) reclaimLocked() {
	now := time.Now()

	for id, w := range c.workers {
		if w.alive && now.Sub(w.lastSeen) > c.cfg.WorkerTimeout {
			w.alive = false
			c.lostWorkers++
			c.logf("worker %s missed heartbeats, presumed lost", id)
		}
	}

	reclaim := func(list []*task) {
		for _, t := range list {
			if t.state != taskRunning {
				continue
			}

			liveAttempts := 0
			for workerID := range t.workers {
				if w, ok := c.workers[workerID]; ok && w.alive {
					liveAttempts++
				} else {
					delete(t.workers, workerID)
				}
			}

			timedOut := now.Sub(t.started) > c.cfg.TaskTimeout
			if liveAttempts == 0 || timedOut {
				t.state = taskIdle
				t.backup = false
				t.workers = make(map[string]int)
				c.reassignments++
				reason := "worker lost"
				if timedOut {
					reason = "timed out"
				}
				c.logf("reassigning %s-%d: %s", t.kind, t.id, reason)
			}
		}
	}

	reclaim(c.mapTasks)
	reclaim(c.reduceTasks)
}

func (c *Coordinator) allDoneLocked(list []*task) bool {
	for _, t := range list {
		if t.state != taskDone {
			return false
		}
	}
	return true
}

func (c *Coordinator) finishLocked() {
	c.closeOnce.Do(func() {
		c.phase = "done"
		c.finishedAt = time.Now()
		close(c.done)
	})
}

func (c *Coordinator) failLocked(err error) {
	c.closeOnce.Do(func() {
		c.phase = "failed"
		c.finishedAt = time.Now()
		c.jobErr = err
		c.logf("aborting job: %v", err)
		close(c.done)
	})
}

// Non-nil once a task has exhausted its attempts, which is the only way the
// job ends other than success.
func (c *Coordinator) Err() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.jobErr
}

func (c *Coordinator) monitor() {
	ticker := time.NewTicker(c.cfg.HeartbeatInterval)
	defer ticker.Stop()

	for {
		select {
		case <-c.done:
			return
		case <-ticker.C:
			c.mu.Lock()
			c.reclaimLocked()
			c.mu.Unlock()
		}
	}
}

func (c *Coordinator) logf(format string, args ...interface{}) {
	if c.cfg.Verbose {
		log.Printf("[coordinator] "+format, args...)
	}
}

func countDone(list []*task) int {
	n := 0
	for _, t := range list {
		if t.state == taskDone {
			n++
		}
	}
	return n
}
