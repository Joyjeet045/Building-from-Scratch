/*
coordinator_test.go - scheduling and fault-tolerance tests

Drives the RPC methods directly, no network needed. Covers the three recovery
paths the blog's fire-and-forget launcher has no answer for: a worker that
stops heartbeating, a task that overruns, and a straggler that gets a backup
whose loser is then rejected.
*/
package coordinator

import (
	"strings"
	"testing"
	"time"

	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/config"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/protocol"
)

func testConfig(mappers, reducers int) *config.Config {
	cfg := config.Default()
	cfg.JobID = "test-job"
	cfg.Root = "testdata-root"
	cfg.NumMappers = mappers
	cfg.NumReducers = reducers
	cfg.TaskTimeout = 100 * time.Millisecond
	cfg.WorkerTimeout = 50 * time.Millisecond
	cfg.HeartbeatInterval = 10 * time.Millisecond
	cfg.Verbose = false
	return cfg
}

func groupsFor(n int) [][]string {
	groups := make([][]string, n)
	for i := range groups {
		groups[i] = []string{"input"}
	}
	return groups
}

func register(t *testing.T, c *Coordinator) string {
	t.Helper()
	var reply protocol.RegisterReply
	if err := c.Register(&protocol.RegisterArgs{Host: "test"}, &reply); err != nil {
		t.Fatalf("register: %v", err)
	}
	return reply.WorkerID
}

func request(t *testing.T, c *Coordinator, workerID string) protocol.RequestTaskReply {
	t.Helper()
	var reply protocol.RequestTaskReply
	if err := c.RequestTask(&protocol.RequestTaskArgs{WorkerID: workerID}, &reply); err != nil {
		t.Fatalf("request: %v", err)
	}
	return reply
}

func report(t *testing.T, c *Coordinator, workerID string, task protocol.RequestTaskReply, success bool) bool {
	t.Helper()
	var reply protocol.ReportTaskReply
	args := protocol.ReportTaskArgs{
		WorkerID: workerID,
		Kind:     task.Kind,
		TaskID:   task.TaskID,
		Attempt:  task.Attempt,
		Success:  success,
		Elapsed:  5 * time.Millisecond,
	}
	if err := c.ReportTask(&args, &reply); err != nil {
		t.Fatalf("report: %v", err)
	}
	return reply.Accepted
}

func TestReduceTasksAreWithheldUntilMapsFinish(t *testing.T) {
	cfg := testConfig(2, 2)
	c := New(cfg, groupsFor(2))
	w := register(t, c)

	first := request(t, c, w)
	if first.Kind != protocol.TaskMap {
		t.Fatalf("first task = %v, want map", first.Kind)
	}

	second := request(t, c, w)
	if second.Kind != protocol.TaskMap {
		t.Fatalf("second task = %v, want map", second.Kind)
	}

	idle := request(t, c, w)
	if idle.Kind != protocol.TaskNone {
		t.Fatalf("with maps outstanding, got %v, want none", idle.Kind)
	}

	report(t, c, w, first, true)
	report(t, c, w, second, true)

	next := request(t, c, w)
	if next.Kind != protocol.TaskReduce {
		t.Fatalf("after maps completed, got %v, want reduce", next.Kind)
	}
}

func TestTaskIsReassignedWhenItsWorkerGoesSilent(t *testing.T) {
	cfg := testConfig(1, 1)
	c := New(cfg, groupsFor(1))

	dead := register(t, c)
	task := request(t, c, dead)
	if task.Kind != protocol.TaskMap {
		t.Fatalf("expected a map task, got %v", task.Kind)
	}

	time.Sleep(cfg.WorkerTimeout * 3)

	alive := register(t, c)
	reassigned := request(t, c, alive)
	if reassigned.Kind != protocol.TaskMap || reassigned.TaskID != task.TaskID {
		t.Fatalf("expected map-%d to be reassigned, got %v-%d",
			task.TaskID, reassigned.Kind, reassigned.TaskID)
	}
	if reassigned.Attempt <= task.Attempt {
		t.Fatalf("attempt did not advance: %d then %d", task.Attempt, reassigned.Attempt)
	}

	if _, reassignments, _ := c.Stats(); reassignments == 0 {
		t.Fatal("expected the reassignment to be counted")
	}
}

func TestLateReportFromASupersededAttemptIsRejected(t *testing.T) {
	cfg := testConfig(1, 1)
	c := New(cfg, groupsFor(1))

	slow := register(t, c)
	slowTask := request(t, c, slow)

	time.Sleep(cfg.WorkerTimeout * 3)

	fast := register(t, c)
	fastTask := request(t, c, fast)

	if !report(t, c, fast, fastTask, true) {
		t.Fatal("the reassigned attempt should be accepted")
	}
	if report(t, c, slow, slowTask, true) {
		t.Fatal("the superseded attempt must be rejected")
	}
}

func TestFailedTaskGoesBackToTheQueue(t *testing.T) {
	cfg := testConfig(1, 1)
	c := New(cfg, groupsFor(1))
	w := register(t, c)

	task := request(t, c, w)
	if report(t, c, w, task, false) {
		t.Fatal("a failure should not be accepted as a result")
	}

	retry := request(t, c, w)
	if retry.Kind != protocol.TaskMap || retry.TaskID != task.TaskID {
		t.Fatalf("expected map-%d to be retried, got %v-%d", task.TaskID, retry.Kind, retry.TaskID)
	}
}

func TestBackupTaskIsLaunchedForAStraggler(t *testing.T) {
	cfg := testConfig(3, 1)
	cfg.EnableBackupTasks = true
	cfg.BackupThreshold = 1.5
	cfg.TaskTimeout = time.Hour
	cfg.WorkerTimeout = time.Hour

	c := New(cfg, groupsFor(3))

	fast := register(t, c)
	straggler := register(t, c)
	spare := register(t, c)

	slowTask := request(t, c, straggler)

	for i := 0; i < 2; i++ {
		task := request(t, c, fast)
		if task.Kind != protocol.TaskMap {
			t.Fatalf("expected a map task, got %v", task.Kind)
		}
		report(t, c, fast, task, true)
	}

	time.Sleep(30 * time.Millisecond)

	backup := request(t, c, spare)
	if backup.Kind != protocol.TaskMap || backup.TaskID != slowTask.TaskID {
		t.Fatalf("expected a backup of map-%d, got %v-%d",
			slowTask.TaskID, backup.Kind, backup.TaskID)
	}

	if _, _, backups := c.Stats(); backups != 1 {
		t.Fatalf("backups launched = %d, want 1", backups)
	}

	if !report(t, c, spare, backup, true) {
		t.Fatal("the backup result should be accepted")
	}
	if report(t, c, straggler, slowTask, true) {
		t.Fatal("the straggler's late result must be rejected")
	}
}

func TestBackupTasksCanBeDisabled(t *testing.T) {
	cfg := testConfig(3, 1)
	cfg.EnableBackupTasks = false
	cfg.TaskTimeout = time.Hour
	cfg.WorkerTimeout = time.Hour

	c := New(cfg, groupsFor(3))
	fast := register(t, c)
	straggler := register(t, c)
	spare := register(t, c)

	request(t, c, straggler)
	for i := 0; i < 2; i++ {
		task := request(t, c, fast)
		report(t, c, fast, task, true)
	}

	time.Sleep(30 * time.Millisecond)

	if got := request(t, c, spare); got.Kind != protocol.TaskNone {
		t.Fatalf("with backups off, got %v, want none", got.Kind)
	}
	if _, _, backups := c.Stats(); backups != 0 {
		t.Fatalf("backups launched = %d, want 0", backups)
	}
}

func TestJobSignalsCompletionAndCountersAreSummed(t *testing.T) {
	cfg := testConfig(2, 2)
	c := New(cfg, groupsFor(2))
	w := register(t, c)

	for {
		task := request(t, c, w)
		if task.Kind == protocol.TaskExit {
			break
		}
		if task.Kind == protocol.TaskNone {
			t.Fatal("a single worker should never be told to wait here")
		}

		var reply protocol.ReportTaskReply
		args := protocol.ReportTaskArgs{
			WorkerID: w,
			Kind:     task.Kind,
			TaskID:   task.TaskID,
			Attempt:  task.Attempt,
			Success:  true,
			Counters: map[string]int64{"records": 10},
			Elapsed:  time.Millisecond,
		}
		if err := c.ReportTask(&args, &reply); err != nil {
			t.Fatalf("report: %v", err)
		}
	}

	select {
	case <-c.Done():
	default:
		t.Fatal("the job should be marked done")
	}

	if got := c.Counters().Get("records"); got != 40 {
		t.Fatalf("summed counter = %d, want 40 across 4 tasks", got)
	}

	var status protocol.StatusReply
	if err := c.Status(&protocol.StatusArgs{}, &status); err != nil {
		t.Fatalf("status: %v", err)
	}
	if !status.Done || status.MapDone != 2 || status.ReduceDone != 2 {
		t.Fatalf("status = %+v, want a completed 2/2 job", status)
	}
}

func TestRepeatedFailuresAbortTheJob(t *testing.T) {
	cfg := testConfig(1, 1)
	cfg.MaxTaskAttempts = 3
	c := New(cfg, groupsFor(1))
	w := register(t, c)

	for i := 0; i < cfg.MaxTaskAttempts; i++ {
		task := request(t, c, w)
		if task.Kind != protocol.TaskMap {
			t.Fatalf("attempt %d: expected a map task, got %v", i, task.Kind)
		}
		report(t, c, w, task, false)
	}

	select {
	case <-c.Done():
	case <-time.After(time.Second):
		t.Fatal("a task past its attempt budget should end the job, not retry forever")
	}

	err := c.Err()
	if err == nil {
		t.Fatal("an aborted job must report why it failed")
	}
	if !strings.Contains(err.Error(), "failed 3 times") {
		t.Fatalf("error should name the attempt count, got %q", err)
	}

	if next := request(t, c, w); next.Kind != protocol.TaskExit {
		t.Fatalf("workers should be told to exit once the job is aborted, got %v", next.Kind)
	}
}

func TestFailuresBelowTheBudgetKeepRetrying(t *testing.T) {
	cfg := testConfig(1, 1)
	cfg.MaxTaskAttempts = 3
	c := New(cfg, groupsFor(1))
	w := register(t, c)

	report(t, c, w, request(t, c, w), false)
	report(t, c, w, request(t, c, w), false)

	if err := c.Err(); err != nil {
		t.Fatalf("two failures are inside the budget, job should still run: %v", err)
	}

	task := request(t, c, w)
	if task.Kind != protocol.TaskMap {
		t.Fatalf("expected the task to be handed out again, got %v", task.Kind)
	}
	report(t, c, w, task, true)

	if err := c.Err(); err != nil {
		t.Fatalf("a task that eventually succeeds must not fail the job: %v", err)
	}
}
