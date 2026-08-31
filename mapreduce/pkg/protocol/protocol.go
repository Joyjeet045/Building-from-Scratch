/*
protocol.go - request and reply types exchanged over net/rpc

TaskKind      - map | reduce | none | exit
Register      - a worker joins the job
RequestTask   - a worker pulls its next task
ReportTask    - a worker reports an outcome
Heartbeat     - liveness ping
Status        - job progress, for the CLI

Workers pull work rather than having it pushed at them. That inverts the
original's fire-and-forget job launch and is what makes reassignment possible,
since the coordinator holds the authoritative view of what is running where.
*/
package protocol

import "time"

type TaskKind int

const (
	TaskNone TaskKind = iota
	TaskMap
	TaskReduce
	TaskExit
)

func (k TaskKind) String() string {
	switch k {
	case TaskNone:
		return "none"
	case TaskMap:
		return "map"
	case TaskReduce:
		return "reduce"
	case TaskExit:
		return "exit"
	}
	return "unknown"
}

const (
	RegisterMethod  = "Coordinator.Register"
	RequestMethod   = "Coordinator.RequestTask"
	ReportMethod    = "Coordinator.ReportTask"
	HeartbeatMethod = "Coordinator.Heartbeat"
	StatusMethod    = "Coordinator.Status"
)

type RegisterArgs struct {
	Host string
}

type RegisterReply struct {
	WorkerID          string
	JobID             string
	Root              string
	NumMappers        int
	NumReducers       int
	HeartbeatInterval time.Duration
}

type RequestTaskArgs struct {
	WorkerID string
}

type RequestTaskReply struct {
	Kind        TaskKind
	Attempt     int
	TaskID      int
	InputFiles  []string
	NumMappers  int
	NumReducers int
	RetryAfter  time.Duration
}

type ReportTaskArgs struct {
	WorkerID string
	Kind     TaskKind
	TaskID   int
	Attempt  int
	Success  bool
	Err      string
	Counters map[string]int64
	Elapsed  time.Duration
}

type ReportTaskReply struct {
	Accepted bool
}

type HeartbeatArgs struct {
	WorkerID string
}

type HeartbeatReply struct {
	Known bool
}

type StatusArgs struct{}

type StatusReply struct {
	Phase           string
	MapDone         int
	MapTotal        int
	ReduceDone      int
	ReduceTotal     int
	WorkersAlive    int
	WorkersLost     int
	BackupsLaunched int
	Reassignments   int
	Done            bool
	Counters        map[string]int64
	Elapsed         time.Duration
}
