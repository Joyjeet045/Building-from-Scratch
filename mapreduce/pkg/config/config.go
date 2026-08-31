/*
config.go - the job spec a user fills in, plus flag parsing for each role

Mode           - coordinator | worker | cluster | local
Config         - the whole job specification
SetupJobConfig - parses the standard flag set
Validate       - rejects a spec before any work is launched

One binary plays every role, selected by -mode, exactly as in the original.
*/
package config

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/interfaces"
)

type Mode string

const (
	ModeCoordinator Mode = "coordinator"
	ModeWorker      Mode = "worker"
	ModeLocal       Mode = "local"
	ModeCluster     Mode = "cluster"
)

type Config struct {
	Mode    Mode
	JobID   string
	JobName string

	InputDir string
	Root     string

	NumMappers  int
	NumReducers int

	Address string

	SpillBytes int

	TaskTimeout       time.Duration
	HeartbeatInterval time.Duration
	WorkerTimeout     time.Duration

	EnableBackupTasks bool
	BackupThreshold   float64
	MaxTaskAttempts   int

	Workers int
	Verbose bool

	Mapper   interfaces.Mapper
	Reducer  interfaces.Reducer
	Combiner interfaces.Combiner
}

func Default() *Config {
	return &Config{
		Mode:              ModeLocal,
		JobName:           "mapreduce",
		Root:              "work",
		NumMappers:        4,
		NumReducers:       2,
		Address:           "127.0.0.1:5330",
		SpillBytes:        32 << 20,
		TaskTimeout:       30 * time.Second,
		HeartbeatInterval: time.Second,
		WorkerTimeout:     5 * time.Second,
		EnableBackupTasks: true,
		BackupThreshold:   1.5,
		MaxTaskAttempts:   4,
		Workers:           runtime.NumCPU(),
		Verbose:           true,
	}
}

func SetupJobConfig() *Config {
	cfg := Default()

	mode := flag.String("mode", string(cfg.Mode), "coordinator | worker | cluster | local")
	flag.StringVar(&cfg.JobName, "name", cfg.JobName, "job name, used in the job directory")
	flag.StringVar(&cfg.JobID, "job-id", "", "job id (defaults to name plus a timestamp)")
	flag.StringVar(&cfg.InputDir, "input", "", "directory of input files")
	flag.StringVar(&cfg.Root, "work", cfg.Root, "shared work directory (an NFS mount in a real cluster)")
	flag.IntVar(&cfg.NumMappers, "mappers", cfg.NumMappers, "number of map tasks")
	flag.IntVar(&cfg.NumReducers, "reducers", cfg.NumReducers, "number of reduce tasks")
	flag.StringVar(&cfg.Address, "addr", cfg.Address, "coordinator host:port")
	flag.IntVar(&cfg.SpillBytes, "spill-bytes", cfg.SpillBytes, "mapper buffer size before spilling to disk")
	flag.DurationVar(&cfg.TaskTimeout, "task-timeout", cfg.TaskTimeout, "reassign a task still running after this long")
	flag.DurationVar(&cfg.WorkerTimeout, "worker-timeout", cfg.WorkerTimeout, "declare a silent worker dead after this long")
	flag.BoolVar(&cfg.EnableBackupTasks, "backup-tasks", cfg.EnableBackupTasks, "duplicate straggling tasks")
	flag.Float64Var(&cfg.BackupThreshold, "backup-threshold", cfg.BackupThreshold, "multiple of median runtime that marks a straggler")
	flag.IntVar(&cfg.MaxTaskAttempts, "max-attempts", cfg.MaxTaskAttempts, "fail the job once one task has failed this many times")
	flag.IntVar(&cfg.Workers, "workers", cfg.Workers, "worker count for local and cluster modes")
	flag.BoolVar(&cfg.Verbose, "v", cfg.Verbose, "verbose logging")
	flag.Parse()

	cfg.Mode = Mode(*mode)
	return cfg
}

func (c *Config) Validate() error {
	switch c.Mode {
	case ModeCoordinator, ModeWorker, ModeLocal, ModeCluster:
	default:
		return fmt.Errorf("config: unknown mode %q", c.Mode)
	}

	if c.Mode != ModeWorker {
		if c.InputDir == "" {
			return fmt.Errorf("config: -input is required in %s mode", c.Mode)
		}
		info, err := os.Stat(c.InputDir)
		if err != nil {
			return fmt.Errorf("config: input directory %q: %w", c.InputDir, err)
		}
		if !info.IsDir() {
			return fmt.Errorf("config: input %q is not a directory", c.InputDir)
		}
		if c.NumMappers < 1 {
			return fmt.Errorf("config: mappers must be at least 1")
		}
		if c.NumReducers < 1 {
			return fmt.Errorf("config: reducers must be at least 1")
		}
	}

	if c.Mode != ModeCoordinator && c.Mapper == nil {
		return fmt.Errorf("config: Mapper must be set")
	}
	if c.Mode != ModeCoordinator && c.Reducer == nil {
		return fmt.Errorf("config: Reducer must be set")
	}
	if c.SpillBytes < 1024 {
		return fmt.Errorf("config: spill-bytes must be at least 1024")
	}
	if c.Workers < 1 {
		return fmt.Errorf("config: workers must be at least 1")
	}
	if c.BackupThreshold < 1 {
		return fmt.Errorf("config: backup-threshold must be at least 1")
	}
	if c.MaxTaskAttempts < 1 {
		return fmt.Errorf("config: max-attempts must be at least 1")
	}
	return nil
}

func (c *Config) ResolveJobID() string {
	if c.JobID == "" {
		c.JobID = fmt.Sprintf("%s-%s", c.JobName, time.Now().Format("2006-01-02-15-04-05"))
	}
	return c.JobID
}
