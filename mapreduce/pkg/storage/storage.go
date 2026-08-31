/*
storage.go - the shared-filesystem layout every role agrees on

Layout         - path scheme for one job
Partition      - routes a key to a reducer, hash(key) mod R
ListInputFiles - sorted input files, so assignment is reproducible
SplitFiles     - deals files into per-mapper groups
WriteAtomic    - write to a temp file then rename, never a partial file

	<root>/<job-id>/
	    map-<m>/part-<r>      intermediate, sorted, one file per reduce partition
	    map-<m>/spill-<n>-<r> transient mapper spills, removed after merging
	    out/part-<r>          final output, one file per reducer

An NFS mount in the original; any path visible to all machines works, which is
what lets the same code run against a local directory. Intermediate output on
shared storage means a completed map task survives the death of the worker
that ran it, so only in-progress work is ever reassigned.
*/
package storage

import (
	"fmt"
	"hash/fnv"
	"os"
	"path/filepath"
	"sort"
)

type Layout struct {
	Root  string
	JobID string
}

func New(root, jobID string) Layout {
	return Layout{Root: root, JobID: jobID}
}

func (l Layout) JobDir() string { return filepath.Join(l.Root, l.JobID) }

func (l Layout) MapDir(mapTask int) string {
	return filepath.Join(l.JobDir(), fmt.Sprintf("map-%d", mapTask))
}

func (l Layout) IntermediatePath(mapTask, partition int) string {
	return filepath.Join(l.MapDir(mapTask), fmt.Sprintf("part-%d", partition))
}

func (l Layout) SpillPath(mapTask, spill, partition int) string {
	return filepath.Join(l.MapDir(mapTask), fmt.Sprintf("spill-%d-%d", spill, partition))
}

func (l Layout) OutputDir() string { return filepath.Join(l.JobDir(), "out") }

func (l Layout) OutputPath(partition int) string {
	return filepath.Join(l.OutputDir(), fmt.Sprintf("part-%d", partition))
}

func (l Layout) Prepare(numMappers int) error {
	if err := os.MkdirAll(l.OutputDir(), 0o755); err != nil {
		return err
	}
	for m := 0; m < numMappers; m++ {
		if err := os.MkdirAll(l.MapDir(m), 0o755); err != nil {
			return err
		}
	}
	return nil
}

func Partition(key string, numReducers int) int {
	if numReducers <= 1 {
		return 0
	}
	h := fnv.New32a()
	_, _ = h.Write([]byte(key))
	return int(h.Sum32() % uint32(numReducers))
}

func ListInputFiles(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var files []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		files = append(files, filepath.Join(dir, entry.Name()))
	}
	sort.Strings(files)
	if len(files) == 0 {
		return nil, fmt.Errorf("storage: no input files in %s", dir)
	}
	return files, nil
}

func SplitFiles(files []string, n int) [][]string {
	if n < 1 {
		n = 1
	}
	if n > len(files) {
		n = len(files)
	}
	groups := make([][]string, n)
	for i, file := range files {
		groups[i%n] = append(groups[i%n], file)
	}
	return groups
}

func WriteAtomic(path string, write func(f *os.File) error) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()

	if err := write(tmp); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return err
	}
	if err := os.Rename(tmpName, path); err != nil {
		os.Remove(tmpName)
		return err
	}
	return nil
}
