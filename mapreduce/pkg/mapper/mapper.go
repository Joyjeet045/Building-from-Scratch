/*
mapper.go - runs one map task and writes its sorted intermediate partitions

Options - what a map task needs: layout, input files, partition count, buffer
Run     - execute map task m over its input files

Reads assigned files line by line into the user's Map, buffers emitted pairs,
and spills a sorted run to disk whenever the buffer exceeds SpillBytes. Spills
are merged at the end into one sorted file per reduce partition, so mapper
memory stays bounded no matter how large the input is. The blog holds
everything in memory and calls this out as a shortcoming.

If the user supplies a Combiner it runs over each key's values before writing,
which is the map-side reduction the blog leaves as an open question.
*/
package mapper

import (
	"bufio"
	"fmt"
	"os"
	"sort"

	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/counters"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/interfaces"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/merge"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/record"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/storage"
)

type Options struct {
	Layout      storage.Layout
	TaskID      int
	InputFiles  []string
	NumReducers int
	SpillBytes  int
	Mapper      interfaces.Mapper
	Combiner    interfaces.Combiner
	Counters    *counters.Set
}

type mapInput struct {
	key   string
	value string
}

func (m mapInput) Key() string   { return m.key }
func (m mapInput) Value() string { return m.value }

type runner struct {
	opts     Options
	buffer   []record.Record
	bytes    int
	spills   int
	spilled  [][]string
	combiner interfaces.Combiner
}

func Run(opts Options) error {
	if opts.NumReducers < 1 {
		return fmt.Errorf("mapper: numReducers must be at least 1")
	}
	if opts.SpillBytes < 1024 {
		opts.SpillBytes = 1024
	}
	if opts.Counters == nil {
		opts.Counters = counters.New()
	}

	r := &runner{opts: opts, combiner: opts.Combiner}

	if err := os.MkdirAll(opts.Layout.MapDir(opts.TaskID), 0o755); err != nil {
		return err
	}
	if err := r.removeStaleSpills(); err != nil {
		return err
	}

	emit := func(key, value string) {
		r.buffer = append(r.buffer, record.Record{Key: key, Value: value})
		r.bytes += len(key) + len(value) + 2
		opts.Counters.Inc("map.emitted", 1)
	}

	for _, path := range opts.InputFiles {
		if err := r.consume(path, emit); err != nil {
			return err
		}
	}

	if len(r.spilled) == 0 {
		return r.writeFinalFromBuffer()
	}
	if len(r.buffer) > 0 {
		if err := r.spill(); err != nil {
			return err
		}
	}
	return r.mergeSpills()
}

func (r *runner) removeStaleSpills() error {
	dir := r.opts.Layout.MapDir(r.opts.TaskID)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		name := entry.Name()
		if len(name) >= 6 && name[:6] == "spill-" {
			if err := os.Remove(dir + string(os.PathSeparator) + name); err != nil {
				return err
			}
		}
	}
	return nil
}

func (r *runner) consume(path string, emit func(key, value string)) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 1<<16), 16<<20)

	for scanner.Scan() {
		r.opts.Counters.Inc("map.input_records", 1)
		r.opts.Mapper.Map(mapInput{key: path, value: scanner.Text()}, emit)

		if r.bytes >= r.opts.SpillBytes {
			if err := r.spill(); err != nil {
				return err
			}
		}
	}
	return scanner.Err()
}

func (r *runner) sortBuffer() {
	sort.Slice(r.buffer, func(i, j int) bool {
		if r.buffer[i].Key != r.buffer[j].Key {
			return r.buffer[i].Key < r.buffer[j].Key
		}
		return r.buffer[i].Value < r.buffer[j].Value
	})
}

func (r *runner) partitionBuffer() [][]record.Record {
	parts := make([][]record.Record, r.opts.NumReducers)
	for _, rec := range r.buffer {
		p := storage.Partition(rec.Key, r.opts.NumReducers)
		parts[p] = append(parts[p], rec)
	}
	return parts
}

func (r *runner) applyCombiner(records []record.Record) []record.Record {
	if r.combiner == nil || len(records) == 0 {
		return records
	}

	out := make([]record.Record, 0, len(records))
	values := make([]string, 0, 8)

	flush := func(key string) {
		combined := r.combiner.Combine(key, values)
		for _, v := range combined {
			out = append(out, record.Record{Key: key, Value: v})
		}
		r.opts.Counters.Inc("combine.in", int64(len(values)))
		r.opts.Counters.Inc("combine.out", int64(len(combined)))
	}

	current := records[0].Key
	for _, rec := range records {
		if rec.Key != current {
			flush(current)
			values = values[:0]
			current = rec.Key
		}
		values = append(values, rec.Value)
	}
	flush(current)
	return out
}

func writeRecords(path string, records []record.Record) error {
	return storage.WriteAtomic(path, func(f *os.File) error {
		w := record.NewWriter(f)
		for _, rec := range records {
			if err := w.Write(rec); err != nil {
				return err
			}
		}
		return w.Flush()
	})
}

func (r *runner) spill() error {
	if len(r.buffer) == 0 {
		return nil
	}
	r.sortBuffer()
	parts := r.partitionBuffer()

	paths := make([]string, r.opts.NumReducers)
	for p := 0; p < r.opts.NumReducers; p++ {
		records := r.applyCombiner(parts[p])
		path := r.opts.Layout.SpillPath(r.opts.TaskID, r.spills, p)
		if err := writeRecords(path, records); err != nil {
			return err
		}
		paths[p] = path
	}

	r.spilled = append(r.spilled, paths)
	r.spills++
	r.buffer = r.buffer[:0]
	r.bytes = 0
	r.opts.Counters.Inc("map.spills", 1)
	return nil
}

func (r *runner) writeFinalFromBuffer() error {
	r.sortBuffer()
	parts := r.partitionBuffer()
	for p := 0; p < r.opts.NumReducers; p++ {
		records := r.applyCombiner(parts[p])
		if err := writeRecords(r.opts.Layout.IntermediatePath(r.opts.TaskID, p), records); err != nil {
			return err
		}
		r.opts.Counters.Inc("map.output_records", int64(len(records)))
	}
	return nil
}

func (r *runner) mergeSpills() error {
	for p := 0; p < r.opts.NumReducers; p++ {
		sources := make([]merge.Source, 0, len(r.spilled))
		for _, paths := range r.spilled {
			source, err := record.OpenSource(paths[p])
			if err != nil {
				return err
			}
			sources = append(sources, source)
		}

		merger := merge.New(sources)
		written := 0
		err := storage.WriteAtomic(r.opts.Layout.IntermediatePath(r.opts.TaskID, p), func(f *os.File) error {
			w := record.NewWriter(f)
			for {
				key, value, ok, err := merger.Next()
				if err != nil {
					return err
				}
				if !ok {
					break
				}
				if err := w.Write(record.Record{Key: key, Value: value}); err != nil {
					return err
				}
				written++
			}
			return w.Flush()
		})
		merger.Close()
		if err != nil {
			return err
		}
		r.opts.Counters.Inc("map.output_records", int64(written))
	}

	for _, paths := range r.spilled {
		for _, path := range paths {
			os.Remove(path)
		}
	}
	return nil
}
