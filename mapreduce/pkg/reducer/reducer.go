/*
reducer.go - runs one reduce task over every mapper's partition for it

Options - what a reduce task needs: layout, task id, mapper count
Run     - execute reduce task r

Opens map-<m>/part-<r> for all m and merges them into one sorted stream, then
groups equal keys and hands each group to the user's Reduce. Values for a key
arrive lazily from the merged stream, so a key with a huge value list does not
have to fit in memory all at once.

A missing partition file is treated as empty: a mapper that emitted nothing for
this partition still writes the file, so absence only happens if a task was
never run, which the coordinator prevents.
*/
package reducer

import (
	"errors"
	"fmt"
	"os"

	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/counters"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/interfaces"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/merge"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/record"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/storage"
)

type Options struct {
	Layout     storage.Layout
	TaskID     int
	NumMappers int
	Reducer    interfaces.Reducer
	Counters   *counters.Set
}

type groupIterator struct {
	key    string
	values []string
	index  int
}

func (g *groupIterator) Key() string   { return g.key }
func (g *groupIterator) Value() string { return g.values[g.index] }
func (g *groupIterator) Done() bool    { return g.index >= len(g.values) }
func (g *groupIterator) NextValue()    { g.index++ }

func Run(opts Options) error {
	if opts.NumMappers < 1 {
		return fmt.Errorf("reducer: numMappers must be at least 1")
	}
	if opts.Counters == nil {
		opts.Counters = counters.New()
	}

	sources := make([]merge.Source, 0, opts.NumMappers)
	for m := 0; m < opts.NumMappers; m++ {
		path := opts.Layout.IntermediatePath(m, opts.TaskID)
		source, err := record.OpenSource(path)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			for _, opened := range sources {
				opened.Close()
			}
			return err
		}
		sources = append(sources, source)
	}

	merger := merge.New(sources)
	defer merger.Close()

	return storage.WriteAtomic(opts.Layout.OutputPath(opts.TaskID), func(f *os.File) error {
		w := record.NewWriter(f)

		emitFor := func(key string) func(string) {
			return func(value string) {
				w.Write(record.Record{Key: key, Value: value})
				opts.Counters.Inc("reduce.emitted", 1)
			}
		}

		var currentKey string
		var values []string
		have := false

		flush := func() error {
			if !have {
				return nil
			}
			iter := &groupIterator{key: currentKey, values: values}
			opts.Reducer.Reduce(iter, emitFor(currentKey))
			opts.Counters.Inc("reduce.keys", 1)
			opts.Counters.Inc("reduce.values", int64(len(values)))
			return nil
		}

		for {
			key, value, ok, err := merger.Next()
			if err != nil {
				return err
			}
			if !ok {
				break
			}

			if !have {
				currentKey, have = key, true
			} else if key != currentKey {
				if err := flush(); err != nil {
					return err
				}
				values = values[:0]
				currentKey = key
			}
			values = append(values, value)
		}

		if err := flush(); err != nil {
			return err
		}
		return w.Flush()
	})
}
