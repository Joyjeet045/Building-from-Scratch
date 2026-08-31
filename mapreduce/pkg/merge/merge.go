/*
merge.go - k-way merge of sorted record streams via a min-heap

Source - one sorted stream of records
Merger - yields the union of its sources in key order

One record per source sits in the heap at a time, so a reducer can consume
every intermediate file for its partition as a single sorted stream with peak
memory proportional to the number of sources, not the data volume.
*/
package merge

import (
	"container/heap"
	"io"
)

type Source interface {
	Next() (key, value string, ok bool, err error)
	Close() error
}

type entry struct {
	key    string
	value  string
	source int
}

type minHeap []entry

func (h minHeap) Len() int { return len(h) }

func (h minHeap) Less(i, j int) bool {
	if h[i].key != h[j].key {
		return h[i].key < h[j].key
	}
	return h[i].source < h[j].source
}

func (h minHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x interface{}) { *h = append(*h, x.(entry)) }

func (h *minHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

type Merger struct {
	sources []Source
	heap    minHeap
	err     error
	primed  bool
}

func New(sources []Source) *Merger {
	return &Merger{sources: sources}
}

func (m *Merger) prime() error {
	m.heap = make(minHeap, 0, len(m.sources))
	for i, source := range m.sources {
		key, value, ok, err := source.Next()
		if err != nil {
			return err
		}
		if ok {
			m.heap = append(m.heap, entry{key: key, value: value, source: i})
		}
	}
	heap.Init(&m.heap)
	m.primed = true
	return nil
}

func (m *Merger) Next() (key, value string, ok bool, err error) {
	if m.err != nil {
		return "", "", false, m.err
	}
	if !m.primed {
		if err := m.prime(); err != nil {
			m.err = err
			return "", "", false, err
		}
	}
	if m.heap.Len() == 0 {
		return "", "", false, nil
	}

	top := heap.Pop(&m.heap).(entry)

	nextKey, nextValue, hasMore, err := m.sources[top.source].Next()
	if err != nil {
		m.err = err
		return "", "", false, err
	}
	if hasMore {
		heap.Push(&m.heap, entry{key: nextKey, value: nextValue, source: top.source})
	}
	return top.key, top.value, true, nil
}

func (m *Merger) Close() error {
	var first error
	for _, source := range m.sources {
		if err := source.Close(); err != nil && first == nil && err != io.EOF {
			first = err
		}
	}
	return first
}
