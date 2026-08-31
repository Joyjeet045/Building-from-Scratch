/*
counters.go - named tallies accumulated per worker and summed by the coordinator

Set    - a concurrency-safe counter map
Merge  - folds another worker's counters in
Snapshot - a plain map for sending over RPC

The MapReduce paper's counter facility, which the blog omits. Workers report
theirs alongside each task result, giving a job-wide view of records read,
emitted, combined away, and skipped.
*/
package counters

import (
	"sort"
	"strings"
	"sync"
)

type Set struct {
	mu     sync.Mutex
	values map[string]int64
}

func New() *Set {
	return &Set{values: make(map[string]int64)}
}

func (s *Set) Inc(name string, delta int64) {
	s.mu.Lock()
	if s.values == nil {
		s.values = make(map[string]int64)
	}
	s.values[name] += delta
	s.mu.Unlock()
}

func (s *Set) Get(name string) int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.values[name]
}

func (s *Set) Snapshot() map[string]int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[string]int64, len(s.values))
	for k, v := range s.values {
		out[k] = v
	}
	return out
}

func (s *Set) Merge(other map[string]int64) {
	if len(other) == 0 {
		return
	}
	s.mu.Lock()
	if s.values == nil {
		s.values = make(map[string]int64)
	}
	for k, v := range other {
		s.values[k] += v
	}
	s.mu.Unlock()
}

func (s *Set) Reset() {
	s.mu.Lock()
	s.values = make(map[string]int64)
	s.mu.Unlock()
}

func (s *Set) String() string {
	snapshot := s.Snapshot()
	names := make([]string, 0, len(snapshot))
	for name := range snapshot {
		names = append(names, name)
	}
	sort.Strings(names)

	var b strings.Builder
	for i, name := range names {
		if i > 0 {
			b.WriteString("  ")
		}
		b.WriteString(name)
		b.WriteByte('=')
		b.WriteString(formatInt(snapshot[name]))
	}
	return b.String()
}

func formatInt(v int64) string {
	if v == 0 {
		return "0"
	}
	negative := v < 0
	if negative {
		v = -v
	}
	var digits []byte
	for count := 0; v > 0; count++ {
		if count > 0 && count%3 == 0 {
			digits = append(digits, ',')
		}
		digits = append(digits, byte('0'+v%10))
		v /= 10
	}
	if negative {
		digits = append(digits, '-')
	}
	for i, j := 0, len(digits)-1; i < j; i, j = i+1, j-1 {
		digits[i], digits[j] = digits[j], digits[i]
	}
	return string(digits)
}
