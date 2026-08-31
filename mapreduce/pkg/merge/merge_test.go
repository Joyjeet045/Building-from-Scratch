/*
merge_test.go - k-way merge correctness and memory behaviour

Checks that the merged stream is globally sorted, that duplicate keys across
sources are all preserved, and that the heap only ever holds one record per
source regardless of how much data flows through.
*/
package merge

import (
	"fmt"
	"sort"
	"testing"
)

type sliceSource struct {
	records [][2]string
	index   int
	closed  bool
	peak    *int
	live    *int
}

func (s *sliceSource) Next() (string, string, bool, error) {
	if s.index >= len(s.records) {
		return "", "", false, nil
	}
	rec := s.records[s.index]
	s.index++
	return rec[0], rec[1], true, nil
}

func (s *sliceSource) Close() error {
	s.closed = true
	return nil
}

type errorSource struct{}

func (errorSource) Next() (string, string, bool, error) {
	return "", "", false, fmt.Errorf("source exploded")
}
func (errorSource) Close() error { return nil }

func drain(t *testing.T, m *Merger) [][2]string {
	t.Helper()
	var out [][2]string
	for {
		key, value, ok, err := m.Next()
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			return out
		}
		out = append(out, [2]string{key, value})
	}
}

func TestMergeProducesGloballySortedStream(t *testing.T) {
	sources := []Source{
		&sliceSource{records: [][2]string{{"apple", "1"}, {"cherry", "1"}, {"elder", "1"}}},
		&sliceSource{records: [][2]string{{"banana", "1"}, {"date", "1"}}},
		&sliceSource{records: [][2]string{{"apple", "2"}, {"fig", "1"}}},
	}

	got := drain(t, New(sources))

	if len(got) != 7 {
		t.Fatalf("got %d records, want 7", len(got))
	}
	for i := 1; i < len(got); i++ {
		if got[i-1][0] > got[i][0] {
			t.Fatalf("not sorted at %d: %q then %q", i, got[i-1][0], got[i][0])
		}
	}
}

func TestMergeKeepsEveryDuplicate(t *testing.T) {
	sources := []Source{
		&sliceSource{records: [][2]string{{"same", "a"}, {"same", "b"}}},
		&sliceSource{records: [][2]string{{"same", "c"}}},
		&sliceSource{records: [][2]string{{"same", "d"}}},
	}

	got := drain(t, New(sources))
	if len(got) != 4 {
		t.Fatalf("got %d records, want 4", len(got))
	}

	values := make([]string, 0, 4)
	for _, rec := range got {
		if rec[0] != "same" {
			t.Fatalf("unexpected key %q", rec[0])
		}
		values = append(values, rec[1])
	}
	sort.Strings(values)
	want := []string{"a", "b", "c", "d"}
	for i := range want {
		if values[i] != want[i] {
			t.Fatalf("values = %v, want %v", values, want)
		}
	}
}

func TestMergeHandlesEmptyAndAbsentSources(t *testing.T) {
	if got := drain(t, New(nil)); len(got) != 0 {
		t.Fatalf("no sources should yield nothing, got %d", len(got))
	}

	sources := []Source{
		&sliceSource{},
		&sliceSource{records: [][2]string{{"only", "1"}}},
		&sliceSource{},
	}
	got := drain(t, New(sources))
	if len(got) != 1 || got[0][0] != "only" {
		t.Fatalf("got %v, want one record for \"only\"", got)
	}
}

func TestMergeHeapHoldsOneRecordPerSource(t *testing.T) {
	const sources, perSource = 5, 1000

	list := make([]Source, sources)
	for i := range list {
		records := make([][2]string, perSource)
		for j := 0; j < perSource; j++ {
			records[j] = [2]string{fmt.Sprintf("key-%06d", j*sources+i), "1"}
		}
		list[i] = &sliceSource{records: records}
	}

	m := New(list)
	count := 0
	for {
		_, _, ok, err := m.Next()
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			break
		}
		count++
		if m.heap.Len() > sources {
			t.Fatalf("heap holds %d entries, expected at most %d", m.heap.Len(), sources)
		}
	}

	if count != sources*perSource {
		t.Fatalf("read %d records, want %d", count, sources*perSource)
	}
}

func TestMergePropagatesSourceErrors(t *testing.T) {
	m := New([]Source{errorSource{}})
	if _, _, _, err := m.Next(); err == nil {
		t.Fatal("expected the source error to surface")
	}
}

func TestCloseClosesEverySource(t *testing.T) {
	a := &sliceSource{}
	b := &sliceSource{}
	if err := New([]Source{a, b}).Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if !a.closed || !b.closed {
		t.Fatal("expected every source to be closed")
	}
}
