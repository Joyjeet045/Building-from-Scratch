/*
storage_test.go - partitioning, input splitting, and atomic write tests

The partitioning test pins the property reducers depend on: a key always maps
to the same partition, so one reducer sees all of its values.
*/
package storage

import (
	"os"
	"path/filepath"
	"testing"
)

func TestPartitionIsStableForAKey(t *testing.T) {
	keys := []string{"the", "quick", "brown", "fox", "", "\t", "ünïcödé"}
	for _, key := range keys {
		want := Partition(key, 8)
		for i := 0; i < 100; i++ {
			if got := Partition(key, 8); got != want {
				t.Fatalf("Partition(%q) returned %d then %d", key, want, got)
			}
		}
	}
}

func TestPartitionStaysInRange(t *testing.T) {
	for _, n := range []int{1, 2, 3, 7, 16} {
		for i := 0; i < 500; i++ {
			key := filepath.Join("key", string(rune('a'+i%26)), string(rune('0'+i%10)))
			p := Partition(key, n)
			if p < 0 || p >= n {
				t.Fatalf("Partition(%q, %d) = %d, out of range", key, n, p)
			}
		}
	}
}

func TestPartitionSpreadsKeys(t *testing.T) {
	const buckets = 4
	seen := make(map[int]int)
	for i := 0; i < 1000; i++ {
		seen[Partition(filepath.Join("word", string(rune(i))), buckets)]++
	}
	if len(seen) != buckets {
		t.Fatalf("keys landed in %d of %d partitions", len(seen), buckets)
	}
}

func TestSplitFilesCoversEveryFileExactlyOnce(t *testing.T) {
	files := []string{"a", "b", "c", "d", "e"}

	for _, n := range []int{1, 2, 3, 5, 9} {
		groups := SplitFiles(files, n)

		seen := make(map[string]int)
		for _, group := range groups {
			if len(group) == 0 {
				t.Errorf("n=%d produced an empty group", n)
			}
			for _, f := range group {
				seen[f]++
			}
		}

		if len(seen) != len(files) {
			t.Errorf("n=%d covered %d files, want %d", n, len(seen), len(files))
		}
		for f, count := range seen {
			if count != 1 {
				t.Errorf("n=%d assigned %q %d times", n, f, count)
			}
		}
	}
}

func TestListInputFilesIsSortedAndSkipsDirectories(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"c.txt", "a.txt", "b.txt"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Mkdir(filepath.Join(dir, "nested"), 0o755); err != nil {
		t.Fatal(err)
	}

	files, err := ListInputFiles(dir)
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(files) != 3 {
		t.Fatalf("got %d files, want 3", len(files))
	}
	for i, want := range []string{"a.txt", "b.txt", "c.txt"} {
		if filepath.Base(files[i]) != want {
			t.Errorf("position %d = %q, want %q", i, filepath.Base(files[i]), want)
		}
	}
}

func TestListInputFilesRejectsEmptyDirectory(t *testing.T) {
	if _, err := ListInputFiles(t.TempDir()); err == nil {
		t.Fatal("expected an error for a directory with no input files")
	}
}

func TestWriteAtomicLeavesNoFileWhenTheWriteFails(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "out")

	err := WriteAtomic(path, func(f *os.File) error {
		f.WriteString("partial")
		return os.ErrInvalid
	})
	if err == nil {
		t.Fatal("expected the write error to surface")
	}
	if _, statErr := os.Stat(path); !os.IsNotExist(statErr) {
		t.Fatal("a failed write must not leave the target file behind")
	}

	entries, _ := os.ReadDir(dir)
	if len(entries) != 0 {
		t.Fatalf("temporary files left behind: %v", entries)
	}
}

func TestWriteAtomicReplacesExistingContent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out")

	for _, want := range []string{"first", "second"} {
		content := want
		if err := WriteAtomic(path, func(f *os.File) error {
			_, err := f.WriteString(content)
			return err
		}); err != nil {
			t.Fatalf("write: %v", err)
		}

		got, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != want {
			t.Fatalf("got %q, want %q", got, want)
		}
	}
}

func TestLayoutPathsAreDistinctPerTask(t *testing.T) {
	l := New("root", "job-1")

	if l.IntermediatePath(0, 1) == l.IntermediatePath(1, 0) {
		t.Fatal("map/partition paths must not collide")
	}
	if l.OutputPath(0) == l.OutputPath(1) {
		t.Fatal("output paths must not collide")
	}
	if l.SpillPath(0, 0, 0) == l.SpillPath(0, 1, 0) {
		t.Fatal("spill paths must not collide")
	}
}
