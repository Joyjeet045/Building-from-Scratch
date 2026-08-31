/*
mapper_test.go - combiner and spill equivalence tests

The two properties worth pinning: turning the combiner on must not change the
final answer, and forcing many spills must produce byte-identical intermediate
files to a single in-memory pass. Both are optimizations that are easy to get
subtly wrong.
*/
package mapper

import (
	"os"
	"strconv"
	"testing"

	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/counters"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/interfaces"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/record"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/storage"
)

type wordSplitter struct{}

func (wordSplitter) Map(input interfaces.MapInput, emit func(key, value string)) {
	word := ""
	for _, c := range input.Value() {
		if c == ' ' {
			if word != "" {
				emit(word, "1")
				word = ""
			}
			continue
		}
		word += string(c)
	}
	if word != "" {
		emit(word, "1")
	}
}

type summingCombiner struct{}

func (summingCombiner) Combine(key string, values []string) []string {
	total := 0
	for _, v := range values {
		n, _ := strconv.Atoi(v)
		total += n
	}
	return []string{strconv.Itoa(total)}
}

func writeCorpus(t *testing.T, lines []string) string {
	t.Helper()
	dir := t.TempDir()
	path := dir + string(os.PathSeparator) + "input.txt"

	content := ""
	for _, line := range lines {
		content += line + "\n"
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func readPartition(t *testing.T, layout storage.Layout, task, partition int) []record.Record {
	t.Helper()
	f, err := os.Open(layout.IntermediatePath(task, partition))
	if err != nil {
		t.Fatalf("open partition: %v", err)
	}
	defer f.Close()

	var out []record.Record
	r := record.NewReader(f)
	for {
		rec, ok, err := r.Next()
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		if !ok {
			return out
		}
		out = append(out, rec)
	}
}

func totalsFrom(t *testing.T, layout storage.Layout, task, numReducers int) map[string]int {
	t.Helper()
	totals := make(map[string]int)
	for p := 0; p < numReducers; p++ {
		for _, rec := range readPartition(t, layout, task, p) {
			n, err := strconv.Atoi(rec.Value)
			if err != nil {
				t.Fatalf("non-numeric value %q", rec.Value)
			}
			totals[rec.Key] += n
		}
	}
	return totals
}

func runMapper(t *testing.T, input string, numReducers, spillBytes int, combiner interfaces.Combiner) (storage.Layout, *counters.Set) {
	t.Helper()
	layout := storage.New(t.TempDir(), "job")
	if err := layout.Prepare(1); err != nil {
		t.Fatal(err)
	}
	set := counters.New()

	err := Run(Options{
		Layout:      layout,
		TaskID:      0,
		InputFiles:  []string{input},
		NumReducers: numReducers,
		SpillBytes:  spillBytes,
		Mapper:      wordSplitter{},
		Combiner:    combiner,
		Counters:    set,
	})
	if err != nil {
		t.Fatalf("mapper.Run: %v", err)
	}
	return layout, set
}

func corpusLines() []string {
	return []string{
		"the quick brown fox",
		"the lazy dog sleeps",
		"the fox jumps over the dog",
		"quick quick slow",
	}
}

func TestCombinerDoesNotChangeTotals(t *testing.T) {
	input := writeCorpus(t, corpusLines())

	plainLayout, _ := runMapper(t, input, 3, 1<<20, nil)
	combinedLayout, _ := runMapper(t, input, 3, 1<<20, summingCombiner{})

	plain := totalsFrom(t, plainLayout, 0, 3)
	combined := totalsFrom(t, combinedLayout, 0, 3)

	if len(plain) != len(combined) {
		t.Fatalf("key counts differ: %d vs %d", len(plain), len(combined))
	}
	for key, want := range plain {
		if combined[key] != want {
			t.Errorf("%q: combined %d, plain %d", key, combined[key], want)
		}
	}
}

func TestCombinerShrinksIntermediateOutput(t *testing.T) {
	input := writeCorpus(t, corpusLines())

	_, plainCounters := runMapper(t, input, 3, 1<<20, nil)
	_, combinedCounters := runMapper(t, input, 3, 1<<20, summingCombiner{})

	plain := plainCounters.Get("map.output_records")
	combined := combinedCounters.Get("map.output_records")

	if combined >= plain {
		t.Fatalf("combiner wrote %d records, no better than %d without it", combined, plain)
	}
	if combinedCounters.Get("combine.in") <= combinedCounters.Get("combine.out") {
		t.Fatal("combine.in should exceed combine.out")
	}
}

func TestSpillingMatchesSinglePass(t *testing.T) {
	lines := make([]string, 0, 400)
	for i := 0; i < 400; i++ {
		lines = append(lines, "alpha beta gamma delta epsilon zeta "+strconv.Itoa(i%17))
	}
	input := writeCorpus(t, lines)

	singleLayout, singleCounters := runMapper(t, input, 4, 1<<24, nil)
	spillLayout, spillCounters := runMapper(t, input, 4, 1024, nil)

	if spillCounters.Get("map.spills") == 0 {
		t.Fatal("expected the small buffer to force spills")
	}
	if singleCounters.Get("map.spills") != 0 {
		t.Fatal("expected the large buffer to avoid spilling")
	}

	for p := 0; p < 4; p++ {
		single := readPartition(t, singleLayout, 0, p)
		spilled := readPartition(t, spillLayout, 0, p)

		if len(single) != len(spilled) {
			t.Fatalf("partition %d: %d records single-pass, %d spilled", p, len(single), len(spilled))
		}
		for i := range single {
			if single[i] != spilled[i] {
				t.Fatalf("partition %d record %d: %+v vs %+v", p, i, single[i], spilled[i])
			}
		}
	}
}

func TestPartitionsAreSortedAndDisjoint(t *testing.T) {
	input := writeCorpus(t, corpusLines())
	layout, _ := runMapper(t, input, 3, 1024, summingCombiner{})

	owner := make(map[string]int)
	for p := 0; p < 3; p++ {
		records := readPartition(t, layout, 0, p)
		for i := 1; i < len(records); i++ {
			if records[i-1].Key > records[i].Key {
				t.Fatalf("partition %d is not sorted at %d", p, i)
			}
		}
		for _, rec := range records {
			if prev, seen := owner[rec.Key]; seen && prev != p {
				t.Fatalf("key %q appears in partitions %d and %d", rec.Key, prev, p)
			}
			owner[rec.Key] = p
		}
	}
}

func TestSpillFilesAreCleanedUp(t *testing.T) {
	lines := make([]string, 0, 200)
	for i := 0; i < 200; i++ {
		lines = append(lines, "one two three four five six seven eight")
	}
	layout, _ := runMapper(t, writeCorpus(t, lines), 2, 1024, nil)

	entries, err := os.ReadDir(layout.MapDir(0))
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if len(entry.Name()) >= 6 && entry.Name()[:6] == "spill-" {
			t.Fatalf("spill file %q was left behind", entry.Name())
		}
	}
}
