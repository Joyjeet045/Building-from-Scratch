/*
mapreduce_test.go - end-to-end job tests

Runs complete jobs against a temporary corpus and checks the answer against
counts computed directly in the test, so the whole pipeline is validated
rather than any single stage. Also pins the property that makes the output
useful: each key lands in exactly one output file.
*/
package mapreduce

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/config"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/interfaces"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/record"
)

type splitter struct{}

func (splitter) Map(input interfaces.MapInput, emit func(key, value string)) {
	for _, word := range strings.Fields(strings.ToLower(input.Value())) {
		emit(word, "1")
	}
}

type adder struct{}

func (adder) Reduce(input interfaces.ReduceInput, emit func(value string)) {
	total := 0
	for !input.Done() {
		n, _ := strconv.Atoi(input.Value())
		total += n
		input.NextValue()
	}
	emit(strconv.Itoa(total))
}

func (adder) Combine(key string, values []string) []string {
	total := 0
	for _, v := range values {
		n, _ := strconv.Atoi(v)
		total += n
	}
	return []string{strconv.Itoa(total)}
}

func buildCorpus(t *testing.T, files, linesPerFile int) (string, map[string]int) {
	t.Helper()
	dir := t.TempDir()
	vocab := []string{"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"}
	expected := make(map[string]int)

	for f := 0; f < files; f++ {
		var b strings.Builder
		for l := 0; l < linesPerFile; l++ {
			for w := 0; w < 6; w++ {
				word := vocab[(f*7+l*3+w)%len(vocab)]
				b.WriteString(word)
				b.WriteByte(' ')
				expected[word]++
			}
			b.WriteByte('\n')
		}
		path := filepath.Join(dir, "doc-"+strconv.Itoa(f)+".txt")
		if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir, expected
}

func readOutputs(t *testing.T, result *Result) (map[string]int, map[string]string) {
	t.Helper()
	totals := make(map[string]int)
	owner := make(map[string]string)

	for _, path := range result.OutputFiles {
		f, err := os.Open(path)
		if err != nil {
			t.Fatalf("open output: %v", err)
		}

		r := record.NewReader(f)
		for {
			rec, ok, err := r.Next()
			if err != nil {
				f.Close()
				t.Fatalf("read output: %v", err)
			}
			if !ok {
				break
			}
			n, err := strconv.Atoi(rec.Value)
			if err != nil {
				f.Close()
				t.Fatalf("non-numeric output %q", rec.Value)
			}
			totals[rec.Key] += n
			owner[rec.Key] = filepath.Base(path)
		}
		f.Close()
	}
	return totals, owner
}

func jobConfig(t *testing.T, input string, mappers, reducers, workers int, combiner interfaces.Combiner) *config.Config {
	t.Helper()
	cfg := config.Default()
	cfg.Mode = config.ModeLocal
	cfg.InputDir = input
	cfg.Root = t.TempDir()
	cfg.NumMappers = mappers
	cfg.NumReducers = reducers
	cfg.Workers = workers
	cfg.Address = "127.0.0.1:0"
	cfg.Verbose = false
	cfg.Mapper = splitter{}
	cfg.Reducer = adder{}
	cfg.Combiner = combiner
	return cfg
}

func TestEndToEndWordCount(t *testing.T) {
	input, expected := buildCorpus(t, 6, 50)

	result, err := Execute(jobConfig(t, input, 4, 3, 4, adder{}))
	if err != nil {
		t.Fatalf("execute: %v", err)
	}

	totals, _ := readOutputs(t, result)
	if len(totals) != len(expected) {
		t.Fatalf("got %d distinct keys, want %d", len(totals), len(expected))
	}
	for word, want := range expected {
		if totals[word] != want {
			t.Errorf("%q = %d, want %d", word, totals[word], want)
		}
	}
}

func TestEachKeyLandsInExactlyOneOutputFile(t *testing.T) {
	input, expected := buildCorpus(t, 4, 40)

	result, err := Execute(jobConfig(t, input, 3, 4, 3, nil))
	if err != nil {
		t.Fatalf("execute: %v", err)
	}

	_, owner := readOutputs(t, result)
	if len(owner) != len(expected) {
		t.Fatalf("got %d keys, want %d", len(owner), len(expected))
	}

	seen := make(map[string]bool)
	for _, file := range owner {
		seen[file] = true
	}
	if len(seen) < 2 {
		t.Fatalf("expected keys spread across partitions, all landed in %v", seen)
	}
}

func TestCombinerDoesNotChangeTheAnswer(t *testing.T) {
	input, expected := buildCorpus(t, 5, 30)

	withCombiner, err := Execute(jobConfig(t, input, 3, 2, 3, adder{}))
	if err != nil {
		t.Fatalf("execute with combiner: %v", err)
	}
	without, err := Execute(jobConfig(t, input, 3, 2, 3, nil))
	if err != nil {
		t.Fatalf("execute without combiner: %v", err)
	}

	a, _ := readOutputs(t, withCombiner)
	b, _ := readOutputs(t, without)

	for word, want := range expected {
		if a[word] != want || b[word] != want {
			t.Errorf("%q: combiner=%d plain=%d want=%d", word, a[word], b[word], want)
		}
	}
	if withCombiner.Counters["map.output_records"] >= without.Counters["map.output_records"] {
		t.Errorf("combiner did not shrink intermediate output: %d vs %d",
			withCombiner.Counters["map.output_records"], without.Counters["map.output_records"])
	}
}

func TestResultIsStableAcrossTaskCounts(t *testing.T) {
	input, expected := buildCorpus(t, 8, 25)

	shapes := [][3]int{{1, 1, 1}, {2, 3, 2}, {8, 1, 4}, {3, 5, 6}}
	for _, shape := range shapes {
		result, err := Execute(jobConfig(t, input, shape[0], shape[1], shape[2], adder{}))
		if err != nil {
			t.Fatalf("mappers=%d reducers=%d: %v", shape[0], shape[1], err)
		}
		totals, _ := readOutputs(t, result)
		for word, want := range expected {
			if totals[word] != want {
				t.Errorf("mappers=%d reducers=%d: %q = %d, want %d",
					shape[0], shape[1], word, totals[word], want)
			}
		}
	}
}

func TestSpillingProducesTheSameAnswer(t *testing.T) {
	input, expected := buildCorpus(t, 4, 200)

	cfg := jobConfig(t, input, 2, 2, 2, nil)
	cfg.SpillBytes = 2048

	result, err := Execute(cfg)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	if result.Counters["map.spills"] == 0 {
		t.Fatal("expected the small buffer to force spills")
	}

	totals, _ := readOutputs(t, result)
	for word, want := range expected {
		if totals[word] != want {
			t.Errorf("%q = %d, want %d", word, totals[word], want)
		}
	}
}

func TestExecuteRejectsAnIncompleteConfig(t *testing.T) {
	cfg := config.Default()
	cfg.Mode = config.ModeLocal
	if _, err := Execute(cfg); err == nil {
		t.Fatal("expected a missing input directory to be rejected")
	}

	input, _ := buildCorpus(t, 1, 1)
	cfg = jobConfig(t, input, 1, 1, 1, nil)
	cfg.Mapper = nil
	if _, err := Execute(cfg); err == nil {
		t.Fatal("expected a missing Mapper to be rejected")
	}
}
