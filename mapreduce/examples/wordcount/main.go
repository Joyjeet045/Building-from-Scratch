/*
main.go - word frequency over a directory of text, the blog's example program

WordCounter - Map, splits a line into words and emits (word, "1")
Adder       - Reduce, sums the counts for one word
Adder       - also Combine, summing map-side before anything is written

Usage semantics follow the paper and the blog: implement Map and Reduce, fill
in a config, call mapreduce.Execute. Because Adder implements Combiner too,
the same summing logic runs on the mapper, which is the optimization the blog
leaves open.
*/
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/config"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/interfaces"
	"github.com/Joyjeet045/Building-from-Scratch/mapreduce/pkg/mapreduce"
)

type WordCounter struct {
	wordRegex *regexp.Regexp
}

func (wc *WordCounter) Map(input interfaces.MapInput, emit func(key, value string)) {
	text := strings.ToLower(input.Value())
	for _, word := range wc.wordRegex.FindAllString(text, -1) {
		emit(word, "1")
	}
}

type Adder struct{}

func (a *Adder) Reduce(input interfaces.ReduceInput, emit func(value string)) {
	total := 0
	for !input.Done() {
		n, err := strconv.Atoi(input.Value())
		if err != nil {
			log.Printf("skipping non-numeric value %q", input.Value())
			input.NextValue()
			continue
		}
		total += n
		input.NextValue()
	}
	emit(strconv.Itoa(total))
}

func (a *Adder) Combine(key string, values []string) []string {
	total := 0
	for _, v := range values {
		n, err := strconv.Atoi(v)
		if err != nil {
			continue
		}
		total += n
	}
	return []string{strconv.Itoa(total)}
}

func main() {
	noCombiner := flag.Bool("no-combiner", false, "disable map-side combining")

	cfg := config.SetupJobConfig()
	cfg.JobName = "wordcount"
	cfg.Mapper = &WordCounter{wordRegex: regexp.MustCompile(`\b[\p{L}\p{N}']+\b`)}
	cfg.Reducer = &Adder{}
	if !*noCombiner {
		cfg.Combiner = &Adder{}
	}

	result, err := mapreduce.Execute(cfg)
	if err != nil {
		log.Fatalf("job failed: %v", err)
	}
	if result == nil {
		return
	}

	fmt.Printf("\njob %s finished in %s\n", result.JobID, result.Elapsed.Round(1e6))
	fmt.Printf("output: %s\n", result.OutputDir)

	names := make([]string, 0, len(result.Counters))
	for name := range result.Counters {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		fmt.Printf("  %-22s %d\n", name, result.Counters[name])
	}

	if result.WorkersLost > 0 || result.Reassignments > 0 || result.BackupsLaunched > 0 {
		fmt.Printf("  %-22s lost=%d reassigned=%d backups=%d\n",
			"fault-tolerance", result.WorkersLost, result.Reassignments, result.BackupsLaunched)
	}

	if _, err := os.Stat(result.OutputDir); err != nil {
		log.Fatalf("output directory missing: %v", err)
	}
}
