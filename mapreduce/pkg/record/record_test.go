/*
record_test.go - round-trip and escaping tests for the on-disk format

Covers the case a naive "key,value" format gets wrong: keys and values that
themselves contain the separator, a newline, or a backslash.
*/
package record

import (
	"bytes"
	"strings"
	"testing"
)

func roundTrip(t *testing.T, records []Record) []Record {
	t.Helper()

	var buf bytes.Buffer
	w := NewWriter(&buf)
	for _, rec := range records {
		if err := w.Write(rec); err != nil {
			t.Fatalf("write: %v", err)
		}
	}
	if err := w.Flush(); err != nil {
		t.Fatalf("flush: %v", err)
	}

	r := NewReader(bytes.NewReader(buf.Bytes()))
	var out []Record
	for {
		rec, ok, err := r.Next()
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		if !ok {
			break
		}
		out = append(out, rec)
	}
	return out
}

func TestRoundTripPlainRecords(t *testing.T) {
	in := []Record{{"alice", "1"}, {"bob", "2"}, {"carol", "3"}}
	out := roundTrip(t, in)

	if len(out) != len(in) {
		t.Fatalf("got %d records, want %d", len(out), len(in))
	}
	for i := range in {
		if out[i] != in[i] {
			t.Errorf("record %d: got %+v, want %+v", i, out[i], in[i])
		}
	}
}

func TestRoundTripSurvivesSeparatorsInsideFields(t *testing.T) {
	in := []Record{
		{"key\twith\ttabs", "value"},
		{"key\nwith\nnewlines", "multi\nline\nvalue"},
		{"back\\slash", "trailing\\"},
		{"carriage\rreturn", "\r\n"},
		{"", "empty key"},
	}

	out := roundTrip(t, in)
	if len(out) != len(in) {
		t.Fatalf("got %d records, want %d", len(out), len(in))
	}
	for i := range in {
		if out[i] != in[i] {
			t.Errorf("record %d: got %q/%q, want %q/%q",
				i, out[i].Key, out[i].Value, in[i].Key, in[i].Value)
		}
	}
}

func TestReaderRejectsMissingSeparator(t *testing.T) {
	r := NewReader(strings.NewReader("no-tab-here\n"))
	if _, _, err := r.Next(); err == nil {
		t.Fatal("expected an error for a line with no separator")
	}
}

func TestReaderRejectsUnknownEscape(t *testing.T) {
	r := NewReader(strings.NewReader("bad\\qkey\tvalue\n"))
	if _, _, err := r.Next(); err == nil {
		t.Fatal("expected an error for an unknown escape sequence")
	}
}

func TestSizeCountsKeyAndValue(t *testing.T) {
	if got := (Record{Key: "ab", Value: "cde"}).Size(); got != 7 {
		t.Errorf("Size() = %d, want 7", got)
	}
}
