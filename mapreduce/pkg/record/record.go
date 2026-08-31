/*
record.go - on-disk key/value format shared by mappers and reducers

Record   - one key/value pair
Writer   - serializes records to a stream
Reader   - deserializes records from a stream

Tab-separated and newline-delimited so files stay greppable. Tabs, newlines
and backslashes inside a key or value are escaped, so a key containing a
separator cannot corrupt the stream the way a naive "key,value" format would.
*/
package record

import (
	"bufio"
	"fmt"
	"io"
	"strings"
)

type Record struct {
	Key   string
	Value string
}

func (r Record) Size() int {
	return len(r.Key) + len(r.Value) + 2
}

func escape(s string) string {
	if !strings.ContainsAny(s, "\\\t\n\r") {
		return s
	}
	var b strings.Builder
	b.Grow(len(s) + 8)
	for _, c := range s {
		switch c {
		case '\\':
			b.WriteString(`\\`)
		case '\t':
			b.WriteString(`\t`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		default:
			b.WriteRune(c)
		}
	}
	return b.String()
}

func unescape(s string) (string, error) {
	if !strings.Contains(s, `\`) {
		return s, nil
	}
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); i++ {
		if s[i] != '\\' {
			b.WriteByte(s[i])
			continue
		}
		i++
		if i >= len(s) {
			return "", fmt.Errorf("record: trailing escape character")
		}
		switch s[i] {
		case '\\':
			b.WriteByte('\\')
		case 't':
			b.WriteByte('\t')
		case 'n':
			b.WriteByte('\n')
		case 'r':
			b.WriteByte('\r')
		default:
			return "", fmt.Errorf("record: unknown escape %q", s[i])
		}
	}
	return b.String(), nil
}

type Writer struct {
	w *bufio.Writer
}

func NewWriter(w io.Writer) *Writer {
	return &Writer{w: bufio.NewWriterSize(w, 1<<16)}
}

func (w *Writer) Write(r Record) error {
	if _, err := w.w.WriteString(escape(r.Key)); err != nil {
		return err
	}
	if err := w.w.WriteByte('\t'); err != nil {
		return err
	}
	if _, err := w.w.WriteString(escape(r.Value)); err != nil {
		return err
	}
	return w.w.WriteByte('\n')
}

func (w *Writer) Flush() error { return w.w.Flush() }

type Reader struct {
	s *bufio.Scanner
}

func NewReader(r io.Reader) *Reader {
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 0, 1<<16), 16<<20)
	return &Reader{s: s}
}

func (r *Reader) Next() (Record, bool, error) {
	for r.s.Scan() {
		line := r.s.Text()
		if line == "" {
			continue
		}
		tab := strings.IndexByte(line, '\t')
		if tab < 0 {
			return Record{}, false, fmt.Errorf("record: missing separator in %q", line)
		}
		key, err := unescape(line[:tab])
		if err != nil {
			return Record{}, false, err
		}
		value, err := unescape(line[tab+1:])
		if err != nil {
			return Record{}, false, err
		}
		return Record{Key: key, Value: value}, true, nil
	}
	return Record{}, false, r.s.Err()
}
