/*
source.go - a record file exposed as a sorted stream for k-way merging

FileSource - reads records from one file, satisfies merge.Source
OpenSource - opens a path as a FileSource

Kept in this package rather than in merge so that merge stays free of any
dependency on the on-disk format; the interface is satisfied structurally.
*/
package record

import "os"

type FileSource struct {
	file   *os.File
	reader *Reader
}

func OpenSource(path string) (*FileSource, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	return &FileSource{file: file, reader: NewReader(file)}, nil
}

func (s *FileSource) Next() (key, value string, ok bool, err error) {
	rec, ok, err := s.reader.Next()
	if err != nil || !ok {
		return "", "", false, err
	}
	return rec.Key, rec.Value, true, nil
}

func (s *FileSource) Close() error { return s.file.Close() }
