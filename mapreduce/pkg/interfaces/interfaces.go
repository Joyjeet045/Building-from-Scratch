/*
interfaces.go - the contract a user's MapReduce program implements

MapInput      - one input record handed to Map
ReduceInput   - iterator over every value sharing one key
Mapper        - turns a record into key/value pairs
Reducer       - folds all values for a key into results
Combiner      - optional map-side pre-aggregation, must be associative
Counters      - named tallies summed across the job

A leaf package with no dependencies, so the framework and the user's binary
can both import it without a cycle.
*/
package interfaces

type MapInput interface {
	Key() string
	Value() string
}

type ReduceInput interface {
	Key() string
	Value() string
	Done() bool
	NextValue()
}

type Mapper interface {
	Map(input MapInput, emit func(key, value string))
}

type Reducer interface {
	Reduce(input ReduceInput, emit func(value string))
}

type Combiner interface {
	Combine(key string, values []string) []string
}

type Counters interface {
	Inc(name string, delta int64)
}
