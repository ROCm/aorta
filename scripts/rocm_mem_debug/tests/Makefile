HIPCC ?= hipcc
HIPFLAGS ?=

POSITIVE_SRCS := $(wildcard positive/*.cpp)
NEGATIVE_SRCS := $(wildcard negative/*.cpp)
POSITIVE_BINS := $(patsubst positive/%.cpp,build/%,$(POSITIVE_SRCS))
NEGATIVE_BINS := $(patsubst negative/%.cpp,build/%,$(NEGATIVE_SRCS))

ALL_BINS := $(POSITIVE_BINS) $(NEGATIVE_BINS)

.PHONY: all clean positive negative

all: $(ALL_BINS)

positive: $(POSITIVE_BINS)

negative: $(NEGATIVE_BINS)

build/%: positive/%.cpp | build
	$(HIPCC) $(HIPFLAGS) -o $@ $<

build/%: negative/%.cpp | build
	$(HIPCC) $(HIPFLAGS) -o $@ $<

build:
	mkdir -p build

clean:
	rm -rf build/*
