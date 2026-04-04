CXX = g++
CXXFLAGS = -Wall -O2 -std=c++17

TRACES_DIR = ../traces_enhanced
PROTO_HDR_DIR = $(TRACES_DIR)/pb_trace/include
PROTO_OBJ_DIR = $(TRACES_DIR)/pb_trace/obj
TRACES_OBJ_DIR = $(TRACES_DIR)/obj

INCLUDES = -I$(PROTO_HDR_DIR) -I$(TRACES_DIR)
LIBS = -lprotobuf -lz -pthread

PROTO_OBJS = $(wildcard $(PROTO_OBJ_DIR)/*.pb.o)
TRACES_OBJS = $(wildcard $(TRACES_OBJ_DIR)/*.o)

.PHONY: all clean test

all: trace-compress

trace-compress: main.o trace_compress.o $(PROTO_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

test_roundtrip: test_roundtrip.o trace_compress.o $(PROTO_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

test: test_roundtrip
	./test_roundtrip

clean:
	rm -f *.o trace-compress test_roundtrip
