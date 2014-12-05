export CC  = gcc
export CXX = g++
export CFLAGS = -O3 -msse2 -I.. 
BIN = bin/pl2m_train bin/pl2m_infer
OBJ = 
.PHONY: clean all

all: $(BIN)

bin/pl2m_infer: src/infer.cpp src/*.h bin
bin/pl2m_train: src/main.cpp src/*.h bin

export LDFLAGS= -pthread -lm -fopenmp 

bin : 
	mkdir bin

$(BIN) : 
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

clean:
	$(RM) -r $(OBJ) $(BIN) bin *~

