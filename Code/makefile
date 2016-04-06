CSRC := CUR.cpp
FSRC := 
CFSRC :=
FFSRC :=

CXX := CC
FC := ftn

CFLAGS := -pg -g -dynamic -O3 -openmp -std=c++11 -mkl
FFLAGS := -pg -g -dynamic -O3 -openmp -mkl

COBJ := $(CSRC:%.cpp=%.o)
FOBJ := $(FSRC:%.f=%.o)
CFLOP := $(CFSRC:%.cpp=%.o)
FFLOP := $(FFSRC:%.f=%.o)

TARGETS := CUR

all: $(TARGETS)

$(COBJ) : %.o : %.cpp
	$(CXX) -c $< -o $@ $(CFLAGS)

$(FOBJ) : %.o : %.f
	$(FC) -c $< -o $@ $(FFLAGS)

$(CFLOP) : %.o : %.cpp
	$(CXX) -c $< -o $@ $(CFLAGS)

$(FFLOP) : %.o : %.f
	$(FC) -c $< -o $@ $(FFLAGS)

CUR: $(COBJ) $(FOBJ)
	$(CXX) $^ -o $@ $(CFLAGS)

clean:
	rm *.o


