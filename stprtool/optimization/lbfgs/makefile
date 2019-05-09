MEX := mex
MEXFLAGS := $(shell if uname -m | grep -q x86_64 ; then echo -largeArrayDims ; fi)
MEXSUFFIX := $(shell if uname -m | grep -q x86_64 ; then echo mexa64 ; else echo mexglx ; fi)

MEXFLAGS += -lm



all:	 lbfgs_example lbfgs.$(MEXSUFFIX)

lbfgs_example: lbfgs.o lbfgs_example.o
	g++ -o lbfgs_example lbfgs.o lbfgs_example.o  -lm 

lbfgs.o : lbfgs.cpp lbfgs.h
	g++ -o lbfgs.o -c lbfgs.cpp

lbfgs_example.o: lbfgs_example.cpp lbfgs.h lbfgs.cpp
	g++ -o lbfgs_example.o -c lbfgs_example.cpp


lbfgs.$(MEXSUFFIX):    lbfgs_mex.cpp lbfgs.cpp lbfgs.h
	$(MEX) $(MEXFLAGS) -DMATLAB -O -output $@ lbfgs_mex.cpp lbfgs.cpp


clean:
	rm -f *.o lbfgs_example *.$(MEXSUFFIX)
