CXX ?= c++
all: tut5_cpu tut5_ipu

tut5_cpu: tut5_complete.cpp
	$(CXX) -g -std=c++11 tut5_complete.cpp -lpoplar -o tut5_cpu

tut5_ipu: tut5_ipu_hardware_complete.cpp
	$(CXX) -g -std=c++11 tut5_ipu_hardware_complete.cpp -lpoplar -o tut5_ipu

clean:
	rm -f tut5_cpu
	rm -f tut5_ipu
