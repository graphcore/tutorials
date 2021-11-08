CXX ?= c++
all: tut6_cpu tut6_ipu

tut6_cpu: tut6_complete.cpp
	$(CXX) -g -std=c++11 tut6_complete.cpp -lpoplar -o tut6_cpu

tut6_ipu: tut6_ipu_hardware_complete.cpp
	$(CXX) -g -std=c++11 tut6_ipu_hardware_complete.cpp -lpoplar -o tut6_ipu

clean:
	rm -f tut6_cpu
	rm -f tut6_ipu
