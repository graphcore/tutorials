CC = g++

all: tut4_ipu_hardware tut4_ipu_model cpp_example

tut4_ipu_hardware: ../tut4_ipu_hardware.cpp
	$(CC) --std=c++11 ../tut4_ipu_hardware.cpp -lpoplar -lpopops -lpoplin -lpoputil -o tut4_ipu_hardware

tut4_ipu_model: ../tut4_ipu_model.cpp
	$(CC) --std=c++11 ../tut4_ipu_model.cpp -lpoplar -lpopops -lpoplin -lpoputil -o tut4_ipu_model

cpp_example: ../libpva/CppExample.cpp
	$(CC) --std=c++11 ../libpva/CppExample.cpp -lpva -ldl -o CppExample

clean:
	rm -f tut4_ipu_hardware tut4_ipu_model CppExample