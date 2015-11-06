all:
	g++ -ggdb -std=c++11 -O3 -w -Wall -IimageLib `pkg-config --cflags opencv` -o `basename main.cpp .cpp` ./main.cpp `pkg-config --libs opencv` -LimageLib -lImg -lpng -lz
