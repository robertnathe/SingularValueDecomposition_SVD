CXX = g++
CXXFLAGS = -std=c++20 -O3 -march=native -fopenmp -DNDEBUG -I/usr/include/eigen3 -Wall -Wextra -Wno-maybe-uninitialized
TARGET = pca
SOURCES = pca.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET) PC_scores.dat
