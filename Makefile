CXX = g++
CXXFLAGS = -std=c++20 -O3 -Wall -Wextra
# Use the proven system path for Eigen headers
INCLUDES = -I/usr/include/eigen3 -I/usr/include
LIBS = -lboost_system -lboost_filesystem -lboost_math_c99 -lcurl
TARGET = BDCSVD_Parallel_HPC_Solver
SOURCE = BDCSVD_Parallel_HPC_Solver.cpp

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

.PHONY: clean

clean:
	rm -f $(TARGET) *.o
