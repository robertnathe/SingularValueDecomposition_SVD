CXX = g++
CXXFLAGS = -std=c++20 -O3 -I/usr/include/eigen3 -I/usr/include -lboost_system -lboost_filesystem -lboost_math_c99 -Wall -Wextra -Wno-maybe-uninitialized
TARGET = Singular_Value_Decomposition_SVD
SOURCES = Singular_Value_Decomposition_SVD.cpp

all: $(TARGET)

$(TARGET):
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)
