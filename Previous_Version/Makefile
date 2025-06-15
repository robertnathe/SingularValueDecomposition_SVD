CXX = g++
CXXFLAGS = -O2 -std=c++17 -I/usr/include/eigen3
LDFLAGS = 
TARGET = output
SOURCES = Singular_Value_Decomposition.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJECTS)
