CXXFLAGS += -g -Wall -std=c++11 -O2
CXXFLAGS += -I./

SRC_MODULES = $(wildcard ./*.cpp)
OBJ_MODULES = $(SRC_MODULES:.cpp=.o)

all: yasli

yasli: $(OBJ_MODULES)
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

clean:
	rm *.o
