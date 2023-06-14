# all: custom_ops.so

# custom_ops.so: group_quantize_decompress/*.cpp
#	g++ -std=c++14 -fPIC \
#		-shared -lpopart -lpoplar -lpoplin -lpopnn -lpopops -lpoputil -lpoprand \
#		-o custom_ops.so

#.PHONY : clean
#clean:
#	-rm custom_ops.so

CXX := g++
CXXFLAGS := -std=c++14 -fPIC -O2 -DONNX_NAMESPACE=onnx -Wall -Wno-sign-compare
LDFLAGS := -shared -lpopart -lpoplar -lpoplin -lpopnn -lpopops -lpoputil -lpoprand -Wl,--no-undefined
SRC_DIR := group_quantize_decompress
#SRCS := $(wildcard $(SRC_DIR)/*.cpp)
SRCS := group_quantize_decompress/group_quantize_decompress.cpp group_quantize_decompress/group_quantize_decompressx.cpp
OBJS := $(SRCS:.cpp=.o)
TARGET := custom_ops.so

all: $(TARGET)

$(TARGET): $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
		$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
		rm -f $(OBJS) $(TARGET)
