CXX = g++
# -O5 doesn’t exist (highest is -O3); also fix missing dash before -mavx2 -mavx512f -mavx512dq 
CXXFLAGS = -O3 -march=native -mtune=native -funroll-loops -DNDEBUG 
LDFLAGS = -fopenmp

all: optimized/gemm_opt

optimized/gemm_opt: optimized/cpp/gemm_opt.cpp
	mkdir -p optimized
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f optimized/gemm_opt

