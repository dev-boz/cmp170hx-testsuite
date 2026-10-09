# CMP 170HX test suite. `make` builds the CUDA tools; `make weight-readback LLAMA_DIR=...` builds the
# llama.cpp-based AI test against your llama.cpp checkout (built with -DGGML_CUDA=ON).
NVCC  ?= nvcc
ARCH  ?= sm_80
CXX   ?= g++
LLAMA_DIR ?=

CUDA_TOOLS = bin/vram-memtest bin/tc-stress bin/sm-verify bin/hbm-map

all: $(CUDA_TOOLS)

bin:
	mkdir -p bin

bin/vram-memtest bin/tc-stress bin/sm-verify: bin/%: src/%.cu | bin
	$(NVCC) -O2 -arch=$(ARCH) -o $@ $<

bin/hbm-map: src/hbm-map.cu | bin
	$(NVCC) -O2 -arch=$(ARCH) -o $@ $< -lcuda

weight-readback: bin/weight-readback

bin/weight-readback: src/weight-readback.cpp | bin
	@test -n "$(LLAMA_DIR)" || { echo "set LLAMA_DIR=/path/to/llama.cpp (built with cmake -DGGML_CUDA=ON)"; exit 1; }
	$(CXX) -O2 -std=c++17 -o $@ $< -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include \
		-L$(LLAMA_DIR)/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$(LLAMA_DIR)/build/bin

clean:
	rm -rf bin

.PHONY: all weight-readback clean
