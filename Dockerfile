# ---------- Build stage ----------
FROM nvidia/cuda:13.4.1-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Go
ARG GO_VERSION=1.27.1

RUN wget -q https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz \
    && rm -rf /usr/local/go \
    && tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz \
    && rm go${GO_VERSION}.linux-amd64.tar.gz

ENV PATH="/usr/local/go/bin:${PATH}"

WORKDIR /app

# ---------- llama.cpp ----------
COPY LlamaFork/llama.cpp /app/llama.cpp

RUN cmake -S /app/llama.cpp \
    -B /app/llama.cpp/build \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_APP=OFF

RUN cmake --build /app/llama.cpp/build \
    --config Release \
    -j"$(nproc)"


# CMD ["/bin/bash"]    

# ---------- Go backend ----------
COPY Local_LLM_Watermarking/go.mod \
    Local_LLM_Watermarking/go.sum \
    /app/backend/

WORKDIR /app/backend

RUN go mod download

COPY Local_LLM_Watermarking/*.go /app/backend/

RUN mkdir -p /opt/cuda-stubs \
    && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /opt/cuda-stubs/libcuda.so.1

ENV CGO_LDFLAGS="-Wl,-rpath-link,/app/llama.cpp/build/bin -Wl,-rpath-link,/opt/cuda-stubs"

# CMD ["/bin/bash"]  

# Build backend
RUN go build -o /app/server .


# # ---------- Runtime stage ----------
FROM nvidia/cuda:13.4.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/server /app/server

# Copy llama.cpp runtime libraries
COPY --from=builder /app/llama.cpp/build/bin/ /app/lib/

ENV LD_LIBRARY_PATH="/app/lib:${LD_LIBRARY_PATH}"

EXPOSE 8080

CMD ["/app/server"]