package main

/*
#cgo CFLAGS: -IC:/Users/JAYANTA/Desktop/llamaClone/llama.cpp/include -IC:/Users/JAYANTA/Desktop/llamaClone/llama.cpp/ggml/include
#cgo LDFLAGS: -L"C:/Users/JAYANTA/Desktop/llamaClone/llama.cpp/build/src/Release" -lllama
#include "llama.h"
*/
import "C"

import "fmt"

func main() {
	fmt.Println("llama.cpp C API loaded")
	fmt.Printf("llama.cpp version: %s\n", C.GoString(C.llama_version()))
}