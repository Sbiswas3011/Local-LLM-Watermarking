package main

/*
#cgo CFLAGS: -IC:/Users/JAYANTA/Desktop/llamaClone/llama.cpp/include -IC:/Users/JAYANTA/Desktop/llamaClone/llama.cpp/ggml/include
#cgo LDFLAGS: -L"C:/Users/JAYANTA/Desktop/llamaClone/llama.cpp/build/src/Release" -lllama
#include "llama.h"
*/
import "C"

import (
	"errors"
	"fmt"
)

func main(){
	fmt.Println("llama.cpp C API loaded")
	fmt.Printf("llama.cpp version: %s\n", C.GoString(C.llama_version()))
	modelPath := "C:/Users/JAYANTA/Desktop/gguf_store/Swift-Qwen3.8-27B-Q4_K_M.gguf"

	runModel(modelPath)
}

func runModel(modelPath string) error{

	C.ggml_backend_load_all();
	model_params := C.llama_model_default_params();
    model_params.n_gpu_layers = C.int(99);

	model := C.llama_model_load_from_file(C.CString(modelPath), model_params)
	if(model == nil){
		return errors.New("Model Not Found")
	}

	vocab := C.llama_model_get_vocab(model)
	ctx_params := C.llama_context_default_params()

	ctx := C.llama_init_from_model(model, ctx_params)
	if(ctx == nil){
		return errors.New("Could not set context")
	}

}