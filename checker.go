package main

/*
#cgo CFLAGS: -IC:/Users/JAYANTA/Desktop/llamaClone/llama.cpp/include -IC:/Users/JAYANTA/Desktop/llamaClone/llama.cpp/ggml/include
#cgo LDFLAGS: -L"C:/Users/JAYANTA/Desktop/llamaClone/llama.cpp/build/src/Release" -lllama
#include "llama.h"
#include <stdlib.h>
static void silent_log_callback(
    enum ggml_log_level level,
    const char * text,
    void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}
static void disable_llama_logs(void) {
    llama_log_set(silent_log_callback, NULL);
}
*/
import "C"

import (
	"unsafe"
)

// func main() {
// 	fmt.Println("llama.cpp C API loaded")
// 	fmt.Printf("llama.cpp version: %s\n", C.GoString(C.llama_version()))
// 	modelPath := "C:/Users/JAYANTA/Desktop/gguf_store/Swift-Qwen3.8-27B-Q4_K_M.gguf"

// 	model_params := C.llama_model_default_params()
// 	model_params.n_gpu_layers = C.int(99)

// 	C.disable_llama_logs()

// 	model := C.llama_model_load_from_file(C.CString(modelPath), model_params)
// 	if model == nil {
// 		fmt.Println("Model Not Found")
// 		return
// 	}
// 	defer C.llama_model_free(model)

// 	vocab := C.llama_model_get_vocab(model)

// 	reader := bufio.NewReader(os.Stdin)

// 	fmt.Println("Enable StreamCheck? (y/n): ")
// 	input, _ := reader.ReadString('\n')

// 	StreamCheckEnabled := strings.TrimSpace(strings.ToLower(input)) == "y"

// 	if StreamCheckEnabled{
// 		// Wait for the file to be cleared
// 		fmt.Println("Waiting for output.txt to be cleared...")

// 		for {
// 			info, err := os.Stat("../output.txt")

// 			if err == nil && info.Size() == 0 {
// 				break
// 			}

// 			time.Sleep(200 * time.Millisecond)
// 		}

// 		fmt.Println("Waiting for generation...")

// 		// Wait until output.txt starts receiving text
// 		var lastSize int64 = 0

// 		for {
// 			info, err := os.Stat("../output.txt")
// 			if err == nil && info.Size() > 0 {
// 				lastSize = info.Size()
// 				break
// 			}

// 			time.Sleep(200 * time.Millisecond)
// 		}

// 		fmt.Println("Generation started. Waiting 5 seconds...")

// 		// Wait 5 seconds after generation starts
// 		time.Sleep(5 * time.Second)

// 		for {
// 			data, err := os.ReadFile("../output.txt")
// 			if err != nil {
// 				fmt.Println("Failed to read output.txt:", err)
// 				return
// 			}

// 			text := string(data)

// 			value := greenPercentage(text, vocab)

// 			fmt.Printf("Green Percentage: %.2f%%\n", value)

// 			// Check whether the file is still growing
// 			info, err := os.Stat("../output.txt")
// 			if err != nil {
// 				return
// 			}

// 			currentSize := info.Size()

// 			if currentSize == lastSize {
// 				fmt.Println("Generation finished.")
// 				break
// 			}

// 			lastSize = currentSize

// 			time.Sleep(5 * time.Second)
// 		}

// 	}else{

// 		data, err := os.ReadFile("../normal.txt")
// 		if err != nil {
// 			fmt.Println("Failed to read output.txt:", err)
// 			return
// 		}

// 		text := string(data)

// 		value := greenPercentage(text, vocab)

// 		fmt.Println("Green Percentage: ", value)
// 	}

// }

func greenPercentage(text string, vocab *C.struct_llama_vocab) float64 {

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	nTokens := -C.llama_tokenize(vocab, cText, C.int32_t(len(text)), nil, 0, true, true)

	tokens := make([]C.llama_token, nTokens)

	C.llama_tokenize(vocab, cText, C.int32_t(len(text)), (*C.llama_token)(unsafe.Pointer(&tokens[0])), C.int32_t(len(tokens)), true, true)

	green := 0

	// for _, token := range tokens {
	// 	tokenInt := int(token)
	// 	if wm.IsGreen(tokenInt) {
	// 		green++
	// 	}
	// }

	if len(tokens) == 0 {
		return 0
	}

	return float64(green) / float64(len(tokens)) * 100
}

type TokenResult struct {
	Token      string
	GreenCount int
	TotalCount int
}

func BasicGreenStreamPercentage(TokenID C.llama_token, Token string, ResultChan chan TokenResult, history unsafe.Pointer, history_size int, n_vocab int) error {
	totalCount := 0
	greenCount := 0
	isGreen := false
	seed := "i_am_a_llm"
	seedC := C.CString(seed)
	defer C.free(unsafe.Pointer(seedC))

	isGreen = bool(C.llama_sampler_check_basic_watermarkv2(TokenID, C.float(0.6), seedC, history, C.size_t(history_size), C.int32_t(n_vocab)))

	if isGreen {
		greenCount++
	}

	totalCount++

	result := TokenResult{
		Token:      Token,
		GreenCount: greenCount,
		TotalCount: totalCount,
	}

	ResultChan <- result

	return nil
}
