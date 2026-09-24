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
	"fmt"
	"math"
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
	Token           string
	GreenCount      int
	TotalCount      int
	GreenPercentage float64
	ZScore          float64
	ContextUsed     int
	TotalContext    int
}

func BasicGreenStreamPercentage(prompt PromptData) (int, int, float64) {

	isGreen := false
	seedC := C.CString(prompt.seed)
	defer C.free(unsafe.Pointer(seedC))

	isGreen = bool(C.llama_sampler_check_basic_watermarkv2(prompt.newTokenID, C.float(prompt.gamma), seedC, prompt.tokenhistoryPtr, C.size_t(prompt.historySize), C.int32_t(prompt.n_vocab)))

	if isGreen {
		prompt.totalGreenTokenCnt++
	}

	prompt.totalTokenCnt++

	expected := float64(prompt.totalTokenCnt) * prompt.gamma
	variance := float64(prompt.totalTokenCnt) * prompt.gamma * (1.0 - prompt.gamma)
	prompt.CurrentZscore = (float64(prompt.totalGreenTokenCnt) - expected) / math.Sqrt(variance)

	result := TokenResult{
		Token:           prompt.piece,
		GreenCount:      prompt.totalGreenTokenCnt,
		TotalCount:      prompt.totalTokenCnt,
		GreenPercentage: float64(prompt.totalGreenTokenCnt * 100 / prompt.totalTokenCnt),
		ZScore:          prompt.CurrentZscore,
		ContextUsed:     int(prompt.nctxUsed),
		TotalContext:    int(prompt.nctx),
	}

	prompt.ResultChan <- result
	// prompt.CloseResultChan <- true

	return prompt.totalTokenCnt, prompt.totalGreenTokenCnt, prompt.CurrentZscore
}

func ProcessText(request ProcessRequest, data PromptData) (int, int, float64, error) {

	data.historySize = request.HistorySize
	data.gamma = request.Gamma
	data.totalTokenCnt = 0
	data.totalGreenTokenCnt = 0
	seedC := C.CString(request.Seed)
	defer C.free(unsafe.Pointer(seedC))

	data.prompt = request.Text
	cPrompt := C.CString(data.prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	nPromptTokens := -C.llama_tokenize(data.vocab, cPrompt, C.int32_t(len(data.prompt)), nil, 0, true, true)

	promptTokens := make([]C.llama_token, nPromptTokens)

	var tokenPtr *C.llama_token
	if nPromptTokens > 0 {
		tokenPtr = (*C.llama_token)(unsafe.Pointer(&promptTokens[0]))
	}

	if len(promptTokens) <= data.historySize {
		return 0, 0, 0, fmt.Errorf("text has too few tokens for history size %d", data.historySize)
	}

	data.totalTokenCnt = data.historySize

	ret := C.llama_tokenize(data.vocab, cPrompt, C.int32_t(len(data.prompt)), tokenPtr, C.int32_t(len(promptTokens)), true, true)
	if ret < 0 {
		return 0, 0, 0.0, fmt.Errorf("failed to tokenize prompt")
	}

	// startToken := 0
	// currentToken := data.historySize
	// startToken := currentToken - data.historySize
	// data.newTokenID = promptTokens[currentToken]

	// gotokenhistory := promptTokens[startToken:currentToken]

	// var tokenhistoryPtr *C.llama_token
	// if len(gotokenhistory) > 0 {
	// 	tokenhistoryPtr = (*C.llama_token)(unsafe.Pointer(&gotokenhistory[0]))
	// }
	// data.tokenhistoryPtr = tokenhistoryPtr

	currentToken := data.historySize

	for {
		// fmt.Println("Current Token & mPromptTokens: ", currentToken, int(nPromptTokens)-1)
		startToken := currentToken - data.historySize
		data.newTokenID = promptTokens[currentToken]

		gotokenhistory := promptTokens[startToken:currentToken]

		var tokenhistoryPtr *C.llama_token
		if len(gotokenhistory) > 0 {
			tokenhistoryPtr = (*C.llama_token)(unsafe.Pointer(&gotokenhistory[0]))
		}
		data.tokenhistoryPtr = tokenhistoryPtr

		isGreen := false

		isGreen = bool(C.llama_sampler_check_basic_watermarkv2(data.newTokenID, C.float(data.gamma), seedC, data.tokenhistoryPtr, C.size_t(data.historySize), C.int32_t(data.n_vocab)))

		if isGreen {
			data.totalGreenTokenCnt++
		}

		data.totalTokenCnt++
		currentToken++

		if currentToken >= int(nPromptTokens)-1 {
			break
		}

	}

	// nCtx := C.llama_n_ctx(Data.ctx)
	// Data.nctx = nCtx

	expected := float64(data.totalTokenCnt) * request.Gamma
	variance := float64(data.totalTokenCnt) * request.Gamma * (1.0 - request.Gamma)
	data.CurrentZscore = (float64(data.totalGreenTokenCnt) - expected) / math.Sqrt(variance)

	return data.totalTokenCnt, data.totalGreenTokenCnt, data.CurrentZscore, nil
}
