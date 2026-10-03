package main

/*
#cgo linux CFLAGS: -I/app/llama.cpp/include -I/app/llama.cpp/ggml/include
#cgo linux LDFLAGS: -L/app/llama.cpp/build/bin -l:libllama.so

#cgo windows,amd64 CFLAGS: -IC:/Users/JAYANTA/Desktop/LLM_work/LlamaFork/llama.cpp/include -IC:/Users/JAYANTA/Desktop/LLM_work/LlamaFork/llama.cpp/ggml/include
#cgo windows,amd64 LDFLAGS: -LC:/Users/JAYANTA/Desktop/LLM_work/LlamaFork/llama.cpp/build/src/Release -l:llama.lib

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
	"bufio"
	"fmt"
	"time"
	"unicode/utf8"
	"os"
	"unsafe"
)

var ModelPath string

type PromptData struct {
	prompt             string
	enableWatermark    bool
	vocab              *C.struct_llama_vocab
	ctx                *C.struct_llama_context
	smpl               *C.struct_llama_sampler
	model              *C.struct_llama_model
	ResultChan         chan TokenResult
	CloseResultChan    chan bool
	history            unsafe.Pointer
	historySize        int
	n_vocab            int
	gamma              float64
	logitbias          float64
	seed               string
	newTokenID         C.llama_token
	piece              string
	tokenhistoryPtr    *C.llama_token
	totalTokenCnt      int
	totalGreenTokenCnt int
	CurrentZscore      float64
	nctx               C.uint32_t
	nctxUsed           C.llama_pos
	resetSampler       bool
}

// var Data = PromptData{}

var Model *C.struct_llama_model
var Ctx_params C.struct_llama_context_params

// var Smpl *C.struct_llama_sampler

func InitModel() (PromptData, error) {
	fmt.Println("llama.cpp C API loaded")
	fmt.Printf("llama.cpp version: %s\n", C.GoString(C.llama_version()))

	ModelPath := os.Getenv("MODEL_PATH")

	if ModelPath == "" {
		ModelPath = "C:/Users/JAYANTA/Desktop/LLM_work/gguf_store/Swift-Qwen3.8-27B-Q4_K_M.gguf"
	}

	model_params := C.llama_model_default_params()
	model_params.n_gpu_layers = C.int(53)

	C.disable_llama_logs()

	Data := PromptData{}

	model := C.llama_model_load_from_file(C.CString(ModelPath), model_params)
	if model == nil {
		return Data, fmt.Errorf("Model Not Found")
	}
	Model = model
	// defer C.llama_model_free(model)

	vocab := C.llama_model_get_vocab(model)
	ctx_params := C.llama_context_default_params()
	ctx_params.n_ctx = 8192
	Ctx_params = ctx_params

	n_vocab := C.llama_vocab_n_tokens(vocab)

	// ctx := C.llama_init_from_model(model, ctx_params)
	// if ctx == nil {
	// 	return Data, fmt.Errorf("Could not set context")
	// }
	// defer C.llama_free(ctx)

	// smpl := C.llama_sampler_chain_init(C.llama_sampler_chain_default_params())
	// if smpl == nil {
	// 	return Data, fmt.Errorf("Could not set sampler")
	// }
	// Smpl = smpl
	// defer C.llama_sampler_free(smpl)
	// C.llama_sampler_chain_add(smpl, C.llama_sampler_init_greedy())
	// C.llama_sampler_chain_add(smpl, C.llama_sampler_init_temp(C.float(0.8)))
	// C.llama_sampler_chain_add(smpl, C.llama_sampler_init_green_red(C.int32_t(4)))
	// C.llama_sampler_chain_add(smpl, C.llama_sampler_init_top_k(40))
	// C.llama_sampler_chain_add(smpl, C.llama_sampler_init_dist(0))

	fmt.Println("Model Loaded Successfully")

	Data = PromptData{
		vocab: vocab,
		model:   model,
		n_vocab: int(n_vocab),
	}

	fmt.Println("Data Variables: ", Data.prompt, Data.enableWatermark, Data.vocab, Data.ctx, Data.smpl, Data.model)

	return Data, nil
}

func StartGenerationwithParams(session *Session, Data PromptData) (bool, error) {

	seedC := C.CString(Data.seed)
	defer C.free(unsafe.Pointer(seedC))
	history := C.llama_token_history_create()
	Data.history = history

	nCtx := C.llama_n_ctx(Data.ctx)
	Data.nctx = nCtx

	if Data.resetSampler {
		if Data.enableWatermark {
			fmt.Println("Watermarking is enabled")
			fmt.Println("Data Logs Before Sampler: ", Data.logitbias, Data.gamma, Data.seed, Data.history)
			C.llama_sampler_chain_add(Data.smpl, C.llama_sampler_init_green_red(C.float(Data.logitbias), C.float(Data.gamma), seedC, history))
			C.llama_sampler_chain_add(Data.smpl, C.llama_sampler_init_top_k(40))
			C.llama_sampler_chain_add(Data.smpl, C.llama_sampler_init_dist(0))
		} else {
			fmt.Println("Watermarking is disabled")
			C.llama_sampler_chain_add(Data.smpl, C.llama_sampler_init_top_k(40))
			C.llama_sampler_chain_add(Data.smpl, C.llama_sampler_init_dist(0))
		}
	}

	generationOver, err := RunConvo(session, Data, true)
	if err != nil {
		return false, fmt.Errorf("failed to create output file: %w", err)
	}

	return generationOver, nil
}

func Generate(prompt PromptData) (string, bool, []TokenResult, error) {

	defer close(prompt.ResultChan)

	cPrompt := C.CString(prompt.prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	// var generatedTokens []TokenResult
	generatedTokens := make([]TokenResult, 0)

	response := ""

	nPromptTokens := -C.llama_tokenize(prompt.vocab, cPrompt, C.int32_t(len(prompt.prompt)), nil, 0, true, true)

	promptTokens := make([]C.llama_token, nPromptTokens)

	// gotokenhistory := make([]C.llama_token, historySize)

	var tokenPtr *C.llama_token
	if nPromptTokens > 0 {
		tokenPtr = (*C.llama_token)(unsafe.Pointer(&promptTokens[0]))
	}

	ret := C.llama_tokenize(prompt.vocab, cPrompt, C.int32_t(len(prompt.prompt)), tokenPtr, C.int32_t(len(promptTokens)), true, true)
	if ret < 0 {
		return "", false, generatedTokens, fmt.Errorf("failed to tokenize prompt")
	}

	start := 0
	if len(promptTokens) > prompt.historySize {
		start = len(promptTokens) - prompt.historySize
	}

	gotokenhistory := promptTokens[start:]

	for _, token := range gotokenhistory {
		fmt.Printf("Adding to prompt.history: %d\n", token)
		C.llama_token_history_add(prompt.history, token)
	}

	var tokenhistoryPtr *C.llama_token
	if len(gotokenhistory) > 0 {
		tokenhistoryPtr = (*C.llama_token)(unsafe.Pointer(&gotokenhistory[0]))
	}
	prompt.tokenhistoryPtr = tokenhistoryPtr

	batch := C.llama_batch_get_one((*C.llama_token)(unsafe.Pointer(&promptTokens[0])), C.int32_t(len(promptTokens)))

	var newTokenID C.llama_token
	// var nCtx C.uint32_t
	var nCtxUsed C.llama_pos

	prompt.totalGreenTokenCnt = 0
	prompt.totalTokenCnt = 0
	var pending []byte
	var immediateStop bool
	// var generatedTokens []TokenResult

	for {
		// Check context size
		// nCtx = C.llama_n_ctx(prompt.ctx)

		startTime := time.Now()

		nCtxUsed = C.llama_memory_seq_pos_max(C.llama_get_memory(prompt.ctx), 0) + 1

		// prompt.nctx = nCtx
		prompt.nctxUsed = nCtxUsed

		if C.uint32_t(nCtxUsed)+C.uint32_t(batch.n_tokens) > prompt.nctx {
			fmt.Print("\n\033[0m")
			fmt.Println("Current nCtxUsed and nCtx is: ", int(nCtxUsed), prompt.nctx)
			return "", false, generatedTokens, fmt.Errorf("context size exceeded")
		}

		// Run the model
		ret := C.llama_decode(prompt.ctx, batch)
		if ret != 0 {
			return "", false, generatedTokens, fmt.Errorf("failed to decode, ret = %d", ret)
		}

		// Sample next token
		newTokenID = C.llama_sampler_sample(prompt.smpl, prompt.ctx, -1)
		prompt.newTokenID = newTokenID
		// TokenIDChan <- newTokenID
		C.llama_token_history_add(prompt.history, C.llama_token(newTokenID))
		C.llama_token_history_remove_oldest(prompt.history)

		// End of generation?
		if C.llama_vocab_is_eog(prompt.vocab, newTokenID) {
			fmt.Print("\n\033[0m")
			fmt.Println("EOG Token was generated: ", int(newTokenID))
			break
		}

		// Convert token -> text
		var buf [256]C.char

		n := C.llama_token_to_piece(prompt.vocab, newTokenID, &buf[0], C.int32_t(len(buf)), 0, true)

		if n < 0 {
			return "", false, generatedTokens, fmt.Errorf("failed to convert token to piece")
		}

		pieceBytes := C.GoBytes(unsafe.Pointer(&buf[0]), n)

		pending = append(pending, pieceBytes...)

		if utf8.Valid(pending) {
			piece := string(pending)
			prompt.piece = piece
			response += piece
			result := TokenResult{}

			prompt.totalTokenCnt, prompt.totalGreenTokenCnt, prompt.CurrentZscore, result, immediateStop = BasicGreenStreamPercentage(prompt, startTime)

			if immediateStop {
				fmt.Println("Generation stopped due to CloseResultChan signal")
				fmt.Println("Current nCtxUsed and nCtx is: ", int(nCtxUsed), prompt.nctx)
				// C.llama_memory_seq_rm(C.llama_get_memory(prompt.ctx), 0, prompt.nctxUsed-1, prompt.nctxUsed)
				return response, true, generatedTokens, nil
			}

			generatedTokens = append(generatedTokens, result)

			fmt.Print(piece)

			pending = pending[:0]
		}

		defaultTokenID := C.llama_token(10)

		gotokenhistory = append(gotokenhistory, newTokenID)

		if len(gotokenhistory) >= prompt.historySize {
			gotokenhistory = gotokenhistory[1:]
		} else {
			missing := prompt.historySize - len(gotokenhistory)

			padding := make([]C.llama_token, missing)

			for i := range padding {
				padding[i] = defaultTokenID
			}

			gotokenhistory = append(padding, gotokenhistory...)
		}

		if len(gotokenhistory) > 0 {
			tokenhistoryPtr = (*C.llama_token)(unsafe.Pointer(&gotokenhistory[0]))
		}
		prompt.tokenhistoryPtr = tokenhistoryPtr

		// Prepare next batch with the sampled token
		batch = C.llama_batch_get_one(&newTokenID, 1)
	}

	fmt.Println("Current nCtxUsed and nCtx is: ", int(nCtxUsed), prompt.nctx)

	return response, true, generatedTokens, nil
}

// var closeChan = make(chan bool)

func RunConvo(session *Session, prompt PromptData, websocket bool) (bool, error) {

	messages := make([]C.llama_chat_message, 0)
	internalmessages := make([]InternalMessage, 0)
	formatted := make([]C.char, int(C.llama_n_ctx(prompt.ctx)))
	prevLen := 0
	genrationOverGlobal := false
	zscore := 0.0
	ctxUsed := 0

	for {

		if !websocket {
			fmt.Print("> ")

			reader := bufio.NewReader(os.Stdin)
			user, err := reader.ReadString('\n')
			if err != nil {
				return false, err
			}
			prompt.prompt = user
			// fmt.Println("Registered Prompt: ", strings.TrimSpace(strings.ToLower(user)))
		}

		internalmessages, _, _ = convertToInternalMessages(
			internalmessages,
			nil,
			prompt,
			"user",
		)

		// Get chat template
		// fmt.Println(prompt.model)
		tmpl := C.llama_model_chat_template(prompt.model, nil)

		// Create C string for user's message
		cUser := C.CString(prompt.prompt)

		// Add user message
		messages = append(messages, C.llama_chat_message{
			role:    C.CString("user"),
			content: cUser,
		})

		// Apply chat template
		newLen := C.llama_chat_apply_template(tmpl, (*C.llama_chat_message)(unsafe.Pointer(&messages[0])), C.size_t(len(messages)), true, &formatted[0], C.int32_t(len(formatted)))

		// Resize if buffer wasn't large enough
		if newLen > C.int32_t(len(formatted)) {
			formatted = make([]C.char, int(newLen))
			newLen = C.llama_chat_apply_template(tmpl, (*C.llama_chat_message)(unsafe.Pointer(&messages[0])), C.size_t(len(messages)), true, &formatted[0], C.int32_t(len(formatted)))
		}

		if newLen < 0 {
			return false, fmt.Errorf("failed to apply chat template")
		}

		// Get only the newly added portion of the formatted prompt
		promptString := C.GoStringN(
			&formatted[prevLen],
			newLen-C.int32_t(prevLen),
		)

		prompt.prompt = promptString

		// Generate response YELLOW
		// fmt.Print("\033[33m")

		// response, err := Generate(prompt.prompt, prompt.vocab, prompt.ctx, prompt.smpl, prompt.enableWatermark, prompt.ResultChan, prompt.history, prompt.historySize, prompt.n_vocab)
		response, generationOver, generatedTokens, err := Generate(prompt)
		if err != nil {
			return false, err
		}

		genrationOverGlobal = generationOver

		internalmessages, zscore, ctxUsed = convertToInternalMessages(
			internalmessages,
			generatedTokens,
			prompt,
			"assistant",
		)

		//End Yellow
		// fmt.Print("\n\033[0m")

		// Add assistant response to messages
		cResponse := C.CString(response)

		messages = append(messages, C.llama_chat_message{
			role:    C.CString("assistant"),
			content: cResponse,
		})

		// Calculate formatted length WITHOUT adding assistant generation prompt
		prevLen = int(C.llama_chat_apply_template(tmpl, (*C.llama_chat_message)(unsafe.Pointer(&messages[0])), C.size_t(len(messages)), false, nil, 0))

		if prevLen < 0 {
			return generationOver, fmt.Errorf("failed to apply chat template")
		}

		if generationOver {
			break
		}
	}

	// session.TokenSpent = int(prompt.nctxUsed)
	session.TokenTotal = int(prompt.nctx)
	session.Zscore = zscore
	session.TokenSpent = ctxUsed
	session.InternalMessages = append(session.InternalMessages, internalmessages...)

	return genrationOverGlobal, nil
}

func convertToInternalMessages(
	history []InternalMessage,
	messages []TokenResult,
	prompt PromptData,
	role string,
) ([]InternalMessage, float64, int) {

	msg := InternalMessage{
		Typ:             "message",
		Role:            role,
		Data:            make([]string, 0),
		Greensplit:      make([]bool, 0),
		Thinkdata:       make([]string, 0),
		Thinkgreensplit: make([]bool, 0),

		Watermarked: prompt.enableWatermark,
		Logitbias:   prompt.logitbias,
		Gamma:       prompt.gamma,
		Seed:        prompt.seed,
		HistorySize: prompt.historySize,
	}

	// User message
	if role == "user" {
		msg.Data = append(msg.Data, prompt.prompt)
		return append(history, msg), 0.0, 0
	}

	// Assistant message
	inThink := false

	for _, token := range messages {

		text := token.Token

		if text == "<think>" {
			inThink = true
			continue
		}

		if text == "</think>" {
			inThink = false
			continue
		}

		// Normal token
		if inThink {
			msg.Thinkdata = append(msg.Thinkdata, text)
			msg.Thinkgreensplit = append(
				msg.Thinkgreensplit,
				token.IsGreen,
			)
		} else {
			msg.Data = append(msg.Data, text)
			msg.Greensplit = append(
				msg.Greensplit,
				token.IsGreen,
			)
		}
	}

	return append(history, msg), messages[len(messages)-1].ZScore, messages[len(messages)-1].ContextUsed
}
