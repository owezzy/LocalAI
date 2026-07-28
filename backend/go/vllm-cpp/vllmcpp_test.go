package main

import (
	"os"
	"path/filepath"
	"testing"
	"unsafe"

	pb "github.com/mudler/LocalAI/pkg/grpc/proto"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestVllmCpp(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "vllm-cpp suite")
}

// The Go POD mirrors must match the C struct layout of vllm.h (ABI v9)
// byte-for-byte: these offsets are the C offsets on LP64 (linux/darwin
// amd64+arm64). A failure here means govllmcpp.go drifted from vllm.h.
var _ = Describe("C ABI struct mirrors", func() {
	It("cModelParams matches vllm_model_params", func() {
		var p cModelParams
		Expect(unsafe.Offsetof(p.ModelPath)).To(Equal(uintptr(0)))
		Expect(unsafe.Offsetof(p.TokenizerConfigPath)).To(Equal(uintptr(8)))
		Expect(unsafe.Offsetof(p.BlockSize)).To(Equal(uintptr(16)))
		Expect(unsafe.Offsetof(p.NumBlocks)).To(Equal(uintptr(20)))
		Expect(unsafe.Offsetof(p.MaxModelLen)).To(Equal(uintptr(24)))
		Expect(unsafe.Offsetof(p.MaxNumSeqs)).To(Equal(uintptr(28)))
		Expect(unsafe.Offsetof(p.ToolParser)).To(Equal(uintptr(32)))
		Expect(unsafe.Offsetof(p.ReasoningParser)).To(Equal(uintptr(40)))
		Expect(unsafe.Offsetof(p.SpeculativeConfig)).To(Equal(uintptr(48)))
		Expect(unsafe.Offsetof(p.EnablePrefixCaching)).To(Equal(uintptr(56)))
		Expect(unsafe.Offsetof(p.MaxNumBatchedTokens)).To(Equal(uintptr(60)))
		Expect(unsafe.Offsetof(p.SchedulingPolicy)).To(Equal(uintptr(64)))
		Expect(unsafe.Offsetof(p.KVTransferConfig)).To(Equal(uintptr(72)))
		Expect(unsafe.Sizeof(p)).To(Equal(uintptr(80)))
	})

	It("cSamplingParams matches vllm_sampling_params (ABI v8)", func() {
		var p cSamplingParams
		Expect(unsafe.Offsetof(p.Temperature)).To(Equal(uintptr(0)))
		Expect(unsafe.Offsetof(p.TopP)).To(Equal(uintptr(4)))
		Expect(unsafe.Offsetof(p.TopK)).To(Equal(uintptr(8)))
		Expect(unsafe.Offsetof(p.MinP)).To(Equal(uintptr(12)))
		Expect(unsafe.Offsetof(p.MaxTokens)).To(Equal(uintptr(16)))
		Expect(unsafe.Offsetof(p.Seed)).To(Equal(uintptr(24)))
		Expect(unsafe.Offsetof(p.HasSeed)).To(Equal(uintptr(32)))
		Expect(unsafe.Offsetof(p.PresencePenalty)).To(Equal(uintptr(36)))
		Expect(unsafe.Offsetof(p.FrequencyPenalty)).To(Equal(uintptr(40)))
		Expect(unsafe.Offsetof(p.RepetitionPenalty)).To(Equal(uintptr(44)))
		Expect(unsafe.Offsetof(p.MinTokens)).To(Equal(uintptr(48)))
		Expect(unsafe.Offsetof(p.IgnoreEOS)).To(Equal(uintptr(52)))
		Expect(unsafe.Offsetof(p.Stop)).To(Equal(uintptr(56)))
		Expect(unsafe.Offsetof(p.NStop)).To(Equal(uintptr(64)))
		Expect(unsafe.Offsetof(p.StructuredJSON)).To(Equal(uintptr(72)))
		Expect(unsafe.Offsetof(p.StructuredRegex)).To(Equal(uintptr(80)))
		Expect(unsafe.Offsetof(p.StructuredChoice)).To(Equal(uintptr(88)))
		Expect(unsafe.Offsetof(p.NStructuredChoice)).To(Equal(uintptr(96)))
		Expect(unsafe.Offsetof(p.StructuredGrammar)).To(Equal(uintptr(104)))
		Expect(unsafe.Offsetof(p.StructuredJSONObject)).To(Equal(uintptr(112)))
		Expect(unsafe.Offsetof(p.LogitsProcessor)).To(Equal(uintptr(120)))
		Expect(unsafe.Offsetof(p.LogitsProcessorUserData)).To(Equal(uintptr(128)))
		Expect(unsafe.Sizeof(p)).To(Equal(uintptr(136)))
	})

	It("cCompletion matches vllm_completion", func() {
		var c cCompletion
		Expect(unsafe.Offsetof(c.Text)).To(Equal(uintptr(0)))
		Expect(unsafe.Offsetof(c.FinishReason)).To(Equal(uintptr(8)))
		Expect(unsafe.Offsetof(c.PromptTokens)).To(Equal(uintptr(16)))
		Expect(unsafe.Offsetof(c.CompletionTokens)).To(Equal(uintptr(20)))
		Expect(unsafe.Sizeof(c)).To(Equal(uintptr(24)))
	})
})

// Pin/mirror skew is the failure mode this backend is most exposed to: the Go
// PODs above are hand-written against one VLLM_ABI_VERSION, and the Makefile
// pins the vllm.cpp commit that produces it. This spec catches drift without
// needing model weights - set VLLM_CPP_LIBRARY to a built libvllm and it binds
// every symbol and compares the library's reported ABI against the mirrors'.
var _ = Describe("real library ABI handshake", func() {
	It("binds every symbol and reports the ABI the mirrors were written against", func() {
		lib := os.Getenv("VLLM_CPP_LIBRARY")
		if lib == "" {
			Skip("VLLM_CPP_LIBRARY not set; skipping the real-library handshake")
		}
		Expect(registerLib(lib)).To(Succeed())
		Expect(vllmABIVersion()).To(Equal(int32(abiVersion)))
		Expect(vllmVersion()).NotTo(BeEmpty())
	})
})

var _ = Describe("parseOptions", func() {
	It("extracts the engine sizing knobs", func() {
		lo := parseOptions(&pb.ModelOptions{Options: []string{
			"block_size:64", "num_blocks:512", "max_num_seqs:32", "unknown:ignored",
		}})
		Expect(lo.blockSize).To(Equal(int32(64)))
		Expect(lo.numBlocks).To(Equal(int32(512)))
		Expect(lo.maxNumSeqs).To(Equal(int32(32)))
	})
	It("ignores malformed and non-positive values", func() {
		lo := parseOptions(&pb.ModelOptions{Options: []string{
			"block_size:abc", "num_blocks:-1", "max_num_seqs", "block_size:0",
		}})
		Expect(lo).To(Equal(loadOptions{}))
	})

	It("carries a speculative_config JSON value through the legacy options list", func() {
		// strings.Cut splits on the FIRST colon only, so a JSON object value
		// survives the "key:value" spelling intact.
		lo := parseOptions(&pb.ModelOptions{Options: []string{
			`speculative_config:{"method":"mtp","num_speculative_tokens":1}`,
		}})
		Expect(lo.speculativeConfig).To(Equal(`{"method":"mtp","num_speculative_tokens":1}`))
	})
})

var _ = Describe("engine_args", func() {
	It("maps every load knob onto the C model params", func() {
		lo := parseOptions(&pb.ModelOptions{EngineArgs: `{
			"block_size": 64,
			"num_blocks": 1024,
			"max_model_len": 16384,
			"max_num_seqs": 32,
			"max_num_batched_tokens": 8192,
			"enable_prefix_caching": true,
			"scheduling_policy": "lpm",
			"tool_parser": "qwen3",
			"reasoning_parser": "deepseek_r1",
			"tokenizer_config": "/models/tok/tokenizer_config.json"
		}`})
		Expect(lo.blockSize).To(Equal(int32(64)))
		Expect(lo.numBlocks).To(Equal(int32(1024)))
		Expect(lo.maxModelLen).To(Equal(int32(16384)))
		Expect(lo.maxNumSeqs).To(Equal(int32(32)))
		Expect(lo.maxNumBatchedTokens).To(Equal(int32(8192)))
		Expect(lo.enablePrefixCaching).To(Equal(int32(1)))
		Expect(lo.schedulingPolicy).To(Equal("lpm"))
		Expect(lo.toolParser).To(Equal("qwen3"))
		Expect(lo.reasoningParser).To(Equal("deepseek_r1"))
		Expect(lo.tokenizerConfigPath).To(Equal("/models/tok/tokenizer_config.json"))
	})

	It("re-marshals a nested speculative_config object to JSON for the engine", func() {
		lo := parseOptions(&pb.ModelOptions{EngineArgs: `{
			"speculative_config": {"method": "mtp", "num_speculative_tokens": 1}
		}`})
		Expect(lo.speculativeConfig).To(MatchJSON(`{"method":"mtp","num_speculative_tokens":1}`))
	})

	It("re-marshals a nested kv_transfer_config object (LMCache) to JSON", func() {
		lo := parseOptions(&pb.ModelOptions{EngineArgs: `{
			"kv_transfer_config": {
				"kv_connector": "LMCacheConnector",
				"kv_role": "kv_both",
				"kv_connector_extra_config": {"host": "127.0.0.1", "port": 65432}
			}
		}`})
		Expect(lo.kvTransferConfig).To(MatchJSON(`{
			"kv_connector":"LMCacheConnector",
			"kv_role":"kv_both",
			"kv_connector_extra_config":{"host":"127.0.0.1","port":65432}
		}`))
	})

	It("accepts a pre-encoded JSON string for the object-valued knobs", func() {
		// A config written by hand (or round-tripped through a flat store) may
		// carry the object as a string; both spellings reach the engine the same.
		lo := parseOptions(&pb.ModelOptions{EngineArgs: `{
			"speculative_config": "{\"method\":\"ngram\",\"num_speculative_tokens\":4}"
		}`})
		Expect(lo.speculativeConfig).To(MatchJSON(`{"method":"ngram","num_speculative_tokens":4}`))
	})

	It("maps enable_prefix_caching false onto the force-OFF tri-state", func() {
		// The C ABI tri-state is 0=model default, 1=on, 2=off, so an explicit
		// `false` must NOT collapse to the 0 that means "let the model decide".
		lo := parseOptions(&pb.ModelOptions{EngineArgs: `{"enable_prefix_caching": false}`})
		Expect(lo.enablePrefixCaching).To(Equal(int32(2)))
	})

	It("leaves the prefix-caching tri-state at the model default when unset", func() {
		lo := parseOptions(&pb.ModelOptions{EngineArgs: `{"max_num_seqs": 4}`})
		Expect(lo.enablePrefixCaching).To(Equal(int32(0)))
	})

	It("accepts the radix-attention alias upstream documents for prefix caching", func() {
		lo := parseOptions(&pb.ModelOptions{EngineArgs: `{"enable_radix_attention": true}`})
		Expect(lo.enablePrefixCaching).To(Equal(int32(1)))
	})

	It("lets engine_args override the legacy options list", func() {
		lo := parseOptions(&pb.ModelOptions{
			Options:    []string{"max_num_seqs:8", "block_size:16"},
			EngineArgs: `{"max_num_seqs": 64}`,
		})
		Expect(lo.maxNumSeqs).To(Equal(int32(64))) // engine_args wins
		Expect(lo.blockSize).To(Equal(int32(16)))  // untouched keys survive
	})

	It("ignores malformed engine_args rather than failing the load", func() {
		lo := parseOptions(&pb.ModelOptions{
			Options:    []string{"max_num_seqs:8"},
			EngineArgs: `{not json`,
		})
		Expect(lo.maxNumSeqs).To(Equal(int32(8)))
	})

	It("ignores unknown keys", func() {
		lo := parseOptions(&pb.ModelOptions{EngineArgs: `{"gpu_memory_utilization": 0.9}`})
		Expect(lo).To(Equal(loadOptions{}))
	})
})

var _ = Describe("samplingFromPredict", func() {
	It("maps the sampling fields onto the C POD", func() {
		sp, _ := samplingFromPredict(&pb.PredictOptions{
			Temperature:      0.7,
			TopP:             0.9,
			TopK:             40,
			MinP:             0.05,
			Tokens:           128,
			Seed:             42,
			Penalty:          1.1,
			PresencePenalty:  0.5,
			FrequencyPenalty: 0.25,
			IgnoreEOS:        true,
		})
		Expect(sp.Temperature).To(BeNumerically("~", 0.7, 1e-6))
		Expect(sp.TopP).To(BeNumerically("~", 0.9, 1e-6))
		Expect(sp.TopK).To(Equal(int32(40)))
		Expect(sp.MinP).To(BeNumerically("~", 0.05, 1e-6))
		Expect(sp.MaxTokens).To(Equal(int32(128)))
		Expect(sp.HasSeed).To(Equal(int32(1)))
		Expect(sp.Seed).To(Equal(uint64(42)))
		Expect(sp.RepetitionPenalty).To(BeNumerically("~", 1.1, 1e-6))
		Expect(sp.PresencePenalty).To(BeNumerically("~", 0.5, 1e-6))
		Expect(sp.FrequencyPenalty).To(BeNumerically("~", 0.25, 1e-6))
		Expect(sp.IgnoreEOS).To(Equal(int32(1)))
	})

	It("keeps the engine defaults for unset fields and stays unseeded", func() {
		sp, keep := samplingFromPredict(&pb.PredictOptions{})
		Expect(sp.TopP).To(BeNumerically("~", 1.0, 1e-6))
		Expect(sp.RepetitionPenalty).To(BeNumerically("~", 1.0, 1e-6))
		Expect(sp.MaxTokens).To(Equal(int32(0))) // unbounded, engine-capped.
		Expect(sp.HasSeed).To(Equal(int32(0)))
		Expect(sp.Stop).To(Equal(uintptr(0)))
		Expect(sp.StructuredGrammar).To(Equal(uintptr(0)))
		Expect(keep).To(BeEmpty())
	})

	It("wires stop prompts and the grammar constraint", func() {
		sp, keep := samplingFromPredict(&pb.PredictOptions{
			StopPrompts: []string{"</s>", "\n\n"},
			Grammar:     "root ::= \"yes\" | \"no\"",
		})
		Expect(sp.NStop).To(Equal(int32(2)))
		Expect(sp.Stop).NotTo(Equal(uintptr(0)))
		Expect(sp.StructuredGrammar).NotTo(Equal(uintptr(0)))
		Expect(keep).NotTo(BeEmpty())
	})
})

var _ = Describe("validModelPath", func() {
	It("accepts a .gguf file", func() {
		dir := GinkgoT().TempDir()
		p := filepath.Join(dir, "model.gguf")
		Expect(os.WriteFile(p, []byte("GGUF"), 0o600)).To(Succeed())
		Expect(validModelPath(p)).To(Succeed())
	})
	It("accepts a directory with config.json", func() {
		dir := GinkgoT().TempDir()
		Expect(os.WriteFile(filepath.Join(dir, "config.json"), []byte("{}"), 0o600)).To(Succeed())
		Expect(validModelPath(dir)).To(Succeed())
	})
	It("refuses a directory without config.json (greedy-probe rule)", func() {
		Expect(validModelPath(GinkgoT().TempDir())).NotTo(Succeed())
	})
	It("refuses a non-gguf file", func() {
		dir := GinkgoT().TempDir()
		p := filepath.Join(dir, "weights.bin")
		Expect(os.WriteFile(p, []byte("x"), 0o600)).To(Succeed())
		Expect(validModelPath(p)).NotTo(Succeed())
	})
	It("refuses a missing path", func() {
		Expect(validModelPath("/nonexistent/model.gguf")).NotTo(Succeed())
	})
})
