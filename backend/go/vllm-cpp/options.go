package main

// Load-time engine configuration, from two config surfaces:
//
//   - `engine_args:` (ModelOptions.EngineArgs, a JSON object) is the canonical
//     one. Keys are spelled exactly as vLLM's own CLI flags, so a config written
//     against vLLM works verbatim here - `speculative_config` and
//     `kv_transfer_config` in particular take the same JSON documents vLLM's
//     --speculative-config / --kv-transfer-config accept, and are handed to the
//     engine unparsed.
//   - `options:` (the free-form "key:value" list) is the older surface this
//     backend shipped with. It is still honoured so existing configs keep
//     working; engine_args wins on any key set in both.
//
// Anything unrecognised is ignored rather than fatal: the engine validates the
// documents it is given and reports a precise error at load, and a config that
// also carries knobs for a different backend must not fail the load here.

import (
	"encoding/json"
	"strconv"
	"strings"

	pb "github.com/mudler/LocalAI/pkg/grpc/proto"
	"github.com/mudler/xlog"
)

type loadOptions struct {
	blockSize  int32 // KV block size (tokens/block); engine default 32.
	numBlocks  int32 // KV blocks to allocate; engine default 256.
	maxNumSeqs int32 // max concurrent sequences; engine default 8.
	// Max sequence length. Also settable through the model config's
	// context_size / max_model_len; see Load for the precedence.
	maxModelLen int32
	// Per-step chunked-prefill token budget (ABI v9). 0 = the engine's
	// bounded per-arch default.
	maxNumBatchedTokens int32
	// Automatic prefix caching tri-state (ABI v7): 0 = the model-capability
	// default, 1 = force on, 2 = force off.
	enablePrefixCaching int32
	// Scheduler admission policy (ABI v9): "" = fcfs, else fcfs|priority|lpm.
	schedulingPolicy string
	// Engine-side parser selection (ABI v4/v5). Empty = the engine
	// auto-detects from the chat template; "none" disables the reasoning
	// split; unknown names fail the first chat call.
	toolParser      string
	reasoningParser string
	// Speculative decoding (ABI v6), as vLLM's --speculative-config JSON:
	// {"method":"mtp"|"dflash"|"ngram", ...}. Empty = no speculation.
	speculativeConfig string
	// External KV connector / LMCache (ABI v9), as vLLM's --kv-transfer-config
	// JSON. Empty = no connector.
	kvTransferConfig string
	// Override for the tokenizer_config.json the chat template is read from
	// (ABI v9). Empty = <model_dir>/tokenizer_config.json.
	tokenizerConfigPath string
}

func parseOptions(opts *pb.ModelOptions) loadOptions {
	lo := loadOptions{}
	applyOptionsList(&lo, opts.GetOptions())
	applyEngineArgs(&lo, opts.GetEngineArgs())
	return lo
}

// applyOptionsList reads the legacy free-form "key:value" list. strings.Cut
// splits on the FIRST colon only, so a JSON object value survives intact.
func applyOptionsList(lo *loadOptions, options []string) {
	for _, o := range options {
		k, v, found := strings.Cut(o, ":")
		if !found {
			continue
		}
		switch strings.TrimSpace(k) {
		case "block_size":
			lo.blockSize = parseInt32(v, lo.blockSize)
		case "num_blocks":
			lo.numBlocks = parseInt32(v, lo.numBlocks)
		case "max_num_seqs":
			lo.maxNumSeqs = parseInt32(v, lo.maxNumSeqs)
		case "max_num_batched_tokens":
			lo.maxNumBatchedTokens = parseInt32(v, lo.maxNumBatchedTokens)
		case "max_model_len":
			lo.maxModelLen = parseInt32(v, lo.maxModelLen)
		case "scheduling_policy", "schedule_policy":
			lo.schedulingPolicy = strings.TrimSpace(v)
		case "tool_parser", "tool_call_parser":
			lo.toolParser = strings.TrimSpace(v)
		case "reasoning_parser":
			lo.reasoningParser = strings.TrimSpace(v)
		case "speculative_config":
			lo.speculativeConfig = strings.TrimSpace(v)
		case "kv_transfer_config":
			lo.kvTransferConfig = strings.TrimSpace(v)
		case "tokenizer_config", "tokenizer_config_path":
			lo.tokenizerConfigPath = strings.TrimSpace(v)
		case "enable_prefix_caching", "enable_radix_attention":
			if b, err := strconv.ParseBool(strings.TrimSpace(v)); err == nil {
				lo.enablePrefixCaching = prefixCachingTriState(b)
			}
		}
	}
}

// applyEngineArgs overlays the `engine_args:` JSON object. A document that does
// not parse is logged and skipped: engine_args is shared with the other engines
// (the vLLM and SGLang backends read the same field), so a stray key must not
// take the model down.
func applyEngineArgs(lo *loadOptions, engineArgs string) {
	if strings.TrimSpace(engineArgs) == "" {
		return
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(engineArgs), &args); err != nil {
		xlog.Warn("[vllm-cpp] ignoring unparseable engine_args", "error", err)
		return
	}
	for k, v := range args {
		switch k {
		case "block_size":
			lo.blockSize = jsonInt32(v, lo.blockSize)
		case "num_blocks":
			lo.numBlocks = jsonInt32(v, lo.numBlocks)
		case "max_num_seqs":
			lo.maxNumSeqs = jsonInt32(v, lo.maxNumSeqs)
		case "max_num_batched_tokens":
			lo.maxNumBatchedTokens = jsonInt32(v, lo.maxNumBatchedTokens)
		case "max_model_len":
			lo.maxModelLen = jsonInt32(v, lo.maxModelLen)
		case "scheduling_policy", "schedule_policy":
			lo.schedulingPolicy = jsonString(v, lo.schedulingPolicy)
		case "tool_parser", "tool_call_parser":
			lo.toolParser = jsonString(v, lo.toolParser)
		case "reasoning_parser":
			lo.reasoningParser = jsonString(v, lo.reasoningParser)
		case "tokenizer_config", "tokenizer_config_path":
			lo.tokenizerConfigPath = jsonString(v, lo.tokenizerConfigPath)
		case "speculative_config":
			lo.speculativeConfig = jsonDocument(v, lo.speculativeConfig, k)
		case "kv_transfer_config":
			lo.kvTransferConfig = jsonDocument(v, lo.kvTransferConfig, k)
		case "enable_prefix_caching", "enable_radix_attention":
			if b, ok := v.(bool); ok {
				lo.enablePrefixCaching = prefixCachingTriState(b)
			}
		default:
			xlog.Debug("[vllm-cpp] ignoring unknown engine_args key", "key", k)
		}
	}
}

// prefixCachingTriState maps a YAML/JSON boolean onto the C ABI's tri-state. An
// explicit `false` must reach the engine as force-OFF (2), NOT as the 0 that
// means "let the model capability decide" - those differ for the hybrid archs,
// which default the cache off, and for the dense ones, which default it on.
func prefixCachingTriState(on bool) int32 {
	if on {
		return prefixCachingOn
	}
	return prefixCachingOff
}

// jsonDocument normalises an object-valued engine_args entry to a JSON string
// for the C ABI. YAML nesting arrives as a map (the natural spelling); a
// pre-encoded JSON string is accepted too, since a config round-tripped through
// a flat store may carry it that way.
func jsonDocument(v any, fallback string, key string) string {
	switch t := v.(type) {
	case string:
		if strings.TrimSpace(t) == "" {
			return fallback
		}
		return t
	default:
		buf, err := json.Marshal(t)
		if err != nil {
			xlog.Warn("[vllm-cpp] ignoring unencodable engine_args value", "key", key, "error", err)
			return fallback
		}
		return string(buf)
	}
}

func jsonString(v any, fallback string) string {
	s, ok := v.(string)
	if !ok {
		return fallback
	}
	return strings.TrimSpace(s)
}

// jsonInt32 accepts the float64 a JSON number decodes to, plus the string
// spelling a YAML config may produce. Non-positive values keep the fallback:
// every knob this covers uses "<= 0 means the engine default".
func jsonInt32(v any, fallback int32) int32 {
	switch t := v.(type) {
	case float64:
		if t <= 0 || t > 1<<31-1 {
			return fallback
		}
		return int32(t)
	case string:
		return parseInt32(t, fallback)
	default:
		return fallback
	}
}

func parseInt32(s string, fallback int32) int32 {
	n, err := strconv.ParseInt(strings.TrimSpace(s), 10, 32)
	if err != nil || n <= 0 {
		return fallback
	}
	return int32(n)
}
