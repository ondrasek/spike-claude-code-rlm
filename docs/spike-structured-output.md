# Spike: Structured Output APIs Across Providers

**Issue:** #10 | **Epic:** #7 | **Date:** 2026-02-23

---

## 1. Per-Provider Findings

### OpenAI (and OpenAI-compatible)

Two modes available via `response_format` parameter:

| Mode | Guarantee | Prompt requirement | Model support |
|------|-----------|-------------------|---------------|
| `json_object` | Valid JSON syntax only (no schema) | Must include "json" in system/user message | gpt-3.5-turbo+ |
| `json_schema` | Valid JSON + exact schema match | None | gpt-4o-2024-08-06+ |

**`json_schema` details:**
- Schema compiled to context-free grammar (CFG) at sampler level — constrained decoding, not post-processing
- Requires `strict: true` in the schema wrapper
- Every object must have `additionalProperties: false` and all properties in `required`
- First-call latency: 2-60s for schema compilation (cached afterward)
- Unsupported: `minLength`/`maxLength`, `minimum`/`maximum`, `pattern`, recursive schemas via external `$ref`, root-level `anyOf`
- Max 100 properties, max 5 nesting levels
- Refusal: `choices[0].message.refusal` (content is `null`)
- Truncation: `finish_reason: "length"` means incomplete JSON

**SDK:**
```python
# json_schema via raw dict
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_schema", "json_schema": {
        "name": "my_schema", "strict": True,
        "schema": { ... }
    }},
    messages=[...],
)
data = json.loads(response.choices[0].message.content)

# json_schema via Pydantic (convenience)
completion = client.beta.chat.completions.parse(
    model="gpt-4o", messages=[...], response_format=MyModel,
)
obj = completion.choices[0].message.parsed  # typed instance
```

### Anthropic

**Two mechanisms** (both GA):

| Mechanism | Response location | `stop_reason` | Model support |
|-----------|------------------|---------------|---------------|
| `output_config.format` (native) | `content[0].text` (JSON string) | `end_turn` | Claude 4.5+, 4.6 |
| Tool-use with `strict: true` | `content[i].input` (dict) | `tool_use` | All Claude 3/4 |

**Native `output_config.format`** (preferred for new code):
```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    output_config={"format": {"type": "json_schema", "schema": { ... }}},
    messages=[...],
)
data = json.loads(response.content[0].text)

# Pydantic convenience
response = client.messages.parse(
    model="claude-sonnet-4-20250514",
    messages=[...], output_format=MyModel,
)
obj = response.parsed_output
```

**Tool-use approach** (legacy, broader model support):
```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    tools=[{"name": "extract", "strict": True, "input_schema": { ... }}],
    tool_choice={"type": "tool", "name": "extract"},
    messages=[...],
)
data = next(b.input for b in response.content if b.type == "tool_use")
```

**Limitations for `strict` / `output_config`:**
- Max 20 strict tools per request
- Max 24 optional parameters across all strict schemas
- No recursive schemas, no numerical constraints
- Grammar compilation cached 24h; first-call latency similar to OpenAI
- Refusal: `stop_reason: "refusal"` — schema not guaranteed
- `tool_choice: any/tool` incompatible with extended thinking

### Ollama

Supported since **v0.5** (December 2024). Uses GBNF grammar-based constrained decoding via llama.cpp.

| Endpoint | Parameter | Modes |
|----------|-----------|-------|
| `/api/chat` (native) | `format` | `"json"` or JSON Schema object directly |
| `/v1/chat/completions` (OpenAI-compat) | `response_format` | `json_object` and `json_schema` |

**Model reliability:**
- **Good:** llama3.1, llama3.2, qwen2.5 (all sizes), mistral
- **Smaller models (7B/8B):** Syntactically valid JSON guaranteed by grammar, but semantic quality degrades. Common issues: whitespace/repetition loops, truncated output, meaningless values.
- **Best practice:** Always include JSON instructions in prompt; use `temperature: 0`.

### OpenRouter

- Supports both `json_object` and `json_schema` via `response_format`
- **Model-dependent:** falls back to `json_object` if underlying provider lacks `json_schema`
- Has a Response Healing plugin (auto-fixes malformed JSON for non-streaming)
- Filter models by `structured_outputs` capability on models page

### HuggingFace

- `json_schema` supported via Inference Providers routing layer
- `json_object` **NOT consistently supported** — HTTP 422 errors reported
- Support is provider- and model-specific (routed through Cerebras, Fireworks, etc.)
- OpenAI SDK `client.beta.chat.completions.parse()` confirmed working

---

## 2. Proposed `StructuredResponse` Dataclass

For the root LM iteration loop. Sub-RLM and verifier remain plain text (no benefit from structured output — they're single-turn, short responses).

```python
@dataclass
class StructuredResponse:
    """Structured LLM response for a single RLM iteration.

    Used when the backend supports structured output to replace
    regex-based code extraction from markdown fences.
    """

    reasoning: str
    """Explanation of the approach, observations, or next steps."""

    code: str | None = None
    """Python code to execute. None when providing a final answer
    or when the LLM needs to explain without code."""

    is_final: bool = False
    """True when the LLM is ready to return its answer."""

    final_answer: str | None = None
    """The completed answer. Only set when is_final is True."""
```

**JSON Schema** (for `response_format` / `output_config`):
```json
{
  "type": "object",
  "properties": {
    "reasoning": { "type": "string" },
    "code": { "type": ["string", "null"] },
    "is_final": { "type": "boolean" },
    "final_answer": { "type": ["string", "null"] }
  },
  "required": ["reasoning", "code", "is_final", "final_answer"],
  "additionalProperties": false
}
```

**Design rationale:**
- **Single `code` field** (not a list): current regex can extract multiple blocks, but structured output works better with one block. The prompt can instruct the LLM to combine code. Simpler schema = better compatibility across providers.
- **`reasoning` always required:** replaces the free-text portions of current markdown responses. Keeps the LLM's chain-of-thought visible for debugging.
- **`is_final` + `final_answer`:** replaces both the `FINAL()` call detection and the `"FINAL" in response` text scan. The LLM sets `is_final: true` and puts the answer in `final_answer` instead of calling `FINAL()`.
- **All fields required, nullable via type union:** satisfies both OpenAI's `additionalProperties: false` + all-`required` constraint and Anthropic's schema rules.

**Note on `anyOf` / type unions:** OpenAI requires nullable fields as `"type": ["string", "null"]` (not `anyOf`). Anthropic supports both. Use the array form for maximum compatibility.

---

## 3. Recommendation: `json_object` vs `json_schema`

**Use `json_schema` (strict mode) as the primary approach.**

Rationale:
- Schema enforcement eliminates parsing failures entirely
- Available on all providers we support: OpenAI (gpt-4o+), Anthropic (Claude 4.5+), Ollama (v0.5+), OpenRouter (model-dependent), HuggingFace (provider-dependent)
- The `StructuredResponse` schema is simple (4 fields, no nesting) — well within all providers' limits

**Fall back to `json_object` + prompt guidance when:**
- Model doesn't support `json_schema` (older OpenAI models, some OpenRouter/HF models)
- Embedding the schema in the prompt and parsing with `json.loads()` + manual validation

**Fall back to regex extraction (current behavior) when:**
- Provider doesn't support either JSON mode (ClaudeCLI backend, CallbackBackend)
- User explicitly disables structured output

**Implementation priority:**
```
json_schema (strict) → json_object + prompt → regex (current)
```

---

## 4. Provider Quirks and Limitations

| Provider | Quirk | Impact |
|----------|-------|--------|
| **OpenAI** | First-call schema compilation: 2-60s latency | Cache warms after first request; negligible for multi-iteration RLM loops |
| **OpenAI** | Prompt must contain "json" for `json_object` mode | Not needed for `json_schema`; only affects fallback path |
| **OpenAI** | Refusal returns `null` content + `refusal` field | Must check `message.refusal` before `json.loads()` |
| **Anthropic** | `output_config.format` only on Claude 4.5+ | Use tool-use with `strict: true` for older models |
| **Anthropic** | `tool_choice: tool` incompatible with extended thinking | If extended thinking is needed, use `output_config.format` instead |
| **Anthropic** | Tool-use `input` is already a dict (not JSON string) | Different parsing path than OpenAI (no `json.loads()` needed) |
| **Anthropic** | Max 24 optional params across strict schemas | Not an issue — our schema has 2 optional fields |
| **Ollama** | Small models (7B/8B): valid syntax but poor semantics | Grammar enforcement doesn't help with answer quality |
| **Ollama** | OpenAI-compat layer `json_schema` wrapper had reliability issues pre-2025 | Use Ollama v0.5+ |
| **OpenRouter** | Falls back to `json_object` silently if model lacks `json_schema` | May get valid JSON without schema enforcement |
| **OpenRouter** | Response Healing plugin auto-fixes malformed JSON | May mask underlying issues; only for non-streaming |
| **HuggingFace** | `json_object` mode returns HTTP 422 on some providers | Skip `json_object` fallback; go straight to regex |
| **HuggingFace** | Support is provider-specific, not model-specific | Cannot guarantee structured output for all HF models |

---

## 5. Impact on Codebase

### Changes needed (mapped to sub-issues)

| Area | Current | With structured output | Issue |
|------|---------|----------------------|-------|
| `LLMBackend` ABC | No `response_format` support | Add optional `structured_output` parameter to `completion()` | #11 |
| `OpenAICompatibleBackend` | Passes `messages` + basic params | Pass `response_format` when structured output enabled | #12 |
| `AnthropicBackend` | Returns `response.content[0].text` | Use `output_config.format` (Claude 4.5+) or tool-use (older) | #13 |
| `RLM._extract_code_blocks()` | Regex: `` ```python\n(.*?)``` `` | Parse `StructuredResponse.code` field; regex as fallback | #14 |
| `RLM.completion()` loop | Checks `FINAL()` + `"FINAL" in response` | Check `StructuredResponse.is_final`; keep old checks as fallback | #14 |
| System prompts | Instruct markdown code fences | Instruct JSON fields; dual prompts for structured/unstructured | #15 |
| `ClaudeCLIBackend` | Shells out to `claude -p` | No structured output support — always use regex fallback | N/A |
| `CallbackBackend` | Test-only; returns plain text | No structured output support — always use regex fallback | N/A |

### Suggested ABC extension

```python
class LLMBackend(ABC):
    @abstractmethod
    def completion(
        self, messages: list[dict[str, str]], model: str, **kwargs: Any
    ) -> CompletionResult: ...

    def supports_structured_output(self) -> bool:
        """Whether this backend supports structured output via response_format."""
        return False
```

`CompletionResult` gains an optional `structured` field:
```python
@dataclass
class CompletionResult:
    text: str
    usage: TokenUsage
    structured: dict[str, Any] | None = None  # parsed StructuredResponse fields
```

The orchestrator checks `backend.supports_structured_output()` to decide the extraction path.
