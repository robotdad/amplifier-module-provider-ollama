# Ollama Image Support Analysis

**Date**: 2026-01-04  
**Status**: Deferred - Not implementing for initial rollout  
**Context**: Part of ImageBlock vision support rollout across providers

## Background

As part of implementing ImageBlock vision support across Amplifier providers, we analyzed whether Ollama should be included in the initial rollout alongside Gemini, Anthropic, and OpenAI.

## Ollama API Format

### Key Difference: Message-Level Images Parameter

Unlike other providers, Ollama uses a **message-level** `images` parameter rather than embedding images in content:

```python
# Gemini, Anthropic, OpenAI (content-level):
{
  "role": "user",
  "content": [
    {"type": "text", "text": "What's in this image?"},
    {"type": "image", "source": {...}}  # ← Image embedded in content
  ]
}

# Ollama (message-level):
{
  "role": "user",
  "content": "What's in this image?",  # ← Plain string only
  "images": ["base64_data_1", "base64_data_2"]  # ← Separate array parameter
}
```

### API Reference

From [Ollama API docs](https://github.com/ollama/ollama/blob/main/docs/api.md):

**Generate endpoint (`/api/generate`)**:
- `images`: (optional) a list of base64-encoded images (for multimodal models such as `llava`)

**Chat endpoint (`/api/chat`)**:
- Message object has `images` field at message level
- Used for multimodal models like llava, bakllava

## Current Implementation

Looking at `_convert_messages()` (lines 1044-1054):

```python
# Handle structured content (list of content blocks from Amplifier)
# Convert to plain string for Ollama which expects string content
if isinstance(content, list):
    text_parts = []
    for block in content:
        if isinstance(block, dict):
            # TextContent block: {"type": "text", "text": "..."}
            if block.get("type") == "text" and "text" in block:
                text_parts.append(block["text"])
            # ToolCallContent or other blocks - skip for now
        elif isinstance(block, str):
            text_parts.append(block)
    content = "\n".join(text_parts) if text_parts else ""
```

**Current behavior**: 
- Structured content is flattened to strings
- ImageBlocks would be silently dropped (same as ToolCallContent)
- This is intentional because Ollama doesn't use content arrays

## Implementation Requirements

To support ImageBlock in Ollama would require:

1. **Extract ImageBlocks** from content list while processing
2. **Build `images` array** at message level:
   ```python
   message = {
       "role": "user",
       "content": "text content here",
       "images": [base64_1, base64_2, ...]  # Extracted from ImageBlocks
   }
   ```
3. **Different pattern** than Gemini/Anthropic/OpenAI
4. **Model detection** - need to know if current model supports vision

## Model Support Variance

**Vision-capable models**:
- `llava:7b`, `llava:13b`, `llava:34b`
- `bakllava`
- `llama3.2-vision`
- `qwen2-vl:7b`

**Text-only models** (majority, including default):
- `llama3.2:3b` (current default)
- `mistral`, `mixtral`
- `codellama`, `deepseek-coder`
- Most other Ollama models

**Challenge**: Provider can't easily determine model capabilities without:
- Querying model info per request (performance impact)
- Maintaining hardcoded list of vision models (brittle)
- Or silently ignoring images for non-vision models (confusing UX)

## Recommendation

**Defer Ollama image support** because:

1. ❌ **Different API architecture** - Message-level `images` array vs content-level embedding
2. ❌ **Implementation complexity** - Requires different pattern than other three providers
3. ❌ **Model variance** - Most Ollama models don't support vision
4. ❌ **Less common use case** - Local models primarily for text, cloud APIs for multimodal
5. ✅ **Can add later** - When users request it for specific vision models

## What Was Implemented Instead

**Initial rollout (complete)**:
- ✅ **Gemini** - PR microsoft/amplifier-module-provider-gemini #5
- ✅ **Anthropic** - PR microsoft/amplifier-module-provider-anthropic #6
- ✅ **OpenAI** - PR microsoft/amplifier-module-provider-openai #3
- ✅ **Azure OpenAI** - Auto-inherits from OpenAI (no code changes needed)

**Priority makes sense**:
1. Cloud APIs with well-defined vision support first
2. Local model providers later when specifically requested

## Future Implementation Path

If/when Ollama image support is needed:

1. **Modify `_convert_messages()`** to:
   - Detect ImageBlocks in content list
   - Extract base64 data into separate `images` array
   - Keep text content as plain string
   
2. **Example implementation**:
   ```python
   # In _convert_messages() for user messages
   if isinstance(content, list):
       text_parts = []
       images = []
       for block in content:
           if block.get("type") == "text":
               text_parts.append(block["text"])
           elif block.get("type") == "image":
               source = block.get("source", {})
               if source.get("type") == "base64":
                   images.append(source.get("data"))
       
       message = {
           "role": "user",
           "content": "\n".join(text_parts)
       }
       if images:
           message["images"] = images
       ollama_messages.append(message)
   ```

3. **Consider model detection**:
   - Check model name for vision indicators (llava, vision, etc.)
   - Or query model info once and cache capabilities
   - Or document that images only work with vision models

4. **Test with vision models**:
   - `llava:7b` or `llava:13b`
   - `llama3.2-vision`
   - Verify images array works correctly

## References

- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- Ollama provider code: `amplifier_module_provider_ollama/__init__.py:1025-1090`
- ImageBlock protocol: `amplifier-core/amplifier_core/message_models.py:82-89`
