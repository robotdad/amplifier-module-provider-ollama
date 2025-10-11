# Ollama Provider Module

Local LLM provider integration for Amplifier using Ollama.

## Features

- Connect to local Ollama server
- Support for all Ollama-compatible models
- Tool calling support
- Streaming responses
- Automatic model pulling (optional)

## Configuration

```python
{
    "host": "http://localhost:11434",  # Ollama server URL (or set OLLAMA_HOST env var)
    "default_model": "llama3.2:3b",    # Default model to use
    "max_tokens": 4096,                # Maximum tokens to generate
    "temperature": 0.7,                # Generation temperature
    "timeout": 120,                    # Request timeout in seconds
    "auto_pull": false                 # Automatically pull missing models
}
```

## Usage

### Prerequisites

#### Option 1: Automated Installation (Recommended)

Use the provided installer script:

```bash
# Install Ollama and pull default model
cd /workspaces/amplifier/amplifier-dev
./scripts/install-ollama.sh --pull-model llama3.2:3b

# Or install Ollama only
./scripts/install-ollama.sh
```

#### Option 2: Manual Installation

1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3.2:3b`
3. Start Ollama server (usually automatic)

### Configuration File

```toml
[provider]
name = "ollama"
model = "llama3.2:3b"
host = "http://localhost:11434"
auto_pull = true
```

### Environment Variables

- `OLLAMA_HOST`: Override default Ollama server URL

## Supported Models

Any model available in Ollama:
- llama3.2:3b (small, fast)
- llama3.2:1b (tiny, fastest)
- mistral (7B)
- mixtral (8x7B)
- codellama (code generation)
- And many more...

See: https://ollama.ai/library

## Error Handling

The provider handles common scenarios gracefully:
- **Server offline**: Mounts successfully, fails on use with clear error
- **Model not found**: Pulls automatically (if auto_pull=true) or provides helpful error
- **Connection issues**: Clear error messages with troubleshooting hints

## Tool Calling

Supports tool calling with compatible models. Tools are automatically formatted in Ollama's expected format (OpenAI-compatible).

## Performance

Local models are typically faster than cloud APIs for:
- Short completions
- Rapid iteration
- Offline/private work

But slower for:
- Large context windows
- Complex reasoning tasks
- Latest capabilities