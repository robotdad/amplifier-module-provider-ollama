"""
Ollama provider module for Amplifier.
Integrates with local Ollama server for LLM completions.
"""

import logging
import os
from collections.abc import Callable
from typing import Any

from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderResponse
from amplifier_core import ToolCall
from amplifier_core.content_models import TextContent
from amplifier_core.content_models import ToolCallContent
from ollama import AsyncClient
from ollama import ResponseError

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """
    Mount the Ollama provider.

    Args:
        coordinator: Module coordinator
        config: Provider configuration including:
            - host: Ollama server URL (default: from OLLAMA_HOST or http://localhost:11434)
            - default_model: Model to use (default: "llama3.2:3b")
            - max_tokens: Maximum tokens (default: 4096)
            - temperature: Generation temperature (default: 0.7)
            - timeout: Request timeout in seconds (default: 120)
            - auto_pull: Whether to auto-pull missing models (default: False)

    Returns:
        Optional cleanup function
    """
    config = config or {}

    # Get configuration with defaults
    host = config.get("host", os.environ.get("OLLAMA_HOST", "http://localhost:11434"))

    provider = OllamaProvider(host, config)
    await coordinator.mount("providers", provider, name="ollama")

    # Test connection but don't fail mount
    if not await provider._check_connection():
        logger.warning(f"Ollama server at {host} is not reachable. Provider mounted but will fail on use.")
    else:
        logger.info(f"Mounted OllamaProvider at {host}")

    # Return cleanup function (ollama client doesn't have explicit close)
    async def cleanup():
        # Ollama AsyncClient uses httpx internally which handles cleanup
        pass

    return cleanup


class OllamaProvider:
    """Ollama local LLM integration."""

    name = "ollama"

    def __init__(self, host: str, config: dict[str, Any] | None = None):
        """
        Initialize Ollama provider.

        Args:
            host: Ollama server URL
            config: Additional configuration
        """
        self.host = host
        self.client = AsyncClient(host=host)
        self.config = config or {}

        # Configuration with sensible defaults
        self.default_model = self.config.get("default_model", "llama3.2:3b")
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.temperature = self.config.get("temperature", 0.7)
        self.timeout = self.config.get("timeout", 120)
        self.auto_pull = self.config.get("auto_pull", False)

    async def _check_connection(self) -> bool:
        """Verify Ollama server is reachable."""
        try:
            await self.client.list()
            return True
        except Exception:
            return False

    async def _ensure_model_available(self, model: str) -> bool:
        """Check if model is available, attempt to pull if not and auto_pull is enabled."""
        try:
            # Try to get model info
            await self.client.show(model)
            return True
        except ResponseError as e:
            if e.status_code == 404:
                if self.auto_pull:
                    logger.info(f"Model {model} not found, pulling...")
                    try:
                        await self.client.pull(model)
                        return True
                    except Exception as pull_error:
                        logger.error(f"Failed to pull model {model}: {pull_error}")
                        return False
                else:
                    logger.warning(f"Model {model} not found. Set auto_pull=True or run 'ollama pull {model}'")
                    return False
            return False

    async def complete(self, messages: list[dict[str, Any]], **kwargs) -> ProviderResponse:
        """
        Generate completion from messages.

        Args:
            messages: Conversation history
            **kwargs: Additional parameters (model, temperature, max_tokens, tools, stream)

        Returns:
            Provider response
        """
        # Get parameters with fallbacks
        model = kwargs.get("model", self.default_model)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        tools = kwargs.get("tools")
        stream = kwargs.get("stream", False)

        # Ensure model is available
        if not await self._ensure_model_available(model):
            raise ValueError(f"Model {model} not found. Run 'ollama pull {model}' or enable auto_pull")

        # Convert messages to Ollama/OpenAI format
        ollama_messages = self._convert_messages(messages)

        # Prepare request parameters
        params = {
            "model": model,
            "messages": ollama_messages,  # Use converted messages
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        # Add tools if provided
        if tools:
            params["tools"] = self._format_tools_for_ollama(tools)

        try:
            # Call Ollama API
            response = await self.client.chat(**params, stream=stream)

            # Parse response
            if stream:
                # For streaming, we need to handle this differently
                # For now, just collect all chunks
                content_parts = []
                async for chunk in response:
                    if "message" in chunk and "content" in chunk["message"]:
                        content_parts.append(chunk["message"]["content"])

                # Build final response
                final_content = "".join(content_parts)

                # Build content_blocks
                content_blocks = []
                if final_content:
                    content_blocks.append(TextContent(text=final_content))

                return ProviderResponse(
                    content=final_content,
                    content_blocks=content_blocks,
                    raw={"message": {"content": final_content, "role": "assistant"}},
                    usage={},
                    tool_calls=None,
                )
            # Non-streaming response
            message = response.get("message", {})
            content = message.get("content", "")

            # Parse tool calls if present
            tool_calls = []
            if "tool_calls" in message:
                for i, tc in enumerate(message["tool_calls"]):
                    function = tc.get("function", {})
                    tool_calls.append(
                        ToolCall(
                            tool=function.get("name", ""),
                            arguments=function.get("arguments", {}),
                            id=tc.get("id", f"call_{i}"),
                        )
                    )

            # Extract usage if available
            usage = {}
            if "prompt_eval_count" in response:
                usage["input"] = response.get("prompt_eval_count", 0)
            if "eval_count" in response:
                usage["output"] = response.get("eval_count", 0)

            # Build content_blocks for structured content
            content_blocks = []

            # Add text content if present
            if content:
                content_blocks.append(TextContent(text=content))

            # Add tool call content blocks
            if tool_calls:
                import json

                for tc in tool_calls:
                    # Ensure input is a dict (should already be from Ollama API)
                    input_dict = tc.arguments if isinstance(tc.arguments, dict) else {}

                    content_blocks.append(ToolCallContent(id=tc.id, name=tc.tool, input=input_dict))

            return ProviderResponse(
                content=content,
                content_blocks=content_blocks,
                raw=response,
                usage=usage,
                tool_calls=tool_calls if tool_calls else None,
            )

        except ResponseError as e:
            if "connection" in str(e).lower():
                raise ConnectionError(f"Cannot connect to Ollama at {self.host}. Is the server running?")
            logger.error(f"Ollama API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise

    def parse_tool_calls(self, response: ProviderResponse) -> list[ToolCall]:
        """
        Parse tool calls from provider response.

        Args:
            response: Provider response

        Returns:
            List of tool calls
        """
        return response.tool_calls or []

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert Amplifier message format to Ollama/OpenAI format.

        Handles the conversion of:
        - Tool calls in assistant messages (Amplifier format -> OpenAI format)
        - Tool result messages
        - Regular user/assistant/system messages
        """
        ollama_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "assistant":
                # Check for tool_calls in Amplifier format
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Convert Amplifier tool_calls to OpenAI format
                    ollama_tool_calls = []
                    for tc in msg["tool_calls"]:
                        ollama_tool_calls.append(
                            {
                                "id": tc.get("id", ""),
                                "type": "function",  # OpenAI requires this
                                "function": {"name": tc.get("tool", ""), "arguments": tc.get("arguments", {})},
                            }
                        )

                    ollama_messages.append({"role": "assistant", "content": content, "tool_calls": ollama_tool_calls})
                else:
                    # Regular assistant message
                    ollama_messages.append({"role": "assistant", "content": content})

            elif role == "tool":
                # Tool result message
                ollama_messages.append(
                    {"role": "tool", "content": content, "tool_call_id": msg.get("tool_call_id", "")}
                )

            else:
                # User, system, etc. - pass through
                ollama_messages.append(msg)

        return ollama_messages

    def _format_tools_for_ollama(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert tools to Ollama format."""
        ollama_tools = []

        for tool in tools:
            # Get schema from tool if available
            input_schema = getattr(tool, "input_schema", {"type": "object", "properties": {}, "required": []})

            ollama_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": input_schema,
                    },
                }
            )

        return ollama_tools
