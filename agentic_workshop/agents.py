"""Small reusable tool-calling agents used by the workshop notebooks."""

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import openai
from openai import OpenAI
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputItem,
)
from pydantic import BaseModel


@dataclass
class Tool:
    """A Python function together with the JSON Schema shown to the model."""

    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Any]

    @classmethod
    def from_pydantic(
        cls,
        args_model: type[BaseModel],
        function: Callable[..., Any],
        *,
        name: str,
        description: str,
    ) -> "Tool":
        """Create the tool schema from a Pydantic arguments model."""
        spec = openai.pydantic_function_tool(
            args_model, name=name, description=description
        )["function"]
        return cls(
            name=name,
            description=description,
            parameters=spec["parameters"],
            function=function,
        )

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": True,
        }


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Use the tools to answer the question."


class BaseAgent(ABC):
    """Model, instructions and tools shared by tool-calling agents."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        tools: list[Tool],
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 10,
    ):
        self.client = client
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    def call_llm(
        self, messages: list[dict | ResponseOutputItem], *, tool_choice: str = "auto"
    ) -> Response:
        return self.client.responses.create(
            model=self.model,
            input=messages,
            tools=[tool.to_openai() for tool in self.tools.values()],
            tool_choice=tool_choice,
        )

    @abstractmethod
    def run(self, task: str | list[dict], **kwargs) -> str: ...


class ToolCallingAgent(BaseAgent):
    """Run a Responses API tool loop, with hooks for logging and task-specific stopping."""

    def __init__(
        self,
        *args,
        verbose: bool = True,
        catch_tool_errors: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.catch_tool_errors = catch_tool_errors

    def run(
        self,
        task: str | list[dict],
        *,
        on_response: Callable[[Response], None] | None = None,
        on_tool_call: Callable[[ResponseFunctionToolCall], None] | None = None,
        stop_when: Callable[[str, Any], bool] | None = None,
    ) -> str:
        """Run until the model answers or `stop_when` returns true.

        After at most `max_steps` tool-enabled calls, make one additional call with
        tools disabled to answer from the evidence gathered so far.

        `task` may be plain text or Responses API user content such as text plus an image.
        Optional hooks let applications collect usage and tool-call metrics without replacing
        the loop.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        for _ in range(self.max_steps):
            response = self.call_llm(messages)
            if on_response:
                on_response(response)
            messages += response.output

            tool_calls = [item for item in response.output if item.type == "function_call"]
            if not tool_calls:
                return response.output_text

            for tool_call in tool_calls:
                if on_tool_call:
                    on_tool_call(tool_call)
                result = self.run_tool(tool_call)
                if self.verbose:
                    print(f"{tool_call.name}({tool_call.arguments})\n  -> {result}\n")
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": result,
                    }
                )
                if stop_when and stop_when(tool_call.name, result):
                    return response.output_text

        messages.append(
            {
                "role": "system",
                "content": (
                    "The tool-use step limit has been reached. Give your final answer now "
                    "using only the information gathered so far. State any missing evidence "
                    "or uncertainty explicitly; do not invent facts or claim actions were "
                    "completed unless the tool results confirm them."
                ),
            }
        )
        response = self.call_llm(messages, tool_choice="none")
        if on_response:
            on_response(response)
        return response.output_text

    def run_tool(self, tool_call: ResponseFunctionToolCall) -> str:
        """Run one tool and return a string suitable for a function-call output."""
        try:
            tool = self.tools[tool_call.name]
            arguments = json.loads(tool_call.arguments)
            result = tool.function(**arguments)
            return result if isinstance(result, str) else json.dumps(result, default=str)
        except Exception as error:
            if not self.catch_tool_errors:
                raise
            return f"Error: {error!r}"
