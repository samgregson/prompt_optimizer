from typing import Any, ParamSpec
from prompt_optimizer.llm_adaptors.llm_adaptor import LLMAdapter

P = ParamSpec("P")


class AnthropicAdapter(LLMAdapter):
    def __init__(self, client: Any, *args: P.args, **kwargs: P.kwargs):
        self.client = client
        self.args = args
        self.kwargs = kwargs
        self.model = (kwargs.pop("model", "claude-2"),)

    def generate_text(self, prompt: str) -> str:
        response = self.client.completion(
            prompt=prompt,
            model=self.model,
            *self.args,
            **self.kwargs,
        )
        return response.completion
