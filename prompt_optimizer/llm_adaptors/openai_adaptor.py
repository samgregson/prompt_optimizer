from typing import Any, ParamSpec
from prompt_optimizer.llm_adaptors.llm_adaptor import LLMAdapter

P = ParamSpec("P")


class OpenAIAdapter(LLMAdapter):
    def __init__(self, client: Any, *args: P.args, **kwargs: P.kwargs):
        self.client = client
        self.args = args
        self.kwargs = kwargs
        self.model = "gpt-4o-mini"
        # self.model = (self.kwargs.pop("model", "gpt-4o-mini"),)
        # self.temperature = (self.kwargs.pop("temperature", "0"),)

    def generate_text(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            # temperature=self.temperature,
            *self.args,
            **self.kwargs,
        )
        return response.choices[0].message.content
