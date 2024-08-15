from typing import Callable, Concatenate, ParamSpec
from prompt_optimizer.llm_adapters.llm_adapter import LLMCallable

P = ParamSpec("P")


class CustomAdapter(LLMCallable):
    def __init__(
        self,
        generate_func: Callable[Concatenate[str, P], str],
        *args: P.args,
        **kwargs: P.kwargs
    ):
        self.generate_func = generate_func
        self.args = args
        self.kwargs = kwargs

    def generate_text(self, prompt: str) -> str:
        return self.generate_func(prompt, *self.args, **self.kwargs)
