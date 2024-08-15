from abc import ABC, abstractmethod


class LLMCallable(ABC):
    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        pass
