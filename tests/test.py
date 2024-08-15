from openai import OpenAI
import pytest
from prompt_optimizer.llm_adapters.openai_adapter import OpenAIAdapter
from prompt_optimizer.prompt_optimizer import PipelineOptimizer


def test_reponse_format():
    # set up optimizer and llm program
    client = OpenAI()
    llm = OpenAIAdapter(client)
    optimizer = PipelineOptimizer(llm=llm)

    @optimizer.llm_node(system_prompt="You are a helpful assistant.")
    def llm_program(query: str):
        return llm.generate_text(query)

    query = "what is the capital of France?"
    golden_answer = "<answer>Paris</answer>"
    initial_response = llm_program(query)

    assert initial_response != golden_answer
    optimizer.optimize(
        iterations=1,
        pipeline_func=llm_program,
        data={"query": query, "golden_answer": golden_answer},
    )
    optimized_response = llm_program(query)
    assert optimized_response == golden_answer


if __name__ == "__main__":
    pytest.main()
