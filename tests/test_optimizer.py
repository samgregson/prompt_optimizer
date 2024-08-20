from typing import List
import pytest
from prompt_optimizer import Optimizer, llm_node
from prompt_optimizer.prompt_optimizer import (
    IMPROVED_PROMPT_TAG,
    Node,
    TransitionVariable,
    node_registry,
)


class MockLLM:
    def generate_text(self, prompt: str) -> str:
        return f"Processed: {prompt}"


class MockLLMOptimizer:
    def generate_text(self, prompt: str) -> str:
        return (
            f"<{IMPROVED_PROMPT_TAG}>here's an improved prompt</{IMPROVED_PROMPT_TAG}>"
        )


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def mock_llm_optimizer():
    return MockLLMOptimizer()


@pytest.fixture
def optimizer(mock_llm):
    return Optimizer(mock_llm)


@pytest.fixture
def optimizer_prompts(mock_llm_optimizer):
    return Optimizer(mock_llm_optimizer)


@pytest.fixture(autouse=True)
def clear_node_registry():
    node_registry.clear()
    yield


def test_node_variables():
    @llm_node(context_params=["x"], input_var="Hello")
    def test_node(x, input_var):
        return f"{x}: {input_var}"

    assert "test_node" in node_registry
    node = node_registry["test_node"]

    assert isinstance(node, Node)
    assert "input_var" in node.optimizable_variables
    variable_value = node.optimizable_variables["input_var"].value
    assert variable_value == "Hello", f"Unexpected value: {variable_value}"
    assert "x" in node.context_params


def test_node_forward_pass():
    @llm_node(input_var="Hello")
    def test_node(input_var):
        return f"Output: {input_var}"

    node = node_registry["test_node"]

    result = node(input_var="World")
    assert isinstance(result, TransitionVariable)
    assert result.value == "Output: Hello", f"Unexpected result: {result.value}"


def test_node_state():
    @llm_node(context_params=["x"], input_var="Hello")
    def test_node(x, input_var):
        return f"Output: {input_var}, Context: {x}"

    node = node_registry["test_node"]

    # run forward pass
    x = "tea time"
    node(x=x, input_var="World")
    state = node.current_state

    assert "x" in state.context.keys()
    assert (
        "tea time" in state.context.values()
    ), f"expected 'tea time' to be in {state.context.values()}"
    assert state.output.value == "Output: Hello, Context: tea time"
    assert len(state.inputs) == 0


def test_node_backward_pass(optimizer):
    @llm_node(context_params=["x"], input_var="Initial")
    def test_node(x, input_var):
        return f"Output: {input_var}"

    optimizer.initialize_nodes()
    node = optimizer.nodes["test_node"]

    x = "tea time"
    node(x=x, input_var="Test")
    node.backward(node.current_state, program_feedback="Feedback")

    assert len(node.optimizable_variables["input_var"]._feedback) == 1
    feedback = node.optimizable_variables["input_var"]._feedback[0]
    assert (
        "<context>{'x': 'tea time'}</context>" in feedback
    ), f"Feedback does not include context: {x}"
    assert (
        "<current_prompt>Initial</current_prompt>" in feedback
    ), "Feedback does not include current prompt: Initial"
    assert (
        "<generated_output>Output: Initial</generated_output>" in feedback
    ), "Feedback does not include output: Output: Initial"


def test_pipeline_forward(optimizer):
    @llm_node(input_var="Hello")
    def node1(x, input_var):
        return f"Node1: {x} {input_var}"

    @llm_node(input_var="World")
    def node2(x, input_var):
        return f"Node2: {x} {input_var}"

    def program(x):
        return node2(node1(x, input_var=""), input_var="")

    result = optimizer.forward(program, x="Test")
    node_1 = node_registry["node1"]
    node_2 = node_registry["node2"]

    assert [node_1.current_state.output] == node_2.current_state.inputs
    assert isinstance(node_1.current_state.output, TransitionVariable)
    assert all(isinstance(v, TransitionVariable) for v in node_2.current_state.inputs)
    assert result == "Node2: Node1: Test Hello World", f"Unexpected result: {result}"


def test_pipeline_backward(optimizer):
    @llm_node(input_var="Hello")
    def node1(x, input_var):
        return f"Node1: {input_var}"

    @llm_node(input_var="World")
    def node2(x, input_var):
        return f"Node2: {input_var}"

    def program(x):
        return node2(node1(x, input_var=""), input_var="")

    node_1 = node_registry["node1"]
    node_2 = node_registry["node2"]
    node_1.set_optimizer(optimizer)
    node_2.set_optimizer(optimizer)

    optimizer.forward(program, x="Test")
    optimizer.backward("Program feedback")

    feedback1 = optimizer.nodes["node1"].optimizable_variables["input_var"]._feedback
    feedback2 = optimizer.nodes["node2"].optimizable_variables["input_var"]._feedback
    # feedback3 = (
    #     optimizer.nodes["node1"]
    #     .current_state.inputs.optimizable_variables["input_var"]
    #     ._feedback
    # )
    # feedback4 = optimizer.nodes["node2"].optimizable_variables["input_var"]._feedback

    assert len(feedback1) == 1
    assert len(feedback2) == 1
    # assert len(feedback3) == 1
    # assert len(feedback4) == 1


def test_optimizer_step(optimizer_prompts):
    @llm_node(input_var="Initial")
    def test_node(input_var):
        return f"Output: {input_var}"

    node = node_registry["test_node"]
    node.set_optimizer(optimizer_prompts)

    optimizer_prompts.forward(test_node, input_var="Test")
    optimizer_prompts.backward("Feedback")
    optimizer_prompts.step()

    updated_value = node.optimizable_variables["input_var"].value
    assert (
        updated_value
        == "here's an improved prompt"  # no text in <improved_prompt> tags
    ), f"Unexpected updated value: {updated_value}"


def test_optimizer_full_cycle(optimizer_prompts):
    optimizer = optimizer_prompts

    @llm_node(input_var="Initial1")
    def node1(input_var):
        return f"Node1: {input_var}"

    @llm_node(input_var="Initial2")
    def node2(input_var):
        return f"Node2: {input_var}"

    def program(x):
        return node2(node1(x))

    optimizer.initialize_nodes()

    evaluation_template = "Evaluate: {program_output}"
    data = {"x": "Test"}

    optimized_variables = optimizer.optimize(
        iterations=2,
        program_func=program,
        evaluation_template=evaluation_template,
        data=data,
    )

    assert "node1" in optimized_variables, "node1 missing from optimized variables"
    assert "node2" in optimized_variables, "node2 missing from optimized variables"

    node1_value = optimized_variables["node1"]["input_var"].value
    node2_value = optimized_variables["node2"]["input_var"].value

    assert (
        node1_value == "here's an improved prompt"  # no text in <improved_prompt> tags
    ), f"Unexpected node1 value: {node1_value}"
    assert (
        node2_value == "here's an improved prompt"  # no text in <improved_prompt> tags
    ), f"Unexpected node1 value: {node2_value}"


def test_chained_nodes_with_extra_args(optimizer_prompts):
    optimizer = optimizer_prompts

    @llm_node(input_var="Start")
    def node1(input_var, extra_arg):
        return f"Node1: {input_var} | {extra_arg}"

    @llm_node(input_var="Middle")
    def node2(input_var, extra_arg):
        return f"Node2: {input_var} | {extra_arg}"

    @llm_node(input_var="End")
    def node3(input_var, extra_arg):
        return f"Node3: {input_var} | {extra_arg}"

    def program(x):
        return node3(node2(node1(x, "First"), "Second"), "Third")

    optimizer.initialize_nodes()

    result = optimizer.forward(program, x="Test")
    assert result == "Node3: End | Third", f"Unexpected result: {result}"

    optimizer.backward("Chained feedback")
    optimizer.step()

    node1_value = optimizer.nodes["node1"].optimizable_variables["input_var"].value
    node2_value = optimizer.nodes["node2"].optimizable_variables["input_var"].value
    node3_value = optimizer.nodes["node3"].optimizable_variables["input_var"].value

    assert (
        node1_value == "here's an improved prompt"
    ), f"Unexpected node1 value: {node1_value}"
    assert (
        node2_value == "here's an improved prompt"
    ), f"Unexpected node2 value: {node2_value}"
    assert (
        node3_value == "here's an improved prompt"
    ), f"Unexpected node3 value: {node3_value}"


if __name__ == "__main__":
    pytest.main()
