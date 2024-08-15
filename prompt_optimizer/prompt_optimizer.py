from dataclasses import dataclass
import logging
from textwrap import dedent
from typing import Callable, Dict, List, Any, ParamSpec, Tuple, Union
from functools import wraps
import inspect
from prompt_optimizer.llm_adaptors.llm_adaptor import LLMAdapter
from prompt_optimizer.utils.extract_xml import extract_text_from_xml
from prompts import (
    prompt_optimiser_prompt,
    input_feedback_prompt,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

node_registry: Dict[str, "Node"] = {}


@dataclass
class NodeOutput:
    """
    Represents the output of a node, storing the node's name, call number,
    and its value.
    """

    node_name: str
    call_number: int
    value: Any


class OptimizableComponent:
    """
    Represents a component within a node that can be optimized based on
    feedback.
    """

    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value
        self.feedback = []
        self.llm: LLMAdapter = None

    def set_llm(self, llm: LLMAdapter):
        self.llm = llm

    def update(self):
        feedback = self._aggregate_feedback()
        prompt = prompt_optimiser_prompt.format(
            component_value=self.value, feedback=feedback
        )
        logging.info(f"Prompt for {self.name}: {prompt}")

        response = self.llm.generate_text(prompt=prompt)
        improved_prompt = extract_text_from_xml(response, "improved_prompt")

        if len(improved_prompt) > 0:
            self.value = improved_prompt[0]

        logging.info(f"Updated value for {self.name}: {self.value}")
        self.feedback.clear()

    def generate_feedback(self, context: str, output: str, output_feedback: str) -> str:
        """
        Generates component feedback based on the given inputs, outputs,
        and output feedback.
        """
        prompt = input_feedback_prompt.format(
            context=context,
            output=output,
            current_prompt=self.value,
            output_feedback=output_feedback,
        )

        feedback = self.llm.generate_text(prompt=prompt)
        logging.info(f"getting feedback for {self.name}\n{feedback}")
        return feedback

    def _aggregate_feedback(self):
        """
        aggregates gradients by LLM sumarisation if required, concatination
        otherwise
        """
        if len(self.feedback) == 0:
            raise ValueError(f"No feedback available for {self.name}")
        agg_feedback = "\n".join(self.feedback)
        if len(agg_feedback) > 1000:
            # summarise gradients
            prompt = dedent(
                f"""
                summarise the following feedback making sure to keep all
                important details:
                {agg_feedback}
                """
            )
            agg_feedback = self.llm.generate_text(prompt=prompt)
        return agg_feedback


class Node:
    def __init__(
        self,
        func: Callable,
        components: Dict[str, Any],
        context_params: List[str],
    ):
        self.func = func
        self.components = {
            name: OptimizableComponent(name, value)
            for name, value in components.items()
        }
        self.name = func.__name__
        self.call_history = []
        self.signature = inspect.signature(func)
        self.context_params = context_params
        self.optimizer: "PipelineOptimizer" = None
        self.llm: LLMAdapter = None

    @property
    def call_number(self) -> int:
        """
        Returns the current call number based on the length of call_history
        """
        return len(self.call_history)

    def __call__(self, *args: Any, **kwargs: Any) -> NodeOutput:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> NodeOutput:
        """
        Performs the forward pass.
        It also updates the call history and handles dependencies between
        nodes.
        """
        logging.info(f"Node {self.name} input args: {args}")
        logging.info(f"Node {self.name} input kwargs: {kwargs}")
        if self.optimizer is not None:
            self.optimizer._pre_node_execution(self.name, self.call_number)
        else:
            logging.warning(f"node `{self.name}` has no optimizer attached")

        bound_args = self._bind_args(args, kwargs)
        logging.info(
            f"Node {self.name} execution started with call number {self.call_number}"
        )
        logging.info(f"Node {self.name} arguments after binding: {bound_args}")

        try:
            result = self.func(*bound_args.args, **bound_args.kwargs)
        except Exception as e:
            result = f"Error: {e}"

        logging.info(f"Node {self.name} execution result: {result}")

        self._update_call_history(bound_args, result)
        if self.optimizer is not None:
            self.optimizer._post_node_execution(self.name, self.call_number)
        return NodeOutput(self.name, self.call_number, result)

    def _bind_args(self, args, kwargs):
        bound_args = self.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        for param_name, param_value in bound_args.arguments.items():
            if isinstance(param_value, NodeOutput):
                self.optimizer._add_dependency(
                    self.name,
                    self.call_number,
                    param_value.node_name,
                    param_value.call_number,
                )
                bound_args.arguments[param_name] = param_value.value
            if param_name in self.components:
                bound_args.arguments[param_name] = self.components[param_name].value

        return bound_args

    def _update_call_history(self, bound_args, result):
        call_data = {
            "context": {
                k: v
                for k, v in bound_args.arguments.items()
                if k in self.context_params
            },
            "output": result,
        }
        self.call_history.append(call_data)

    def backward(self, call_number: int):
        """Compute gradients for each component during the backward pass."""
        logging.info(f"backwards pass through {self.name}")
        call_data = self.call_history[call_number]
        node_context = call_data["context"]
        node_output = call_data["output"]

        for component in self.components.values():
            dependencies = self.optimizer.dependencies.get((self.name, call_number), [])
            output_feedback = []
            if len(dependencies) == 0:
                output_feedback.append(self.optimizer.feedback)
            for dep_node, dep_comp in dependencies:
                feedback = self.optimizer.nodes[dep_node].components[dep_comp].feedback
                output_feedback.append(feedback)
            feedback = component.generate_feedback(
                node_context, node_output, output_feedback
            )
            component.feedback.append(feedback)  #

    def set_optimizer(self, optimizer: "PipelineOptimizer"):
        self.optimizer = optimizer
        for component in self.components.values():
            component.set_llm(optimizer.llm)


P = ParamSpec("P")


def llm_node(context_params: List[str] | None = None, **components):
    """Decorator to define a node in the pipeline."""
    if context_params is None:
        context_params = []

    def decorator(func):
        node = Node(
            func=func,
            components=components,
            context_params=context_params,
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            result = node(*args, **kwargs)
            return result

        # add the node to the node_registry
        node_registry[func.__name__] = node
        logging.info(f"Node {func.__name__} added to registry")

        return wrapper

    return decorator


class PipelineOptimizer:
    """
    Manages the nodes, their execution order, dependencies,
    and optimization process.
    """

    def __init__(self, llm: LLMAdapter):
        self.llm = llm
        self.nodes: Dict[str, Node] = {}
        self.execution_order: List[Tuple[str, int]] = []
        self.dependencies: Dict[Tuple[str, int], set] = {}
        self.feedback = ""  # AKA the Loss

    def _initialize_nodes(self):
        logging.info("####################################################")
        logging.info("################ initializing nodes ################")
        logging.info("####################################################")

        for name, node in node_registry.items():
            node.set_optimizer(self)
            self.nodes[name] = node

    def _pre_node_execution(self, node_name: str, call_number: int):
        """Tracks the execution order of nodes."""
        self.execution_order.append((node_name, call_number))

    def _post_node_execution(self, node_name: str, call_number: int):
        """Placeholder for any post-execution logic."""
        pass

    def _add_dependency(
        self, from_node: str, from_call: int, to_node: str, to_call: int
    ):
        """Adds a dependency between nodes."""
        from_key = (from_node, from_call)
        to_key = (to_node, to_call)
        if from_key not in self.dependencies:
            self.dependencies[from_key] = set()
        self.dependencies[from_key].add(to_key)

    def _reset_dependancies(self):
        """
        Clears the execution history and dependencies.
        """
        self.execution_order.clear()
        self.dependencies.clear()

    def zero_grad(self):
        for node in self.nodes.values():
            for component in node.components.values():
                component.feedback.clear()

    def forward(self, pipeline_func: Callable, *args, **kwargs) -> Any:
        """
        Runs the pipeline and clears history before execution, returning the
        unwrapped final output.
        """
        self._reset_dependancies()
        logging.info(f"Pipeline input args: {args}")
        logging.info(f"Pipeline input kwargs: {kwargs}")
        allowed_keys = inspect.signature(pipeline_func).parameters.keys()
        logging.info(f"Pipeline keys: {allowed_keys}")
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
        final_output = pipeline_func(*args, **filtered_kwargs)
        if isinstance(final_output, NodeOutput):
            return final_output.value  # Unwrap the final NodeOutput
        return final_output

    def _topological_sort(self) -> List[Tuple[str, int]]:
        """Performs a topological sort on the nodes based on dependencies."""
        sorted_nodes = []
        visited = set()

        def dfs(node):
            visited.add(node)
            for dep in self.dependencies.get(node, []):
                if dep not in visited:
                    dfs(dep)
            sorted_nodes.append(node)

        for node in self.execution_order:
            if node not in visited:
                dfs(node)

        return list(reversed(sorted_nodes))

    def backward(self):
        sorted_nodes = self._topological_sort()
        logging.info(f"sorted_nodes: {sorted_nodes}")

        for node_name, call_number in reversed(sorted_nodes):
            self.nodes[node_name].backward(call_number=call_number)

    def step(self):
        for node in self.nodes.values():
            for component in node.components.values():
                component.update()

    def optimize(
        self,
        iterations: int,
        pipeline_func: Callable,
        evaluation_template: str,
        data: Union[List[Dict[str, Any]], Dict[str, Any]],
    ) -> Dict[str, Dict[str, OptimizableComponent]]:
        """
        Optimizes the pipeline based on feedback from the evaluation function.
        """
        self._initialize_nodes()

        for iter in range(iterations):
            logging.info(f"#### Iteration {iter} ####")
            data_list = [data] if isinstance(data, dict) else data
            for item in data_list:
                logging.info(f"Data item for iteration {iter}: {item}")
                try:
                    pipeline_output: NodeOutput = self.forward(pipeline_func, **item)
                    logging.info(
                        f"Pipeline output for iteration {iter}: {pipeline_output}"
                    )
                    evaluation_prompt = evaluation_template.format(
                        pipeline_ouput=pipeline_output, **item
                    )
                    logging.info(f"eval prompt: {evaluation_prompt}")
                    self.feedback = self.llm.generate_text(prompt=evaluation_prompt)
                    logging.info(f"Feedback for iteration {iter}: {self.feedback}")
                    self.backward()
                    self.step()
                    self.zero_grad()
                except Exception as e:
                    logging.error(f"An error occurred: {e}")
                    break

        return {name: node.components for name, node in self.nodes.items()}


# Example usage
if __name__ == "__main__":
    from textwrap import dedent
    from openai import OpenAI
    from patch_openai import patch_openai
    from prompt_optimizer.prompt_optimizer import PipelineOptimizer
    from prompt_optimizer.llm_adaptors.openai_adaptor import OpenAIAdapter

    client = OpenAI()
    client = patch_openai(client)

    system_prompt = "you are a helpful assistent"
    query = "what is the capital of France?"
    golden_answer = "<answer>Paris</answer>"

    llm = OpenAIAdapter(client)
    optimizer = PipelineOptimizer(llm=llm)

    @optimizer.llm_node(context_params="query", system_prompt=system_prompt)
    def answer_query(query: str, system_prompt: str):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
        )
        return response.choices[0].message.content.strip()

    exact_match_evaluator = dedent(
        """
        Your task is to judge the quality if the response given by an AI assistent.
        Below are the <question>, <assistent_response> and <expected_answer>.
        A correct answer would be an exact string match between the <assistent_response> and the
        <expected_answer>

        <question>{query}</question>
        <assistent_response>{pipeline_ouput}</assistent_response>
        <expected_answer>{golden_answer}</expected_answer>
        """
    )

    optimizer.optimize(
        iterations=1,
        pipeline_func=answer_query,
        evaluation_template=exact_match_evaluator,
        data={"query": query, "golden_answer": golden_answer},
    )

    answer_query(query)
