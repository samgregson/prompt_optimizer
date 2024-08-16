import inspect
import logging
from abc import ABC
from dataclasses import dataclass
from functools import wraps
from textwrap import dedent
from typing import Any, Callable, Dict, List, ParamSpec, Set, Union

from prompt_optimizer.llm_adapters.llm_adapter import LLMCallable
from prompt_optimizer.prompts import (
    variable_feedback_prompt,
    variable_optimiser_prompt,
)
from prompt_optimizer.utils.extract_from_xml import extract_from_xml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

node_registry: Dict[str, "Node"] = {}


class Variable(ABC):
    """
    Abstract base class for variables within a node.
    """

    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value
        self.feedback = []
        self.llm: LLMCallable = None

    def set_llm(self, llm: LLMCallable):
        self.llm = llm

    def generate_feedback(self, context: str, output: str, output_feedback: str) -> str:
        """
        Generates variable feedback based on the given inputs, outputs,
        and output feedback.
        """
        prompt = variable_feedback_prompt.format(
            context=context,
            output=output,
            current_prompt=self.value,
            output_feedback=output_feedback,
        )
        try:
            feedback = self.llm.generate_text(prompt=prompt)
            logging.info(f"Generated feedback for `{self.name}`: {feedback}")
            return feedback
        except Exception as e:
            logging.error(f"Error generating feedback for `{self.name}`: {str(e)}")
            return ""

    def _aggregate_feedback(self):
        """
        Aggregates feedback by LLM sumarisation if required, concatination
        otherwise
        """
        agg_feedback = "\n".join(self.feedback)
        if len(agg_feedback) > 1000:
            prompt = dedent(
                f"""
                summarise the following feedback making sure to keep all
                important details:
                {agg_feedback}
                """
            )
            try:
                agg_feedback = self.llm.generate_text(prompt=prompt)
            except Exception as e:
                logging.error(f"Error summarising feedback for `{self.name}`: {str(e)}")
        return agg_feedback


class TransitionVariable(Variable):
    """
    Represents the output of a node, storing the node's name, call number,
    and its value.
    """

    def __init__(self, node: "Node", value: Any):
        self.node = node
        self.value = value


class OptimizableVariable(Variable):
    """
    Represents an input variable within a node that can be optimized based on
    feedback.
    """

    def update(self):
        """updates the variable based on aggregated feedback"""
        if not self.feedback:
            logging.error(f"No feedback available for ``{self.name}``. Skipping update")
            return

        feedback = self._aggregate_feedback()
        prompt = variable_optimiser_prompt.format(
            variable_value=self.value, feedback=feedback
        )

        try:
            response = self.llm.generate_text(prompt=prompt)
            improved_value = self._extract_improved_value(response)

            if improved_value:
                self.value = improved_value
            else:
                logging.warning(f"Failed to extract improved value for `{self.name}`")
        except Exception as e:
            logging.error(f"Error updating variable `{self.name}`: {str(e)}")

        logging.info(f"Updated value for `{self.name}`: {self.value}")
        self.feedback.clear()

    def _extract_improved_value(self, response: str) -> str:
        """Extracts the improved value from the response."""
        improved_prompt = extract_from_xml(response, "improved_prompt")
        return improved_prompt[0] if improved_prompt else None


@dataclass
class NodeState:
    node: "Node"
    context: Dict[str, Any]
    output: TransitionVariable


class Node:
    def __init__(
        self,
        func: Callable,
        variables: Dict[str, Any],
        context_params: List[str],
    ):
        self.func = func
        self.variable_dict = {
            name: OptimizableVariable(name, value) for name, value in variables.items()
        }
        self.name = func.__name__
        self.signature = inspect.signature(func)
        self.context_params = context_params
        self.optimizer: "Optimizer" = None
        self.llm: LLMCallable = None
        self.current_state: NodeState = (None,)

    def __call__(self, *args: Any, **kwargs: Any) -> TransitionVariable:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> TransitionVariable:
        """
        Performs the forward pass, updating the call history and handling
        relationships between nodes.
        """

        bound_args = self._bind_args(args, kwargs)
        logging.info(f"Node `{self.name}` bound args: {bound_args}")

        try:
            result = self.func(*bound_args.args, **bound_args.kwargs)
        except Exception as e:
            result = f"Error executing Node `{self.name}`: {e}"

        logging.info(f"Node `{self.name}` execution result: {result}")

        self._set_current_state(bound_args, result)
        self.optimizer._post_node_execution(self)

        return TransitionVariable(self, result)

    def _bind_args(self, args, kwargs):
        """
        Binds arguments to the function signature and handles relationships
        """
        bound_args = self.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        for arg_name, input in bound_args.arguments.items():
            if isinstance(input, TransitionVariable):
                # extract (unwrap) Variable value
                bound_args.arguments[arg_name] = input.value
            if arg_name in self.variable_dict:
                # override OptimizableVariable value
                bound_args.arguments[arg_name] = self.variable_dict[arg_name].value

        return bound_args

    def _set_current_state(self, bound_args: inspect.BoundArguments, result):
        """Creates a NodeVisit instance with the current execution data."""
        self.current_state = NodeState(
            node=self,
            context={
                k: v
                for k, v in bound_args.arguments.items()
                if k in self.context_params
            },
            output=result,
        )

    def backward(self, state: NodeState, program_feedback: str = None):
        """
        Compute feedback for each Variable and NodeOutput in a backward pass.
        """
        logging.info(f"backwards pass through `{self.name}`")
        node_context = state.context
        node_output = state.output

        # propegate feedback to input variables
        for variable in self.variable_dict.values():
            output_feedback = self._collect_output_feedback(program_feedback)
            variable_feedback = variable.generate_feedback(
                node_context, node_output, output_feedback
            )
            variable.feedback.append(variable_feedback)
        # TODO: propegate feedback to other node input

    def _collect_output_feedback(self, program_feedback):
        """Collects feedback from child nodes or the optimizer"""
        if program_feedback:
            return program_feedback
        else:
            raise NotImplementedError("sequential programs not yet supported")

    def set_optimizer(self, optimizer: "Optimizer"):
        """Sets the optimizer for this Node and its components"""
        self.optimizer = optimizer
        for variable in self.variable_dict.values():
            variable.set_llm(optimizer.llm)


P = ParamSpec("P")


def llm_node(context_params: List[str] | None = None, **variables):
    """Decorator to define a node in the program."""
    if context_params is None:
        context_params = []

    def decorator(func):
        node = Node(
            func=func,
            variables=variables,
            context_params=context_params,
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            return node(*args, **kwargs)

        # add the node to the node_registry
        node_registry[func.__name__] = node
        logging.info(f"Node `{func.__name__}` added to registry")

        return wrapper

    return decorator


class Optimizer:
    """
    Manages the nodes, their execution order, edges,
    and optimization process.
    """

    def __init__(self, llm: LLMCallable):
        self.llm = llm
        self.nodes: Dict[str, Node] = {}
        self.execution_stack: List[NodeState] = []

    def _initialize_nodes(self):
        """Adds Nodes to Optimizer and assigns Optimizer to Nodes"""
        logging.info("Attaching nodes to optimizer")

        for name, node in node_registry.items():
            node.set_optimizer(self)
            self.nodes[name] = node

    def _post_node_execution(self, node: Node):
        """Add node state to stack"""
        self.execution_stack.append(node.current_state)

    def zero_grad(self):
        for node in self.nodes.values():
            for variable in node.variable_dict.values():
                variable.feedback.clear()

    def forward(self, program_func: Callable, *args, **kwargs) -> Any:
        """
        Runs the program and clears history before execution, returning the
        unwrapped final output.
        """
        allowed_keys = inspect.signature(program_func).parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
        final_output = program_func(*args, **filtered_kwargs)
        if isinstance(final_output, TransitionVariable):
            return final_output.value
        return final_output

    def backward(self, program_feedback: str):
        """Compute feedback for each variable during the backward pass."""
        if not self.execution_stack:
            raise ValueError("Execution stack is empty. Cannot perform backward pass.")

        # Process the first state with program feedback
        state = self.execution_stack.pop()
        state.node.backward(state, program_feedback)

        # Process remaining states without program feedback
        for state in reversed(self.execution_stack):
            state.node.backward(state)

        # Clear the stack after processing
        self.execution_stack.clear()

    def step(self):
        for node in self.nodes.values():
            for variable in node.variable_dict.values():
                variable.update()

    def optimize(
        self,
        iterations: int,
        program_func: Callable,
        evaluation_template: str,
        data: Union[List[Dict[str, Any]], Dict[str, Any]],
    ) -> Dict[str, Dict[str, OptimizableVariable]]:
        """
        Optimizes the program based on feedback from the evaluation function.
        """
        self._initialize_nodes()

        for iter in range(iterations):
            logging.info(f"#### Iteration {iter} ####")
            data_list = [data] if isinstance(data, dict) else data
            for item in data_list:
                logging.info(f"## Data item: {item} ##")
                try:
                    program_output: TransitionVariable = self.forward(
                        program_func, **item
                    )
                    logging.info(
                        f"Program output for iteration {iter}: {program_output}"
                    )
                    evaluation_prompt = evaluation_template.format(
                        program_ouput=program_output, **item
                    )
                    logging.info(
                        f"Eval prompt for iteration {iter}: {evaluation_prompt}"
                    )
                    feedback = self.llm.generate_text(prompt=evaluation_prompt)
                    logging.info(f"Feedback for iteration {iter}: {feedback}")
                    self.backward(feedback)
                    self.step()
                    self.zero_grad()
                except Exception as e:
                    logging.error(f"An error occurred: {e}")
                    break

        return {name: node.variable_dict for name, node in self.nodes.items()}

    def get_prompt_info(self):
        info = ""
        for node_key, node in self.nodes.items():
            info += f"Node: `{node_key}`\n"
            for variable in node.variable_dict.values():
                info += f"  {variable.name}: {variable.value}\n"
        return info
