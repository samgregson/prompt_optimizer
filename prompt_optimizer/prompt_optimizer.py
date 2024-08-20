import inspect
import logging
from abc import ABC
from dataclasses import dataclass
from functools import wraps
from textwrap import dedent
from typing import Any, Callable, Dict, List, OrderedDict, ParamSpec, Union

from prompt_optimizer.llm_adapters.llm_adapter import LLMCallable
from prompt_optimizer.prompts import (
    variable_feedback_prompt,
    variable_optimiser_prompt,
)
from prompt_optimizer.utils.extract_from_xml import extract_from_xml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

node_registry: Dict[str, "Node"] = {}
IMPROVED_PROMPT_TAG = "improved_prompt"


class Variable(ABC):
    """
    Abstract base class for variables within a node.
    """

    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value
        self._feedback = []

    def generate_feedback(
        self, context: str, output: str, output_feedback: str, llm: LLMCallable
    ) -> str:
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
            feedback = llm.generate_text(prompt=prompt)
            logging.info(f"Generated feedback for `{self.name}`: {feedback}")
            self._feedback.append(feedback)
        except Exception as e:
            logging.error(f"Error generating feedback for `{self.name}`: {str(e)}")

    def get_feedback(self, llm: LLMCallable):
        """
        Aggregates feedback by LLM sumarisation if required, concatination
        otherwise
        """
        agg_feedback = "\n".join(self._feedback)
        if len(agg_feedback) > 1000:
            prompt = dedent(
                f"""
                summarise the following feedback making sure to keep all
                important details:
                {agg_feedback}
                """
            )
            try:
                agg_feedback = llm.generate_text(prompt=prompt)
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
        self.name = f"`{node.name}` output"
        self._feedback = []


class OptimizableVariable(Variable):
    """
    Represents an input variable within a node that can be optimized based on
    feedback.
    """

    def update(self, llm: LLMCallable):
        """updates the variable based on aggregated feedback"""
        if not self._feedback:
            logging.error(f"No feedback available for ``{self.name}``. Skipping update")
            return

        feedback = self.get_feedback(llm=llm)
        prompt = variable_optimiser_prompt.format(
            variable_value=self.value, feedback=feedback
        )

        try:
            response = llm.generate_text(prompt=prompt)
            improved_value = self._extract_improved_value(response)

            if improved_value:
                self.value = improved_value
            else:
                logging.warning(f"Failed to extract improved value for `{self.name}`")
        except Exception as e:
            logging.error(f"Error updating variable `{self.name}`: {str(e)}")

        logging.info(f"Updated value for `{self.name}`: {self.value}")
        self._feedback.clear()

    def _extract_improved_value(self, response: str) -> str:
        """Extracts the improved value from the response."""
        improved_prompt = extract_from_xml(response, IMPROVED_PROMPT_TAG)
        return improved_prompt[0] if improved_prompt else None


@dataclass
class NodeState:
    node: "Node"
    context: Dict[str, Any]
    inputs: List[TransitionVariable]
    output: TransitionVariable


class Node:
    def __init__(
        self,
        func: Callable,
        variables: Dict[str, Any],
        context_params: List[str],
    ):
        self.func = func
        self.optimizable_variables = {
            name: OptimizableVariable(name, value) for name, value in variables.items()
        }
        self.name = func.__name__
        self.signature = inspect.signature(func)
        self.context_params = context_params
        self.optimizer: "Optimizer" = None
        self.current_state: NodeState = None
        self.output = TransitionVariable(self, None)

    def __call__(self, *args: Any, **kwargs: Any) -> TransitionVariable:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> TransitionVariable:
        """
        Performs the forward pass, updating the call history and handling
        relationships between nodes.
        """

        all_kwargs = self._get_all_kwargs(args, kwargs)
        context = self._extract_context(all_kwargs)
        input_variables = self._extract_input_variables(all_kwargs)
        all_kwargs = self._unwrap_input_arguments(all_kwargs)
        all_kwargs = self._override_optimizable_variables(all_kwargs)

        logging.info(f"Node `{self.name}` bound args: {all_kwargs}")

        try:
            result = self.func(**all_kwargs)
            logging.info(f"Node `{self.name}` execution result: {result}")
        except Exception as e:
            result = f"Error executing Node `{self.name}`: {e}"

        self.output.value = result
        # output = TransitionVariable(self, result)
        self._set_current_state(input_variables, context, self.output)

        if self.optimizer is None:
            logging.warning(f"node `{self.name}` has no optimizer attached")
        else:
            self.optimizer._add_state_to_stack(self.current_state)

        return self.output

    def _get_all_kwargs(self, args, kwargs):
        """
        Binds arguments to the function signature
        """
        bound_args = self.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        return bound_args.arguments

    def _unwrap_input_arguments(self, all_kwargs: OrderedDict[str, Any]):
        """
        Extracts just the value from TransitionVariables
        """
        for arg_name, input in all_kwargs.items():
            if isinstance(input, TransitionVariable):
                # extract (unwrap) Variable value
                all_kwargs[arg_name] = input.value

        return all_kwargs

    def _override_optimizable_variables(self, all_kwargs: OrderedDict[str, Any]):
        """
        Overrides the values of optimizable variables in the given bound arguments.
        """
        for arg_name, input in all_kwargs.items():
            if arg_name in self.optimizable_variables:
                # override OptimizableVariable value
                all_kwargs[arg_name] = self.optimizable_variables[arg_name].value

        return all_kwargs

    def _set_current_state(
        self,
        input_variables: List[TransitionVariable],
        context: Any,
        result: TransitionVariable,
    ):
        """Creates a NodeVisit instance with the current execution data."""

        self.current_state = NodeState(
            node=self,
            context=context,
            inputs=input_variables,
            output=result,
        )

    def _extract_context(self, all_kwargs: OrderedDict[str, Any]):
        """Extract context as defined by `context_params` as Dict from kwargs"""
        context = {}
        for key, value in all_kwargs.items():
            if key in self.context_params:
                value_string = (
                    value.value if isinstance(value, TransitionVariable) else str(value)
                )
                context[key] = value_string
        return context

    def _extract_input_variables(self, all_kwargs: OrderedDict[str, Any]):
        inputVariables = []
        for arg_name, input in all_kwargs.items():
            if isinstance(input, TransitionVariable):
                inputVariables.append(input)
        return inputVariables

    def backward(self, state: NodeState, program_feedback: str = None):
        """
        Compute feedback for each Variable and NodeOutput in a backward pass.
        """
        logging.info(f"backwards pass through `{self.name}`")
        node_context = state.context
        node_output = state.output.value

        output_feedback = self._collect_output_feedback(
            state=state, llm=self.optimizer.llm, program_feedback=program_feedback
        )

        # propegate feedback to input variables
        for variable in self.optimizable_variables.values():
            variable.generate_feedback(
                context=node_context,
                output=node_output,
                llm=self.optimizer.llm,
                output_feedback=output_feedback,
            )
        # propegate feedback to input variables
        for variable in state.inputs:
            variable.generate_feedback(
                context=node_context,
                output=node_output,
                llm=self.optimizer.llm,
                output_feedback=output_feedback,
            )

    def _collect_output_feedback(
        self, state: NodeState, llm: LLMCallable, program_feedback: str
    ):
        """Collects feedback from child nodes or the optimizer"""
        if program_feedback:
            return program_feedback
        else:
            feedback = state.output.get_feedback(llm)
            if not feedback:
                raise ValueError(f"No output feedback exists for `{self.name}")
            return feedback

    def set_optimizer(self, optimizer: "Optimizer"):
        """Sets the optimizer for this Node and its components"""
        self.optimizer = optimizer
        optimizer.nodes[self.name] = self


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

    def initialize_nodes(self):
        """Adds Nodes to Optimizer and assigns Optimizer to Nodes"""
        logging.info(f"Attaching nodes to optimizer: {' '.join(node_registry.keys())}")

        for name, node in node_registry.items():
            node.set_optimizer(self)

    def _add_state_to_stack(self, state: NodeState):
        """Add node state to stack"""
        self.execution_stack.append(state)

    def zero_grad(self):
        for node in self.nodes.values():
            for variable in node.optimizable_variables.values():
                variable._feedback.clear()

    def forward(self, program_func: Callable, *args, **kwargs) -> str:
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
            for variable in node.optimizable_variables.values():
                variable.update(llm=self.llm)

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
        self.initialize_nodes()

        for iter in range(iterations):
            logging.info(f"#### Iteration {iter} ####")
            data_list = [data] if isinstance(data, dict) else data
            for item in data_list:
                logging.info(f"## Data item: {item} ##")
                try:
                    program_output: str = self.forward(program_func, **item)
                    logging.info(
                        f"Program output for iteration {iter}: {program_output}"
                    )
                    evaluation_prompt = self._format_evaluation_prompt(
                        evaluation_template, item, program_output
                    )
                    feedback = self.llm.generate_text(prompt=evaluation_prompt)
                    logging.info(f"Feedback for iteration {iter}: {feedback}")
                    self.backward(feedback)
                    self.step()
                    self.zero_grad()
                except Exception as e:
                    logging.error(f"An error occurred: {e}")
                    raise e

        return {name: node.optimizable_variables for name, node in self.nodes.items()}

    def _format_evaluation_prompt(
        self, evaluation_template: str, item: Dict[str, Any], program_output: str
    ):
        try:
            evaluation_prompt = evaluation_template.format(
                program_output=program_output, **item
            )
            return evaluation_prompt
        except Exception as e:
            error = dedent(
                f"""
                Error formatting evaluation prompt
                Check for typos in your dataset and template
                Eval template:{evaluation_template}
                args: 'program_output', {item}
                """
            )
            logging.error(error)
            raise e

    def get_prompt_info(self):
        info = ""
        for node_key, node in self.nodes.items():
            info += f"Node: `{node_key}`\n"
            for variable in node.optimizable_variables.values():
                info += f"  {variable.name}: {variable.value}\n"
        return info
