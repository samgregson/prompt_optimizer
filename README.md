# Prompt Optimizer

Prompt Optimizer is a flexible and extensible framework designed to optimize variables within an LLM program based on feedback. This project is heavily inspired by [TextGrad](https://github.com/zou-group/textgrad) but aims to be more adaptable and easier to integrate into existing projects. It leverages decorators for seamless integration and allows you to bring your own LLM providers. Additionally, it integrates well with libraries like LangSmith for observability and Instructor for structured output.

## Features

- **Flexible Integration**: Easily append to existing projects using decorators.
- **Custom LLM Providers**: Bring your own LLM providers for generating and refining prompts.
- **Observability**: Integrates with libraries like LangSmith for observability and supports structured output using Instructor.
- **Feedback-Based Optimization**: Optimizes variables based on "llm as judge" feedback.

## Usage

### Defining Nodes
Use the `llm_node` decorator to define nodes in your program. Each node can have optimizable variables (prompts) that will be refined based on feedback.

``` Python
from prompt_optimizer import llm_node

@llm_node(context_params=["param1", "param2"], variable1="value1", variable2="value2")
def example_node(param1, param2, variable1, variable2):
    # llm logic here
    return result
```

### Creating the Optimizer
Create an instance of Optimizer and initialize it with your LLM provider.

**Note**: these `LLMCallable`s are only used for feedback generation, the program is completely defined by you!

``` py
# Instantiate an optimizer and pass in an `LLMCallable` with **kwargs as per OpenAI or Anthropic
llm = OpenAIAdapter(client, model="gpt-4o")
optimizer = Optimizer(llm)
```

Or bring your own LLM provider, or any function that returns a string.

``` py
# Define your custom text generation function
def custom_generate_text(prompt: str, model_name: str) -> str:
    # Simulate a text generation process
    return f"Generated response for '{prompt}' using model '{model_name}'"

# Use CustomAdapter with your custom function
custom_llm = CustomAdapter(generate_func=custom_generate_text, model_name="custom-model")
optimizer = Optimizer(custom_llm)
```

### Running the Optimization
Define your program function and run it using the optimizer.
``` py
# Define a template for evaluating the final answer
exact_match_evaluator = dedent(
    """
    Your task is to judge the quality of the response given by an AI assistant.
    Below are the <question>, <assistant_response> and <expected_answer>.
    A correct answer would be an exact string match between the <assistant_response> and the
    <expected_answer>.

    <question>{query}</question>
    <assistant_response>{program_output}</assistant_response>
    <expected_answer>{golden_answer}</expected_answer>
    """
)

# Define your dataset
data = [{
    "query": "what is the capital of France?",
    "golden_answer": "<answer>Paris</answer>"
}]

# Run optimization
optimizer.optimize(
    iterations=1,
    program_func=answer_query,
    evaluation_template=exact_match_evaluator,
    data=data,
)
```
## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.
