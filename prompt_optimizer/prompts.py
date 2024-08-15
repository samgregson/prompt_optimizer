from textwrap import dedent


input_feedback_prompt = dedent(
    """
    - You are tasked with providing detailed feedback for the prompt
    of an LLM call
    - The goal is to refine and improve the prompt to align the
    generated output based on the feedback that is given
    - Your feedback should be comprehensive and directly address the output
    feedback given, considering the context, generated output and the specific
    prompt.
    - The provided output is not necessarily the desired output; it is merely
    the output that was generated
    - include specific suggestions for improvement to better align the output
    to the ouput feedback
    - provide your feedback response in <Feedback></Feedback> tags

    *Important*: At this stage, you should refrain from providing new examples
    of the input. Your task is solely to provide feedback.

    <context>{context}</context>
    <generated_output>{output}</generated_output>
    <current_prompt>{current_prompt}</current_prompt>
    <output_feedback>{output_feedback}</output_feedback>
    """
)

final_output_evaluation_prompt = dedent(
    """
    You are an evaluator tasked with assessing the quality of a generated
    Grasshopper script. Most importantly the script should acheive the task
    as given by the task description. A golden example is provided for
    reference as to what an ideal solution could look like.
    *Note*: The output may be valid even if it is not an exact match
    to the golden example.Your assessment should focus on the quality and
    validity of the script within the Grasshopper environment.

    ## Parameters that will be provided:
    - <golden_example>, provided for reference
    - generated <output>, to be evaluated
    - <task_description>

    Provide a detailed assessment of the generated output, highlighting
    any potential issues, and overall quality and validity.

    Your evaluation should cover the following criteria:

    ### Accuracy:
    - The output should correctly implement the logic and functionality as
    intended.
    - It should produce the expected results when compared against the golden
    example.

    ### Completeness:
    - The output should cover all required aspects and components as specified
    in the task.
    - No essential parts should be missing.

    ### Validity:
    - It should not produce any errors or warnings when executed.

    ### Task Description Alignment:
    - The output should address all the objectives outlined in the task
    description.
    - It should cover all inputs or conditions specified by the task
    description.

    Evaluate the quality and validity of the <output>, against the description
    of the <task_description>. A golden example has been provided given for
    reference as a potential valid solution.

    <task_description>
    {description}
    </task_description>

    <output>
    {output}
    </output>

    <golden_example>
    {golden_example}
    </golden_example>
    """
)

prompt_optimiser_prompt = dedent(
    """
    You are an expert in optimizing technical documentation.
    Your task is to improve a prompt based on provided feedback.
    Your goal is to refine the prompt to better align with the feedback,
    ensuring that the improvements are targeted and relevant.

    <original_prompt>
    {component_value}
    </original_prompt>

    <feedback>
    {feedback}
    </feedback>

    # Task
    1. analyze the feedback and context
    2. identify areas for improvement in the prompt
    3. provide specific, actionable improvements
    4. ensure that the improved prompt is returned in full with the
    suggested changes within <improved_prompt></improved_prompt>
    tags.
    5. ensure that the improvements directly address the feedback provided.
    6. ensure that the prompt remains generally useful for different queries
    """
)
