from textwrap import dedent


variable_feedback_prompt = dedent(
    """
    - You are tasked with providing detailed feedback for the prompt
    of an LLM call
    - The goal is to refine and improve the prompt to align the generated
    output based on the feedback that is given
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

variable_optimiser_prompt = dedent(
    """
    You are an expert prompt engineer.
    Your task is to improve an LLM prompt based on provided feedback.
    Your goal is to refine the prompt to better align with the feedback,
    ensuring that the improvements are targeted and relevant.

    <original_prompt>
    {variable_value}
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

input_feedback_prompt = dedent(
    """
    The goal of the task is to provide detailed feedback on the input context
    used in an LLM call. The purpose of this feedback is to refine and improve
    the input context so that the generated output aligns better with the
    given feedback.

    Here's a breakdown of what needs to be done:

    1. **Analyze the Context**: Look at the provided context, the generated
    output, the current input, and the feedback on the output.
    2. **Provide Comprehensive Feedback**: Offer detailed feedback that
    directly addresses the output feedback, considering all the provided
    information.
    3. **Suggest Improvements**: Include specific suggestions on how to improve
    the input context to better align the generated output with the feedback.
    4. **Use Specific Tags: Ensure the feedback is enclosed within <Feedback>
    </Feedback> tags.
    5. **Avoid New Examples**: Do not provide new examples of the input at this
    stage; focus solely on giving feedback.

    The ultimate aim is to help refine the input context to achieve a more
    desirable output from the LLM based on the feedback provided.

    <context>{context}</context>
    <generated_output>{output}</generated_output>
    <current_input>{current_input}</current_input>
    <output_feedback>{output_feedback}</output_feedback>
    """
)
