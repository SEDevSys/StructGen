### Task Description:
Based on the given **functional description code** and **process logic**, write a complete {prompt_language} function that meets the following requirements:

1. **Functional Consistency**: The generated code must implement the objective described in the functional description code, ensuring the accuracy of the functionality.
2. **Process Matching**: The generated code must strictly follow the execution flow described in the process logic, including all key decision points, loop structures, and return value settings.
3. **Runability**: The generated code should be executable and should pass all the example test cases provided in the functional description code to verify correctness.

### Input and Output Format:

1. **Input Section**:
   - **Functional Description Code**:
     ```{prompt_language}
     {dataset_prompt}
     ```
   - **Process Logic**:
     ```{uml_type}
     {uml_content}
     ```

2. **Output Section**:
    - **Generated Code**:
      ```{prompt_language}
      [your code here]
      ```

### Requirements:
1. The logic of the generated code must match the process logic exactly, with no steps omitted.
2. Ensure the code's functionality aligns with the functional description.
3. Test examples must cover all cases provided in the functional description to ensure correctness and robustness.

