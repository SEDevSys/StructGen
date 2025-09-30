import copy
import threading
from functools import wraps

from utils import *


def test_generated_test_cases(test_cases_input, code_str):
    if isinstance(test_cases_input, str):
        try:
            test_cases = ast.literal_eval(test_cases_input)
            if not isinstance(test_cases, list):
                raise ValueError("test_cases_input The string must represent a list.")
        except Exception as e:
            return "failed"
    elif isinstance(test_cases_input, list):
        test_cases = test_cases_input
    else:
        return "failed"

    failed_tests = []
    namespace = {}

    try:
        exec(code_str, namespace)
    except Exception as e:
        return f"failed: An error occurred while executing code_str: {e}"

    for i in range(len(test_cases) - 1, -1, -1):
        test_case = test_cases[i].strip()
        if "assert" in test_case:
            last_assert_index = test_case.rfind("assert")
            test_case = test_case[last_assert_index + len("assert") :].strip()
            test_cases[i] = test_case

        full_statement = f"{code_str}\nassert {test_case}"

        try:
            # TODO Multi-threading has default timeout setting
            with swallow_io():
                with time_limit(float(3.0)):
                    exec(full_statement, namespace)
        except TimeoutException:
            failed_tests.append(f"Test {test_case} Failed: timed out")
        except BaseException as e:
            failed_tests.append(f"Test {test_case} Failed: {e}")

    return "passed" if not failed_tests else failed_tests


def get_configuration_parameters(code_model):
    config = get_config()

    model_name = code_model
    sanitized = ""  # _sanitized
    uml_type = config.get("UML", "uml_type")
    dataset_name = config.get("datasets", "dataset_name")
    output_file = config.get("basic", "output_file_jsonl").format(
        dataset_name=dataset_name,
        model_name=model_name,
        uml_type=uml_type,
        design_repair_num=config.get("feedback", "max_plantUML_attempts"),
        code_repair_num=int(config.get("feedback", "max_function_code_attempts")) - 1,
    )
    result_file_path = config.get("UML", "uml_csv_file").format(
        uml_type=uml_type, _sanitized=sanitized, dataset_name=dataset_name
    )
    return config, model_name, dataset_name, output_file, result_file_path


def generate_uml(client, problem, model_name, config, additional_text=False):
    plantUML_attempts = 0
    uml_type = config.get("UML", "uml_type")
    max_plantuml_attempts = config.getint("feedback", "max_plantUML_attempts")

    while plantUML_attempts < max_plantuml_attempts:
        plantUML_attempts += 1
        error_additional_text = (
            problem["error_additional_text"]
            if additional_text and problem["error_additional_text"]
            else ""
        )
        result_uml = generate_completion(
            client,
            problem,
            model_name,
            generate_type="generate_uml",
            additional_text=error_additional_text,
        )
        result_uml, extract_uml_status = extract_uml(result_uml)
        if extract_uml_status:
            problem[uml_type] = result_uml
            break
        else:
            problem[uml_type] = (
                problem[uml_type] if additional_text else f"extract {uml_type} failed"
            )


def generate_and_validate_function_code(
    client, problem, model_name, config, additional_text=False
):
    function_code_attempts = 0
    prompt_language = (
        problem["task_id"].split("/")[0].lower()
        if "/" in problem["task_id"] and not problem["task_id"].startswith("HumanEval")
        else "python"
    )
    max_function_code_attempts = config.getint("feedback", "max_function_code_attempts")
    if "error_additional_text" not in problem:
        problem["error_additional_text"] = None
    if "passed" not in problem:
        problem["passed"] = "False"
    # TODO Actually only one code repair iteration, because first is generation, second is repair, and third exits the loop
    while function_code_attempts < max_function_code_attempts:
        function_code_attempts += 1
        error_additional_text = (
            problem["error_additional_text"]
            if additional_text and problem["error_additional_text"]
            else ""
        )
        result_function_code = generate_completion(
            client,
            problem,
            model_name,
            generate_type="generate_function_code",
            additional_text=error_additional_text,
        )
        result_function_code, extract_function_code_status = extract_function(
            problem, result_function_code, prompt_language
        )
        # Test cases example
        # ["has_close_elements([1.0, 2.0, 3.0], 0.5) == False"]
        test_cases_copy = copy.deepcopy(problem["generate_test_cases"])
        test_result = test_generated_test_cases(test_cases_copy, result_function_code)
        problem["completion"] = result_function_code

        if test_result == "passed":
            problem["passed"] = "True"
            break
        else:
            problem["passed"] = "False"


def filter_problems(problems, config, condition_key, condition_value):
    max_task_id = config.getint("test", "temp_num")
    filtered = {
        task_id: details
        for task_id, details in problems.items()
        if (
            condition_key not in details  # key does not exist
            or not details.get(condition_key)  # key exists but value is empty
            or "".join(details.get(condition_key)).strip() == condition_value
        )
        and int(re.search(r"\d+", str(task_id)).group()) < max_task_id
    }
    return filtered


def process_single_thread(
    error_problems,
    handler_func,
    client,
    designer_model,
    config,
    additional_text,
    result_file_path,
    problems,
    condition_key,
):
    total_tasks = len(error_problems)
    processed_tasks = 0
    save_threshold = total_tasks // 5  # Save every 20% of tasks

    for task_id in tqdm(
        error_problems, desc=f"Processing {condition_key} related errors"
    ):
        try:
            tqdm.write("Processing task_id: {}".format(task_id))
            handler_func(
                client, error_problems[task_id], designer_model, config, additional_text
            )
            processed_tasks += 1

            if processed_tasks >= save_threshold:
                save_results(
                    merge_dict_values(problems, error_problems), result_file_path
                )
                processed_tasks = 0
        except Exception as e:
            print(f"Error occurred while processing task {task_id}: {str(e)}")
    final_result = merge_dict_values(problems, error_problems)
    problems = final_result
    save_results(problems, result_file_path)


def timeout_decorator(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutError("Function call timed out")]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                raise TimeoutError(
                    f"Function {func.__name__} timed out after {seconds} seconds"
                )
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]

        return wrapper

    return decorator


def process_tasks_with_fallback(
    error_problems,
    handler_func,
    client,
    designer_model,
    config,
    additional_text,
    result_file_path,
    problems,
    condition_key,
):
    process_single_thread(
        error_problems,
        handler_func,
        client,
        designer_model,
        config,
        additional_text,
        result_file_path,
        problems,
        condition_key,
    )


def handle_error_cases(
    client: OpenAI,
    problems: Dict,
    condition_key: str,
    condition_value,
    handler_func,
    designer_model: str,
    config: configparser.ConfigParser,
    additional_text=False,
    rerun=False,
) -> Dict:
    config, model_name, dataset_name, output_file, result_file_path = (
        get_configuration_parameters(designer_model)
    )

    if condition_key not in next(iter(problems.values()), {}):
        rerun = True

    error_problems = (
        problems
        if rerun
        else filter_problems(problems, config, condition_key, condition_value)
    )

    process_tasks_with_fallback(
        error_problems,
        handler_func,
        client,
        designer_model,
        config,
        additional_text,
        result_file_path,
        problems,
        condition_key,
    )

    return problems


def process_eval(problems, code_model, language):
    config, model_name, dataset_name, output_file, result_file_path = (
        get_configuration_parameters(code_model)
    )
    uml_type = config.get("UML", "uml_type")
    client = get_client(config)
    designer_model = config.get("feedback", "designer_model")

    while (
        len(filter_problems(problems, config, uml_type, f"extract {uml_type} failed"))
        > 1
    ):
        problems = handle_error_cases(
            client,
            problems,
            uml_type,
            f"extract {uml_type} failed",
            generate_uml,
            code_model,
            config,
        )

    feedback_uml_attempts = 0
    max_feedback_uml_attempts = config.getint("feedback", "max_plantUML_attempts")
    while feedback_uml_attempts <= max_feedback_uml_attempts:
        print("Generating and testing code...")
        problems = handle_error_cases(
            client,
            problems,
            "passed",
            "False",
            generate_and_validate_function_code,
            code_model,
            config,
            additional_text=True,
        )
        # Save results
        sample = convert_data_format(problems, uml_type)
        write_jsonl(output_file, sample)

        error_num = 0
        for task_id, item_dict in problems.items():
            if item_dict["passed"] == "False":
                error_num += 1

        error_num = 0
        for task_id, item_dict in problems.items():
            if item_dict["passed"] == "False":
                error_num += 1
        print("Number of errors after testing:", error_num)

        if feedback_uml_attempts == max_feedback_uml_attempts:
            save_results(problems, result_file_path)
            break

        print("Starting plantUML repair...")
        problems = handle_error_cases(
            client,
            problems,
            "passed",
            "False",
            generate_uml,
            code_model,
            config,
            additional_text=True,
        )

        feedback_uml_attempts += 1

    print("Successfully save the result to file")


def extract_function_function_name_params(code):
    match = re.search(r"def\s+(\w+)\s*\((.*?)\)\s*:", code)
    if match:
        func_name = match.group(1)
        func_params = match.group(2)
        return func_name, func_params
    else:
        return None, None


def main():
    language = "python"
    CODE_MODEL_LIST = [
        # "deepseek-chat",
        # "gpt-3.5-turbo-1106",
        # "gpt-3.5-turbo-16k",
        # "gpt-4-turbo-1106",
        "deepseek-coder-6.7b-instruct",
        # "Mistral-7B-Instruct-v0.3",
        # "CodeLlama-7b-Instruct-hf",
        # "gemma-2-9b-it",
        # "starcoder2-7b",
        # "Qwen2.5-Coder-7B-Instruct",
        # "CodeLlama-13b-Instruct-hf",
    ]
    for code_model in CODE_MODEL_LIST:
        config, model_name, dataset_name, output_file, result_file_path = (
            get_configuration_parameters(code_model)
        )
        result_dict = load_result_csv_2_dict(result_file_path)

        if config.get("datasets", "dataset_name") == "mbpp":
            if os.path.exists(result_file_path):
                problems = convert_mbpp_2_humaneval_dict(
                    "./data/mbpp/mbpp.jsonl",
                    result_file_path,
                )
            else:
                problems = transform_json_file_to_dict("./data/mbpp/mbpp.jsonl")

        elif config.get("datasets", "dataset_name") == "humaneval":
            for task_id, item_dict in result_dict.items():
                if (
                    "generate_test_cases" in item_dict
                    and item_dict["generate_test_cases"]
                ):
                    item_dict["generate_test_cases"] = json.loads(
                        item_dict["generate_test_cases"]
                    )
            dataset_dict = transform_json_file_to_dict(
                "./data/humaneval/HumanEval.jsonl"
            )
            problems = merge_dict_values(dataset_dict, result_dict)
        else:
            dataset_dict = transform_json_file_to_dict(
                f"./data/humaneval-x/humaneval_{language}.jsonl"
            )
            problems = merge_dict_values(result_dict, dataset_dict)

        for task_id, item_dict in problems.items():
            if config.get("datasets", "dataset_name") != "humaneval-x":
                ensure_def_in_canonical_solution(item_dict)
            else:
                if language == "js":
                    language_temp = "javascript"
                else:
                    language_temp = language
                problems = {
                    task_id: item_dict
                    for task_id, item_dict in problems.items()
                    if task_id.split("/")[0].lower() == language_temp
                }

            if config.get("datasets", "dataset_name") == "mbpp":
                if "canonical_solution" not in item_dict:
                    item_dict["canonical_solution"] = item_dict["code"]
                    del item_dict["code"]
                if "prompt" not in item_dict:
                    item_dict["prompt"] = item_dict["text"]
                    del item_dict["text"]
                item_dict["task_id"] = str(task_id)
                function_name, paras = extract_function_function_name_params(
                    item_dict["canonical_solution"]
                )
                item_dict["entry_point"] = function_name

            if "test_setup_code" in item_dict:
                del item_dict["test_setup_code"]
            if "test_list" in item_dict:
                del item_dict["test_list"]
            if "challenge_test_list" in item_dict:
                del item_dict["challenge_test_list"]

        print(
            f"\n===============================\n"
            f"designer_model:{config.get('feedback', 'designer_model')}\n"
            f"coder_model:{code_model}\n"
            f"output_file:{output_file}\n"
            f"==============================="
        )

        process_eval(problems, code_model, language)


if __name__ == "__main__":
    main()
    print("Remember to download the obtained plantuml_mbpp locally")
