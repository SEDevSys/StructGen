import ast
import cmath
import configparser
import contextlib
import csv
import gzip
import heapq as hq
import io
import json
import math
import os
import re
import signal
import sys
from collections import Counter, defaultdict, ChainMap as ct
from itertools import tee
from math import tan, pi
from typing import Dict
from typing import Iterable, Any

import regex
from openai import OpenAI


def get_config():
    config = configparser.ConfigParser()
    config.read("./generate/config.ini")
    return config


def get_client(config):
    base_url = config.get("ollama", "base_url")
    api_key = config.get("ollama", "api_key")
    return OpenAI(base_url=base_url, api_key=api_key)


def read_md_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            result = file.read()
        return result
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode("utf-8"))


def read_csv_to_dict(file_path):
    csv.field_size_limit(500 * 1024 * 1024)
    with open(file_path, mode="r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        headers = reader.fieldnames

        if not headers:
            raise ValueError("CSV file is empty or has no headers.")

        if len(headers) == 1:
            return {row[headers[0]] for row in reader}

        elif len(headers) == 2:
            return {row[headers[0]]: row[headers[1]] for row in reader}

        else:
            return {
                row[headers[0]]: {key: row[key] for key in headers[1:]}
                for row in reader
            }


def read_json_to_dict(file_path):
    with open(file_path, "r") as f:
        json_data = json.load(f)

    return json_data


def transform_mbpp_2_humaneval(obj1):
    import re

    def extract_function_signature(code):
        match = re.search(r"def\s+(\w+)\s*\((.*?)\)\s*:", code)
        if match:
            func_name = match.group(1)
            func_params = match.group(2)
            return func_name, func_params
        else:
            return None, None

    def build_test_function(test_list, entry_point):
        test_code = f"def check(candidate):\n\n"
        for test in test_list:
            test_replaced = test.replace(entry_point, "candidate")
            test_code += f"    {test_replaced}\n"
        return test_code

    def format_prompt(text, test_list, entry_point, func_params):
        prompt = f'def {entry_point}({func_params}):\n"""\n{text}\n'
        if test_list[0] != "test test cases failed" and len(test_list[0]) < 100:
            prompt += "Your code should satisfy these tests:\n"
            for test in test_list:
                prompt += (
                    f"{test.replace(entry_point, entry_point).replace('assert', '')}\n"
                )
        prompt += '"""\n'
        return prompt

    func_name, func_params = extract_function_signature(obj1["code"])
    if not func_name or not func_params:
        raise ValueError(
            "Cannot extract function definition from code, please check input format."
        )

    obj2 = {
        "task_id": str(obj1["task_id"]),
        "prompt": format_prompt(
            obj1["text"], obj1["public_test"], func_name, func_params
        ),
        "canonical_solution": obj1["code"],
        "entry_point": func_name,
        "test": build_test_function(obj1["test_list"], func_name),
    }

    return obj2


def merge_dict_values(dict1, dict2):
    merged_dict = {}

    all_keys = set(dict1.keys()).union(set(dict2.keys()))

    for key in all_keys:
        merged_value = {}
        if key in dict1:
            merged_value.update(dict1[key])
        if key in dict2:
            merged_value.update(dict2[key])

        merged_dict[key] = merged_value

    return merged_dict


def load_result_csv_2_dict(result_path):
    if not os.path.exists(result_path):
        return defaultdict(dict)
    else:
        return read_csv_to_dict(result_path)


def transform_json_file_to_dict(file_path):
    result = {}
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if "task_id" in obj:
                        result[obj["task_id"]] = obj
                    else:
                        print("Skipping object without 'task_id':", obj)
        return result
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON format: {e}")
        return None


def convert_mbpp_2_humaneval_dict(file_path, result_file_path):
    result_dict = load_result_csv_2_dict(result_file_path)
    mbpp_data = transform_json_file_to_dict(file_path)

    problems = {}
    for task_id, item_dict in mbpp_data.items():
        task_id = str(task_id)
        item_dict["task_id"] = task_id
        if result_dict[task_id]["generate_test_cases"]:
            test_cases_list = json.loads(result_dict[task_id]["generate_test_cases"])
        else:
            test_cases_list = ["test test cases failed"]
        item_dict["public_test"] = test_cases_list
        problems[task_id] = transform_mbpp_2_humaneval(item_dict)
        result_dict[task_id]["generate_test_cases"] = test_cases_list
    problems = merge_dict_values(result_dict, problems)
    return problems


def dict_to_csv(dict_list: dict, csv_output_path: str):
    data_rows = []
    for task_id, content in dict_list.items():
        row = {"task_id": task_id}
        row.update(content)
        if "generate_test_cases" in content:
            row["generate_test_cases"] = json.dumps(content["generate_test_cases"])
        data_rows.append(row)

    headers = ["task_id"]
    for row in data_rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)

    with open(csv_output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        for row in data_rows:
            row_with_defaults = {header: row.get(header, "") for header in headers}
            writer.writerow(row_with_defaults)


def construct_prompt(user_prompt, prompt_language):
    messages = [
        {
            "role": "system",
            "content": f"You are a professional {prompt_language} programmer.",
        },
        {"role": "user", "content": user_prompt},
    ]
    return messages


def generate_completion(client, problem, model_name, generate_type, additional_text=""):
    config = get_config()
    uml_type = config.get("UML", "uml_type")
    prompt_file_path = config.get("prompt", generate_type).format(uml_type=uml_type)
    prompt_language = (
        problem["task_id"].split("/")[0].lower()
        if "/" in problem["task_id"] and not problem["task_id"].startswith("HumanEval")
        else "python"
    )

    if generate_type == "generate_test_cases":
        user_prompt = read_md_file(prompt_file_path).format(
            dataset_solution=problem["canonical_solution"],
            function_name=problem["entry_point"],
        )
    elif generate_type == "generate_uml":
        user_prompt = read_md_file(prompt_file_path).format(
            dataset_prompt=problem["prompt"],
            prompt_language=prompt_language,
        )
    elif generate_type == "generate_function_code":
        user_prompt = read_md_file(prompt_file_path).format(
            dataset_prompt=problem["prompt"],
            uml_type="dot" if uml_type == "graphviz" else uml_type,
            uml_content=problem[uml_type],
            prompt_language=prompt_language,
        )
    elif generate_type == "generate_function_code_archcode":
        archcode_requirements = read_json_to_dict(
            "/home/wsy/code_generation/human-eval/data/humaneval/ARCHCODE_requirements.json"
        )
        user_prompt = read_md_file(prompt_file_path).format(
            dataset_prompt=problem["prompt"],
            uml_type="dot" if uml_type == "graphviz" else uml_type,
            uml_content=problem[uml_type],
            ArchCode_Requirments=archcode_requirements[problem["task_id"]],
        )
    else:
        raise ValueError(
            "Invalid generate_type. Must be 'generate_function' or 'generate_plantUML'."
        )

    user_prompt += f"\n{additional_text}"
    prompt = construct_prompt(user_prompt, prompt_language)
    result = generate_one_completion(client, prompt, config, model_name)

    return result


def generate_one_completion(client, messages, config, model_name):

    if model_name == "gemma-2-9b-it":
        messages = [
            {
                "role": "user",
                "content": "You are a professional Python programmer. \n"
                + messages[1]["content"],
            }
        ]

    model_mapping = {
        "Qwen2.5-Coder-7B-Instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "CodeLlama-13b-Instruct-hf": "codellama/CodeLlama-13b-Instruct-hf",
        "deepseek-coder-6.7b-instruct": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "Mistral-7B-Instruct-v0.3": "LLM-Research/Mistral-7B-Instruct-v0.3",
        "CodeLlama-7b-Instruct-hf": "codellama/CodeLlama-7b-Instruct-hf",
        "gemma-2-9b-it": "LLM-Research/gemma-2-9b-it",
        "starcoder2-7b": "bigcode/starcoder2-7b",
        "deepseek-chat": "deepseek-chat",
    }

    model_name = model_mapping.get(model_name, model_name)

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=config.getfloat("eval", "temperature"),
            top_p=config.getfloat("eval", "top_p"),
            max_tokens=config.getint("eval", "max_tokens"),
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating completion: {e}")
        return f"Error generating completion: {e}"


def extract_function_with_python(text):
    lines = text.splitlines()
    result = []
    inside_function = False

    for line in lines:
        if line.strip().startswith("import") or line.strip().startswith("from"):
            result.append(line)
        elif line.strip().startswith("def "):
            result.append(line)
            inside_function = True
        elif inside_function:
            if line.strip() == "" or line.startswith(" "):
                result.append(line)
            else:
                break

    return "\n".join(result)


def extract_function_with_java(text):
    lines = text.splitlines()
    result = []
    inside_function = False
    brace_count = 0

    for line in lines:
        stripped_line = line.strip()

        if (
            stripped_line.startswith("import")
            or stripped_line.startswith("from")
            or stripped_line.startswith("#include")
            or stripped_line.startswith("using")
        ):
            result.append(line)
            continue

        if (
            stripped_line.startswith("def ")
            or stripped_line.startswith("function ")
            or "function" in stripped_line
            or stripped_line.startswith("const ")
            or stripped_line.startswith("let ")
            or stripped_line.startswith("var ")
            or (
                (
                    "public " in stripped_line
                    or "private " in stripped_line
                    or "protected " in stripped_line
                    or stripped_line.startswith("class ")
                )
                and "(" in stripped_line
            )
        ):
            result.append(line)
            inside_function = True
            brace_count += stripped_line.count("{")
            brace_count -= stripped_line.count("}")
            continue

        if inside_function:
            result.append(line)
            brace_count += stripped_line.count("{")
            brace_count -= stripped_line.count("}")

            if brace_count == 0:
                if stripped_line.endswith("}") or not line.startswith(" "):
                    inside_function = False

            if (
                "console.log(" in stripped_line
                or "System.out.println" in stripped_line
                or "cout <<" in stripped_line
                or stripped_line.startswith("print(")
            ):
                break

    while result and not result[-1].strip():
        result.pop()

    return "\n".join(result)


def extract_function_with_cpp(text):
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    if not text or text.isspace():
        return ""

    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    text = text.replace("\\n", "\n")
    text = text.replace('\\"', '"')
    text = text.replace('""', '"')

    lines = text.splitlines()
    result = []
    inside_function = False
    brace_count = 0
    current_includes = []
    current_function = []

    cpp_types = [
        "bool",
        "void",
        "int",
        "float",
        "double",
        "string",
        "char",
        "vector<",
        "auto",
        "long",
        "unsigned",
        "short",
        "const",
        "static",
        "class",
        "struct",
        "template",
        "std::",
        "boost::",
    ]

    for line in lines:
        stripped_line = line.strip()

        if not stripped_line:
            if inside_function:
                current_function.append(line)
            elif current_includes:
                current_includes.append(line)
            continue

        if (
            "main(" in stripped_line
            or "cout" in stripped_line
            or "test" in stripped_line.lower()
            or "assert" in stripped_line
        ):
            continue

        if stripped_line.startswith("#include") or stripped_line.startswith("using"):
            current_includes.append(line)
            continue

        is_function_start = False
        if not inside_function:
            words = stripped_line.split()
            if words:
                is_function_start = (
                    any(
                        first_word.startswith(type_name)
                        for type_name in cpp_types
                        for first_word in words[:2]
                    )
                    and "(" in stripped_line
                    and ";" not in stripped_line
                    and "{" in stripped_line
                )

        if is_function_start:
            if current_includes:
                result.extend(current_includes)
                result.append("")
                current_includes = []

            current_function.append(line)
            inside_function = True
            brace_count = stripped_line.count("{")
            brace_count -= stripped_line.count("}")
            continue

        if inside_function:
            current_function.append(line)
            brace_count += stripped_line.count("{")
            brace_count -= stripped_line.count("}")

            if brace_count == 0:
                inside_function = False
                result.extend(current_function)
                current_function = []
                result.append("")

    if current_function:
        result.extend(current_function)

    while result and not result[-1].strip():
        result.pop()

    return "\n".join(result)


def add_missing_imports(code: str) -> str:
    required_imports = ["import math", "from typing import List"]

    lines = code.splitlines()

    existing_imports = {
        line.strip()
        for line in lines
        if line.strip().startswith("import") or line.strip().startswith("from")
    }

    missing_imports = [imp for imp in required_imports if imp not in existing_imports]

    if missing_imports:
        insertion_index = 0
        for i, line in enumerate(lines):
            if line.strip():
                if (
                    line.startswith("#!")
                    or line.startswith('"""')
                    or line.startswith("'''")
                ):
                    continue
                insertion_index = i
                break

        updated_code = (
            lines[:insertion_index] + missing_imports + [""] + lines[insertion_index:]
        )
        return "\n".join(updated_code)

    return code


def extract_generated_test_cases(big_text):

    block_pattern = re.compile(
        r"<test_cases>(.*?)</test_cases>", re.IGNORECASE | re.DOTALL
    )
    blocks = block_pattern.findall(big_text)

    if not blocks:
        return ["extract test cases failed"]

    all_extracted = []

    single_test_case_pattern = re.compile(
        r"^\s*(?:assert\s+)?"
        r"(?P<func>[A-Za-z_]\w*)"
        r"\s*\(\s*(?P<args>.*?)\)\s*"
        r"==\s*(?P<output>.*)\s*$",
        re.MULTILINE,
    )

    for block_content in blocks:
        lines = block_content.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = single_test_case_pattern.match(line)
            if match:
                func_name = match.group("func")
                args_str = match.group("args").strip()
                output_str = match.group("output").strip()

                test_case_str = f"{func_name}({args_str}) == {output_str}"
                all_extracted.append(test_case_str)

    if not all_extracted:
        return ["extract test cases failed"]

    return all_extracted


def extract_function_code(problem: dict, text: str, prompt_language: str) -> str:
    function_pattern = rf"```{prompt_language}\s+([\s\S]*?)```"

    code_blocks = re.findall(function_pattern, text, re.MULTILINE)
    problem["response"] = code_blocks

    if not code_blocks:
        return "# extract function code failed"

    function_map = {
        "java": extract_function_with_java,
        "javascript": extract_function_with_java,
        "cpp": extract_function_with_cpp,
        "python": extract_function_with_python,
    }

    function_code = function_map[prompt_language](code_blocks[0])
    if not function_code:
        return "# extract function code failed"

    return function_code


def extract_function(problem, completion, prompt_language):

    extracted_function_code = extract_function_code(
        problem, completion, prompt_language
    )

    if extracted_function_code == "# extract function code failed":
        print("extract function code failed")
        return extracted_function_code, False

    return extracted_function_code, True


def extract_uml(response: str):
    config = get_config()
    uml_type = config.get("UML", "uml_type")
    if uml_type == "graphviz":
        uml_type_extract = "dot"
    else:
        uml_type_extract = uml_type
    if not isinstance(response, str):
        return "Input must be a string", False

    try:
        matches = re.findall(rf"```{uml_type_extract}(.*?)```", response, re.DOTALL)
        if matches:
            return matches[0].strip(), True
        else:
            return f"extract {uml_type} failed", False
    except Exception as e:
        return f"Error during extraction: {str(e)}", False


def rename_2_entry_point(code: str, new_name: str) -> str:
    lines = code.strip().split("\n")

    functions = []
    current_func = None

    for line in lines:
        line = line.strip()
        if line.startswith("def ") and line.endswith(":"):
            func_name = line.split("(")[0][4:]
            if current_func:
                functions.append(current_func)
            current_func = {"name": func_name, "body": [], "start_line": line}
        elif current_func is not None:
            current_func["body"].append(line)

    if current_func:
        functions.append(current_func)

    if not functions:
        print(f"No functions found in the provided code. new_name is {new_name}")
        return code

    function_names = {func["name"] for func in functions}
    for func in functions:
        if not any(
            other_func in "\n".join(func["body"]) for other_func in function_names
        ):
            renamed_code = code.replace(f"def {func['name']}(", f"def {new_name}(")
            return renamed_code

    largest_func = max(functions, key=lambda func: len(func["body"]))
    renamed_code = code.replace(f"def {largest_func['name']}(", f"def {new_name}(")
    return renamed_code


def log_tqdm_progress_to_file(tqdm_bar, file_path="progress_log.txt"):
    with open(file_path, "a") as file:
        file.write(f"Progress: {tqdm_bar.n}/{tqdm_bar.total}\n")


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):
    _stream = "stdin"


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def extract_testcases_from_prompt(prompt):
    patterns = [
        re.compile(r"(?P<func_call>\w+\([^)]*\))\s*->\s*(?P<expected_output>[^.\n]+)"),
        re.compile(
            r">>>\s*(?P<func_call>\w+\([^)]*\))\s*\n\s*(?P<expected_output>[^.\n]+)"
        ),
    ]

    all_matches = []

    for pattern in patterns:
        for match in pattern.finditer(prompt):
            func_call = match.group("func_call").strip()
            expected_out = match.group("expected_output").strip()
            full_text = match.group(0).strip()

            if full_text.startswith("def ") or full_text.endswith(":"):
                continue

            all_matches.append((func_call, expected_out))

    unique_pairs = list(dict.fromkeys(all_matches))

    results = [f"{fc} == {eo}" for (fc, eo) in unique_pairs]
    return results


def ensure_def_in_canonical_solution(problem: dict):

    if "canonical_solution" not in problem:
        return problem

    code = problem["canonical_solution"]
    if f"def {problem['entry_point']}" not in code:
        new_code = f"{problem['prompt']}\n{code}"
        problem["canonical_solution"] = new_code


def get_return_type(function_str: str) -> type:
    try:
        tree = ast.parse(function_str)
        return_nodes = []

        class ReturnVisitor(ast.NodeVisitor):
            def visit_Return(self, node):
                if node.value is not None:
                    return_nodes.append(node.value)

        ReturnVisitor().visit(tree)

        if return_nodes:
            last_return = return_nodes[-1]

            if isinstance(last_return, ast.Str):
                return str
            elif isinstance(last_return, (ast.Num, ast.Constant)):
                return (
                    type(last_return.n)
                    if hasattr(last_return, "n")
                    else type(last_return.value)
                )
            elif isinstance(last_return, ast.List):
                return list
            elif isinstance(last_return, ast.Dict):
                return dict
            elif isinstance(last_return, ast.Set):
                return set
            elif isinstance(last_return, ast.Tuple):
                return tuple
            elif isinstance(last_return, ast.NameConstant):
                if last_return.value in (True, False):
                    return bool
                elif last_return.value is None:
                    return type(None)
            elif isinstance(
                last_return, (ast.BinOp, ast.JoinedStr, ast.FormattedValue)
            ):
                return str

        return Any

    except Exception as e:
        print(f"Error analyzing return type: {e}")
        return Any


def format_value(value: Any, return_type: type = None) -> str:
    try:
        evaluated_value = safe_eval(value)

        if return_type is not None:
            if isinstance(evaluated_value, return_type):
                value = evaluated_value
            else:
                value = convert_to_type(evaluated_value, return_type)
        else:
            value = evaluated_value

        if isinstance(value, str):
            if not is_numeric_string(value):
                return f"'{value}'"
            return value
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, type(None)):
            return "None"
        elif isinstance(value, (list, tuple, set, dict)):
            return str(value)
        else:
            return f"'{str(value)}'"

    except Exception as e:
        print(f"Error in format_value: {str(e)}")
        try:
            return str(value)
        except:
            return ""


def create_global_scope():
    global_scope = {
        "re": re,
        "sys": sys,
        "io": io,
        "math": math,
        "cmath": cmath,
        "pi": pi,
        "tan": tan,
        "heapq": hq,
        "hq": hq,
        "Counter": Counter,
        "defaultdict": defaultdict,
        "ChainMap": ct,
        "ct": ct,
        "regex": regex,
        "tee": tee,
    }

    global_scope.update(
        {
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "all": all,
            "any": any,
            "round": round,
        }
    )

    return global_scope


def update_generate_test_cases(item_dict, function_str, function_name):
    global_scope = create_global_scope()
    local_scope = {}

    try:
        return_type = get_return_type(function_str)
        exec(function_str, global_scope, local_scope)
        target_function = local_scope[function_name]
        global_scope[function_name] = target_function

    except Exception as e:
        return [f"Error during function definition or loading: {str(e)}"]

    updated_test_cases = []
    test_cases = item_dict if isinstance(item_dict, list) else []

    for test_case in test_cases:
        if "assert" in test_case:
            last_assert_index = test_case.rfind("assert")
            test_case = test_case[last_assert_index + len("assert") :].strip()

        if " == " not in test_case:
            continue

        left_part, expected_output = test_case.split(" == ", 1)
        left_part = left_part.strip()

        try:
            stdout_backup = sys.stdout
            sys.stdout = io.StringIO()

            exec(f"print({left_part})", global_scope)
            actual_output = sys.stdout.getvalue().strip()

            sys.stdout = stdout_backup

            formatted_output = format_value(actual_output, return_type)
            new_test_case = f"{left_part} == {formatted_output}"
            updated_test_cases.append(new_test_case)

        except Exception as e:
            sys.stdout = stdout_backup
            print(f"Error processing test case '{test_case}': {e}")
            updated_test_cases.append(test_case)

    return updated_test_cases if updated_test_cases else ["No valid test cases found"]


def save_results(problems, result_file_path):
    dict_to_csv(problems, result_file_path)
    print("Results have been saved to file\n")


def convert_data_format(data_list, uml_type):
    result = []
    for task_id, data in data_list.items():
        new_data = {
            "task_id": task_id,
            "completion": data.get("completion"),
            f"{uml_type}": data.get(uml_type),
        }
        result.append(new_data)
    return result


def concatenate_test_results(test_result):
    if not isinstance(test_result, list):
        return ""
    result_str = "### Error-prone situations:\n"
    for idx, item in enumerate(test_result):
        error_msg = item
        result_str += f"{idx + 1}. {error_msg}\n"
    return result_str


def filter_problems(problems, config, condition_key, condition_value):
    max_task_id = config.getint("eval", "max_task_id")

    filtered = {
        task_id: details
        for task_id, details in problems.items()
        if (
            condition_key not in details
            or not details.get(condition_key)
            or "".join(details.get(condition_key)).strip() == condition_value
        )
        and int(re.search(r"\d+", str(task_id)).group()) < max_task_id
    }
    return filtered


def safe_eval(value):
    try:
        return ast.literal_eval(str(value))
    except:
        return value


def is_numeric_string(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def convert_to_type(value, target_type):
    try:
        if target_type == bool:
            return bool(value)
        elif target_type == int:
            return int(float(value))
        elif target_type == float:
            return float(value)
        elif target_type == str:
            return str(value)
        elif target_type == list:
            return list(value)
        elif target_type == tuple:
            return tuple(value)
        elif target_type == set:
            return set(value)
        elif target_type == dict:
            return dict(value)
        else:
            return value
    except:
        return value


def extract_uml_code(text: str) -> str:
    uml_pattern = r"```plantuml\s+([\s\S]*?)```"

    code_blocks = re.findall(uml_pattern, text, re.MULTILINE)

    if not code_blocks:
        return "# extract uml code failed"

    uml_code = code_blocks[0].strip()

    return uml_code


def execute_docker_evaluation(test_name, language):
    import subprocess

    container_name = "8928e25148ad"
    password = "18428149886wang"
    language_name = test_name.split("_")[0]

    command = f"bash /workspace/CodeGeeX/scripts/evaluate_humaneval_x.sh /workspace/CodeGeeX/datax/{test_name}.jsonl {language_name} 8"

    full_command = f"sudo docker exec {container_name} {command}"

    process = subprocess.Popen(
        ["sudo", "-S"] + full_command.split(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout, stderr = process.communicate(input=password + "\n")

    print(stdout)
    return stdout, stderr


def separate_import_and_methods(java_code: str) -> dict:
    import_pattern = r"^\s*import\s+.+?;"
    imports = re.findall(import_pattern, java_code, flags=re.MULTILINE)

    code_without_imports = re.sub(import_pattern, "", java_code, flags=re.MULTILINE)

    main_pattern = r"public\s+static\s+void\s+main\s*\(.*?\)\s*\{.*?(?:\}.*?|$)"
    code_without_main = re.sub(
        main_pattern, "", code_without_imports, flags=re.DOTALL
    ).strip()

    return {"imports": imports, "remaining_code": code_without_main}


def modify_last_function(java_code: str) -> str:
    function_pattern = r"([a-zA-Z0-9_<>]+)\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)\s*\{"

    matches = list(re.finditer(function_pattern, java_code))

    if not matches:
        return java_code

    last_function = matches[-1]

    return_type_start, return_type_end = last_function.start(1), last_function.end(1)
    function_name_start, function_name_end = last_function.start(2), last_function.end(
        2
    )

    modified_code = (
        java_code[:return_type_start]
        + "void"
        + java_code[return_type_end:function_name_start]
        + "kkkkk"
        + java_code[function_name_end:]
    )

    return modified_code


def clean_java_comments(java_code: str) -> str:
    lines = java_code.split("\n")
    result_lines = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            result_lines.append(lines[i])
            i += 1
            continue

        if line.startswith("/**"):
            comment_lines = []
            comment_lines.append(lines[i])
            i += 1

            while i < len(lines) and "*/" not in lines[i]:
                comment_lines.append(lines[i])
                i += 1

            if i < len(lines) and "*/" in lines[i]:
                comment_lines.append(lines[i])
                if is_valid_block_comment(comment_lines):
                    result_lines.extend(comment_lines)
                i += 1
            continue

        if line.startswith("//"):
            result_lines.append(lines[i])
            i += 1
            continue

        if "*/" in line and not any(
            l.strip().startswith("/**") for l in result_lines[-5:]
        ):
            i += 1
            continue

        if line.startswith("*") and not any(
            l.strip().startswith("/**") for l in result_lines[-5:]
        ):
            i += 1
            continue

        result_lines.append(lines[i])
        i += 1

    return "\n".join(result_lines)


def is_valid_block_comment(comment_lines: list) -> bool:
    if not comment_lines:
        return False

    if not comment_lines[0].strip().startswith("/**"):
        return False

    if not comment_lines[-1].strip().endswith("*/"):
        return False

    for line in comment_lines[1:-1]:
        stripped = line.strip()
        if not stripped.startswith("*"):
            return False

    return True


def convert_jsonl(result_dict, language, input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            data = json.loads(line.strip())
            result = separate_import_and_methods(data["completion"])
            import_string = "\n".join(result["imports"])
            code_string = clean_java_comments(result["remaining_code"])
            new_data = {
                "task_id": f"{language}/{data['task_id'].split('/')[-1]}",
                "generation": data["completion"],
                "prompt": import_string
                + "\n"
                + modify_last_function(result_dict[data["task_id"]]["prompt"])
                + "}\n"
                + code_string
                + "}",
            }

            fout.write(json.dumps(new_data) + "\n")
