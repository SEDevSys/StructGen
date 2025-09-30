# StructGen 📐💻

<p align="center">
  <a href="#overview">📖 Overview</a> •
  <a href="#features">✨ Features</a> •
  <a href="#environment-setup">🧪 Environment Setup</a> •
  <a href="#configuration">⚙️ Configuration</a> •
  <a href="#quick-start">🚀 Quick Start</a> •
  <a href="#customization">🛠️ Customization</a> •
</p>

## 📖 Overview

**StructGen** is a novel framework for function-level code generation that leverages the power of Large Language
Models (LLMs) and Unified Modeling Language (UML) activity diagrams to generate high-quality, structured code. This
framework introduces a two-phase approach that separates design thinking from code implementation, resulting in more
robust and maintainable code generation.

The framework consists of two distinct phases:

- **🎨 Design Phase**: An LLM acts as a Designer, analyzing requirements and creating UML activity diagrams that model
  the logical flow and structure of the solution through sequential, selective, and iterative patterns.

- **⌨️ Coding Phase**: A separate LLM functions as a Coder, translating the design schemes into executable code while
  maintaining adherence to the structural guidance provided by the UML diagrams.

## ✨ Features

- **🏗️ Structure-Guided Generation**: Uses UML activity diagrams to provide structural guidance for code generation
- **🔄 Two-Phase Architecture**: Separates design thinking from implementation for better code quality
- **🌐 Multi-Model Support**: Compatible with various LLMs including GPT-3.5, GPT-4, DeepSeek Coder, and more
- **📚 Multi-Dataset Support**: Works with HumanEval, MBPP, and HumanEval-X datasets
- **🔧 Flexible Configuration**: Extensive configuration options through `config.ini`
- **📝 Multiple UML Types**: Supports PlantUML and Graphviz diagram formats
- **🧪 Comprehensive Testing**: Built-in test case generation and validation
- **🔁 Iterative Refinement**: Automatic feedback loop for improving generated code
- **📊 Detailed Logging**: Comprehensive logging and result tracking

## 🧪 Environment Setup

StructGen is developed and tested on Ubuntu 24.04.1 LTS with Python 3.8. Follow these steps to set up your environment:

### Prerequisites

- Python 3.8
- Conda (recommended) or pip
- Git

### Installation

1. **Create and activate a new conda environment:**

```bash
conda create -n StructGen python=3.8
conda activate StructGen
```

2. **Clone the repository:**

```bash
git clone https://github.com/SEDevSys/StructGen
cd StructGen
```

3. **Install required dependencies:**

```bash
pip install -r requirements.txt
```

4. **Set up the HumanEval testing framework:**

```bash
git clone https://github.com/openai/human-eval
pip install -e human-eval
```

## ⚙️ Configuration

### API Configuration

Configure your LLM API settings in `generate/config.ini`:

```ini
[ollama]
base_url = http://your-api-endpoint/v1
api_key = your-api-key
```

### Key Configuration Sections

- **`[eval]`**: Model parameters and generation settings
- **`[datasets]`**: Dataset selection (humaneval, mbpp, humaneval-x)
- **`[UML]`**: UML diagram type and file paths
- **`[feedback]`**: Iterative refinement parameters
- **`[prompt]`**: Template file locations

### HumanEval Framework Modification

To ensure compatibility with the evaluation framework, modify the `execute.py` file in your `human-eval` directory:

```python
# In the unsafe_execute function, replace the check_program assignment with:
check_program = (
        completion
        + "\n"
        + problem["test"]
        + "\n"
        + f"check({problem['entry_point']})"
)
```

## 🚀 Quick Start

### Basic Usage

Generate code using the default configuration:

```bash
python generate_feedback.py
```

### Output

The framework generates several outputs:

- **CSV Files**: Intermediate results and UML diagrams stored in CSV format
- **JSONL Files**: Final code generation results in JSONL format
- **Log Files**: Detailed execution logs for debugging and analysis
- **UML Diagrams**: Generated activity diagrams for each problem

Results are saved in the configured output directories as specified in `config.ini`.

## 🛠️ Customization

### Adding New Models

To add support for new LLMs, modify the model mapping in `utils.py`:

```python
model_mapping = {
    "your-new-model": "provider/your-new-model",
    # ... existing mappings
}
```

### Custom Prompt Templates

Create new prompt templates in the `generate/` directory and update the configuration:

```ini
[prompt]
generate_uml = /path/to/your/custom_uml_template.md
generate_function_code = /path/to/your/custom_code_template.md
```

### Extending UML Support

To add new UML diagram types, implement the corresponding extraction logic in `utils.py` and update the configuration
options.

## 🔧 Troubleshooting

### Common Issues

1. **API Connection Errors**: Verify your API endpoint and key in `config.ini`
2. **Import Errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`
3. **Permission Errors**: Check file permissions and ensure write access to output directories
4. **Memory Issues**: Adjust `max_tokens` and `max_workers` in configuration for large datasets

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

**StructGen** - Bridging the gap between design thinking and code implementation through structured, LLM-powered
generation. 