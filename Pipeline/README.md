# CoT2EL Pipeline for MCQA Tasks

This project provides a comprehensive, 5-stage pipeline for generating, structuring, normalizing, and filtering Chain-of-thought reasoning for Multiple Choice Question Answering (MCQA) datasets using Large Language Models like DeepSeek.

The entire pipeline is configuration-driven, making it easy to adapt to new datasets, models, and processing logic without changing the core code.


## Project Structure

```bash 
deepseek_mcqa_pipeline/ 
├── main.py # Main entry point to run the pipeline 
├── generator.py # Core logic for API calls (Stages 1-3) 
├── data_loader.py # Utility for loading datasets 
├── prompt_manager.py # Centralized prompt templates 
├── post_processor.py # Logic for normalization and filtering (Stages 4-5) 
├── configs/ 
│ ├── cqa_config.yaml # Configuration for CommonsenseQA 
│ ├── siqa_config.yaml # Configuration for Social IQA 
│ └── varierr_config.yaml # Configuration for VariErr NLI 
└── README.md 
``` 


## The 5-Stage Pipeline

1.  **Stage 1: CoT Generation**: For each question and answer choice, this stage generates CoTs using a specified model (e.g., `deepseek-chat`).
2.  **Stage 2: Explanation Extraction**: Using a reasoning model (e.g., `deepseek-reasoner`), this stage takes the generated CoTs and extracts specific sentences that either support or oppose each answer choice.
3.  **Stage 3: Structuring**: The unstructured text from Stage 2 is converted into a structured JSON object with `support` and `oppose` keys for each option.
4.  **Stage 4: Normalization**: The keys of the JSON objects are cleaned and standardized into a consistent format (e.g., 'A', 'B', 'C'), resolving inconsistencies from the LLM's output.
5.  **Stage 5: Discourse Filtering**: The final step filters the normalized evidence against externally provided discourse units (segments and connectives), retaining only the most salient and structurally sound statements. The discourse segmenters are trained based on [DisCoDisCo](https://github.com/gucorpling/DisCoDisCo).

## Setup

1.  **Install dependencies:**
    ```bash
    pip install pyyaml openai tqdm
    ```

2.  **Configure your tasks:**
    -   Navigate to the `configs/` directory.
    -   Edit the `*.yaml` files to match your project.
    -   **Crucially, you must fill in your `api_key` and the correct, absolute file paths for your datasets (`input_file`, `discourse_file`) and your desired `output_dir`.**

## How to Run

Use the `main.py` script from your terminal. You must specify which task to run using the `--config` argument.

```bash
# Run the full pipeline for the CommonsenseQA task
python main.py --config configs/cqa_config.yaml

# Run the full pipeline for the Social IQA task
python main.py --config configs/siqa_config.yaml
```

## Advanced Usage: Starting from a Specific Stage

If you have already completed some stages, you can save time by starting the pipeline from a later stage using the `--start_stage` argument.

```bash
# Run only the final filtering stage (Stage 5) for the CQA task
# This assumes that stages 1-4 have already been completed successfully.
python main.py --config configs/cqa_config.yaml --start_stage 5
```

## How to Add a New MCQA Dataset

- Create a new YAML file in the `configs/` directory (e.g., `new_dataset_config.yaml`). Fill in all the required paths and parameters.

- Add new prompt templates to `prompt_manager.py` if your task requires unique prompts.

- Update the logic in `generator.py` and `post_processor.py` if your dataset has a unique structure that needs special handling. The code is designed to be adaptable, primarily requiring changes in how data fields are accessed.

- Run the pipeline with your new configuration file.