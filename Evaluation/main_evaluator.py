import argparse
import yaml
import os
import pandas as pd
from evaluator import Evaluator
from data_loader import load_dataset
from data_processor import DataProcessor
from metrics_calculator import MetricsCalculator

def main():
    parser = argparse.ArgumentParser(description="Run the HLV Evaluation Pipeline.")
    parser.add_argument('--config', type=str, required=True, help='Path to the evaluation configuration file.')
    parser.add_argument('--mode', type=str, required=True, choices=['baseline', 'with_explanations', 'calculate'], help='Execution mode.')
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    task_name = config['task_name']
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    if args.mode in ['baseline', 'with_explanations']:
        print(f"--- Running Evaluation in '{args.mode}' mode ---")
        evaluator = Evaluator(config)
        
        if args.mode == 'baseline':
            input_file = config['input_baseline_file']
            output_file = os.path.join(output_dir, f"{task_name}_baseline_raw_output.jsonl")
            explanation_data = None
        else: # with_explanations
            input_file = config['input_baseline_file']
            explanation_file = config['input_explanation_file']
            output_file = os.path.join(output_dir, f"{task_name}_with_explanations_raw_output.jsonl")
            print(f"Loading explanations from: {explanation_file}")
            explanation_data = load_dataset(explanation_file)

        print(f"Loading baseline data from: {input_file}")
        baseline_data = load_dataset(input_file)
        
        evaluator.run_evaluation(baseline_data, explanation_data, output_file)

    elif args.mode == 'calculate':
        print("--- Running Metric Calculation ---")
        processor = DataProcessor(config)
        calculator = MetricsCalculator()

        # 1. Process Gold Standard Data
        gold_standard_file = config['gold_standard_file']
        print(f"Processing gold standard data from: {gold_standard_file}")
        processed_gold_df = processor.process_gold_standard(gold_standard_file)

        # 2. Process LLM outputs for all settings and calculate metrics
        all_metrics_results = {}
        evaluation_settings = config['evaluation_settings']

        for setting_name, setting_details in evaluation_settings.items():
            raw_llm_output_file = os.path.join(output_dir, setting_details['raw_output_file'])
            if not os.path.exists(raw_llm_output_file):
                print(f"Warning: Raw output file not found for setting '{setting_name}'. Skipping.")
                continue

            print(f"\nProcessing and calculating metrics for: {setting_name}")
            processed_llm_df = processor.process_llm_output(raw_llm_output_file)
            
            # Calculate metrics
            metrics = calculator.calculate_all_metrics(processed_llm_df, processed_gold_df)
            all_metrics_results[setting_name] = metrics

        # 3. Save results to Excel
        output_excel_path = os.path.join(output_dir, f"{task_name}_metrics_report.xlsx")
        df_results = pd.DataFrame.from_dict(all_metrics_results, orient='index')
        df_results.to_excel(output_excel_path)
        print(f"\nMetrics report saved successfully to: {output_excel_path}")

if __name__ == '__main__':
    main()