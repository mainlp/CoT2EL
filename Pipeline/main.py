import argparse
import yaml
import os
from generator import Generator
from data_loader import load_dataset
from post_processor import PostProcessor

def main():
    parser = argparse.ArgumentParser(description="A multi-stage pipeline for generating and processing MCQA explanations.")
    parser.add_argument('--config', type=str, required=True, help='Path to the task configuration file.')
    parser.add_argument('--start_stage', type=int, default=1, help='Which stage to start from (1 to 5).')

    args = parser.parse_args()

    # load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    gen = Generator(config)
    processor = PostProcessor(config)

    # --- Stage 1 & 2: Generation & Evidence Extraction ---
    if args.start_stage <= 2 and 'generation_stage_1_and_2' in config:
        print(f"Loading dataset from {config['input_file']}...")
        dataset = load_dataset(config['input_file'])
        gen.run_generation_stage_1_and_2(dataset)

    # --- Stage 3: Structuring ---
    if args.start_stage <= 3 and 'structuring_stage_3' in config:
        stage2_config = config.get('generation_stage_1_and_2', {})
        stage2_output_path = os.path.join(config['output_dir'], stage2_config.get('output_file'))
        
        if not os.path.exists(stage2_output_path):
            raise FileNotFoundError(f"Stage 2 output not found at {stage2_output_path}. Please run stages 1 & 2 first.")
        
        print(f"Loading data from {stage2_output_path} for Stage 3...")
        stage2_results = load_dataset(stage2_output_path)
        gen.run_structuring_stage_3(stage2_results)

    # --- Stage 4: Normalization ---
    if args.start_stage <= 4 and 'post_processing_stage_4' in config:
        stage3_config = config.get('structuring_stage_3', {})
        stage3_output_path = os.path.join(config['output_dir'], stage3_config.get('output_file'))

        if not os.path.exists(stage3_output_path):
            raise FileNotFoundError(f"Stage 3 output not found at {stage3_output_path}. Please run stage 3 first.")

        print(f"Loading data from {stage3_output_path} for Stage 4...")
        stage3_results = load_dataset(stage3_output_path)

        print(f"Loading original dataset from {config['input_file']} for key normalization...")
        original_dataset = load_dataset(config['input_file'])

        processor.run_normalization(stage3_results, original_dataset)

    # --- Stage 5: Filtering ---
    if args.start_stage <= 5 and 'filtering_stage_5' in config:
        stage4_config = config.get('post_processing_stage_4', {})
        stage4_output_path = os.path.join(config['output_dir'], stage4_config.get('output_file'))

        if not os.path.exists(stage4_output_path):
            raise FileNotFoundError(f"Stage 4 output not found at {stage4_output_path}. Please run stage 4 first.")

        print(f"Loading data from {stage4_output_path} for Stage 5...")
        stage4_results = load_dataset(stage4_output_path)

        discourse_file_path = config['filtering_stage_5']['discourse_file']
        if not os.path.exists(discourse_file_path):
             raise FileNotFoundError(f"Discourse file not found at {discourse_file_path}.")
        print(f"Loading discourse units from {discourse_file_path}...")
        discourse_data = load_dataset(discourse_file_path)

        processor.run_filtering(stage4_results, discourse_data)

    print("All tasks finished successfully.")

if __name__ == '__main__':
    main()