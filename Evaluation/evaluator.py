import os
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import prompt_factory_eval

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.task_name = config['task_name']
        print("Loading evaluation model... This may take a moment.")
        self.model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            cache_dir=config.get('cache_dir'),
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['model_name'],
            cache_dir=config.get('cache_dir')
        )
        print("Model loaded successfully.")

    def _get_llm_response(self, input_prompt, rank_type):
        """Gets logits, full text, or score from the LLM."""
        messages = [{"role": "user", "content": input_prompt}]
        text_template = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_template], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=256,
            output_logits=True,
            return_dict_in_generate=True
        )

        if rank_type == 'logits':
            # Token IDs for A, B, C, D, E for many tokenizers
            token_ids = [32, 33, 34, 36, 35] # A, B, C, D, E
            if self.task_name == 'cqa':
                return [generated_ids.logits[0][0][tid].item() for tid in token_ids]
            else:
                return [generated_ids.logits[0][0][tid].item() for tid in token_ids[:3]]
        
        output_sequence = generated_ids.sequences
        output_sequence = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, output_sequence)
        ]
        response = self.tokenizer.batch_decode(output_sequence, skip_special_tokens=True)[0]
        return response

    def run_evaluation(self, baseline_data, explanation_data, output_file):
        """Runs the LLM-as-a-Judge evaluation."""
        
        all_results = []
        
        for i, item_data in tqdm(enumerate(baseline_data), total=len(baseline_data), desc=f"Evaluating {self.task_name}"):
            result_record = item_data.copy()
            add_explanations = None
            
            if explanation_data:
                explanation_record = explanation_data[i].get('filtered_evidence')
                if not explanation_record:
                    print(f"Warning: No filtered evidence found for item {i}. Skipping explanation.")
                else:
                    # Using a simple representation of explanations
                    add_explanations = json.dumps(explanation_record)

            # Evaluate for each ranking type: logits, full, score
            for rank_type in ['logits', 'full', 'score']:
                prompts = prompt_factory_eval.generate_prompt(
                    self.task_name, rank_type, item_data, add_explanations
                )
                
                if rank_type == 'score':
                    scores = [self._get_llm_response(p, rank_type) for p in prompts]
                    result_record[rank_type] = scores
                else:
                    response = self._get_llm_response(prompts, rank_type)
                    result_record[rank_type] = response
            
            all_results.append(result_record)

        # Save raw results
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in all_results:
                f.write(json.dumps(record) + '\n')
        print(f"Raw evaluation results saved to {output_file}")