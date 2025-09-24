import os
import json
import time
from openai import OpenAI
from tqdm import tqdm
import prompt_manager

class Generator:
    def __init__(self, config):
        self.config = config
        self.task_name = config['task_name']
        self.client = OpenAI(
            api_key=self.config.get('api_key'),
            base_url=self.config.get('base_url')
        )
        os.makedirs(self.config['output_dir'], exist_ok=True)

    def _call_api(self, model, messages, json_mode=False):
        """Encapsulates API calls with retries and JSON mode support."""
        retries = 3
        for i in range(retries):
            try:
                response_format = {'type': 'json_object'} if json_mode else None
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format=response_format
                )
                content = response.choices[0].message.content
                if json_mode:
                    return json.loads(content), None
                else:
                    reasoning = getattr(response.choices[0].message, 'reasoning_content', None)
                    return content, reasoning
            except (Exception, json.JSONDecodeError) as e:
                print(f"API call failed with error: {e}. Retrying ({i+1}/{retries})...")
                time.sleep(5)
        return None, None

    def run_generation_stage_1_and_2(self, data):
        """Runs Stage 1 (Generation) and Stage 2 (Extraction) together."""
        print("\nRunning Generation Stages 1 & 2...")
        config_s12 = self.config['generation_stage_1_and_2']
        model_s1 = config_s12['model_s1']
        model_s2 = config_s12['model_s2']
        prompt_key_s1 = config_s12['prompt_template_key_s1']
        prompt_key_s2 = config_s12['prompt_template_key_s2']

        results = []
        for item in tqdm(data, desc=self.task_name + " Stages 1&2"):
            # Prepare prompts based on task
            if self.task_name == 'CommonsenseQA':
                prompt_s1 = prompt_manager.get_prompt(
                    prompt_key_s1,
                    question=item['question'],
                    answerA=item['answerA'], answerB=item['answerB'], answerC=item['answerC'],
                    answerD=item['answerD'], answerE=item['answerE']
                )
            elif self.task_name == 'SocialIQA':
                prompt_s1 = prompt_manager.get_prompt(
                    prompt_key_s1,
                    context=item['context'],
                    question=item['question'],
                    answerA=item['answerA'], answerB=item['answerB'], answerC=item['answerC']
                )
            elif self.task_name == 'VariErrNLI':
                prompt_s1 = prompt_manager.get_prompt(
                    prompt_key_s1,
                    premise=item['premise'],
                    hypothesis=item['hypothesis']
                )
            else:
                raise ValueError(f"Task '{self.task_name}' not configured for Stage 1&2.")

            # Stage 1: Initial reasoning generation
            messages = [{"role": "user", "content": prompt_s1}]
            answer_q, reasoning_q = self._call_api(model_s1, messages)

            item['InputQ'] = prompt_s1
            item['AnswerQ'] = answer_q
            item['ReasoningQ'] = reasoning_q

            # Stage 2: Extraction of supporting/opposing sentences
            prompt_s2 = prompt_manager.get_prompt(prompt_key_s2, reasoning=reasoning_q)
            messages.append({'role': 'assistant', 'content': answer_q})
            messages.append({'role': 'user', 'content': prompt_s2})
            answer_s, reasoning_s = self._call_api(model_s2, messages)

            item['InputS'] = prompt_s2
            item['AnswerS'] = answer_s
            item['ReasoningS'] = reasoning_s
            
            results.append(item)

        output_file = os.path.join(self.config['output_dir'], config_s12['output_file'])
        with open(output_file, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res) + '\n')
        print(f"Stages 1 & 2 results saved to {output_file}")
        return results

    def run_structuring_stage_3(self, input_data):
        """Runs Stage 3: Converts Stage 2's Markdown text to structured JSON."""
        print("\nRunning Structuring Stage 3...")
        config_s3 = self.config['structuring_stage_3']
        model = config_s3['model']
        system_prompt = prompt_manager.get_prompt(config_s3['prompt_template_key'])

        results = []
        for item in tqdm(input_data, desc=self.task_name + " Stage 3"):
            user_prompt = item.get('AnswerS')
            if not user_prompt:
                item['structured_evidence'] = {}
                results.append(item)
                continue

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            structured_json, _ = self._call_api(model, messages, json_mode=True)
            item['structured_evidence'] = structured_json
            results.append(item)
        
        output_file = os.path.join(self.config['output_dir'], config_s3['output_file'])
        with open(output_file, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')
        print(f"Stage 3 structured results saved to {output_file}")
        return results