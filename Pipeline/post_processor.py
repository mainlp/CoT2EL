import json
from tqdm import tqdm
import os
from difflib import SequenceMatcher

class PostProcessor:
    def __init__(self, config):
        self.config = config
        self.task_name = config['task_name']
        self.standard_keys = config.get('post_processing_stage_4', {}).get('standard_keys', [])

    def _normalize_single_dict(self, original_dict, original_data_record=None):
        """Normalizes the keys of a single JSON object (dictionary)."""
        key_map = {}
        for i, key in enumerate(self.standard_keys):
            key_map[key] = [f'({key})', f'{key}.', f'Option {key}']
            if self.task_name == 'VariErrNLI':
                conditions = {'A': 'entailment', 'B': 'neutral', 'C': 'contradiction'}
                if key in conditions:
                    key_map[key].append(conditions[key])
            elif (self.task_name in ['SocialIQA', 'CommonsenseQA']) and original_data_record:
                answer_key = f'answer{key.upper()}' # CQA uses answerA, answerB...
                if answer_key in original_data_record:
                    key_map[key].append(original_data_record[answer_key].lower())

        normalized_dict = {}
        if not isinstance(original_dict, dict):
            return {}
            
        for original_key, value in original_dict.items():
            found = False
            for std_key, variations in key_map.items():
                if any(var.lower() in original_key.lower() for var in variations):
                    normalized_dict[std_key] = value
                    found = True
                    break
            if not found:
                # Fallback for keys that don't match standard patterns but might contain the answer text
                if (self.task_name in ['SocialIQA', 'CommonsenseQA']) and original_data_record:
                     for std_key in self.standard_keys:
                         ans_key = f'answer{std_key.upper()}'
                         if original_data_record.get(ans_key, '').lower() in original_key.lower():
                             normalized_dict[std_key] = value
                             found = True
                             break
                if not found:
                    print(f"Warning: Could not normalize key '{original_key}'. It will be dropped.")

        return normalized_dict

    def run_normalization(self, structured_data, original_data):
        """Runs Stage 4: Normalization of JSON keys."""
        print("\nRunning Normalization Stage 4...")
        original_data_map = {i: item for i, item in enumerate(original_data)}
        
        normalized_results = []
        for i, item in tqdm(enumerate(structured_data), total=len(structured_data), desc=self.task_name + " Stage 4"):
            if 'structured_evidence' in item:
                normalized_evidence = self._normalize_single_dict(item['structured_evidence'], original_data_map.get(i))
                item['normalized_evidence'] = normalized_evidence
            item.pop('structured_evidence', None)
            normalized_results.append(item)

        output_config = self.config['post_processing_stage_4']
        output_file = os.path.join(self.config['output_dir'], output_config['output_file'])
        with open(output_file, 'w', encoding='utf-8') as f:
            for res in normalized_results:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')
        
        print(f"Stage 4 normalized results saved to {output_file}")
        return normalized_results

    def _find_best_match(self, check_snt, discourse_units):
        """Finds the best matching sentence in discourse units using SequenceMatcher."""
        if not discourse_units or not check_snt:
            return None
        best_score = 0
        output_match_snt = None
        for unit in discourse_units:
            matcher = SequenceMatcher(None, check_snt, unit)
            ratio = matcher.ratio()
            if ratio > best_score and ratio > 0.6: # Similarity threshold
                best_score = ratio
                output_match_snt = unit
        return output_match_snt

    def _filter_dict_with_discourse_units(self, normalized_dict, discourse_units):
        """Filters a single normalized dictionary using discourse units."""
        filtered_dict = {}
        if not normalized_dict or not isinstance(normalized_dict, dict):
            return filtered_dict
            
        for key, value in normalized_dict.items():
            filtered_dict[key] = {'support': [], 'oppose': []}
            if isinstance(value, dict):
                for sentiment in ['support', 'oppose']:
                    if value.get(sentiment) and isinstance(value[sentiment], list):
                        for sentence in value[sentiment]:
                            best_match = self._find_best_match(sentence, discourse_units)
                            if best_match:
                                filtered_dict[key][sentiment].append(best_match)
        return filtered_dict

    def run_filtering(self, normalized_data, discourse_data):
        """Runs Stage 5: Filtering evidence against discourse units."""
        print("\nRunning Filtering Stage 5...")
        
        filtered_results = []
        for i, item in tqdm(enumerate(normalized_data), total=len(normalized_data), desc=self.task_name + " Stage 5"):
            normalized_evidence = item.get('normalized_evidence')
            if i < len(discourse_data):
                discourse_record = discourse_data[i]
                segments = discourse_record.get('segments', [])
                connectives = discourse_record.get('connectives', [])
                union_units = list(set(segments) | set(connectives))
                
                filtered_evidence = self._filter_dict_with_discourse_units(normalized_evidence, union_units)
                item['filtered_evidence'] = filtered_evidence
            else:
                item['filtered_evidence'] = {} # No discourse data available for this item

            item.pop('normalized_evidence', None)
            filtered_results.append(item)

        output_config = self.config['filtering_stage_5']
        output_file = os.path.join(self.config['output_dir'], output_config['output_file'])
        with open(output_file, 'w', encoding='utf-8') as f:
            for res in filtered_results:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')
        
        print(f"Stage 5 filtered results saved to {output_file}")
        return filtered_results