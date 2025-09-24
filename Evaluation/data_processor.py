import json
import ast
import numpy as np
import pandas as pd
from data_loader import load_dataset

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.task_name = config['task_name']

    # --- Gold Standard Processing ---
    def _transfer_votings_to_dist(self, votes_str, item_dict):
        """Converts vote string to a normalized probability distribution."""
        try:
            votes_dict = ast.literal_eval(votes_str)
        except (ValueError, SyntaxError):
            return [0] * len(self._get_label_list())
        
        scores = []
        for label in self._get_label_list():
            answer_key = f'answer{label}'
            answer_text = item_dict.get(answer_key, '')
            scores.append(votes_dict.get(answer_text, 0))
            
        total = sum(scores)
        return [s / total for s in scores] if total > 0 else [0] * len(scores)

    def _transfer_ratings_to_scores(self, item_dict):
        """Averages human ratings to get a score per option."""
        rating_map = {"5 - Very Likely": 5, "4 - Likely": 4, "3 - Plausible": 3, "2 - Technically Possible": 2, "1 - Impossible": 1}
        avg_scores = []
        for label in self._get_label_list():
            ratings = item_dict.get(f'answer{label}_ratings', [])
            if not ratings:
                avg_scores.append(0)
                continue
            scores = [rating_map[r['rating']] for r in ratings]
            avg_scores.append(np.mean(scores))
        return avg_scores

    def process_gold_standard(self, gold_file_path):
        """Processes the raw gold standard file into a clean DataFrame."""
        gold_data = load_dataset(gold_file_path)
        processed_records = []
        for item in gold_data:
            record = item.copy()
            if 'votes_distribution' in item:
                record['distribution'] = self._transfer_votings_to_dist(item['votes_distribution'], item)
            if 'answerA_ratings' in item:
                 record['score'] = self._transfer_ratings_to_scores(item)
            processed_records.append(record)
        return pd.DataFrame(processed_records)

    # --- LLM Output Processing ---
    def _normalize_dist(self, logits):
        """Normalizes logits to a probability distribution."""
        logits = np.array([abs(x) for x in logits])
        total = logits.sum()
        return (logits / total).tolist() if total > 0 else [0] * len(logits)

    def _process_scores(self, score_list):
        """Extracts the first digit from score strings."""
        processed = []
        for score in score_list:
            score_str = str(score).strip()
            if score_str and score_str[0].isdigit():
                processed.append(int(score_str[0]))
            else:
                processed.append(1) # Default score
        return processed

    def _assign_scores_from_rank(self, rank_str):
        """Converts a ranked string like 'B C A' to a list of scores."""
        label_list = self._get_label_list()
        n = len(label_list)
        scores = {label: 1 for label in label_list}
        
        # Clean up the rank string
        ranked_labels = [char for char in rank_str.upper() if char in label_list]
        
        seen = set()
        unique_ranks = []
        for label in ranked_labels:
            if label not in seen:
                seen.add(label)
                unique_ranks.append(label)

        for i, label in enumerate(unique_ranks):
            scores[label] = n - i
            
        return [scores[label] for label in label_list]

    def process_llm_output(self, llm_output_file):
        """Processes the raw output from the evaluator LLM into a clean DataFrame."""
        llm_data = load_dataset(llm_output_file)
        processed_records = []
        for item in llm_data:
            record = item.copy()
            if 'logits' in item:
                record['distribution'] = self._normalize_dist(item['logits'])
            if 'score' in item:
                record['score'] = self._process_scores(item['score'])
            if 'full' in item:
                record['rank'] = self._assign_scores_from_rank(item['full'])
            processed_records.append(record)
        return pd.DataFrame(processed_records)

    def _get_label_list(self):
        return ['A', 'B', 'C', 'D', 'E'] if self.task_name == 'cqa' else ['A', 'B', 'C']