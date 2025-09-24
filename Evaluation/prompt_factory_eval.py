def generate_prompt(task, rank_type, item_data, add_explanations=None):
    """
    Generates prompts for the LLM-as-a-Judge based on the task and evaluation type.
    """
    explanation_text = f"\\nExplanations: {add_explanations}" if add_explanations else ""

    if task == 'VariErrNLI':
        premise = item_data['premise']
        hypothesis = item_data['hypothesis']
        options = "A. Entailment \\nB. Neutral \\nC. Contradiction"
        context = f"Context: {premise} \\nStatement: {hypothesis}"
        
        if rank_type == 'logits':
            return f"Please determine whether the following statement is true (entailment), undetermined (neutral), or false (contradiction) given the context below. Consider relevant perspectives, possible explanations, or reasoning patterns in the following explanations. Select ONE of the listed options and start your answer with a single letter.\\n{context}\\n{options}{explanation_text}\\nAnswer:"
        elif rank_type == 'full':
            return f"Please assess whether the following statement is true (entailment), undetermined (neutral), or false (contradiction) given the context below. Consider relevant perspectives, possible explanations, or reasoning patterns in the following explanations. Rank all the following options from most appropriate to least appropriate. Only output the letters representing the options, separated by spaces.\\n{context}\\n{options}{explanation_text}\\nAnswer:"
        elif rank_type == 'score':
            prompts = []
            for label in ['Entailment', 'Neutral', 'Contradiction']:
                prompts.append(f"Please rate the following answer based on its plausibility in representing the relationship between the context and the statement on the 5-Point Scale rating as below. Consider relevant perspectives, possible explanations, or reasoning patterns in the following explanations. Only output a single integer corresponding to your evaluation.\\n{context}\\nAnswer: {label}\\nPlausibility Ratings:\\n1 = Impossible\\n2 = Technically Possible\\n3 = Plausible\\n4 = Likely\\n5 = Very Likely{explanation_text}\\nRating:")
            return prompts

    elif task in ['cqa', 'siqa']:
        question = item_data['question']
        context = f"Scenario: {item_data['context']}" if task == 'siqa' else f"Question: {question}"
        answers = {f"{k.replace('answer','')}": v for k, v in item_data.items() if k.startswith('answer')}
        options = "\\n".join([f"{key}. {value}" for key, value in answers.items()])

        if rank_type == 'logits':
            return f"Please read the following, consider relevant perspectives, possible explanations, or reasoning patterns in the following explanations. Choose the most appropriate answer from the options provided and start your answer with a single letter.\\n{context}\\n{options}{explanation_text}\\nAnswer:"
        elif rank_type == 'full':
            return f"Please read the following, consider relevant perspectives, possible explanations, or reasoning patterns in the following explanations. Rank all the following options from best to worst base on relevance and appropriateness. Only output the letters representing the options, separated by spaces.\\n{context}\\n{options}{explanation_text}\\nAnswer:"
        elif rank_type == 'score':
            prompts = []
            for ans_text in answers.values():
                prompts.append(f"Please read the following, consider relevant perspectives, possible explanations, or reasoning patterns in the following explanations. Rate the plausibility of the answer on the 5-Point Scale rating as below. Only output a single integer corresponding to your evaluation.\\n{context}\\nAnswer: {ans_text}\\nPlausibility Ratings:\\n1 = Impossible\\n2 = Technically Possible\\n3 = Plausible\\n4 = Likely\\n5 = Very Likely{explanation_text}\\nRating:")
            return prompts
    
    raise ValueError(f"Unknown task for prompt generation: {task}")