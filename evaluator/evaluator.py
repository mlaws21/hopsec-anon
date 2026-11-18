import re

def contains_lowercase_in_parentheses(s):
    if re.search(r'\([a-z]\)', s):
        return True
    return False

def normalize_option_answer(answer):
    answer = re.sub(r'[()]', '', answer)
    return answer.strip().lower()

def clean_and_extract(predicted_answer):
    predicted_answer = predicted_answer.replace('.', ' ').replace(',', ' ')
    match = re.match(r'\(([a-z])\)', predicted_answer.strip())
    if match:
        return match.group(1)
    return predicted_answer.strip().replace(" ", "").lower()


def evaluate_answer(ground_truth_answer, predicted_answer):
    ground_truth_answer = ground_truth_answer.strip().lower()
    predicted_answer = predicted_answer.strip().lower()

    # Yes/No Binary
    if ground_truth_answer in ['yes', 'no', 'valid', 'invalid', 'true', 'false']:
        return ground_truth_answer == predicted_answer
    
    # Option (A), (B)...
    elif contains_lowercase_in_parentheses(ground_truth_answer):
        predicted_answer = clean_and_extract(predicted_answer)
        return normalize_option_answer(ground_truth_answer) == normalize_option_answer(predicted_answer)
    
    return ground_truth_answer == predicted_answer

def test_function():
    return 
