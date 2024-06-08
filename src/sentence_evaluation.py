
from utils.gpt_utils import get_batch_list, extract_content_from_gpt_response
import numpy as np

if __name__ == "__main__":
    # models = ['gpt-4-turbo-2024-04-09', 'gpt-4o-2024-05-13', 'gpt-3.5-turbo-0125']
    models = ['gpt-4-turbo-2024-04-09']
    for m in models:
        data = get_batch_list(f'data/sentence_evaluate/{m}_ranking_batch_requests_output.jsonl')
        results = []
        for d in data:
            customed_id, content = extract_content_from_gpt_response(d, 'ranking')
            # content.remove(0)
            results.append(content)
        results = np.array(results)
        
        
        for i in range(4):
            first_column = results[:, i]
            unique, counts = np.unique(first_column, return_counts=True)
            count_dict = dict(zip(unique, counts))
            
            # print(f"Model: {m}, {i+1}")
            string = f"{str(i)}    "
            
            for key, value in count_dict.items():
                # print(f"{value}")
                string += f"& {value / 166:.4f}"
            
            print(f"{string} \\\\")