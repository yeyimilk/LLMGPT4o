import jsonlines
from data_loader import load_top_k_chinese_per_length
import json

def get_gpt4o_results_list(file_path = 'data/gpt_batch_requests_output.jsonl'):
    data_list = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data_list.append(obj)
    return data_list


if __name__ == '__main__':
    top_k_c = load_top_k_chinese_per_length()
    gpt_results_list = get_gpt4o_results_list()

    results = {}
    for gpt_response in gpt_results_list:
        custom_id = gpt_response['custom_id']
        key = custom_id.split('_')[0]
        type_index = custom_id.split('_')[1]
        
        response = gpt_response['response']
        message = response['body']['choices'][0]['message']['content']
        
        prompt_type = type_index[0]
        reponse_index = type_index[1]
        
        original_text = top_k_c[key]
        
        if key in results.keys():
            if prompt_type == '0':
                results[key]['meaning'].append(message)
            else:
                results[key]['translation'].append(message)
        
        else:
            results[key] = {
                'original_text': original_text,
                'meaning': [],
                'translation': []
            }
            if prompt_type == '0':
                results[key]['meaning'].append(message)
            else:
                results[key]['translation'].append(message)
    
    json.dump(results, open("data/gpt4o_results.json", "w", encoding='utf-8'), ensure_ascii=False, indent=4)
    
    
    
    




        

