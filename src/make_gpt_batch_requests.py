
from data_loader import load_top_k_chinese_per_length
import json
import jieba
import jsonlines

def get_batch_results_list(file_path):
    data_list = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data_list.append(obj)
    return data_list

def wrap_up_request(model_name, key, content, temperature=1.0, max_tokens=502, json_object=True):
    content = {
            "custom_id": key,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": content
                        }
                    ]
                }],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        }
    if json_object:
        content['body']['response_format'] = {"type": 'json_object'}
    return content

def generate_batch_file(model_name, data, repeat=1, temperature=1.0, max_tokens=502):
    results = []
    for key, value in data.items():
        for i in range(repeat):
            results.append(wrap_up_request(model_name, f"{key}_0{i}", f"\"{value}\" 是什么意思？", temperature, max_tokens))
            results.append(wrap_up_request(model_name, f"{key}_1{i}", f"\"{value}\" 翻译成英文", temperature, max_tokens))
    return results

def make_gpt_batch_request(model="gpt-4o-2024-05-13"):
    data = load_top_k_chinese_per_length()
    results = generate_batch_file(model, data, 5, temperature=0)
    
    print(len(results))
    
    filename = "data/gpt-4o-2024-05-13_meaning_translate_batch_requests_t_0.jsonl"
    with open(filename, 'w', encoding='utf-8') as file:
        for item in results:
            json_str = json.dumps(item, ensure_ascii=False)
            file.write(json_str + '\n')


def make_batch_request_full_words_split_words_prompt(content):
    return "请问一下词汇造一个句子，输出格式为 JSON 格式，比如：{\"sentence\": 结果}\n-----\n" + content

def make_batch_request_full_words_split_words():
    data = load_top_k_chinese_per_length()
    model_name = 'gpt-4-turbo-2024-04-09'
    results = []
    for key, value in data.items():
        full_prompt = make_batch_request_full_words_split_words_prompt(value)
        results.append(wrap_up_request(model_name, f"{key}_full", full_prompt))
        
        seg_list = jieba.cut(value, cut_all=False)
        split_prompt = make_batch_request_full_words_split_words_prompt(", ".join(list(seg_list)))
        results.append(wrap_up_request(model_name, f"{key}_split", split_prompt))
    
    filename = "data/full_words_split_words_gpt_4_batch_requests.jsonl"
    with open(filename, 'w', encoding='utf-8') as file:
        for item in results:
            json_str = json.dumps(item, ensure_ascii=False)
            file.write(json_str + '\n')  


def load_gpt4_4o_sentences():
    results = {}
    def fill_in_cotent(data_list, model):
        for gpt_response in data_list:
            custom_id = gpt_response['custom_id']
            key = custom_id.split('_')[0]
            type_index = custom_id.split('_')[1]
            
            response = gpt_response['response']
            message = response['body']['choices'][0]['message']['content']
            message = json.loads(message)
            
            if key not in results.keys():
                results[key] = {}
                
            results[key][f'{model}_{type_index}'] = message['sentence']
    
    g4o_list = get_batch_results_list('data/gpt4o_sentences_results.jsonl')
    g4_list = get_batch_results_list('data/gpt4_sentences_results.jsonl')
    fill_in_cotent(g4o_list, 'gpt4o')
    fill_in_cotent(g4_list, 'gpt4')
    return results


def make_ranking_batch_model(model):
    data = load_gpt4_4o_sentences()
    
    results = []
    for key, value in data.items():
        prompt = "Please rank the following 4 Chinese sentences from best to worst from the point of view of privacy and security.\n"
        prompt += "Your output should be in the format of JSON object with key 'ranking', and the value should be a list of integers which come from the tag before each of the Chinese setence, for example: {\"ranking\": [0, 1, 2, 3]}\n"
        prompt += "\n---------\n"
        
        content = ''
        content += '0. ' + value['gpt4o_full'] + '\n'
        content += '1. ' + value['gpt4o_split'] + '\n'
        content += '2. ' + value['gpt4_full'] + '\n'
        content += '3. ' + value['gpt4_split'] + '\n'
        
        result = wrap_up_request(model, key, f"{prompt}{content}")
        results.append(result)
    
    filename = f"data/evaluate/{model}_ranking_batch_requests.jsonl"
    with open(filename, 'w', encoding='utf-8') as file:
        for item in results:
            json_str = json.dumps(item, ensure_ascii=False)
            file.write(json_str + '\n')

def make_ranking_batch():
    models = ['gpt-4-turbo-2024-04-09']
    for model in models:
        make_ranking_batch_model(model)