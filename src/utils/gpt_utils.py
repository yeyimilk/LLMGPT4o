
import jsonlines
import json

def get_batch_list(file_path):
    data_list = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data_list.append(obj)
    return data_list

def extract_content_from_gpt(content, json_key=None):
    content = content['body']['choices'][0]['message']['content']
    if json_key:
        content = json.loads(content)[json_key]
    return content

def extract_content_from_gpt_response(response, json_key=None):
    customed_id = response['custom_id']
    return customed_id, extract_content_from_gpt(response['response'], json_key)