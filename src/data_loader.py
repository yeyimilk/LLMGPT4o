
import json
import re
import regex



def contains_chinese(text):
    return re.search(r'[\u4e00-\u9fff]', text) is not None

def contains_korean(value):
    if regex.search(r'\p{IsHangul}', value):
        return True
    return False

def load_top_k_chinese(k):
    with open("data/tiktoken.json", "r") as f:
        data = json.load(f)
    
    results = {}
    
    for key, value in data.items():
        if len(results.keys())>= k:
            break
        
        if contains_chinese(value):
           results[key] = value 
    
    return results


def load_top_k_token(k, chinese=True, korean=False):
    with open("data/tiktoken.json", "r") as f:
        data = json.load(f)
    
    results = {}
    
    for key, value in data.items():
        if len(results.keys())>= k:
            break
        if korean and contains_korean(value):
            results[key] = value
        elif chinese and contains_chinese(value):
            results[key] = value
    
    return results

def load_top_k_chinese_per_length(k=2000, t_count=20):
    top_k = load_top_k_chinese(k)
    
    size =  0
    count = 0
    results = {}
    for key, value in top_k.items():
        if len(value) == size:
            if count >= t_count:
                continue
            else:
                count += 1
                results[key] = value
        else:
            results[key] = value
            count = 1
            size = len(value)
    
    return results

def load_top_k_token_per_length(k=2000, t_count=20, chinese=True, korean=False):
    top_k = load_top_k_token(k, chinese, korean)
    
    size =  0
    count = 0
    results = {}
    for key, value in top_k.items():
        if len(value) == size:
            if count >= t_count:
                continue
            else:
                count += 1
                results[key] = value
        else:
            results[key] = value
            count = 1
            size = len(value)
    
    return results



    
if __name__ == "__main__":        
    data = load_top_k_token_per_length(2000, 20, chinese=True, korean=False)
    print(data)