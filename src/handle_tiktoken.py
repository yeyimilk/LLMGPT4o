
import json
from data_loader import load_top_k_chinese_per_length
import jieba

def extract_all_tiktoken():    
    tokens = load_top_k_chinese_per_length()
    results = {}
    
    for key, value in tokens.items():
        seg_list = jieba.cut(value, cut_all=False)
        results[key] = ", ".join(list(seg_list))
        
    json.dump(results, open("data/jieba_tiktoken.json", "w", encoding='utf-8'), ensure_ascii=False, indent=4)

if __name__ == "__main__":
    extract_all_tiktoken()