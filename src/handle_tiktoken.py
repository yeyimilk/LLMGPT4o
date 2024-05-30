import tiktoken
import json
from data_loader import load_top_k_chinese_per_length, contains_chinese
import jieba

def extract_all_tiktoken():
    # oT = tiktoken.get_encoding("o200k_base")
    # clT = tiktoken.get_encoding("cl100k_base")
    
    tokens = load_top_k_chinese_per_length()
    results = {}
    
    for key, value in tokens.items():
        seg_list = jieba.cut(value, cut_all=False)
        results[key] = ", ".join(list(seg_list))
        
    json.dump(results, open("data/jieba_tiktoken.json", "w", encoding='utf-8'), ensure_ascii=False, indent=4)
    
    # indexes = oT.encode("什么是 微信公众号天天中彩票")
    # for i in indexes:
    #     print(i, "===>", T.decode([i]))

    # results = {}

    # for i in range(0, 199998):
    #     result = T.decode([i])
    #     results[f"{i}"] = result

    # results = {k: v for k, v in sorted(results.items(), key=lambda item: len(item[1]), reverse=True)}

    # json.dump(results, open("data/tiktoken.json", "w", ), ensure_ascii=False, indent=4)

    # return results



if __name__ == "__main__":
    extract_all_tiktoken()
    # data = json.load(open("data/cl100k_tiktoken.json", "r", encoding='utf-8'))
    # print(data['181081']['85241'])
    # print(contains_chinese(data['181081']['85241']))
    # clT = tiktoken.get_encoding("cl100k_base")
    
    # for i in range(500):
    #     print(i, clT.decode([i]))
    
    # print(clT.decode([102]))