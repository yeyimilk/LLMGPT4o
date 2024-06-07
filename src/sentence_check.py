
from utils.gpt_utils import get_batch_list, extract_content_from_gpt_response
import numpy as np
from data_loader import load_top_k_chinese_per_length
import jieba
import csv
import pandas as pd

def generate_csv():
    models = ['gpt-4-turbo-2024-04-09', 'gpt-4o-2024-05-13']
    for m in models:
        top_k_chinese = load_top_k_chinese_per_length()
        
        data = get_batch_list(f'data/sentence/{m}_full_split_words_batch_requests_output.jsonl')
        results = []
        full_count = 0
        split_count = 0
        for d in data:
            
            result = []
            
            customed_id, content = extract_content_from_gpt_response(d, 'sentence')
            index_id = customed_id.split('_')[0]
            word_type = customed_id.split('_')[1]
            
            full_word = top_k_chinese[index_id]
            full_word = full_word.replace(' ', '')
            seg_list = jieba.cut(full_word, cut_all=False)

            result.append(customed_id)
            
            all_apeared = 0

            if word_type == 'full':
                all_apeared = 1 if full_word in content else 0
                full_count += 1 if full_word in content else 0
                result.append(full_word)
                
            else:                    
                seg_list = list(seg_list)
                result.append("; ".join(seg_list))
                seg_list = [seg for seg in seg_list if seg != '' and seg != '_']
                
                all_true = all([seg in content for seg in seg_list])
                split_count += 1 if all_true else 0
                all_apeared = 1 if all_true else 0
                  
            result.append(content)
            result.append(all_apeared)
            
            results.append(result)
        
        headers = ['id', 'original', 'response', 'appeared', 'accuracy', 'relevant']
        f_name = f'data/sentence/{m}_full_split_words_results.csv'
        with open(f_name, 'w', encoding="utf-8-sig") as f:
            write = csv.writer(f)
            write.writerow(headers)
            write.writerows(results)
            

def evaluate_csv():
    models = ['gpt-4-turbo-2024-04-09', 'gpt-4o-2024-05-13']
    for m in models:
        f_name = f'data/sentence/{m}_full_split_words_results.csv'
        data = pd.read_csv(f_name)
        full = data[data['id'].str.contains('full')]
        split = data[data['id'].str.contains('split')]
        
        print(m)
        print("====Full Words====")
        print(full['accuracy'].value_counts().sort_index())
        print("====Split Words====")
        print(split['accuracy'].value_counts().sort_index())
        print("\n")
        
        
    
if __name__ == "__main__":
    # generate_csv()
    evaluate_csv()