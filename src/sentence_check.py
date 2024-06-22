
from utils.gpt_utils import get_batch_list, extract_content_from_gpt_response
import numpy as np
from data_loader import load_top_k_chinese_per_length
import jieba
import csv
import pandas as pd
import matplotlib.pyplot as plt

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
            

def evaluate_csv(ids_list=None):
    models = ['gpt-4-turbo-2024-04-09', 'gpt-4o-2024-05-13']
    
    arrays = [None] * 6
    
    score_5_count = {}
    
    for m in models:
        f_name = f'data/sentence/{m}_full_split_words_results.csv'
        data = pd.read_csv(f_name)
        
        if ids_list is not None:
            data['rid'] = data['id'].str.split('_').str[0]
            data = data[data['rid'].isin(ids_list)]
        
        full = data[data['id'].str.contains('full')]
        split = data[data['id'].str.contains('split')]
        
        full_value_counts = full['accuracy'].value_counts().sort_index()
        split_value_counts = split['accuracy'].value_counts().sort_index()
        
        score_5_count[m] = {
            'full': full_value_counts.get(5, 0),
            'split': split_value_counts.get(5, 0),
        }
    
    return score_5_count
        
        
def evalute_only_all_appear():
    gpt4o = pd.read_csv('data/sentence/gpt-4o-2024-05-13_full_split_words_results.csv')
    gpt4 = pd.read_csv('data/sentence/gpt-4-turbo-2024-04-09_full_split_words_results.csv')
    
    def extract_ids_for_appeared(id_appeared, df):
        # dictionary to save id and appeared value
        for index, row in df.iterrows():
            r_id = row['id'].split('_')[0]
            if r_id not in id_appeared:
                id_appeared[r_id] = 0
            
            id_appeared[r_id] += row['appeared']
            
        return id_appeared

    id_appeared = {}
    
    extract_ids_for_appeared(id_appeared, gpt4o)
    extract_ids_for_appeared(id_appeared, gpt4)
    
    # only keep when value is 4
    id_appeared = {k: v for k, v in id_appeared.items() if v == 4}
    
    evaluate_csv(list(id_appeared.keys()))
    
def evaluate_with_tokens_size(t_size=20):
    top_k_chinese = load_top_k_chinese_per_length()
    data = {}
    for key, value in top_k_chinese.items():
        if len(value) < t_size:
            data[key] = value
    
    ids = list(data.keys())
    return evaluate_csv(ids), len(ids)
    

def draw_score_figure():
    gpt4o_full = []
    gpt4o_split = []
    gpt4_full = []
    gpt4_split = []
    totals = []
    
    for i in range(3, 13):
        results, total = evaluate_with_tokens_size(i)
        
        gpt4o = results['gpt-4o-2024-05-13']
        gpt4 = results['gpt-4-turbo-2024-04-09']
        print(f"Token size: {i-1}-{total}, gpt4o: full: {gpt4o['full']}, split: {gpt4o['split']} gpt4: full: {gpt4['full']}, split: {gpt4['split']}")

        gpt4o_full.append(gpt4o['full'])
        gpt4o_split.append(gpt4o['split'])
        gpt4_full.append(gpt4['full'])
        gpt4_split.append(gpt4['split'])
        totals.append(total)
    
    x = np.linspace(2, 11, 10)
    plt.plot(x, gpt4o_full, label='G4o-L')
    plt.plot(x, gpt4o_split, label='G4o-S')
    plt.plot(x, gpt4_full, label='G4-L')
    plt.plot(x, gpt4_split, label='G4-S')
    plt.plot(x, totals, label='Total')
    
    plt.xlabel('Token Size')
    plt.ylabel('Count')
    plt.title('Score 5 Sentence Count Over Token Size')
    plt.legend()
    
    # plt.show()
    plt.savefig('imgs/token_size_score_5.png', dpi=200)

if __name__ == "__main__":
    generate_csv()    
    evaluate_csv()
    evalute_only_all_appear()
    draw_score_figure()
    
    
    