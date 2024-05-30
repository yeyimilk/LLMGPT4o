
from utils.gpt_utils import get_batch_list, extract_content_from_gpt_response
import numpy as np
from data_loader import load_top_k_chinese_per_length
import jieba

if __name__ == "__main__":
    models = ['gpt-4-turbo-2024-04-09', 'gpt-4o-2024-05-13']
    for m in models:
        
        top_k_chinese = load_top_k_chinese_per_length()
        
        data = get_batch_list(f'data/sentence/{m}_full_split_words_batch_requests_output.jsonl')
        results = []
        full_count = 0
        split_count = 0
        for d in data:
            customed_id, content = extract_content_from_gpt_response(d, 'sentence')
            index_id = customed_id.split('_')[0]
            word_type = customed_id.split('_')[1]
            
            full_word = top_k_chinese[index_id]
            full_word = full_word.replace(' ', '')
            seg_list = jieba.cut(full_word, cut_all=False)

            if word_type == 'full':
                full_count += 1 if full_word in content else 0
            else:
                all_true = all([seg in content for seg in seg_list if seg != ''])
                split_count += 1 if all_true else 0
        
        size = len(data)
        print(f"Model: {m}")
        print(f"Size: {size}, Full: {full_count}, {full_count/size:.4f}, Split: {split_count}, {split_count/size :.4f}\n")