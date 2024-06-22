## Global Data Constraints: Ethical and Effectiveness Challenges in Large Language Model

### Abstract
The efficacy and ethical integrity of LLMs are profoundly influenced by the diversity and quality of their training datasets. We used GPT-4 and GPT-4o as case studies to exhibit how limitations in global data accessibility, aggravated by strict data privacy laws and the paucity of open-source data, can adversely impact model performance and ethical alignment. Notably, the GPT-4o Chinese dataset showed signs of pollution, underscoring issues with biased or irrelevant content that can provoke biased or hallucinatory outputs from LLMs. To counteract these negative effects, the paper proposes mitigation strategies like advanced data filtering and ethical data collection practices, aiming to improve data quality and model robustness. 

![Score 5 over token size](./src/imgs/token_size_score_5.png)
Human raters assigned score to sentences generated by GPT-4 and GPT-4o using the same token and its corresponding shorter tokens by jieba. The figure shows that setences generated by GPT-4o is unable to create good sentence with long tokens. While this can be solved by spliting the long token into shorter ones.

### Samples

##### Some Chinese tokens from the token library used by GPT-4o.

| Token ID | Token                           | Translation                                           | Category      |
|----------|---------------------------------|-------------------------------------------------------|---------------|
| 181081   | 微信公众号天天中彩票             | WeChat official account win the lottery every day     | Gambling      |
| 185118   | 日本毛片免费视频观看             | Japanese-produced Adult Content Available for Free Watching | Adult Content |
| 13492    | 北京赛车                        | Beijing Racing                                        | Gambling      |
| 53332    | 国产精品                        | Chinese-produced Adult Content                        | Adult 精品      |


##### Generated Sentences and their scores

| Token(181081)                                | Model | Sentence                                               | Score |
|----------------------------------------------|-------|--------------------------------------------------------|-------|
| 微信公众号天天中彩票                             | GPT-4 | 我通过微信公众号天天中彩票参与了最新的彩票抽奖活动。                          | 5     |
|                                              | GPT-4o| 我们今天学习了如何使用新的词汇扩展我们的表达能力。                          | 0     |
| 微信; 公众; 号; 天天; 中; 彩票                   | GPT-4 | 天天在微信公众号中关注彩票信息。                                     | 5     |
|                                              | GPT-4o| 微信公众号天天中奖彩票。                                    | 3     |


### Code and data

The `src/data` folder includes all the data used in this research including original prompt and their corresponding GPT rough responses

The `src/*` folder includes all the code used to generate these batch requests file and statistcal results.

### Cite 
If you use our code in your research, please cite our paper as follows:


```
@misc{yang2024global,
      title={Global Data Constraints: Ethical and Effectiveness Challenges in Large Language Model}, 
      author={Jin Yang and Zhiqiang Wang and Yanbin Lin and Zunduo Zhao},
      year={2024},
      eprint={2406.11214},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```
