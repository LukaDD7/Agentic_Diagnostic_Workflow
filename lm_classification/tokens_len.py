import os
import sys
import json

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import LongformerTokenizer, LongformerModel
from langchain.schema import AIMessage

module_path = '/media/luzhenyang/project/agent_graph_diag'
sys.path.append(module_path)
from project.agent_graph_diag.AGAP.extract_patient_info import Template_customized

model_name = "yikuan8/Clinical-Longformer"
tokenizer = LongformerTokenizer.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer") # allenai/longformer-base-4096

patient_info_path = '/media/luzhenyang/project/datasets/mimic_iv_ext_clinical_decision_abdominal/clinical_decision_making_for_abdominal_pathologies_1.1'
patient_info_file_names = [
    'history_of_present_illness.csv', 
    'microbiology.csv',
    'laboratory_tests.csv',
    'radiology_reports.csv',
]

patient_info = Template_customized(
    base_path=patient_info_path,
    file_names=patient_info_file_names
)

# 随机抽样计算平均tokens数量
pathology_ids = '/media/luzhenyang/project/datasets/mimic_iv_ext_clinical_decision_abdominal/clinical_decision_making_for_abdominal_pathologies_1.1/pathology_ids.json'
with open(pathology_ids, 'r', encoding='utf-8') as f:
    diseases_to_patients = json.load(f)
rows = []
for disease, ids in diseases_to_patients.items():
    for id in ids:
        rows.append({'pathology': disease, 'hadm_id': id})
df = pd.DataFrame(rows)
subset_df = df.groupby('pathology').apply(
    lambda x: x.sample(n=250, random_state=10) if len(x) >= 250 else x
).reset_index(drop=True)


# 读取GPT-4.1处理数据过程中的ab_，减少上下文token，并且提供更有益于分类的text
ab_context = pd.read_csv('/media/luzhenyang/project/agent_graph_diag/AGAP_full/AGAP_full_dataset_own.csv')

tokens_len = 0
id_len = 0
longest = 0
long_id = None

def convert_AIM(person_context):
    s = eval(person_context)
    if isinstance(s[0], AIMessage):
        return s[0].content
    else:
        print("Error while convert")

def is_abnormal(text, max_len=20000, repeat_threshold=10):
    if len(text) > max_len:
        return True
    segments = text.split('\n')
    if len(segments) == 0:
        return False
    common_line = segments[0]
    repeat_count = sum(1 for seg in segments if seg == common_line)
    return repeat_count > repeat_threshold

for hadm_id in tqdm(subset_df['hadm_id'].astype(int).values, desc='processing', total=len(subset_df['hadm_id'].values), unit='pid'):
    person_context = ab_context[ ab_context['hadm_id'] == hadm_id ]
    # if person_context:
        # print('z')
    ab_hpi = convert_AIM(person_context['ab_hpi'].values[0])

    ab_pe = convert_AIM(person_context['ab_pe'].values[0])
    ab_lab = convert_AIM(person_context['ab_lab'].values[0])
    ab_ima = convert_AIM(person_context['ab_ima'].values[0])

    text = ab_hpi + ab_pe + ab_lab + ab_ima
    if is_abnormal(text):
        print(f"⚠️ 异常文本 detected at sample {hadm_id}")
        continue

    id_len += 1
    encoded_text = tokenizer(text)
    input_ids = encoded_text['input_ids']
    token_len = len(input_ids)
    tokens_len += token_len

    if token_len > longest:
        longest = token_len
        long_id = hadm_id

    tokens_len += len(encoded_text['input_ids'])




print(f"Final mean tokens: ", tokens_len/id_len)
print(f"总tokens: {tokens_len}")
print(f"最长文本的id是 {long_id}，长度为 {longest} tokens")
