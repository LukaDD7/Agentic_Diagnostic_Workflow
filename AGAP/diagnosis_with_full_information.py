from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode # tools_condition
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from torch.utils.data import Dataset, DataLoader
from trustcall import create_extractor

import os
import csv
import json
import pandas as pd
from tqdm import tqdm
from typing import Annotated
from typing_extensions import TypedDict
from typing import Union, Literal, List
from pydantic import BaseModel, Field

from project.agent_graph_diag.AGAP.extract_patient_info import Template_customized, MedicalDataset
from project.agent_graph_diag.AGAP.prompts import Prompts

# global constant
patient_info_path = '/media/luzhenyang/project/datasets/mimic_iv_ext_clinical_decision_abdominal/clinical_decision_making_for_abdominal_pathologies_1.1'
patient_info_file_names = [
            'history_of_present_illness.csv', 
            'microbiology.csv',
            'laboratory_tests.csv',
            'radiology_reports.csv',
        ]

class PlainLlm:
    def __init__(self, prompt_mode='full'):
        self.prompt_mode = prompt_mode
        self.prompts = Prompts(mode=prompt_mode)
        self.agent = self.ins_llm()
        self.chain = self.prompts.full_info_prompt | self.agent

    def load_patient_info(self, hadm_id: int) -> dict:
        """Retrive patient info using hadm_id"""
        patient_info = Template_customized(
            base_path=patient_info_path,
            file_names=patient_info_file_names
        )
        return  {
            'hpi': patient_info.extract_hpi(hadm_id),
            'pe': patient_info.extract_pe(hadm_id),
            'lab': patient_info.laboratory_test_mapping(hadm_id),
            'imaging': patient_info.extract_rr(hadm_id)
        }
    
    def ins_llm(self):
        return AzureChatOpenAI(
            azure_deployment='gpt-4.1',
            api_version='2024-12-01-preview',
            # azure_openai_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            # azure_openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=3,
        )

    def run_subset(self, seed=42):
        # input process
        pathology_ids = '/media/luzhenyang/project/datasets/mimic_iv_ext_clinical_decision_abdominal/clinical_decision_making_for_abdominal_pathologies_1.1/pathology_ids.json'
        with open(pathology_ids, 'r', encoding='utf-8') as f:
            diseases_to_patients = json.load(f)
        rows = []
        for disease, ids in diseases_to_patients.items():
            for id in ids:
                rows.append({'pathology': disease, 'hadm_id': id})
        df = pd.DataFrame(rows)
        subset_df = df.groupby('pathology').apply(
            lambda x: x.sample(n=20, random_state=seed) if len(x) >= 20 else x
        ).reset_index(drop=True)
        subset_df.to_csv(f'subset_ids_{seed}.csv', index=False, encoding='utf-8')

        # output process
        output_file_path = f'/media/luzhenyang/project/agent_graph_diag/results/full_information_subset_{seed}_{self.prompt_mode}.csv'
        fieldnames = [
            'hadm_id', 'diagnosis', 'response'
        ]
        existing_hadm_ids = set()
        if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0:
            with open(output_file_path, 'r', newline='', encoding='utf-8') as csvfile:
                existing_df = pd.read_csv(csvfile)
                existing_hadm_ids = set(existing_df['hadm_id'].astype(int).values)

        if not os.path.exists(output_file_path):
            with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        # hadm_ids = [int(x) for x in subset_df['hadm_id'].values]
        for hadm_id in tqdm(subset_df['hadm_id'].astype(int).values, desc='processing', total=len(subset_df['hadm_id'].values), unit='pid'):
            if hadm_id in existing_hadm_ids:
                continue
            hadm_id_pyint = int(hadm_id)
            patient_info = self.load_patient_info(hadm_id=hadm_id)
            result = self.chain.invoke(
                input={
                    'hpi': [HumanMessage(content=patient_info['hpi'])],
                    'pe': [HumanMessage(content=patient_info['pe'])],
                    'lab': [HumanMessage(content=patient_info['lab'])], 
                    'imaging': [HumanMessage(content=patient_info['imaging'])]
            })
            with open(output_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'hadm_id': hadm_id_pyint, # ensure save as python int
                    'diagnosis': subset_df[ subset_df['hadm_id'] == hadm_id ]['pathology'].iloc[0], 
                    'response': result.content, 
                })


class CotLLM:
    def __init__(self, prompt_mode='cot'):
        self.prompt_mode = prompt_mode
        self.prompts = Prompts(mode=prompt_mode)
        self.agent = self.ins_llm()
        self.chain = self.prompts.cot_prompt | self.agent

    def load_patient_info(self, hadm_id: int) -> dict:
        """Retrive patient info using hadm_id"""
        patient_info = Template_customized(
            base_path=patient_info_path,
            file_names=patient_info_file_names
        )
        return  {
            'hpi': patient_info.extract_hpi(hadm_id),
            'pe': patient_info.extract_pe(hadm_id),
            'lab': patient_info.laboratory_test_mapping(hadm_id),
            'imaging': patient_info.extract_rr(hadm_id)
        }
    
    def ins_llm(self):
        return AzureChatOpenAI(
            azure_deployment='gpt-4.1',
            api_version='2024-12-01-preview',
            # azure_openai_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            # azure_openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=3,
        )

    def run_subset(self, seed=42):
        # input process
        pathology_ids = '/media/luzhenyang/project/datasets/mimic_iv_ext_clinical_decision_abdominal/clinical_decision_making_for_abdominal_pathologies_1.1/pathology_ids.json'
        with open(pathology_ids, 'r', encoding='utf-8') as f:
            diseases_to_patients = json.load(f)
        rows = []
        for disease, ids in diseases_to_patients.items():
            for id in ids:
                rows.append({'pathology': disease, 'hadm_id': id})
        df = pd.DataFrame(rows)
        subset_df = df.groupby('pathology').apply(
            lambda x: x.sample(n=20, random_state=seed) if len(x) >= 20 else x
        ).reset_index(drop=True)
        subset_df.to_csv(f'subset_ids_{seed}.csv', index=False, encoding='utf-8')

        # output process
        output_file_path = f'/media/luzhenyang/project/agent_graph_diag/cot_results/subset_{seed}_{self.prompt_mode}.csv'
        fieldnames = [
            'hadm_id', 'diagnosis', 'response', 'input_tokens', 'output_tokens', 'total_tokens',
        ]
        existing_hadm_ids = set()
        if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0:
            with open(output_file_path, 'r', newline='', encoding='utf-8') as csvfile:
                existing_df = pd.read_csv(csvfile)
                existing_hadm_ids = set(existing_df['hadm_id'].astype(int).values)

        if not os.path.exists(output_file_path):
            with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        # hadm_ids = [int(x) for x in subset_df['hadm_id'].values]
        for hadm_id in tqdm(subset_df['hadm_id'].astype(int).values, desc='processing', total=len(subset_df['hadm_id'].values), unit='pid'):
            if hadm_id in existing_hadm_ids:
                continue
            hadm_id_pyint = int(hadm_id)
            patient_info = self.load_patient_info(hadm_id=hadm_id)
            result = self.chain.invoke(
                input={
                    'hpi': [HumanMessage(content=patient_info['hpi'])],
                    'pe': [HumanMessage(content=patient_info['pe'])],
                    'lab': [HumanMessage(content=patient_info['lab'])], 
                    'imaging': [HumanMessage(content=patient_info['imaging'])]
            })
            with open(output_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'hadm_id': hadm_id_pyint, # ensure save as python int
                    'diagnosis': subset_df[ subset_df['hadm_id'] == hadm_id ]['pathology'].iloc[0], 
                    'response': result.content, 
                    'input_tokens':result.usage_metadata['input_tokens'], 
                    'output_tokens':result.usage_metadata['output_tokens'], 
                    'total_tokens':result.usage_metadata['total_tokens'], 
                })

def main():
    plain_llm = CotLLM(prompt_mode='cot')
    for seed in [71]:  # [20, 23, 9, 7, 1]   [4,10,96,42,71]
        print(f"working on seed:{seed} in CoT way")
        plain_llm.run_subset(seed=seed)

if __name__ == "__main__":
    main()


















