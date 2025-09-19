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
import time
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

class State(TypedDict):
    messages: Annotated[list, add_messages]
    hadm_id: str
    patient_info: dict
    ab_hpi: Union[str, AIMessage]
    ab_pe: Union[str, AIMessage]
    ab_lab: Union[str, AIMessage]
    ab_ima: Union[str, AIMessage]

class abnormal_findings_format(BaseModel):
    basic_info: str = Field(..., description='e.g. age, sex, imaging modality, etc.')
    findings: str = Field(..., description='key abnormal information, e.g. RLQ Pain, gallbladder wall thickening, etc.')

class diagnostic_process_format(BaseModel):
    diagnostic_pathway: str = Field(..., description='the path that the pathology established or excluded')
    evidence: str = Field(..., description='evidence supporting the diagnosis according to the diagnostic criteria')

class NodesAndEdeges:
    def __init__(self, nodes, edges, prompt_mode='lenient'):
        self.diagnostic_criteria = self.load_diagnosis_creteria()
        self.prompt_mode = prompt_mode
        self.prompts = Prompts(mode=prompt_mode)
        self.init_agents()
        self.init_trustcall_llm()
        self.convert_tools_as_node()
        self.action_construct_info_extractor_chain()
        self.action_construct_diagnosis_eli_chain()
        self.nodes_mapping = {
            'START': START, 
            'END': END, 
            'Brain': None, 
            'tool_node_load_patient_info': self.load_patient_info, 
            'agent_node_hpi': self.extract_abnormal_hpi, 
            'agent_node_pe': self.extract_abnormal_pe, 
            'agent_node_lab': self.extract_abnormal_lab, 
            'agent_node_imaging': self.extract_abnormal_imaging,
            'tool_node_eliminate_appendicitis': self.eliminate_appendicitis, 
            'tool_node_eliminate_cholecystitis': self.eliminate_cholecystitis, 
            'tool_node_eliminate_pancreatitis': self.eliminate_pancreatitis, 
            'tool_node_eliminate_diverticulitis': self.eliminate_diverticulitis, 
            'tool_node_action_empty_0': self.action_empty, 
            'tool_review': self.tool_node_review, 
        }
        self.no_added_nodes = {
            'START': START, 
            'END': END, 
            'Brain': None,
        }
        self.graph_init(nodes, edges)

    def graph_init(self, nodes, edges):
        self.graph_builder = StateGraph(State)
        self.brain_with_tool = self.agent_brain.bind_tools([review_tool])
        self.graph_builder.add_node('Brain', self.brain)
        for node in nodes:
            if node not in self.no_added_nodes.keys():
                self.graph_builder.add_node(node, self.nodes_mapping[node])
        for head, tails in edges.items():
            if isinstance(head, (tuple)) and isinstance(tails, list):
                self.graph_builder.add_edge(list(head), tails)
            elif isinstance(tails, str):
                if tails == 'tool_review':
                    self.graph_builder.add_conditional_edges(
                        'Brain', 
                        self.tools_condition,
                        {
                            'tool_review': 'tool_review',
                            '__end__': END
                        }
                    )
                    self.graph_builder.add_edge('tool_review', 'Brain')
                elif head == 'START': 
                    self.graph_builder.add_edge(START, tails)
                else:
                    self.graph_builder.add_edge(head, tails)
            else:
                for tail in tails:
                    self.graph_builder.add_edge(head, tail)

        memory = MemorySaver()
        self.graph = self.graph_builder.compile(checkpointer=memory)
        return

    def tools_condition(self,state: State):
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get('messages', []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tool_review"
        return '__end__'

    def init_agents(self):
        self.agent_ex_hpi = self.ins_llm()
        self.agent_ex_pe = self.ins_llm()
        self.agent_ex_lab = self.ins_llm()
        self.agent_ex_ima = self.ins_llm()
        self.agent_diag_app = self.ins_llm()
        self.agent_diag_cho = self.ins_llm()
        self.agent_diag_pan = self.ins_llm()
        self.agent_diag_div = self.ins_llm()
        self.agent_brain = self.ins_llm()
        self.agent_str_ab = self.ins_llm()
        self.agent_str_diag = self.ins_llm()
        return 

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
    
    def init_trustcall_llm(self):
        self.tc_ab_hpi = create_extractor(
            llm=self.agent_str_ab,
            tools=[abnormal_findings_format]
        )
        self.tc_ab_pe = create_extractor(
            llm=self.agent_str_ab,
            tools=[abnormal_findings_format]
        )
        self.tc_ab_lab = create_extractor(
            llm=self.agent_str_ab,
            tools=[abnormal_findings_format]
        )
        self.tc_ab_ima = create_extractor(
            llm=self.agent_str_ab,
            tools=[abnormal_findings_format]
        )
        self.tc_diag_app = create_extractor(
            llm=self.agent_str_diag,
            tools=[diagnostic_process_format]
        )
        self.tc_diag_cho = create_extractor(
            llm=self.agent_str_diag,
            tools=[diagnostic_process_format]
        )
        self.tc_diag_pan = create_extractor(
            llm=self.agent_str_diag,
            tools=[diagnostic_process_format]
        )
        self.tc_diag_div = create_extractor(
            llm=self.agent_str_diag,
            tools=[diagnostic_process_format]
        )
        return

    def load_diagnosis_creteria(self):
        appendicitis_diag_criteria = open('Appendicitis_diagnostic_criteria.txt', 'r', encoding='utf-8').read()
        cholecystitis_diag_criteria = open('Cholecystitis_diagnostic_criteria.txt', 'r', encoding='utf-8').read()
        pancreatitis_diag_criteria = open('Pancreatitis_diagnostic_criteria.txt', 'r', encoding='utf-8').read()
        diverticulitis_diag_criteria = open('Diverticulitis_diagnostic_criteria.txt', 'r', encoding='utf-8').read()
        diag_diff = open('diag_differ.txt', 'r', encoding='utf-8').read()
        return {
            'appendicitis_diag_criteria': appendicitis_diag_criteria,
            'cholecystitis_diag_criteria': cholecystitis_diag_criteria,
            'pancreatitis_diag_criteria': pancreatitis_diag_criteria,
            'diverticulitis_diag_criteria': diverticulitis_diag_criteria,
            'differential_diagnosis': diag_diff
        }

    def action_construct_info_extractor_chain(self):
        self.history_chain = self.prompts.history_prompt | self.agent_ex_hpi
        self.physical_exam_chain = self.prompts.physical_exam_prompt | self.agent_ex_pe
        self.lab_results_chain = self.prompts.lab_results_prompt | self.agent_ex_lab
        self.radiology_chain = self.prompts.radiology_prompt | self.agent_ex_ima

    def action_construct_diagnosis_eli_chain(self):
        self.eliminate_appendicitis_chain = self.prompts.eliminate_appendicitis_prompt | self.agent_diag_app
        self.eliminate_cholecystitis_chain = self.prompts.eliminate_cholecystitis_prompt | self.agent_diag_cho
        self.eliminate_pancreatitis_chain = self.prompts.eliminate_pancreatitis_prompt | self.agent_diag_pan
        self.eliminate_diverticulitis_chain = self.prompts.eliminate_diverticulitis_prompt | self.agent_diag_div

    def load_patient_info(self, state: State) -> dict:
        """Retrive patient info using hadm_id"""
        patient_info = Template_customized(
            base_path=patient_info_path,
            file_names=patient_info_file_names
        )
        hadm_id = state['hadm_id']
        return {
            'patient_info': {
            'hpi': patient_info.extract_hpi(hadm_id),
            'pe': patient_info.extract_pe(hadm_id),
            'lab': patient_info.laboratory_test_mapping(hadm_id),
            'imaging': patient_info.extract_rr(hadm_id)
            }
        }
    
    def extract_abnormal_hpi(self, state: State):
        """extract abnormal findings from history of present illness"""
        # return {
        #     'ab_hpi': [self.history_chain.invoke({
        #         'hpi': [HumanMessage(content=state['patient_info']['hpi'])]
        #     }).content],
        #     }
        return {
            'ab_hpi': [self.history_chain.invoke({
                'hpi': [HumanMessage(content=state['patient_info']['hpi'])]
            })],
            }

    def extract_abnormal_pe(self, state: State):
        """extract abnormal findings from physical examination"""
        # return {
        #     'ab_pe': [self.physical_exam_chain.invoke(
        #         {'pe': [HumanMessage(content=state['patient_info']['pe'])]}).content],
        # }
        return {
            'ab_pe': [self.physical_exam_chain.invoke(
                {'pe': [HumanMessage(content=state['patient_info']['pe'])]})],
        }

    def extract_abnormal_lab(self, state: State):
        """extract abnormal findings from laboratory tests"""
        # return {'ab_lab': [self.lab_results_chain.invoke({
        #     'lab': [HumanMessage(content=state['patient_info']['lab'])],
        # }).content],
        # }
        return {'ab_lab': [self.lab_results_chain.invoke({
            'lab': [HumanMessage(content=state['patient_info']['lab'])]})],
        }
    
    def extract_abnormal_imaging(self, state: State):
        """extract abnormal findings from radiology reports"""
        # return {
        #     'ab_ima': [self.radiology_chain.invoke({'imaging': [HumanMessage(content=state['patient_info']['imaging'])],
        # }).content],
        # }
        return {
            'ab_ima': [self.radiology_chain.invoke(
                {'imaging': [HumanMessage(content=state['patient_info']['imaging'])],
        })],
        }
    
    def eliminate_appendicitis(self, state: State):
        """establish or exclude the diagnosis of appendicitis"""
        if self.prompt_mode != 'own':
            return {'messages': [
                self.eliminate_appendicitis_chain.invoke({
                    'appendicitis_diag_criteria': [HumanMessage(content=self.diagnostic_criteria['appendicitis_diag_criteria'])], 
                    'ab_hpi': [HumanMessage(content=state['ab_hpi'][0])],
                    'ab_pe': [HumanMessage(content=state['ab_pe'][0])], 
                    'ab_lab': [HumanMessage(content=state['ab_lab'][0])], 
                    'ab_ima': [HumanMessage(content=state['ab_ima'][0])], 
                })
            ]}
        elif self.prompt_mode == 'own':
            return {'messages': [
                self.eliminate_appendicitis_chain.invoke({
                    'ab_hpi': [HumanMessage(content=state['ab_hpi'][0].content)],
                    'ab_pe': [HumanMessage(content=state['ab_pe'][0].content)], 
                    'ab_lab': [HumanMessage(content=state['ab_lab'][0].content)], 
                    'ab_ima': [HumanMessage(content=state['ab_ima'][0].content)], 
                })
            ]}
    
    def eliminate_cholecystitis(self, state: State):
        """establish or exclude the diagnosis of cholecystitis"""
        if self.prompt_mode != 'own':
            return {'messages': [
                self.eliminate_cholecystitis_chain.invoke({
                    'cholecystitis_diag_criteria': [HumanMessage(content=self.diagnostic_criteria['cholecystitis_diag_criteria'])], 
                    'ab_hpi': [HumanMessage(content=state['ab_hpi'][0])],
                    'ab_pe': [HumanMessage(content=state['ab_pe'][0])], 
                    'ab_lab': [HumanMessage(content=state['ab_lab'][0])], 
                    'ab_ima': [HumanMessage(content=state['ab_ima'][0])], 
                })
            ]}
        elif self.prompt_mode == 'own':
            return {'messages': [
                self.eliminate_cholecystitis_chain.invoke({
                    'ab_hpi': [HumanMessage(content=state['ab_hpi'][0].content)],
                    'ab_pe': [HumanMessage(content=state['ab_pe'][0].content)], 
                    'ab_lab': [HumanMessage(content=state['ab_lab'][0].content)], 
                    'ab_ima': [HumanMessage(content=state['ab_ima'][0].content)], 
                })
            ]}
    
    def eliminate_pancreatitis(self, state: State):
        """establish or exclude the diagnosis of pancreatitis"""
        if self.prompt_mode != 'own':
            return {'messages': [
                self.eliminate_pancreatitis_chain.invoke({
                    'pancreatitis_diag_criteria': [HumanMessage(content=self.diagnostic_criteria['pancreatitis_diag_criteria'])], 
                    'ab_hpi': [HumanMessage(content=state['ab_hpi'][0])],
                    'ab_pe': [HumanMessage(content=state['ab_pe'][0])], 
                    'ab_lab': [HumanMessage(content=state['ab_lab'][0])], 
                    'ab_ima': [HumanMessage(content=state['ab_ima'][0])], 
                })
            ]}
        elif self.prompt_mode == 'own':
            return {'messages': [
                self.eliminate_pancreatitis_chain.invoke({ 
                    'ab_hpi': [HumanMessage(content=state['ab_hpi'][0].content)],
                    'ab_pe': [HumanMessage(content=state['ab_pe'][0].content)], 
                    'ab_lab': [HumanMessage(content=state['ab_lab'][0].content)], 
                    'ab_ima': [HumanMessage(content=state['ab_ima'][0].content)], 
                })
            ]}
    
    def eliminate_diverticulitis(self, state: State):
        """establish or exclude the diagnosis of diverticulitis"""
        if self.prompt_mode != 'own':
            return {'messages': [
                self.eliminate_diverticulitis_chain.invoke({
                    'diverticulitis_diag_criteria': [HumanMessage(content=self.diagnostic_criteria['diverticulitis_diag_criteria'])], 
                    'ab_hpi': [HumanMessage(content=state['ab_hpi'][0])],
                    'ab_pe': [HumanMessage(content=state['ab_pe'][0])], 
                    'ab_lab': [HumanMessage(content=state['ab_lab'][0])], 
                    'ab_ima': [HumanMessage(content=state['ab_ima'][0])], 
                })
            ]}
        elif self.prompt_mode == 'own':
            return {'messages': [
                self.eliminate_diverticulitis_chain.invoke({ 
                    'ab_hpi': [HumanMessage(content=state['ab_hpi'][0].content)],
                    'ab_pe': [HumanMessage(content=state['ab_pe'][0].content)], 
                    'ab_lab': [HumanMessage(content=state['ab_lab'][0].content)], 
                    'ab_ima': [HumanMessage(content=state['ab_ima'][0].content)], 
                })
            ]}

    def str_ab_hpi(self, state: State):
        result = self.tc_ab_hpi.invoke({'messages': [state['ab_hpi']]})
        return {
            'messages': result["responses"][0]
        }
    
    def str_ab_pe(self, state: State):
        result = self.tc_ab_pe.invoke({'messages': [state['ab_pe']]})
        return {
            'messages': result["responses"][0]
        }

    def str_ab_lab(self, state: State):
        result = self.tc_ab_lab.invoke({'messages': [state['ab_lab']]})
        return {
            'messages': result["responses"][0]
        }

    def str_ab_ima(self, state: State):
        result = self.tc_ab_ima.invoke({'messages': [state['ab_ima']]})
        return {
            'messages': result["responses"][0]
        }

    def str_diag_app(self, state: State):
        result = self.tc_diag_app.invoke({'messages': [state['diag_app']]})
        return {
            'messages': result["responses"][0]
        }
    
    def str_diag_cho(self, state: State):
        result = self.tc_diag_cho.invoke({'messages': [state['diag_cho']]})
        return {
            'messages': result["responses"][0]
        }
    
    def str_diag_pan(self, state: State):
        result = self.tc_diag_pan.invoke({'messages': [state['diag_pan']]})
        return {
            'messages': result["responses"][0]
        }

    def str_diag_div(self, state: State):
        result = self.tc_diag_div.invoke({'messages': [state['diag_div']]})
        return {
            'messages': result["responses"][0]
        }

    def action_empty(self, state: State):
        """empty action"""
        return {
            'messages': [HumanMessage(content='waiting parallel processing...')]
        }

    def brain(self, state: State):
        state['messages'].append(self.prompts.review_prompt)
        return {
            'messages': self.brain_with_tool.invoke(state['messages'])
        }

    def convert_tools_as_node(self):
        self.tool_node_review = ToolNode(tools=[review_tool], name='tool_review')


@tool
def review_tool(diagnosis_condition: str):
    "return differential_diagnosis criteria for finally formulating diagnosis"
    diag_diff = open('diag_differ.txt', 'r', encoding='utf-8').read()
    return {'messages': [HumanMessage(content=diag_diff)]}

class AgentGraph:
    def __init__(self, nodes: list, edges: dict, prompt_mode='lenient'):
        """
        Args:
            nodes: the nodes in the graph, the node name of the node and the function name of the node
        """
        self.prompt_mode = prompt_mode
        self.nodes_and_edges = NodesAndEdeges(nodes, edges, prompt_mode=self.prompt_mode)

    def display(self):
        save_path = '/data/luzhenyang/project/agent_graph_diag/graph_without_stru.png'
        try:
            graph = self.nodes_and_edges.graph.get_graph()
            output_path = "graph.png"  # 可自定义路径

            # 使用 Graphviz 渲染为 PNG
            graph.draw_png(output_file_path=save_path)

            print(f"✅ Graph saved to {save_path}")

        except ImportError:
            print(
                "❌ You need to install pygraphviz and Graphviz system dependencies.\n"
                "See: https://github.com/pygraphviz/pygraphviz/blob/main/INSTALL.txt"
            )
        except Exception as e:
            print(f"❌ Failed to generate or save graph: {e}")

    def run(self, hadm_id: int): 
        config = {'configurable': {'thread_id': "1"}}
        for event in self.nodes_and_edges.graph.stream(
            {
                'messages': [HumanMessage(content='graph start...')],
                'hadm_id': hadm_id,
                'diagnostic_criteria': self.nodes_and_edges.diagnostic_criteria,
            },
            config=config,
            stream_mode='values'
        ):
            if "messages" in event:
                event["messages"][-1].pretty_print()
    
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
        output_file_path = f'/media/luzhenyang/project/agent_graph_diag/results/agent_graph_subset_{seed}_{self.prompt_mode}.csv'
        fieldnames = [
            'hadm_id', 'diagnosis', 
            'ab_hpi', 'ab_pe', 'ab_lab', 'ab_ima', 
            'diag_app', 'diag_cho', 'diag_div', 'diag_pan', 
            'final_review'
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

        thread_id = -1
        # hadm_ids = [int(x) for x in subset_df['hadm_id'].values]
        for hadm_id in tqdm(subset_df['hadm_id'].astype(int).values, desc='processing', total=len(subset_df['hadm_id'].values), unit='pid'):
            if hadm_id in existing_hadm_ids:
                continue
            hadm_id_pyint = int(hadm_id) # for seria
            thread_id += 1
            config = {'configurable': {'thread_id': f"{thread_id}"}}
            result = self.nodes_and_edges.graph.invoke(
                input={
                    'hadm_id': hadm_id_pyint,
                    'diagnostic_criteria': self.nodes_and_edges.diagnostic_criteria,
                },
                config=config,
            )
            with open(output_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                tool_called_sign = False
                for message in result['messages']:
                    try:
                        tool_calls = message.tool_calls
                        if tool_calls:
                            tool_called_sign = True
                    except AttributeError:
                        pass

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if tool_called_sign:    # called tool
                    print('TOOL_CALLED')
                    writer.writerow({
                        'hadm_id': hadm_id_pyint, # ensure save as python int
                        'diagnosis': subset_df[ subset_df['hadm_id'] == hadm_id ]['pathology'].iloc[0], 
                        'ab_hpi': result['ab_hpi'], 
                        'ab_pe': result['ab_pe'], 
                        'ab_lab': result['ab_lab'], 
                        'ab_ima': result['ab_ima'], 
                        'final_review': result['messages'][-1].content, 
                        'diag_pan': result['messages'][-7].content, 
                        'diag_div': result['messages'][-8].content, 
                        'diag_cho': result['messages'][-9].content, 
                        'diag_app': result['messages'][-10].content, 
                    })
                else:
                    writer.writerow({
                        'hadm_id': hadm_id_pyint, # ensure save as python int
                        'diagnosis': subset_df[ subset_df['hadm_id'] == hadm_id ]['pathology'].iloc[0], 
                        'ab_hpi': result['ab_hpi'], 
                        'ab_pe': result['ab_pe'], 
                        'ab_lab': result['ab_lab'], 
                        'ab_ima': result['ab_ima'], 
                        'final_review': result['messages'][-1].content, 
                        'diag_pan': result['messages'][-3].content, 
                        'diag_div': result['messages'][-4].content, 
                        'diag_cho': result['messages'][-5].content, 
                        'diag_app': result['messages'][-6].content, 
                    })

    def run_full_dataset(self):
        # input process
        pathology_ids = '/media/luzhenyang/project/datasets/mimic_iv_ext_clinical_decision_abdominal/clinical_decision_making_for_abdominal_pathologies_1.1/pathology_ids.json'
        with open(pathology_ids, 'r', encoding='utf-8') as f:
            diseases_to_patients = json.load(f)
        rows = []
        for disease, ids in diseases_to_patients.items():
            for id in ids:
                rows.append({'pathology': disease, 'hadm_id': id})
        df = pd.DataFrame(rows)
        # subset_df = df.groupby('pathology').apply(
        #     lambda x: x.sample(n=20, random_state=seed) if len(x) >= 20 else x
        # ).reset_index(drop=True)
        # subset_df.to_csv(f'subset_ids_{seed}.csv', index=False, encoding='utf-8')

        # output process
        output_file_path = f'/media/luzhenyang/project/agent_graph_diag/AGAP_full/AGAP_full_dataset_{self.prompt_mode}.csv'
        fieldnames = [
            'hadm_id', 'diagnosis', 
            'ab_hpi', 'ab_pe', 'ab_lab', 'ab_ima', 
            'diag_app', 'diag_cho', 'diag_div', 'diag_pan', 
            'final_review', 'other_process',
            'input_tokens', 'output_tokens', 'total_tokens',
            'timetaken', 'tool_called', 
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

        thread_id = -1
        # hadm_ids = [int(x) for x in subset_df['hadm_id'].values]
        for hadm_id in tqdm(df['hadm_id'].astype(int).values, desc='processing', total=len(df['hadm_id'].values), unit='pid'):
            if hadm_id in existing_hadm_ids:
                continue
            hadm_id_pyint = int(hadm_id) # for seria
            thread_id += 1
            config = {'configurable': {'thread_id': f"{thread_id}"}}
            # 计时
            time_start = time.perf_counter()
            result = self.nodes_and_edges.graph.invoke(
                input={
                    'hadm_id': hadm_id_pyint,
                    'diagnostic_criteria': self.nodes_and_edges.diagnostic_criteria,
                },
                config=config,
            )
            time_end = time.perf_counter()
            time_taken = time_end - time_start
            # 结束⌛️
            with open(output_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                tool_called_sign = False
                for message in result['messages']:
                    try:
                        tool_calls = message.tool_calls
                        if tool_calls:
                            tool_called_sign = True
                    except AttributeError:
                        pass

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if tool_called_sign:    # called tool
                    print('TOOL_CALLED')
                    input_tokens_acc, output_tokens_acc, total_tokens_acc = 0, 0, 0
                    for ab_ in ['ab_hpi', 'ab_pe', 'ab_lab', 'ab_ima']:
                        msg = result[ab_]
                        input_tokens_acc += msg[0].usage_metadata['input_tokens'] # ['prompt_tokens']
                        output_tokens_acc += msg[0].usage_metadata['output_tokens'] # response_metadata completion_tokens
                        total_tokens_acc += msg[0].usage_metadata['total_tokens'] # total_tokens
                    for epo in result['messages']:
                        if isinstance(epo, AIMessage):
                            input_tokens_acc += epo.usage_metadata['input_tokens']
                            output_tokens_acc += epo.usage_metadata['output_tokens']
                            total_tokens_acc += epo.usage_metadata['total_tokens']

                    saved_indices = [-1, -7, -8, -9, -10]
                    all_msgs = result['messages']
                    N = len(all_msgs)

                    # 将负索引转换为对应的正索引（防止越界）
                    saved_pos_indices = {N + idx if idx < 0 else idx for idx in saved_indices}

                    # 保留所有不在 saved_pos_indices 中的消息
                    other_process = [
                        all_msgs[i] for i in range(N)
                        if i not in saved_pos_indices
                    ]

                    writer.writerow({
                        'hadm_id': hadm_id_pyint, # ensure save as python int
                        'diagnosis': df[ df['hadm_id'] == hadm_id ]['pathology'].iloc[0], 
                        'ab_hpi': result['ab_hpi'], 
                        'ab_pe': result['ab_pe'], 
                        'ab_lab': result['ab_lab'], 
                        'ab_ima': result['ab_ima'], 
                        'final_review': all_msgs[-1].content, 
                        'diag_pan': all_msgs[-7].content, 
                        'diag_div': all_msgs[-8].content, 
                        'diag_cho': all_msgs[-9].content, 
                        'diag_app': all_msgs[-10].content, 
                        'other_process': other_process,
                        'input_tokens': input_tokens_acc, # TODO 补上ab_的tokens
                        'output_tokens': output_tokens_acc,
                        'total_tokens': total_tokens_acc, 
                        'timetaken': round(time_taken, 2),
                        'tool_called': tool_called_sign
                    })
                else:
                    input_tokens_acc, output_tokens_acc, total_tokens_acc = 0, 0, 0
                    for ab_ in ['ab_hpi', 'ab_pe', 'ab_lab', 'ab_ima']:
                        msg = result[ab_]
                        input_tokens_acc += msg[0].usage_metadata['input_tokens'] # ['prompt_tokens']
                        output_tokens_acc += msg[0].usage_metadata['output_tokens'] # response_metadata completion_tokens
                        total_tokens_acc += msg[0].usage_metadata['total_tokens'] # total_tokens
                    for epo in result['messages']:
                        if isinstance(epo, AIMessage):
                            input_tokens_acc += epo.usage_metadata['input_tokens']
                            output_tokens_acc += epo.usage_metadata['output_tokens']
                            total_tokens_acc += epo.usage_metadata['total_tokens']

                    saved_indices = [-1, -3, -4, -5, -6]
                    all_msgs = result['messages']
                    N = len(all_msgs)

                    # 将负索引转换为对应的正索引（防止越界）
                    saved_pos_indices = {N + idx if idx < 0 else idx for idx in saved_indices}

                    # 保留所有不在 saved_pos_indices 中的消息
                    other_process = [
                        all_msgs[i] for i in range(N)
                        if i not in saved_pos_indices
                    ]

                    writer.writerow({
                        'hadm_id': hadm_id_pyint, # ensure save as python int
                        'diagnosis': df[ df['hadm_id'] == hadm_id ]['pathology'].iloc[0], 
                        'ab_hpi': result['ab_hpi'], 
                        'ab_pe': result['ab_pe'], 
                        'ab_lab': result['ab_lab'], 
                        'ab_ima': result['ab_ima'], 
                        'final_review': result['messages'][-1].content, 
                        'diag_pan': result['messages'][-3].content, 
                        'diag_div': result['messages'][-4].content, 
                        'diag_cho': result['messages'][-5].content, 
                        'diag_app': result['messages'][-6].content, 
                        'other_process': other_process,
                        'input_tokens': input_tokens_acc,
                        'output_tokens': output_tokens_acc,
                        'total_tokens': total_tokens_acc, 
                        'timetaken': round(time_taken, 2), 
                        'tool_called': tool_called_sign
                    })
        

def main():
    nodes = [
        'START', 'END', 
        'tool_node_load_patient_info', 'agent_node_hpi', 'agent_node_pe', 'agent_node_lab', 'agent_node_imaging', 
        'tool_node_action_empty_0', 
        # 'str_ab_hpi', 'str_ab_pe', 'str_ab_lab', 'str_ab_ima', 
        'tool_node_eliminate_appendicitis', 'tool_node_eliminate_cholecystitis', 
        'tool_node_eliminate_pancreatitis', 'tool_node_eliminate_diverticulitis', 
        # 'str_diag_app', 'str_diag_cho', 'str_diag_pan', 'str_diag_div',
        'tool_review', 'Brain',
    ] 
    edges = {
        'START': 'tool_node_load_patient_info', 
        'tool_node_load_patient_info': ['agent_node_hpi', 'agent_node_pe', 'agent_node_lab', 
                                        'agent_node_imaging'], 
        'agent_node_hpi': 'tool_node_action_empty_0', 
        'agent_node_pe': 'tool_node_action_empty_0', 
        'agent_node_lab': 'tool_node_action_empty_0', 
        'agent_node_imaging': 'tool_node_action_empty_0', 
        'tool_node_action_empty_0': ['tool_node_eliminate_appendicitis', 'tool_node_eliminate_cholecystitis', 
                                     'tool_node_eliminate_pancreatitis', 'tool_node_eliminate_diverticulitis'], 
        'tool_node_eliminate_appendicitis': 'Brain', 
        'tool_node_eliminate_cholecystitis': 'Brain', 
        'tool_node_eliminate_pancreatitis': 'Brain', 
        'tool_node_eliminate_diverticulitis': 'Brain', 
        'Brain': 'tool_review',
    }
    graph = AgentGraph(nodes=nodes, edges=edges, prompt_mode='own')
    graph.run_full_dataset()

    # for ty_prompt in ['lenient', 'own', 'strict']:
    #     graph = AgentGraph(nodes=nodes, edges=edges, prompt_mode=ty_prompt)
    #     for i in [20, 23, 9, 7, 1]:       # [10, 4, 96, 42, 71]
    #         print(f"now  working on seed:{i}, type:{ty_prompt}")
    #         graph.run_subset(seed=i)

    # res = graph.run(int(25841363))
    # print(res)
    # graph.display()

if __name__ == "__main__":
    main()


# 带结构化输出的graph，GPT4.1可以准确遵循指令，不输出无关内容，所以无需结构化输出
# edges = {
#         'START': 'tool_node_load_patient_info', 
#         'tool_node_load_patient_info': ['agent_node_hpi', 'agent_node_pe', 'agent_node_lab', 
#                                         'agent_node_imaging'], 
#         'agent_node_hpi': 'str_ab_hpi', 
#         'agent_node_pe': 'str_ab_pe', 
#         'agent_node_lab': 'str_ab_lab', 
#         'agent_node_imaging': 'str_ab_ima',
#         ('str_ab_hpi', 'str_ab_pe', 'str_ab_lab', 'str_ab_ima'): 'tool_node_action_empty_0', 
#         'tool_node_action_empty_0': ['tool_node_eliminate_appendicitis', 'tool_node_eliminate_cholecystitis', 
#                                      'tool_node_eliminate_pancreatitis', 'tool_node_eliminate_diverticulitis'], 
#         'tool_node_eliminate_appendicitis': 'str_diag_app', 
#         'tool_node_eliminate_cholecystitis': 'str_diag_cho', 
#         'tool_node_eliminate_pancreatitis': 'str_diag_pan', 
#         'tool_node_eliminate_diverticulitis': 'str_diag_div', 
#         ('str_diag_cho', 'str_diag_app', 'str_diag_pan', 'str_diag_div'): 'Brain',
#         'Brain': 'tool_review',
#     }























