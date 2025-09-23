# Structuring Large Language Models as Agentic Workflows Improves Diagnostic Performance and Physician Assistance

This repository contains the code accompanying the paper:  
**"Structuring Large Language Models as Agentic Workflows Improves Diagnostic Performance and Physician Assistance"**.  

---

## Requirements
The code is released under an open-source license and enables full replication of our analyses once access to the MIMIC-IV-CDM dataset has been obtained. To reproduce the experiments, two requirements must be fulfilled: 
  1.Dataset access: Users must obtain independent access to the MIMIC-IV-CDM dataset via PhysioNet, in compliance with the data use agreement. 
  2.Cloud service configuration: Some components of our workflow require connecting to a private cloud service API in order to enable large language model (LLM) interaction with the MIMIC-IV-CDM dataset. This API service must adhere to the PhysioNet data use policy, including disabling any content retention, logging, or review functions that would result in secondary storage or inspection of protected data.
---

## Main Components

### Agentic Diagnostic Workflow (ADW)
- The main implementation of ADW is in **`AGAP/auto_diagnosis.py`**.
- **`subset_ids_1.csv`**  is one of the sampled case id subset for reproducing the experiment easier.
- At **lines 699–700**, the workflow is initialized and executed:  
  ```python
  graph = AgentGraph(nodes=nodes, edges=edges, prompt_mode='strict')
  graph.run_subset(seed=1)
  ```
- The parameter **`prompt_mode`** controls the prompting variant. Available options:  
  - `'strict'`  
  - `'lenient'`  
  - `'own'`  

### Full Dataset Execution
- **`AGAP_own_full_dataset.py`**: Runs ADW on the full dataset with the specified settings.
- **`prompts.py`**: All the LLM method prompts are here.

### Baseline Comparisons
- **`diagnosis_with_full_information.py`**: Runs Chain-of-Thought (CoT) and Vanilla Prompting baselines.  

### Deep Learning Baselines
- **`lm_classification/`**: Contains code for **Clinical Longformer** and **Longformer** baselines.  
- **`data_processing.ipynb`**: Handles preprocessing of the dataset for classification models.  

### Human-in-the-Loop Experiment
- First, set up the data in path: /Human_Examination/data, detailed see file: streamlit_app.spec(or streamlit_wo_ADW_app.spec).
- Second, install the application with env ins_human_interface: pyinstaller streamlit_app.spec

---

## Citation
If you use this code, please cite our paper:  
```
@article{,
  title={Structuring Large Language Models as Agentic Workflows Improves Diagnostic Performance and Physician Assistance},
  author={...},
  journal={...},
  year={2025}
}
