# Structuring Large Language Models as Agentic Workflows Improves Diagnostic Performance and Physician Assistance

This repository contains the code accompanying the paper:  
**"Structuring Large Language Models as Agentic Workflows Improves Diagnostic Performance and Physician Assistance"**.  

---

## Requirements
To run the code, you need:  
1. [Insert your first requirement from *Code Availability* here]  
2. [Insert your second requirement from *Code Availability* here]  

---

## Main Components

### Agentic Diagnostic Workflow (ADW)
- The main implementation of ADW is in **`AGAP/auto_diagnosis.py`**.  
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

### Baseline Comparisons
- **`diagnosis_with_full_information.py`**: Runs Chain-of-Thought (CoT) and Vanilla Prompting baselines.  

### Long Document Classifiers
- **`lm_classification/`**: Contains code for **Clinical Longformer** and **Longformer** baselines.  
- **`data_processing.ipynb`**: Handles preprocessing of the dataset for classification models.  

---

## Citation
If you use this code, please cite our paper:  
```
@article{your_paper,
  title={Structuring Large Language Models as Agentic Workflows Improves Diagnostic Performance and Physician Assistance},
  author={...},
  journal={...},
  year={2025}
}
