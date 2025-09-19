from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class Prompts:
    def __init__(self, mode='prompt_flexible'):
        print('提示词初始化')
        if mode == 'strict':
            self.prompt_strict()
        elif mode == 'lenient':
            self.prompt_lenient()
        elif mode == 'own':
            self.prompt_own()
        elif mode == 'full':
            self.prompt_FI()
        elif mode == 'cot':
            self.CoT_prompt()
        elif mode == 'prompt_flexible':
            self.prompt_flexible()

    def prompt_strict(self):
        self.history_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""The diagnostic pathway:
1)Patient History, 2)Physical Examination, 3)Laboratory Tests, 4)Radiology Imaging Investigation
## Initial Complaint: Abdominal Pain

<Current Task>
■ STEP 1 - Problem-specific history

<Full Records>
History of Present Illness: {hpi}
                         
<Task Requirements>
- Extract clinical abnormal information from History of Present Illness, organize chronologically
- Filter out normal and non-critical information in the records
- Do not make any diagnosis or diagnostic reasoning
- Mark abnormal points with symbol [!]
- Focus on following sign:
    {
    "Pain characteristics": ["Location", "Radiation area", "Aggravating/relieving factors"],
    "Triggering events": ["Diet/behavior"],
    "Past history/high-risk factors": ["Disease history", "Lifestyle habits"],
    "others": ["Fever", "vomiting", "Fecaluria/pneumaturia/pyuria/stool per vagina", "constipation", "irritable bowel syndrome",]
    }

<Response Template>
    Basic patient info: e.g., 40-years old female, pregnant, etc.
    Key findings: e.g. RLQ, fever, etc.
"""),
            MessagesPlaceholder(variable_name="hpi"),
        ])  # ------------------------------------------------------------------------------------------------------------
        self.physical_exam_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""The diagnostic pathway:
1)Patient History, 2)Physical Examination, 3)Laboratory Tests, 4)Radiology Imaging Investigation
## Initial Complaint: Abdominal Pain

<Current Task>
■ STEP 2 - Physical Examination(PE)

<Full Records>
Physical Examination:{pe}

<Task Requirements>
- Extract abnormal clinical information and Pathognomonic Sign from Physical Examination
- Filter out normal and non-critical information in the records
- Do not make any diagnosis or diagnostic reasoning
- Mark abnormal points with symbol [!]
- The most important region you need to focus on is "Abdomen" 
- Focus on following sign:
    {  
    "Abdominal Palpation": ["Pain or Tenderness location", "Rebound tenderness", "Guarding"],  
    "Special Signs": ["Murphy's sign", "McBurney's point", "Boas sign", "Cullen/Grey-Turner signs", \
                    "Left/Right lower quadrant mass", "Dullness on percussion", "right iliac fossa pain"],  
    "Systemic Signs": ["Fever", "Hypotension", "Jaundice", "vomiting or food intolerance", "Nausea"],  
    "Complications": ["Peritoneal irritation", "Abscess mass", "Pneumaturia/vaginal defecation"]  
    }

"""),
            MessagesPlaceholder(variable_name="pe"),
        ])  # ------------------------------------------------------------------------------------------------------------
        self.lab_results_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""The diagnostic pathway: 
1)Patient History, 2)Physical Examination, 3)Laboratory Tests, 4)Radiology Imaging Investigation
## Initial Complaint: Abdominal Pain

<Current Task>
■ STEP 3 - Laboratory Tests

<Full Records>
Laboratory Tests: {lab}

<Task Requirements>
- Extract abnormal clinical results from Laboratory Tests
- Filter out normal and non-critical information in the records
- Do not make any diagnosis or diagnostic reasoning
- Mark abnormal points with symbol [!]
- Do not use full-width spaces or Unicode spaces such as \\u2003. Use standard ASCII spaces only.
- Focus on following tests:
    {  
    "Inflammatory Markers": ["Elevated white blood cell count(WBC)", "Elevated C-reactive Protein(CRP)", "Interleukin-6", "Resistin", "Procalcitonin(PCT)"],  
    "Organ-Specific Tests": ["Serum Amylase/Lipase", "Bilirubin/ALP"],   
    "Complications": ["Hypocalcemia", "Abscess"], 
    "others": 
        [
        "Hematocrit", "lactate dehydrogenase", 
        "serum triglyceride and calcium levels (In the absence of gallstones or significant history of alcohol use)",
        "complete blood count", 
        "procalcitonin and calprotectin (pediatric patients with suspected acute appendicitis)", 
        ]
    }

"""),
            MessagesPlaceholder(variable_name="lab"),
        ])  # ------------------------------------------------------------------------------------------------------------
        self.microbio_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""The diagnostic pathway: 
1)Patient History, 2)Physical Examination, 3)Laboratory Tests, 4)Microbiology Tests, 5)Radiology Imaging Investigation
## Initial Complaint: Abdominal Pain

<Current Task>
■ STEP 4 - Microbiology Tests

<Full Records>
Microbiology Tests: {microbio}

<Task Requirements>
- Extract abnormal clinical information from Microbiology Tests
- Filter out normal and non-critical information in the records
- Do not make any diagnosis or diagnostic reasoning
- Mark abnormal points with symbol [!]
- Focus on following tests:
    {  
    "Inflammatory Markers": ["WBC", "CRP", "PCT"],  
    "Organ-Specific Tests": ["Amylase/Lipase", "Bilirubin/ALP"],  
    "Imaging Findings": ["Ultrasound", "CT findings"],  
    "Complications": ["Hypocalcemia", "Abscess"]  
    } 

"""),
            MessagesPlaceholder(variable_name="microbio"),
        ])  # ------------------------------------------------------------------------------------------------------------
        self.radiology_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""The diagnostic pathway: 
1)Patient History, 2)Physical Examination, 3)Laboratory Tests, 4)Radiology Imaging Investigation
## Initial Complaint: Abdominal Pain

<Current Task>
■ FINAL STEP - Radiology Imaging Investigation

<Full Records>
{imaging}   

<Task Instructions>
Your task is to extract only **abnormal and relevant imaging findings** from the radiology report.                        

<Requirements>
- Extract only abnormal findings related to **organ-specific features** and **pathological signs**.
- Ignore normal descriptions and unrelated incidental findings.
- Do not provide any diagnostic interpretation or conclusion.
- Mark each abnormal finding with the symbol: [!]
- Do not mention any disease name.
- Focus especially on the following imaging features:
    {
"Key Findings": [
  "gallstones", "sludge", "gallbladder wall thickening", "pericholecystic fluid",
  "biliary obstruction", "gallbladder stasis", "radiopaque cholelithiasis", 
  "gallbladder wall edema", "biliary ductal dilation",

  "colonic wall thickening", "fat stranding", "diverticulosis", "diverticula",
  "pericolic abscess", "fistula", "extraluminal gas or fluid",
  
  "dilated appendix", "appendiceal wall thickening", "periappendiceal fat stranding",
  "appendicolith", "intraluminal appendicolith", "periappendiceal fluid collection",
  "appendiceal rupture", "appendiceal perforation", "mesenteric stranding",
  
  "pancreatic ductal dilation", "peripancreatic stranding", "peripancreatic fluid collection",
  "focal pancreatic lesions", "necrosis", "fluid around pancreas", "Common bile duct (CBD) dilation",
  "stone in the very distal common bile duct", "small gallstones within gallbladder"

  "bowel obstruction", "luminal obstruction", "gas or fluid collection", "air", "abscess"
],

"Key Regions of Interest": [
  "gallbladder", "bile ducts", "cystic duct", "liver",
  "appendix", "periappendiceal region", "cecum",
  "sigmoid colon", "right colon", "descending colon",
  "pancreas", "peripancreatic fat", "retroperitoneum"
],

"Recommendations to Note": [
  "Significant CT region (ABDOMEN): HEPATOBILIARY, PANCREAS, GASTROINTESTINAL, pelvis 
]
}


"""),
            MessagesPlaceholder(variable_name="imaging"),
        ])  # ------------------------------------------------------------------------------------------------------------

        # Eliminate prompts
        self.eliminate_appendicitis_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""\
[Role]: You are a Clinical Diagnosis Assistant. 

[Task Requirements]: There is a patient \
presenting to emergency department associated with Abdominal Pain. \
The abnormal findings of the patient had been extracted in the previous steps.  \
**We need you establish or exclude the diagnosis of acute appendicitis.** Please strictly comply the diagnostic rules I provided to you: 
- Using of a combination of clinical parameters (e.g., AIR, AAS scores) and ultrasound to improve diagnostic sensitivity and specificity. 
- Focus solely on diagnosing the disease I specified; do not include any treatment considerations or severity assessment. 
- Base the assessment strictly on the diagnostic criteria and scoring systems provided.
- If the AIR Score and/or AAS Score classifies the patient as "high-risk," make a direct diagnosis of appendicitis. 
- If the patient falls into the "intermediate-risk" category, first incorporate radiological imaging findings, followed \
    by laboratory tests and other clinical indicators. 
- If the patient is classified as "low-risk," and no other enough evidences support diagnosis, appendicitis should be excluded.

## Abnormal Information of Patient: 
Patient History: {ab_hpi}
Physical Exmination: {ab_pe}
Laboratory Tests: {ab_lab}
Radiology Imaging: {ab_imga}

## Clinical Diagnostic Criteria and Clinical Scoring Systems:
{appendicitis_diag_criteria}

[Response Template/format]: 
- <think> ... </think>: your reasoning process
- <Diagnostic Pathway>: the path you establish or exclude the diagnosis (supported by the diagnostic criteria)
- <Final Answer>: whether the disease established or not, e.g., appendicitis eliminated, appendicitis established
- <Confidence>: the confidence level of the diagnosis, e.g., 80% 
- <Others>: any other information you think should be included

"""),
            MessagesPlaceholder(variable_name='ab_hpi'), 
            MessagesPlaceholder(variable_name='ab_pe'), 
            MessagesPlaceholder(variable_name='ab_lab'), 
            MessagesPlaceholder(variable_name='ab_ima'), 
            MessagesPlaceholder(variable_name="appendicitis_diag_criteria"),
        ])
        self.eliminate_cholecystitis_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""\
[Role]: You are a Clinical Diagnosis Assistant. 

[Task Requirements]: There is a patient \
presenting to emergency department associated with Abdominal Pain. \
The abnormal findings of the patient had been extracted in the previous steps.  \
**We need you establish or exclude the diagnosis of acute calculus cholecystitis.** Please strictly comply the diagnostic rules I provided to you: 
- As no feature has sufficient diagnostic power to establish or exclude the diagnosis of ACC, it is recommended not to rely on a single clinical or laboratory finding. 
- Using a combination of detailed history, complete clinical examination, laboratory tests and imaging. 
- Murphy’s sign would not be accessible due to data processing.
- Hepatobiliary iminodiacetic acid (HIDA) scan has the highest sensitivity and specificity for the diagnosis \
    of ACC as compared to other imaging modalities. Diagnostic accuracy of computed tomography (CT) is poor. \
    Magnetic resonance imaging (MRI) is as accurate as abdominal US.
- Focus solely on diagnosing the disease I specified; do not include any treatment considerations or severity assessment. 
- Using the criteria I provided to you.
- If the diagnostic threshold is met, establish a direct diagnosis of cholecystitis. 
- If the threshold is not met, first consider radiological imaging findings, followed by laboratory tests and other clinical indicators.
- If the diagnostic threshold is not met and no other enough evidences support diagnosis, cholecystitis should be excluded. 

## Abnormal Information of Patient: 
Patient History: {ab_hpi}
Physical Exmination: {ab_pe}
Laboratory Tests: {ab_lab}
Radiology Imaging: {ab_ima}

## Clinical Diagnostic Criteria:
{cholecystitis_diag_criteria}

[Response Template/format]: 
- <think> ... </think>: your reasoning process
- <Diagnostic Pathway>: the path you establish or exclude the diagnosis (supported by the diagnostic criteria)
- <Final Answer>: whether the disease established or not, e.g., cholecystitis eliminated, cholecystitis established
- <Confidence>: the confidence level of the diagnosis, e.g., 80% 
- <Others>: any other information you think should be included

"""),
            MessagesPlaceholder(variable_name='ab_hpi'), 
            MessagesPlaceholder(variable_name='ab_pe'), 
            MessagesPlaceholder(variable_name='ab_lab'), 
            MessagesPlaceholder(variable_name='ab_ima'), 
            MessagesPlaceholder(variable_name="cholecystitis_diag_criteria"),
        ])
        self.eliminate_pancreatitis_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""\
[Role]: You are a Clinical Diagnosis Assistant. 

[Task Requirements]: There is a patient presenting to emergency department associated with Abdominal Pain. \
The abnormal findings of the patient had been extracted in the previous steps.  \
**We need you establish or exclude the diagnosis of acute pancreatitis.** Please strictly comply the diagnostic rules I provided to you: 
- Focus solely on diagnosing the disease I specified; do not include any treatment considerations or severity assessment. 
- Using the criteria I provided to you.
- If the diagnostic threshold is met, establish a direct diagnosis of pancreatitis. 
- If the threshold is not met, first consider radiological imaging findings, followed by laboratory tests and other clinical indicators.
- If the diagnostic threshold is not met and no other enough evidences support diagnosis, pancreatitis should be excluded. 

## Abnormal Information of Patient: 
Patient History: {ab_hpi}
Physical Exmination: {ab_pe}
Laboratory Tests: {ab_lab}
Radiology Imaging: {ab_ima}

## Clinical Diagnostic Criteria:
{pancreatitis_diag_criteria}

[Response Template/format]: 
- <think> ... </think>: your reasoning process
- <Diagnostic Pathway>: the path you establish or exclude the diagnosis (supported by the diagnostic criteria)
- <Final Answer>: whether the disease established or not, e.g., pancreatitis eliminated, pancreatitis established
- <Confidence>: the confidence level of the diagnosis, e.g., 80% 
- <Others>: any other information you think should be included

"""),
            MessagesPlaceholder(variable_name='ab_hpi'), 
            MessagesPlaceholder(variable_name='ab_pe'), 
            MessagesPlaceholder(variable_name='ab_lab'), 
            MessagesPlaceholder(variable_name='ab_ima'), 
            MessagesPlaceholder(variable_name="pancreatitis_diag_criteria"),
            # MessagesPlaceholder(variable_name="clinical_guidelines"),
            HumanMessage(content="<think>\n")
        ])
        self.eliminate_diverticulitis_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""\
[Role]: You are a Clinical Diagnosis Assistant. 

[Task Requirements]: There is a patient \
presenting to emergency department associated with Abdominal Pain. \
The abnormal findings of the patient had been extracted in the previous steps.  \
**We need you establish or exclude the diagnosis of acute (sigmoid colon) diverticulitis.** Please strictly comply to the diagnostic rules I provided: 
- Focus solely on diagnosing the acute sigmoid diverticulitis; do not include any treatment considerations or severity assessment.
- Using the criteria I provided to you.
- If the diagnostic threshold is met, establish a direct diagnosis of acute diverticulitis. 
- If the threshold is not met, claim it and deliver the evidence according to the criteria.
- If the diagnostic threshold is not met and no other enough evidences support diagnosis, diverticulitis should be excluded. 

## Abnormal Information of Patient: 
Patient History: {ab_hpi}
Physical Exmination: {ab_pe}
Laboratory Tests: {ab_lab}
Radiology Imaging: {ab_ima}

## Clinical Diagnostic Criteria (ASCRS Guidelines):
{diverticulitis_diag_criteria}
                         
[Response Template/format]: 
- <think> ... </think>: your reasoning process
- <Diagnostic Pathway>: the path you establish or exclude the diagnosis (supported by the diagnostic criteria)
- <Final Answer>: whether the disease established or not, e.g., diverticulitis eliminated, diverticulitis established
- <Confidence>: the confidence level of the diagnosis, e.g., 80% 
- <Others>: any other information you think should be included

"""),
            MessagesPlaceholder(variable_name='ab_hpi'), 
            MessagesPlaceholder(variable_name='ab_pe'), 
            MessagesPlaceholder(variable_name='ab_lab'), 
            MessagesPlaceholder(variable_name='ab_ima'), 
            MessagesPlaceholder(variable_name="diverticulitis_diag_criteria"),
        ])

        # Review and action prompt
        self.review_prompt = HumanMessage(content="""[Role]: You are a Clinical Diagnosis Assistant. \
                The context presented you the diagnostic results of 4 diseases, following the task requirements below.

                [Task Requirements]: 
                - Review the diagnostic results in context but not diagnosis again!!! Do not repeat the diagnostic process in previous steps!!!
                - After review the context, state the diagnosis results in few words, e.g., appendictitis, diverticulitis and pancreatitis excluded, cholecystitis established.
                - If there was a single diagnosis established, summarize the results and response.
                - If there is a need to differential diagnosis (more than one diagnosis is possible in the four pathologies), call the tool named {{tool_review}} for providing \
                    differential diagnosis knowledge among appendicitis, cholecystitis, pancreatitis, and diverticulitis.
                - If no concrete diagnosis was formulated, review the most likely one if existed in previous steps, summarize it and lower the confidence.
                - If there was no diagnosis established, summarize the results and response. Do not make any diagnosis or diagnostic reasoning.
                                          
                [Response Format 1 (without tool usage)]:
                - <think> ... </think>: your reasoning process
                - <Summarization>: summarize the 4 results, e.g., appendictitis excluded because of the ..., cholecystitis established because of the ...
                - <Final Diagnosis>: the disease name only, e.g. pancreatitis
                - <Confidence>: confidence level of the results, e.g., 90% 
                - <Tool Usage Statement>: no tool called
                                        
                [Response Format 2 (with tool called)]: 
                - <think> ... </think>: your reasoning process
                - <Step State>: Agent Brain Step
                - <Summarization>: according to the tool called result, establishing the final diagnosis (only one disease should left)
                - <Final Diagnosis>: the disease name only, e.g. pancreatitis
                - <Confidence>: confidence level of the results, e.g., 60% (with tool called, the confidence level should be lower than without.)
                - <Tool Usage Statement>: tool called

                """)

    def prompt_lenient(self):
        self.history_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""The diagnostic pathway:
1)Patient History, 2)Physical Examination, 3)Laboratory Tests, 4)Radiology Imaging Investigation
## Initial Complaint: Abdominal Pain

<Current Task>
■ STEP 1 - Problem-specific history

<Full Records>
History of Present Illness: {hpi}
                         
<Task Requirements>
- Extract clinical abnormal information from History of Present Illness, organize chronologically
- Filter out normal and non-critical information in the records
- Do not make any diagnosis or diagnostic reasoning
- Mark abnormal points with symbol [!]
- Focus on following sign:
    {
    "Pain characteristics": ["Location", "Radiation area", "Aggravating/relieving factors"],
    "Triggering events": ["Diet/behavior"],
    "Past history/high-risk factors": ["Disease history", "Lifestyle habits"],
    "others": ["Fever", "vomiting", "Fecaluria/pneumaturia/pyuria/stool per vagina", "constipation", "irritable bowel syndrome",]
    }

<Response Template>
    Basic patient info: e.g., 40-years old female, pregnant, etc.
    Key findings: e.g. RLQ, fever, etc.
"""),
            MessagesPlaceholder(variable_name="hpi"),
        ])  # ------------------------------------------------------------------------------------------------------------
        self.physical_exam_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""The diagnostic pathway:
1)Patient History, 2)Physical Examination, 3)Laboratory Tests, 4)Radiology Imaging Investigation
## Initial Complaint: Abdominal Pain

<Current Task>
■ STEP 2 - Physical Examination(PE)

<Full Records>
Physical Examination:{pe}

<Task Requirements>
- Extract abnormal clinical information and Pathognomonic Sign from Physical Examination
- Filter out normal and non-critical information in the records
- Do not make any diagnosis or diagnostic reasoning
- Mark abnormal points with symbol [!]
- The most important region you need to focus on is "Abdomen" 
- Focus on following sign:
    {  
    "Abdominal Palpation": ["Pain or Tenderness location", "Rebound tenderness", "Guarding"],  
    "Special Signs": ["Murphy's sign", "McBurney's point", "Boas sign", "Cullen/Grey-Turner signs", \
                    "Left/Right lower quadrant mass", "Dullness on percussion", "right iliac fossa pain"],  
    "Systemic Signs": ["Fever", "Hypotension", "Jaundice", "vomiting or food intolerance", "Nausea"],  
    "Complications": ["Peritoneal irritation", "Abscess mass", "Pneumaturia/vaginal defecation"]  
    }

"""),
            MessagesPlaceholder(variable_name="pe"),
        ])  # ------------------------------------------------------------------------------------------------------------
        self.lab_results_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""The diagnostic pathway: 
1)Patient History, 2)Physical Examination, 3)Laboratory Tests, 4)Radiology Imaging Investigation
## Initial Complaint: Abdominal Pain

<Current Task>
■ STEP 3 - Laboratory Tests

<Full Records>
Laboratory Tests: {lab}

<Task Requirements>
- Extract abnormal clinical results from Laboratory Tests
- Filter out normal and non-critical information in the records
- Do not make any diagnosis or diagnostic reasoning
- Mark abnormal points with symbol [!]
- Do not use full-width spaces or Unicode spaces such as \\u2003. Use standard ASCII spaces only.
- Focus on following tests:
    {  
    "Inflammatory Markers": ["Elevated white blood cell count(WBC)", "Elevated C-reactive Protein(CRP)", "Interleukin-6", "Resistin", "Procalcitonin(PCT)"],  
    "Organ-Specific Tests": ["Serum Amylase/Lipase", "Bilirubin/ALP"],   
    "Complications": ["Hypocalcemia", "Abscess"], 
    "others": 
        [
        "Hematocrit", "lactate dehydrogenase", 
        "serum triglyceride and calcium levels (In the absence of gallstones or significant history of alcohol use)",
        "complete blood count", 
        "procalcitonin and calprotectin (pediatric patients with suspected acute appendicitis)", 
        ]
    }

"""),
            MessagesPlaceholder(variable_name="lab"),
        ])  # ------------------------------------------------------------------------------------------------------------
        self.microbio_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""The diagnostic pathway: 
1)Patient History, 2)Physical Examination, 3)Laboratory Tests, 4)Microbiology Tests, 5)Radiology Imaging Investigation
## Initial Complaint: Abdominal Pain

<Current Task>
■ STEP 4 - Microbiology Tests

<Full Records>
Microbiology Tests: {microbio}

<Task Requirements>
- Extract abnormal clinical information from Microbiology Tests
- Filter out normal and non-critical information in the records
- Do not make any diagnosis or diagnostic reasoning
- Mark abnormal points with symbol [!]
- Focus on following tests:
    {  
    "Inflammatory Markers": ["WBC", "CRP", "PCT"],  
    "Organ-Specific Tests": ["Amylase/Lipase", "Bilirubin/ALP"],  
    "Imaging Findings": ["Ultrasound", "CT findings"],  
    "Complications": ["Hypocalcemia", "Abscess"]  
    } 

"""),
            MessagesPlaceholder(variable_name="microbio"),
        ])  # ------------------------------------------------------------------------------------------------------------
        self.radiology_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""The diagnostic pathway: 
1)Patient History, 2)Physical Examination, 3)Laboratory Tests, 4)Radiology Imaging Investigation
## Initial Complaint: Abdominal Pain

<Current Task>
■ FINAL STEP - Radiology Imaging Investigation

<Full Records>
{imaging}   

<Task Instructions>
Your task is to extract only **abnormal and relevant imaging findings** from the radiology report.                        

<Requirements>
- Extract only abnormal findings related to **organ-specific features** and **pathological signs**.
- Ignore normal descriptions and unrelated incidental findings.
- Do not provide any diagnostic interpretation or conclusion.
- Mark each abnormal finding with the symbol: [!]
- Do not mention any disease name.
- Focus especially on the following imaging features:
    {
"Key Findings": [
  "gallstones", "sludge", "gallbladder wall thickening", "pericholecystic fluid",
  "biliary obstruction", "gallbladder stasis", "radiopaque cholelithiasis", 
  "gallbladder wall edema", "biliary ductal dilation",

  "colonic wall thickening", "fat stranding", "diverticulosis", "diverticula",
  "pericolic abscess", "fistula", "extraluminal gas or fluid",
  
  "dilated appendix", "appendiceal wall thickening", "periappendiceal fat stranding",
  "appendicolith", "intraluminal appendicolith", "periappendiceal fluid collection",
  "appendiceal rupture", "appendiceal perforation", "mesenteric stranding",
  
  "pancreatic ductal dilation", "peripancreatic stranding", "peripancreatic fluid collection",
  "focal pancreatic lesions", "necrosis", "fluid around pancreas", "Common bile duct (CBD) dilation"

  "bowel obstruction", "luminal obstruction", "gas or fluid collection", "air", "abscess"
],

"Key Regions of Interest": [
  "gallbladder", "bile ducts", "cystic duct", "liver",
  "appendix", "periappendiceal region", "cecum",
  "sigmoid colon", "right colon", "descending colon",
  "pancreas", "peripancreatic fat", "retroperitoneum"
],

"Recommendations to Note": [
  "Significant CT region (ABDOMEN): HEPATOBILIARY, PANCREAS, GASTROINTESTINAL, pelvis 
]
}


"""),
            MessagesPlaceholder(variable_name="imaging"),
        ])  # ------------------------------------------------------------------------------------------------------------

        # Eliminate prompts
        self.eliminate_appendicitis_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""\
[Role]: You are a Clinical Diagnosis Assistant. 

[Task Requirements]: There is a patient \
presenting to emergency department associated with Abdominal Pain. \
The abnormal findings of the patient had been extracted in the previous steps.  \
We need you establish or exclude the diagnosis of acute appendicitis. Please comply the diagnostic rules I provided to you: 
- Using of a combination of clinical parameters (e.g., AIR, AAS scores) and radiology imaging findings to improve diagnostic sensitivity and specificity. 
- Focus solely on diagnosing the disease I specified; do not include any treatment considerations or severity assessment. 
- Do not apply criteria rigidly. A moderate WBC elevation, combined with compatible symptoms and imaging, may still support a diagnosis. 
- Please comprehensively evaluate the diagnosis likelihood: 
    1.Definite: Strong clinical, lab, and imaging support
    2.Probable: Most but not all criteria are met
    3.Possible: Partial criteria or suspicious findings
    4.Unlikely: Insufficient evidence or alternative diagnosis more likely
                         
## Abnormal Information of Patient: 
Patient History: {ab_hpi}
Physical Exmination: {ab_pe}
Laboratory Tests: {ab_lab}
Radiology Imaging: {ab_imga}

## Clinical Diagnostic Criteria and Clinical Scoring Systems:
{appendicitis_diag_criteria}

[Response Template/format]: 
- <think> ... </think>: your reasoning process
- <evidence>: the evidence supproting your evaluation
- <Final Answer>: whether the disease established or not, e.g., appendicitis eliminated, appendicitis established
- <Confidence>: the confidence level of the diagnosis, e.g., 80% 
- <Others>: any other information you think should be included

"""),
            MessagesPlaceholder(variable_name='ab_hpi'), 
            MessagesPlaceholder(variable_name='ab_pe'), 
            MessagesPlaceholder(variable_name='ab_lab'), 
            MessagesPlaceholder(variable_name='ab_ima'), 
            MessagesPlaceholder(variable_name="appendicitis_diag_criteria"),
        ])
        self.eliminate_cholecystitis_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""\
[Role]: You are a Clinical Diagnosis Assistant. 

[Task Requirements]: There is a patient \
presenting to emergency department associated with Abdominal Pain. \
The abnormal findings of the patient had been extracted in the previous steps.  \
We need you establish or exclude the diagnosis of acute cholecystitis. Please comply the diagnostic rules I provided to you: 
- Using a combination of history of present illness, physical examination, laboratory tests and imaging. 
- Focus solely on diagnosing the disease I specified; do not include any treatment considerations or severity assessment. 
- Using the criteria I provided to you. 
- Murphy’s sign would not be accessible due to data processing.
- Do not apply criteria rigidly. A moderate WBC elevation, combined with compatible symptoms and imaging, may still support a diagnosis. 
- Please comprehensively evaluate the diagnosis likelihood: 
    1.Definite: Strong clinical, lab, and imaging support
    2.Probable: Most but not all criteria are met
    3.Possible: Partial criteria or suspicious findings
    4.Unlikely: Insufficient evidence or alternative diagnosis more likely

## Abnormal Information of Patient: 
Patient History: {ab_hpi}
Physical Exmination: {ab_pe}
Laboratory Tests: {ab_lab}
Radiology Imaging: {ab_ima}

## Clinical Diagnostic Criteria:
{cholecystitis_diag_criteria}

[Response Template/format]: 
- <think> ... </think>: your reasoning process
- <evidence>: the evidence supproting your evaluation
- <Final Answer>: whether the disease established or not, e.g., cholecystitis eliminated, cholecystitis established
- <Confidence>: the confidence level of the diagnosis, e.g., 80% 
- <Others>: any other information you think should be included

"""),
            MessagesPlaceholder(variable_name='ab_hpi'), 
            MessagesPlaceholder(variable_name='ab_pe'), 
            MessagesPlaceholder(variable_name='ab_lab'), 
            MessagesPlaceholder(variable_name='ab_ima'), 
            MessagesPlaceholder(variable_name="cholecystitis_diag_criteria"),
        ])
        self.eliminate_pancreatitis_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""\
[Role]: You are a Clinical Diagnosis Assistant. 

[Task Requirements]: There is a patient presenting to emergency department associated with Abdominal Pain. \
The abnormal findings of the patient had been extracted in the previous steps.  \
We need you establish or exclude the diagnosis of acute pancreatitis. Please strictly comply the diagnostic rules I provided to you: 
- Focus solely on diagnosing the disease I specified; do not include any treatment considerations or severity assessment. 
- Using the criteria I provided to you.
- Do not apply criteria rigidly. For example, if lipase is slightly below 3× but there is classic pain and supporting CT findings, a diagnosis may still be warranted. 
- Please comprehensively evaluate the diagnosis likelihood: 
    1.Definite: Strong clinical, lab, and imaging support
    2.Probable: Most but not all criteria are met
    3.Possible: Partial criteria or suspicious findings
    4.Unlikely: Insufficient evidence or alternative diagnosis more likely

## Abnormal Information of Patient: 
Patient History: {ab_hpi}
Physical Exmination: {ab_pe}
Laboratory Tests: {ab_lab}
Radiology Imaging: {ab_ima}

## Clinical Diagnostic Criteria:
{pancreatitis_diag_criteria}

[Response Template/format]: 
- <think> ... </think>: your reasoning process
- <evidence>: the evidence supproting your evaluation
- <Final Answer>: whether the disease established or not, e.g., pancreatitis eliminated, pancreatitis established
- <Confidence>: the confidence level of the diagnosis, e.g., 80% 
- <Others>: any other information you think should be included

"""),
            MessagesPlaceholder(variable_name='ab_hpi'), 
            MessagesPlaceholder(variable_name='ab_pe'), 
            MessagesPlaceholder(variable_name='ab_lab'), 
            MessagesPlaceholder(variable_name='ab_ima'), 
            MessagesPlaceholder(variable_name="pancreatitis_diag_criteria"),
        ])
        self.eliminate_diverticulitis_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""\
[Role]: You are a Clinical Diagnosis Assistant. 

[Task Requirements]: There is a patient \
presenting to emergency department associated with Abdominal Pain. \
The abnormal findings of the patient had been extracted in the previous steps.  \
**We need you establish or exclude the diagnosis of acute diverticulitis.** Please strictly comply to the diagnostic rules I provided: 
- Focus solely on diagnosing the acute diverticulitis; do not include any treatment considerations or severity assessment.
- Using the criteria I provided to you.
- Do not apply criteria rigidly. A moderate WBC elevation, combined with compatible symptoms and imaging, may still support a diagnosis.
- Please comprehensively evaluate the diagnosis likelihood: 
    1.Definite: Strong clinical, lab, and imaging support
    2.Probable: Most but not all criteria are met
    3.Possible: Partial criteria or suspicious findings
    4.Unlikely: Insufficient evidence or alternative diagnosis more likely

## Abnormal Information of Patient: 
Patient History: {ab_hpi}
Physical Exmination: {ab_pe}
Laboratory Tests: {ab_lab}
Radiology Imaging: {ab_ima}

## Clinical Diagnostic Criteria (ASCRS Guidelines):
{diverticulitis_diag_criteria}
                         
[Response Template/format]: 
- <think> ... </think>: your reasoning process
- <evidence>: the evidence supproting your evaluation
- <Final Answer>: whether the disease established or not, e.g., diverticulitis eliminated, diverticulitis established
- <Confidence>: the confidence level of the diagnosis, e.g., 80% 
- <Others>: any other information you think should be included

"""),
            MessagesPlaceholder(variable_name='ab_hpi'), 
            MessagesPlaceholder(variable_name='ab_pe'), 
            MessagesPlaceholder(variable_name='ab_lab'), 
            MessagesPlaceholder(variable_name='ab_ima'), 
            MessagesPlaceholder(variable_name="diverticulitis_diag_criteria"),
        ])

        # Review and action prompt
        self.review_prompt = HumanMessage(content="""[Role]: You are a Clinical Diagnosis Assistant. \
                The context presented you the diagnostic results of 4 diseases, following the task requirements below.

                [Task Requirements]: 
                - Review the diagnostic results in context but not diagnosis again!!! Do not repeat the diagnostic process in previous steps!!!
                - After review the context, state the diagnosis results in few words, e.g., appendictitis, diverticulitis and pancreatitis excluded, cholecystitis established.
                - If there was a single diagnosis (or no diagnosis) established, summarize the results and response.
                - If there is a need to differential diagnosis (more than one diagnosis is possible in the four pathologies), call the tool named {{tool_review}} for providing \
                    differential diagnosis knowledge among appendicitis, cholecystitis, pancreatitis, and diverticulitis.
                - If no concrete diagnosis was formulated, such as had multiple "possible" result, review the evidence and make ensure to formulate a primary diagnosis. Call the tool if needed.
                
                [Response Format 1 (without tool usage)]:
                - <think> ... </think>: your reasoning process
                - <Summarization>: summarize the 4 results, e.g., appendictitis excluded because of the ..., cholecystitis established because of the ...
                - <Final Diagnosis>: the disease name only, e.g. pancreatitis
                - <Confidence>: confidence level of the results, e.g., 90% 
                - <Tool Usage Statement>: no tool called
                                        
                [Response Format 2 (with tool called)]: 
                - <think> ... </think>: your reasoning process
                - <Step State>: Agent Brain Step
                - <Summarization>: according to the tool called result, establishing the final diagnosis (only one disease should left)
                - <Final Diagnosis>: the disease name only, e.g. pancreatitis
                - <Confidence>: confidence level of the results, e.g., 60% (with tool called, the confidence level should be lower than without.)
                - <Tool Usage Statement>: tool called

                """)

    def prompt_own(self):
        self.history_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""The diagnostic pathway:
1)Patient History, 2)Physical Examination, 3)Laboratory Tests, 4)Radiology Imaging Investigation
## Initial Complaint: Abdominal Pain

<Current Task>
■ STEP 1 - Problem-specific history

<Full Records>
History of Present Illness: {hpi}
                         
<Task Requirements>
- Apply your clinical knowledge to extract abnormal findings from the History of Present Illness and arrange them chronologically
- Filter out normal and non-critical information in the records
- Do not make any diagnosis or diagnostic reasoning
- Mark abnormal points with symbol [!]

<Response Template>
    Basic patient info: basic but useful information of the patient, such as age, sex etc. 
    Key findings: the abnormalities you found
"""),
            MessagesPlaceholder(variable_name="hpi"),
        ]) 

        self.physical_exam_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""The diagnostic pathway:
1)Patient History, 2)Physical Examination, 3)Laboratory Tests, 4)Radiology Imaging Investigation
## Initial Complaint: Abdominal Pain

<Current Task>
■ STEP 2 - Physical Examination(PE)

<Full Records>
Physical Examination:{pe}

<Task Requirements>
- Apply your clinical knowledge to extract abnormal clinical information and Pathognomonic Sign from Physical Examination
- Filter out normal and non-critical information in the records
- Do not make any diagnosis or diagnostic reasoning
- Mark abnormal points with symbol [!]

"""),
            MessagesPlaceholder(variable_name="pe"),
        ])  
        
        self.lab_results_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""The diagnostic pathway: 
1)Patient History, 2)Physical Examination, 3)Laboratory Tests, 4)Radiology Imaging Investigation
## Initial Complaint: Abdominal Pain

<Current Task>
■ STEP 3 - Laboratory Tests

<Full Records>
Laboratory Tests: {lab}

<Task Requirements>
- Apply your clinical knowledge to extract abnormal clinical results from Laboratory Tests
- Filter out normal and non-critical information in the records
- Do not make any diagnosis or diagnostic reasoning
- Mark abnormal points with symbol [!]
- Do not use full-width spaces or Unicode spaces such as \\u2003. Use standard ASCII spaces only.

"""),
            MessagesPlaceholder(variable_name="lab"),
        ])  

        self.microbio_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""The diagnostic pathway: 
1)Patient History, 2)Physical Examination, 3)Laboratory Tests, 4)Microbiology Tests, 5)Radiology Imaging Investigation
## Initial Complaint: Abdominal Pain

<Current Task>
■ STEP 4 - Microbiology Tests

<Full Records>
Microbiology Tests: {microbio}

<Task Requirements>
- Apply your clinical knowledge to extract abnormal clinical information from Microbiology Tests
- Filter out normal and non-critical information in the records
- Do not make any diagnosis or diagnostic reasoning
- Mark abnormal points with symbol [!]

"""),
            MessagesPlaceholder(variable_name="microbio"),
        ])  

        self.radiology_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""The diagnostic pathway: 
1)Patient History, 2)Physical Examination, 3)Laboratory Tests, 4)Radiology Imaging Investigation
## Initial Complaint: Abdominal Pain

<Current Task>
■ FINAL STEP - Radiology Imaging Investigation

<Full Records>
{imaging}                       

<Task Requirements>
- Apply your clinical knowledge to extract only abnormal findings from radiology imaging reports
- Ignore normal descriptions and unrelated incidental findings.
- Do not provide any diagnostic interpretation or conclusion.
- Mark each abnormal finding with the symbol: [!]

"""),
            MessagesPlaceholder(variable_name="imaging"),
        ])  

        # Eliminate prompts
        self.eliminate_appendicitis_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""\
[Role]: You are a Clinical Diagnosis Assistant. 

[Task Requirements]: There is a patient \
presenting to emergency department associated with Abdominal Pain. \
The abnormal findings of the patient had been extracted in the previous steps.  \
I need you to apply your clinical knowledge to either establish or rule out the diagnosis of acute appendicitis. 
- Please comprehensively evaluate the diagnosis likelihood: 
    1.Definite: Strong clinical, lab, and imaging support
    2.Probable: Most but not all criteria are met
    3.Possible: Partial criteria or suspicious findings
    4.Unlikely: Insufficient evidence or alternative diagnosis more likely
                         
## Abnormal Information of Patient: 
Patient History: {ab_hpi}
Physical Exmination: {ab_pe}
Laboratory Tests: {ab_lab}
Radiology Imaging: {ab_imga}

[Response Template/format]: 
- <think> ... </think>: your reasoning process
- <evidence>: the evidence supproting your evaluation
- <Final Answer>: whether the disease established or not, e.g., appendicitis eliminated(unlikely), appendicitis established(probable)
- <Confidence>: the confidence level of the diagnosis, e.g., 80% 
- <Others>: any other information you think should be included

"""),
            MessagesPlaceholder(variable_name='ab_hpi'), 
            MessagesPlaceholder(variable_name='ab_pe'), 
            MessagesPlaceholder(variable_name='ab_lab'), 
            MessagesPlaceholder(variable_name='ab_ima'), 
        ])
        self.eliminate_cholecystitis_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""\
[Role]: You are a Clinical Diagnosis Assistant. 

[Task Requirements]: There is a patient \
presenting to emergency department associated with Abdominal Pain. \
The abnormal findings of the patient had been extracted in the previous steps.  \
I need you to apply your clinical knowledge to either establish or rule out the diagnosis of acute calculus cholecystitis. 
- Murphy’s sign would not be accessible due to data processing.
- Please comprehensively evaluate the diagnosis likelihood: 
    1.Definite: Strong clinical, lab, and imaging support
    2.Probable: Most but not all criteria are met
    3.Possible: Partial criteria or suspicious findings
    4.Unlikely: Insufficient evidence or alternative diagnosis more likely

## Abnormal Information of Patient: 
Patient History: {ab_hpi}
Physical Exmination: {ab_pe}
Laboratory Tests: {ab_lab}
Radiology Imaging: {ab_ima}

[Response Template/format]: 
- <think> ... </think>: your reasoning process
- <evidence>: the evidence supproting your evaluation
- <Final Answer>: whether the disease established or not, e.g., appendicitis eliminated(unlikely), appendicitis established(probable)
- <Confidence>: the confidence level of the diagnosis, e.g., 80% 
- <Others>: any other information you think should be included

"""),
            MessagesPlaceholder(variable_name='ab_hpi'), 
            MessagesPlaceholder(variable_name='ab_pe'), 
            MessagesPlaceholder(variable_name='ab_lab'), 
            MessagesPlaceholder(variable_name='ab_ima'), 
        ])
        self.eliminate_pancreatitis_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""\
[Role]: You are a Clinical Diagnosis Assistant. 

[Task Requirements]: There is a patient presenting to emergency department associated with Abdominal Pain. \
The abnormal findings of the patient had been extracted in the previous steps.  \
I need you to apply your clinical knowledge to either establish or rule out the diagnosis of acute pancreatitis. 
- Please comprehensively evaluate the diagnosis likelihood: 
    1.Definite: Strong clinical, lab, and imaging support
    2.Probable: Most but not all criteria are met
    3.Possible: Partial criteria or suspicious findings
    4.Unlikely: Insufficient evidence or alternative diagnosis more likely

## Abnormal Information of Patient: 
Patient History: {ab_hpi}
Physical Exmination: {ab_pe}
Laboratory Tests: {ab_lab}
Radiology Imaging: {ab_ima}

[Response Template/format]: 
- <think> ... </think>: your reasoning process
- <evidence>: the evidence supproting your evaluation
- <Final Answer>: whether the disease established or not, e.g., appendicitis eliminated(unlikely), appendicitis established(probable)
- <Confidence>: the confidence level of the diagnosis, e.g., 80% 
- <Others>: any other information you think should be included

"""),
            MessagesPlaceholder(variable_name='ab_hpi'), 
            MessagesPlaceholder(variable_name='ab_pe'), 
            MessagesPlaceholder(variable_name='ab_lab'), 
            MessagesPlaceholder(variable_name='ab_ima'), 
        ])
        self.eliminate_diverticulitis_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""\
[Role]: You are a Clinical Diagnosis Assistant. 

[Task Requirements]: There is a patient \
presenting to emergency department associated with Abdominal Pain. \
The abnormal findings of the patient had been extracted in the previous steps.  \
I need you to apply your clinical knowledge to either establish or rule out the diagnosis of acute diverticulitis. 
- Please comprehensively evaluate the diagnosis likelihood: 
    1.Definite: Strong clinical, lab, and imaging support
    2.Probable: Most but not all criteria are met
    3.Possible: Partial criteria or suspicious findings
    4.Unlikely: Insufficient evidence or alternative diagnosis more likely


## Abnormal Information of Patient: 
Patient History: {ab_hpi}
Physical Exmination: {ab_pe}
Laboratory Tests: {ab_lab}
Radiology Imaging: {ab_ima}
                         
[Response Template/format]: 
- <think> ... </think>: your reasoning process
- <evidence>: the evidence supproting your evaluation
- <Final Answer>: whether the disease established or not, e.g., appendicitis eliminated(unlikely), appendicitis established(probable)
- <Confidence>: the confidence level of the diagnosis, e.g., 80% 
- <Others>: any other information you think should be included

"""),
            MessagesPlaceholder(variable_name='ab_hpi'), 
            MessagesPlaceholder(variable_name='ab_pe'), 
            MessagesPlaceholder(variable_name='ab_lab'), 
            MessagesPlaceholder(variable_name='ab_ima'), 
        ])

        # Review and action prompt
        self.review_prompt = HumanMessage(content="""[Role]: You are a Clinical Diagnosis Assistant. \
                The context presented you the diagnostic results of 4 diseases, following the task requirements below.

                [Task Requirements]: 
                - Review the diagnostic results in context but not diagnosis again!!! Do not repeat the diagnostic process in previous steps!!!
                - Ensure that only one primary diagnosis was formulated.
                - Apply your clinical knowledge to tease out the results, summarize them according to the format I provided.
                - If there is a need to differential diagnosis(more than one diagnosis is possible among appendicitis, cholecystitis, pancreatitis, and diverticulitis,\
                    and you can not tease out the conditions by yourself), you can call the tool named {{tool_review}} which can return some \
                    differential diagnosis knowledge.

                [Response Format 1 (without tool usage)]:
                - <think> ... </think>: your reasoning process
                - <Summarization>: summarize the 4 results, e.g., appendictitis excluded because of the ..., cholecystitis established because of the ...
                - <Final Diagnosis>: the disease name only, e.g. pancreatitis
                - <Confidence>: confidence level of the results, e.g., 90% 
                - <Tool Usage Statement>: no tool called
                                        
                [Response Format 2 (with tool called)]: 
                - <think> ... </think>: your reasoning process
                - <Step State>: Agent Brain Step
                - <Summarization>: according to the tool called result, establishing the final diagnosis (only one disease should left)
                - <Final Diagnosis>: the disease name only, e.g. pancreatitis
                - <Confidence>: confidence level of the results, e.g., 60% (with tool called, the confidence level should be lower than without.)
                - <Tool Usage Statement>: tool called

                """)

    def prompt_FI(self):
        """The prompts that with full information from patient, and try to invoke the LLMs directly answer the diagnosis result"""
        self.full_info_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""
                You are a Clinical Diagnosis Assistant. Now, I am providing a patient information range from \
                History of Present Illness, Physical Examination, Laboratory Tests, and Radiology Imaging Reports.
                Your task is to tease out whether the patient have the following pathologies: 1.Appendicitis, \
                2.Cholecystitis, 3.Diverticulitis, 4.Pancreatitis. Note that you should establish a primary \
                diagnosis or exclude all the disease. If any other pathology suspected, carefully detail it.
                Note that Murphy’s sign would not be accessible due to data processing.
                
                [Response Template/format]: 
                - <think> ... </think>: your reasoning process
                - <evidence>: the evidence supproting your evaluation
                - <Final Answer>: whether the disease established or not, e.g., appendicitis eliminated(unlikely), \
                appendicitis established(probable)
                - <Confidence>: the confidence level of the diagnosis, e.g., 80% 
                - <Others>: any other information you think should be included
                         
                ## Patient Information: 
                History of Present Illness: {hpi}
                Physical Exmination: {pe}
                Laboratory Tests: {lab}
                Radiology Imaging Reports: {imaging}
            """),
            MessagesPlaceholder(variable_name='hpi'), 
            MessagesPlaceholder(variable_name='pe'), 
            MessagesPlaceholder(variable_name='lab'), 
            MessagesPlaceholder(variable_name='imaging'), 
        ])

    def CoT_prompt(self):
        self.cot_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""
You are a Clinical Diagnosis Assistant with expert-level reasoning skills.  
Your task is to make a step-by-step diagnostic evaluation for a patient presenting with abdominal pain.  
The diagnostic workflow includes multiple stages. Follow them strictly and do not skip steps.  

### STEP 1 — Extract abnormal findings from History of Present Illness (HPI)
Input:
History of Present Illness: {hpi}

Instructions:
- Extract abnormal findings only
- Ignore normal or non-informative content
- Arrange findings chronologically
- Mark abnormal points with [!]
- Do NOT make any diagnosis or reasoning

Response Format:
Basic patient info: ...
Key findings:
- [!] ...
- [!] ...

### STEP 2 — Extract abnormal findings from Physical Examination
Input:
Physical Examination: {pe}

Instructions:
- Same as above
- Mark abnormal points with [!]

Response Format:
Key findings:
- [!] ...
- [!] ...

### STEP 3 — Extract abnormal findings from Laboratory Tests
Input:
Laboratory Tests: {lab}

Instructions:
- Same as above
- Mark abnormal points with [!]

Response Format:
Key findings:
- [!] ...
- [!] ...

### STEP 4 — Extract abnormal findings from Radiology reports
Input:
Radiology Imaging Reports: {imaging}

Instructions:
- Same as above
- Mark abnormal points with [!]

Response Format:
Key findings:
- [!] ...
- [!] ...

### STEP 5 — Diagnostic reasoning for Appendicitis
Instructions:
You have extracted abnormal clinical findings in the previous steps, including:
- History of Present Illness (HPI)
- Physical Examination (PE)
- Laboratory Tests
- Radiology Imaging

Now, based on those abnormal findings, evaluate whether **acute appendicitis** is present or can be ruled out.

Guidance:
- Base your reasoning strictly on the abnormalities identified earlier
- Do NOT guess or invent information not previously extracted
- Classify the diagnostic likelihood using one of the following:
    • Definite — Strong clinical, lab, and imaging support  
    • Probable — Most but not all criteria are met  
    • Possible — Some suggestive signs, but not enough evidence  
    • Unlikely — Insufficient support or more consistent with alternative diagnoses

Response Format:
<think> ...reasoning process... </think>  
<evidence>: Refer to key abnormalities from HPI / PE / Lab / Imaging  
<Final Answer>: whether the disease established or not, e.g., appendicitis eliminated (unlikely), appendicitis established (probable)
<Confidence>: [e.g., 85%]  
<Others>: Any additional notes or alternative considerations  

### STEP 6 — Diagnostic reasoning for Cholecystitis
(same as above)

### STEP 7 — Diagnostic reasoning for Diverticulitis
(same as above)

### STEP 8 — Diagnostic reasoning for Pancreatitis
(same as above)

### STEP 9 — Final diagnosis conclusion
Instructions:
- Do NOT perform diagnostic reasoning again
- Review the four previous conclusions
- Select only ONE final primary diagnosis
- Mention any suspected others if relevant

Response Format:
<think> Reviewing diagnostic likelihoods... </think>
<Summarization>:
- Appendicitis: unlikely
- Cholecystitis: possible
- Diverticulitis: probable
- Pancreatitis: ruled out

<Final Diagnosis>: Diverticulitis
<Confidence>: 90%

Note: Complete all steps at once
"""),
            MessagesPlaceholder(variable_name='hpi'), 
            MessagesPlaceholder(variable_name='pe'), 
            MessagesPlaceholder(variable_name='lab'), 
            MessagesPlaceholder(variable_name='imaging'), 
        ])

    def prompt_flexible(self):
        self.holistic_hypo = ChatPromptTemplate.from_messages([
            HumanMessage(content="""                   
You are a diagnostic reasoning assistant. Given a full clinical case including patient history, physical exam, labs, and imaging, your task is to:

1. Generate 1-2 differential diagnoses that could explain the patient's presentation.
2. For each diagnosis, include:
   -  Supporting evidence: signs, symptoms, labs, imaging that support this diagnosis
   -  Contradictory evidence: findings that argue against this diagnosis or favor an alternative
   -  Clinical reasoning: briefly explain why this diagnosis fits and what would make it more/less likely
   -  Priority level: based on a combination of diagnostic likelihood and clinical urgency (e.g., High, Medium, Low)

3. If there are any prior tests or interventions (e.g., CT, lipase, antibiotics), infer what the likely diagnostic hypothesis was at the time that led to ordering them (reverse clinical reasoning).

4. Format the output as structured JSON.

Example output:
{
  "Possible_Diagnoses": [
    {
      "Name": "Acute Pancreatitis",
      "Supportive_Evidence": [
        "Severe epigastric pain radiating to back",
        "Lipase 5x upper limit of normal",
        "CT shows peripancreatic edema"
      ],
      "Against_Evidence": [
        "No nausea/vomiting",
        "Normal ALT"
      ],
      "Clinical_Reasoning": "The combination of epigastric pain and elevated lipase strongly suggests pancreatitis. CT findings confirm inflammatory changes. However, absence of ALT elevation slightly weakens the biliary etiology.",
      "Priority": "High"
    },
    {
      "Name": "Acute Cholecystitis",
      "Supportive_Evidence": [...],
      "Against_Evidence": [...],
      "Clinical_Reasoning": "...",
      "Priority": "Medium"
    }
  ],
  "Inferred_Reasoning_For_Tests": [
    {
      "Test": "CT Abdomen",
      "Rationale": "Likely ordered to evaluate for appendicitis or pancreatitis given focal abdominal pain and leukocytosis."
    },
    {
      "Test": "Lipase",
      "Rationale": "Ordered due to suspicion of pancreatitis based on epigastric pain pattern."
    }
  ]
}

Now process the following case:
Chief Complaint: The patient presents with acute abdominal pain.
History of Present Illness: {hpi}
Physical Examination: {pe}
Laboratory Tests: {lab}
Radiology Imaging Reports: {imaging}
"""),
            MessagesPlaceholder(variable_name='hpi'), 
            MessagesPlaceholder(variable_name='pe'), 
            MessagesPlaceholder(variable_name='lab'), 
            MessagesPlaceholder(variable_name='imaging'), 
        ])

        self.deciding_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""
Below is your complete working memory: 
--- Context Window Start ---
{context}
--- Context Window End ---
                         

You are a clinical diagnostic reasoning agent acting as a second-reader assistant. You have received a set of differential diagnosis hypotheses generated by another assistant, each with:
-  Supporting evidence
-  Contradictory evidence
-  Clinical reasoning
-  Priority level

You also received reasoning about why certain diagnostic tests (e.g., CT, ultrasound, labs) were ordered — to help you understand the clinical thinking behind them.

Now, based on these hypotheses and your own judgment, you must decide what to do next in order to refine, verify, or finalize the diagnosis.

You have access to the following internalized **tools**, which represent different diagnostic behaviors you can take:
Only invoke one tool at one time.

---
Available Actions (you must choose one):
1. Tool name: {{planning}} 
   Use when: You want to plan how to verify a particular hypothesis.  
   Output: Suggest diagnostic steps, identify supporting and opposing evidence to classify the diagnosis \
        (definite/probable/possible/unlikely), and estimate your confidence (0-100%) in reasoning correctly about this.

2. Tool name: {{add_verification}}
   Use when: You have completed verification and want to store the result in `state['verification_result']`.

3. Tool name: {{reflecting}}
   Use when: You want to reflect on the overall reasoning so far, identify missing steps, contradictions, or possible errors. \
        This will help you course-correct or identify forgotten diagnoses.

4. Tool name: {{final_diagnosis}}
   Use when: You believe you have completed all necessary reasoning and can safely finalize the primary diagnosis. You must give:
   - The single most likely primary diagnosis
   - Diagnostic certainty (definite/probable/possible/unlikely)
   - Your own confidence (0-100%) that you can justify this decision
   - This tool would stop the process and return the result directly to the user!

---
You may also retrieve any patient information through the following tools:
- Tool name: {{hpi}}, with your intent to read the history of present illness
- Tool name: {{pe}}, with your intent to read the history of present illness
- Tool name:{{lab}}, with your intent to read the history of present illness
- Tool name:{{imaging}}, with your intent to read the radiology imaging reports

    When calling any of these tools, you must explicitly state:
        Which hypothesis (diagnosis) you are trying to verify or refute.
        What kind of evidence (symptoms, signs, tests) you expect to find or are looking for.
    For example:
        “To verify the hypothesis of acute appendicitis, I will retrieve the HPI and look for signs of migrating pain, anorexia, and fever.”
    
    The result of this tool call will be automatically inserted into your context for further reasoning. Be precise in your intent.

---
Choose your next step like a thoughtful clinician. Select the action that will most effectively progress the diagnosis based on:
- The priority and uncertainty of the working hypotheses
- Available vs missing data
- Your confidence in clinical interpretation
- Whether any prior reasoning seems flawed or incomplete

---
- All actions must be invoked at least one time
- Use as many tools as useful and whenever possible to get more information about the patient.
- Execute all tools you consider useful.
- Avoid finalizing the diagnosis too early unless you are confident that no further information or reasoning is needed.
- Do not provide a final diagnosis unless explicitly instructed. You must first verify all relevant hypotheses and complete appropriate reasoning steps.
- Be aware of potential cognitive biases during your reasoning and reflection.
- Specifically, watch for:
    1) Premature closure: concluding a diagnosis too early without full evidence
    2) Anchoring bias: over-relying on the initial information or leading hypothesis. These should be explicitly checked and corrected during your reflecting step.

"""),
            MessagesPlaceholder(variable_name='messages')
        ])

        self.planning = ChatPromptTemplate.from_messages([
            HumanMessage(content="""
You are a diagnostic reasoning agent. You are now tasked with planning how to verify a specific diagnostic hypothesis, based on all available patient information.

Your input includes a candidate diagnosis with some supportive and contradictory evidence, as well as preliminary clinical reasoning.

Your task is to:
1. Decide whether this diagnosis is currently Definite, Probable, Possible, or Unlikely.
2. Outline a flexible diagnostic reasoning path to further verify or refute this hypothesis.
3. Indicate your confidence in your current knowledge and reasoning, on a scale from 0% to 100%.

Do NOT rigidly rely on textbook criteria. Instead, reason like a skilled clinician who adapts to atypical presentations and incomplete information. Focus on **clinical logic**, not **fixed thresholds**.

Output format (in JSON):
{
  "Diagnosis_Name": "string",
  "Reasoning_Path": "string - your step-by-step plan to verify this diagnosis",
  "Current_Conclusion": "Definite / Probable / Possible / Unlikely",
  "Confidence": float (e.g., 0.75)
}

Inputs:
name: {name}
supp_evidence: {supp_evidence}
cont_evidence: {cont_evidence}
reasoning: {reasoning}
priority: {priority}
"""),
            MessagesPlaceholder(variable_name='name'),
            MessagesPlaceholder(variable_name='supp_evidence'),
            MessagesPlaceholder(variable_name='cont_evidence'),
            MessagesPlaceholder(variable_name='reasoning'),
            MessagesPlaceholder(variable_name='priority')
        ])

        self.hpi_tool_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""
You are reviewing the History of Present Illness (HPI) to evaluate a diagnostic hypothesis.

Your current reasoning intent is:
{{agent_intent}}

Carefully review the following HPI content:
{{hpi}}

Based on your diagnostic intent, extract **relevant clinical findings** from the HPI that either:
- Support your current hypothesis
- Contradict your current hypothesis
- Raise new possibilities that you had not considered

You should only focus on **findings relevant to your current intent**, not all abnormalities.

Format your output as:

{
  "Intent_Clarification": "...",
  "Supporting_Evidence": [...],
  "Contradictory_Evidence": [...],
  "Noteworthy_Other_Findings": [...], 
  "Confidence_Impact": "Increased" / "Decreased" / "Unchanged",
  "Updated_Intent": "..."  // (Optional: refine your hypothesis or add new differential)
}
"""),
            MessagesPlaceholder(variable_name='agent_intent'),
            MessagesPlaceholder(variable_name='hpi')
        ])

        self.pe_tool_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""
You are reviewing the Physical Examination (pe) to evaluate a diagnostic hypothesis.

Your current reasoning intent is:
{{agent_intent}}

Carefully review the following HPI content:
{{pe}}

Based on your diagnostic intent, extract **relevant clinical findings** from the PE that either:
- Support your current hypothesis
- Contradict your current hypothesis
- Raise new possibilities that you had not considered

You should only focus on **findings relevant to your current intent**, not all abnormalities.

Format your output as:

{
  "Intent_Clarification": "...",
  "Supporting_Evidence": [...],
  "Contradictory_Evidence": [...],
  "Noteworthy_Other_Findings": [...], 
  "Confidence_Impact": "Increased" / "Decreased" / "Unchanged",
  "Updated_Intent": "..."  // (Optional: refine your hypothesis or add new differential)
}
"""),
            MessagesPlaceholder(variable_name='agent_intent'),
            MessagesPlaceholder(variable_name='pe')
        ])

        self.lab_tool_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""
You are reviewing the laboratory tests (lab) to evaluate a diagnostic hypothesis.

Your current reasoning intent is:
{{agent_intent}}

Carefully review the following HPI content:
{{lab}}

Based on your diagnostic intent, extract **relevant clinical findings** from the lab that either:
- Support your current hypothesis
- Contradict your current hypothesis
- Raise new possibilities that you had not considered

You should only focus on **findings relevant to your current intent**, not all abnormalities.

Format your output as:

{
  "Intent_Clarification": "...",
  "Supporting_Evidence": [...],
  "Contradictory_Evidence": [...],
  "Noteworthy_Other_Findings": [...], 
  "Confidence_Impact": "Increased" / "Decreased" / "Unchanged",
  "Updated_Intent": "..."  // (Optional: refine your hypothesis or add new differential)
}
"""),
            MessagesPlaceholder(variable_name='agent_intent'),
            MessagesPlaceholder(variable_name='lab')
        ])

        self.imaging_tool_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""
You are reviewing the radiology imaging reports to evaluate a diagnostic hypothesis.

Your current reasoning intent is:
{{agent_intent}}

Carefully review the following HPI content:
{{imaging}}

Based on your diagnostic intent, extract **relevant clinical findings** from the imaging reports that either:
- Support your current hypothesis
- Contradict your current hypothesis
- Raise new possibilities that you had not considered

You should only focus on **findings relevant to your current intent**, not all abnormalities.

Format your output as:

{
  "Intent_Clarification": "...",
  "Supporting_Evidence": [...],
  "Contradictory_Evidence": [...],
  "Noteworthy_Other_Findings": [...], 
  "Confidence_Impact": "Increased" / "Decreased" / "Unchanged",
  "Updated_Intent": "..."  // (Optional: refine your hypothesis or add new differential)
}
"""),
            MessagesPlaceholder(variable_name='agent_intent'),
            MessagesPlaceholder(variable_name='imaging')
        ])

        self.reflecting_tool_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""
You are a clinical diagnostic assistant. Your current task is to reflect on the diagnostic reasoning process so far.

You are given:
- Context: Overall diagnostic process.
- Initial diagnostic hypotheses (already explored).
- Verification results of each diagnosis (including their evidence and confidence).

Please now conduct a structured reflection. Consider the following:

1. **Completeness Check**:
   - Have all initial hypotheses been adequately verified?
   - Are any hypotheses missing?
   - Are the verification results robust, or is there conflicting or weak evidence?

2. **Diagnostic Soundness**:
   - Are any verification results unreliable or questionable?
   - Is there a need for further differential diagnosis?
     - If yes: Propose which additional hypotheses to consider.
     - What should be checked next (labs, imaging, symptoms) to help distinguish them?

3. **Final Consideration**:
   - If there is one strong diagnosis with high evidence and clinical urgency:
     - You may recommend making it the final primary diagnosis.
     - However, **explicitly mention any remaining conflicts or doubts**.
   - Otherwise, advise the next step (e.g., more verification or data retrieval).

Please respond in structured paragraphs with clarity and clinical reasoning. Be cautious of cognitive biases such as premature closure or anchoring.

Input Data:
- Context: {{context}}
- Verification Results: {{verification_result}}

DO NOT output a diagnosis unless criteria for final diagnosis are clearly met.
"""),
            MessagesPlaceholder(variable_name='context'),
            MessagesPlaceholder(variable_name='verification_result')
        ])

        self.final_diagnosis_tool_prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""
You are a clinical diagnostic assistant. You are about to make the final diagnosis based on your full review of the case.

You have already:
- Reviewed the patient's full history, physical exam, labs, and imaging.
- Generated and verified multiple diagnostic hypotheses.
- Reflected on the diagnostic process to ensure completeness and soundness.

Now, you must:
1. Output exactly **one** primary diagnosis.
2. Assign a diagnostic certainty level:
   - "definite": clear, overwhelming evidence
   - "probable": strong but not absolute support
   - "possible": plausible but not well-supported
   - "unlikely": weak support, but still a consideration
3. Provide a **confidence score** (0-100%) indicating how sure you are.
4. Justify your choice clearly, citing key supporting and conflicting evidence.
5. Mention any remaining diagnostic uncertainties, if any.

⚠️ Only call this tool when you're reasonably confident and all relevant hypotheses have been considered. Premature closure or anchoring bias must be avoided.

Format your output as structured JSON:
{
  "Primary_Diagnosis": "Acute Pancreatitis",
  "Certainty_Level": "probable",
  "Confidence": 85%,
  "Justification": "Severe epigastric pain with lipase >5x ULN and CT findings. No strong conflicting evidence",
  "Diagnostic path": [the completed way you evaluated the final diagnosis]
}
                         
Input: 
## Tool-calling input: {{tool_call_input}} \n\n
## The context window: {{messages}}
"""),
            MessagesPlaceholder(variable_name='tool_call_input'),
            MessagesPlaceholder(variable_name='messages')
        ])




