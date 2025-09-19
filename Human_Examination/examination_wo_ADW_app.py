# examination_wo_ADW_app.py (修正版)

import streamlit as st
import pandas as pd
import time
import os
import re
import sys  # <--- 导入 sys
import datetime
import importlib

# --- 辅助函数：处理打包后的路径 (必须添加) ---
def resource_path(relative_path):
    """ 获取资源的绝对路径，无论是开发环境还是PyInstaller打包后 """
    try:
        # PyInstaller 创建一个临时文件夹，并将路径存储在 _MEIPASS 中
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- 导入你的自定义模块 ---
import extract_patient_info
importlib.reload(extract_patient_info)

# --- 配置 ---
# 使用 resource_path 包装所有数据文件路径
# 这些文件需要被打包到 'data' 子目录下
CASE_ID_FILE = resource_path('data/pre_experiment_case_list.csv')
# 这个版本不需要 ADW_RESULTS_FILE，所以可以注释或删除
# ADW_RESULTS_FILE = resource_path('data/AGAP_fullset_filtered.csv')

# 这个列表用于初始化 extractor
PATIENT_INFO_FILE_NAMES = [
    'history_of_present_illness.csv', 
    'physical_examination.csv', 
    'laboratory_tests.csv',
    'radiology_reports.csv',
]

# --- 静态资源 ---
DISEASE_OPTIONS = ['Appendicitis', 'Cholecystitis', 'Diverticulitis', 'Pancreatitis', 'Other', 'Uncertain']
CONFIDENCE_LEVELS = ['0 (-Uncertain-)', '1 (Low confidence)', '2 (Moderate confidence)','3 (High confidence)']

# --- 数据提取和加载函数 ---
# format_text 函数保持不变...
def format_text(text_input, ab=False):
    if not text_input:
        return ""
    if ab:
        try:
            processed_text = eval(text_input)
            if isinstance(processed_text, list):
                processed_text = "\n".join(str(item) for item in processed_text)
        except:
            processed_text = str(text_input)
    else:
        processed_text = str(text_input)
    processed_text = processed_text.replace('\\n', '\n')
    processed_text = re.sub(r'\n\s*\n+', '\n\n', processed_text).strip()
    lines = processed_text.split('\n')
    formatted_lines = []
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            formatted_lines.append("")
            continue
        if re.match(r'^[-*+]\s+', line_strip):
            formatted_lines.append(line)
        else:
            formatted_lines.append(line + "  ")
    return "\n".join(formatted_lines)


@st.cache_resource # 使用cache_resource来缓存你的类实例
def get_patient_info_extractor():
    """初始化并缓存你的数据提取类实例"""
    # ✅ 传递打包后数据文件所在的目标文件夹名
    return extract_patient_info.Template_customized(
        base_path_relative_to_exe="data", 
        file_names=PATIENT_INFO_FILE_NAMES
    )

def extract_full_case_info(hadm_id, extractor):
    """根据hadm_id提取单个病例的四项关键信息"""
    hadm_id = int(hadm_id) # 确保是整数类型
    hpi = extractor.extract_hpi(hadm_id)
    pe = extractor.extract_pe(hadm_id)
    lab = extractor.laboratory_test_mapping_v2_llm(hadm_id)
    imaging = extractor.extract_rr(hadm_id)
    return {
        "hpi": hpi, "pe": pe, "lab": lab, "imaging": imaging
    }

@st.cache_data
def load_data():
    """加载病例列表"""
    try:
        case_df = pd.read_csv(CASE_ID_FILE) # 这里的CASE_ID_FILE已经是绝对路径了
        return case_df, None # 这个版本不需要 adw_df
    except FileNotFoundError as e:
        st.error(f"Critical file not found: {e}")
        return None, None

def get_info_for_step(case_info, step):
    """根据当前步骤返回应显示的信息"""
    info_to_show = {'step': 'HPI', 'info': f"**History of Present Illness:**\n{case_info['hpi']}"}
    if step >= 2:
        info_to_show = {'step': 'PE', 'info': f"**Physical Examination:**\n{case_info['pe']}"}
    if step >= 3:
        info_to_show = {'step': 'Lab Tests', 'info': f"**Laboratory Tests:**\n{case_info['lab']}"}
    if step >= 4:
        info_to_show = {'step': 'Imaging Reports', 'info': f"**Imaging Reports:**\n{case_info['imaging']}"}
    return info_to_show
    
# 主应用、Session State 初始化等代码保持不变...
st.set_page_config(layout="wide")
st.title("Clinical Diagnostic Process Simulation (Unaided)")

# --- Session State 初始化 ---
if 'app_state' not in st.session_state:
    st.session_state.app_state = "login"
# ... 其他 session state 初始化 ...
if 'rater_id' not in st.session_state:
    st.session_state.rater_id = ""
if 'current_case_idx' not in st.session_state:
    st.session_state.current_case_idx = 0
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'step_start_time' not in st.session_state:
    st.session_state.step_start_time = None
if 'results' not in st.session_state:
    st.session_state.results = []

# --- 登录界面 ---
if st.session_state.app_state == "login":
    st.header("Welcome, Physician")
    rater_id_input = st.text_input("Please enter your assigned Rater ID to begin (e.g., 'Physician_A'):")
    if st.button("Start or Resume Exam"):
        if rater_id_input:
            st.session_state.rater_id = rater_id_input.strip()
            
            # --- 修正结果文件的保存路径 ---
            if getattr(sys, 'frozen', False):
                output_dir = os.path.dirname(sys.executable)
            else:
                output_dir = os.path.abspath(".")
            rater_output_file = os.path.join(output_dir, f"results_{st.session_state.rater_id}_unaided.csv")

            # --- 进度加载逻辑 ---
            start_case_idx = 0
            start_step = 1
            if os.path.exists(rater_output_file):
                try:
                    completed_df = pd.read_csv(rater_output_file)
                    if not completed_df.empty:
                        last_entry = completed_df.iloc[-1]
                        last_case_id = last_entry['case_id']
                        last_step = last_entry['step']
                        case_list_df, _ = load_data()
                        case_ids_in_order = case_list_df['hadm_id'].tolist()
                        last_case_idx = case_ids_in_order.index(last_case_id)
                        
                        if last_step < 4:
                            start_case_idx = last_case_idx
                            start_step = last_step + 1
                        else:
                            start_case_idx = last_case_idx + 1
                            start_step = 1
                        st.info(f"Welcome back, {st.session_state.rater_id}! Resuming from Case {start_case_idx + 1}, Step {start_step}.")
                except (pd.errors.EmptyDataError, KeyError, ValueError, FileNotFoundError):
                    st.info(f"Welcome, {st.session_state.rater_id}! Starting a new session.")

            st.session_state.current_case_idx = start_case_idx
            st.session_state.current_step = start_step
            st.session_state.app_state = "evaluation"
            st.session_state.step_start_time = time.time()
            st.rerun()
        else:
            st.warning("Evaluator ID is required.")

# --- 评估主界面 ---
elif st.session_state.app_state == "evaluation":
    case_list_df, _ = load_data()
    if case_list_df is None:
        st.stop()

    if st.session_state.current_case_idx >= len(case_list_df):
        st.session_state.app_state = "done"
        st.rerun()

    current_case = case_list_df.iloc[st.session_state.current_case_idx]
    hadm_id = current_case['hadm_id']
    
    extractor = get_patient_info_extractor()
    if extractor is None:
        st.error("Data extractor could not be initialized. Please check the logs.")
        st.stop()
        
    full_case_info = extract_full_case_info(hadm_id, extractor)

    st.header(f"Case {st.session_state.current_case_idx + 1} of {len(case_list_df)} - Step {st.session_state.current_step} of 4")
    st.subheader(f"Case ID: `{hadm_id}` | Difficulty: `{current_case['case_difficulty']}`")
    st.markdown("---")

    col1 = st.columns(1)
    with col1[0]:
        st.subheader("Patient Information")
        for step in range(1, st.session_state.current_step + 1):
            if step < 5:
                info_for_this_step = get_info_for_step(full_case_info, step)
                with st.expander(f"{info_for_this_step['step']}", expanded=True):
                    st.markdown(f"<div style='height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 10px;'>{info_for_this_step['info'].replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Your Assessment")

    with st.form(key=f"form_{hadm_id}_{st.session_state.current_step}"):
        diagnosis = st.radio(
            "Based on the information available **at this step**, what is your primary diagnosis?",
            options=DISEASE_OPTIONS,
            horizontal=True,
            key=f"diag_{hadm_id}_{st.session_state.current_step}"
        )
        confidence = st.radio(
            "How confident are you in this assessment?",
            options=CONFIDENCE_LEVELS,
            horizontal=True,
            key=f"conf_{hadm_id}_{st.session_state.current_step}"
        )
        submitted = st.form_submit_button("Submit and Proceed")

        if submitted:
            # 测试交互项排斥检测
            diag = st.session_state[f"diag_{hadm_id}_{st.session_state.current_step}"]
            conf = st.session_state[f"conf_{hadm_id}_{st.session_state.current_step}"]
            if diag == 'Uncertain' and not conf.startswith('0'):
                st.error("❌ When diagnosis is 'Uncertain', confidence must be '0 (-Uncertain-)'")
            elif diag != 'Uncertain' and conf.startswith('0'):
                st.error("❌ When diagnosis is not 'Uncertain', confidence cannot be '0 (-Uncertain-)'")
            else:
                end_time = time.time()
                time_taken = end_time - st.session_state.step_start_time
                ground_truth_disease = current_case['disease']

                st.session_state.results.append({
                    'rater_id': st.session_state.rater_id,
                    'experience_level': "Junior",
                    'case_id': hadm_id,
                    'case_difficulty': current_case['case_difficulty'],
                    'ground_truth': ground_truth_disease,
                    'step': st.session_state.current_step,
                    'condition': 'Unaided',
                    'diagnosis_choice': diagnosis,
                    'confidence_score': confidence.split(" ")[0],
                    'time_taken_sec': round(time_taken, 2)
                })
                
                # --- 修正结果文件的保存路径 ---
                if getattr(sys, 'frozen', False):
                    output_dir = os.path.dirname(sys.executable)
                else:
                    output_dir = os.path.abspath(".")
                rater_output_file = os.path.join(output_dir, f"results_{st.session_state.rater_id}_unaided.csv")

                new_result_df = pd.DataFrame([st.session_state.results[-1]])
                new_result_df.to_csv(rater_output_file, mode='a', header=not os.path.exists(rater_output_file), index=False)
                
                if st.session_state.current_step < 4:
                    st.session_state.current_step += 1
                else:
                    st.session_state.current_case_idx += 1
                    st.session_state.current_step = 1
                
                st.session_state.step_start_time = time.time()
                st.rerun()

# --- 完成界面 ---
elif st.session_state.app_state == "done":
    st.success("Exam Complete!")
    st.balloons()
    st.write(f"Thank you, {st.session_state.rater_id}! Your contributions are invaluable. Please send the result file back to the research team.")