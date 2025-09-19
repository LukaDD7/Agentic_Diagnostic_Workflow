import streamlit as st
import pandas as pd
import time
import os
import re
import sys
import datetime
import importlib

# --- 辅助函数：处理打包后的路径 ---
def resource_path(relative_path):
    """ 获取资源的绝对路径，无论是开发环境还是PyInstaller打包后 """
    try:
        # PyInstaller 创建一个临时文件夹，并将路径存储在 _MEIPASS 中
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- 导入你的自定义模块 ---
# 注意：如果 extract_patient_info.py 也需要访问文件，它也需要使用 resource_path
import extract_patient_info
importlib.reload(extract_patient_info)

# --- 配置 ---
# 使用 resource_path 包装所有文件路径！
# 你的 .spec 文件把 data 目录下的文件放到了 'data' 子目录中
# 所以路径应该是 'data/文件名.csv'
CASE_ID_FILE = resource_path('data/pre_experiment_case_list.csv')
ADW_RESULTS_FILE = resource_path('data/AGAP_fullset_filtered.csv') # <--- 假设这个文件也被放在data目录

# 这是一个绝对路径，必须修改！
# 打包时，你需要将这个路径下的所有文件都包含在 .spec 的 datas 中
# 并且在代码中通过 resource_path 访问它们
# PATIENT_INFO_PATH = '/media/luzhenyang/project/datasets/mimic_iv_ext_clinical_decision_abdominal/clinical_decision_making_for_abdominal_pathologies_1.1' # <--- 必须删除或修改
PATIENT_INFO_PATH = resource_path('data') # <--- 改成打包后的相对路径
PATIENT_INFO_FILE_NAMES = [
    'history_of_present_illness.csv', 
    'physical_examination.csv', 
    'laboratory_tests.csv',
    'radiology_reports.csv',
]

# --- 静态资源 ---
DISEASE_OPTIONS = ['Appendicitis', 'Cholecystitis', 'Diverticulitis', 'Pancreatitis', 'Other', 'Uncertain']
CONFIDENCE_LEVELS = ['0 (-Uncertain-)', '1 (Low confidence)', '2 (Moderate confidence)','3 (High confidence)']
# CONFIDENCE_LEVELS = list(range(1, 8)) # 1 to 7

# --- 数据提取和加载 ---
def format_text(text_input, ab=False):
    """
    健壮地处理 \\n -> 换行 + Markdown 格式，在 st.markdown 中兼容展示。
    """
    if not text_input:
        return ""

    # Step 1: 解析字符串结构
    if ab:
        try:
            processed_text = eval(text_input)
            if isinstance(processed_text, list):
                processed_text = "\n".join(str(item) for item in processed_text)
        except:
            processed_text = str(text_input)
    else:
        processed_text = str(text_input)

    # Step 2: 转换 '\\n' 为 '\n'
    processed_text = processed_text.replace('\\n', '\n')

    # Step 3: 合并多余空行，标准化换行
    processed_text = re.sub(r'\n\s*\n+', '\n\n', processed_text).strip()

    # Step 4: 加 Markdown 换行规则
    lines = processed_text.split('\n')
    formatted_lines = []
    for line in lines:
        line_strip = line.strip()

        # 空行 → 保留
        if not line_strip:
            formatted_lines.append("")
            continue

        # markdown 列表项 → 保留
        if re.match(r'^[-*+]\s+', line_strip):
            formatted_lines.append(line)
        else:
            # 普通文本 → 行尾加两个空格
            formatted_lines.append(line + "  ")

    return "\n".join(formatted_lines)


# --- 数据提取和加载函数 ---

@st.cache_resource # 使用cache_resource来缓存你的类实例
def get_patient_info_extractor():
    """初始化并缓存你的数据提取类实例"""
    return extract_patient_info.Template_customized(
        base_path_relative_to_exe=PATIENT_INFO_PATH,
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
        "hpi": hpi,
        "pe": pe,
        "lab": lab,
        "imaging": imaging
    }

def adw_output_processing(cell):
    if not isinstance(cell, str):
        return ""
    try:
        print(cell)
        eval_res = eval(cell)
        return eval_res[0].content
    except Exception as e:
        return f"[ERROR: {e}]"

@st.cache_data
def load_data():
    """加载病例列表和ADW结果"""
    try:
        case_df = pd.read_csv(CASE_ID_FILE)
        adw_df = pd.read_csv(ADW_RESULTS_FILE).set_index('hadm_id')
        # cols = ['ab_hpi', 'ab_pe', 'ab_lab', 'ab_ima']
        # adw_df[cols] = adw_df[cols].applymap(adw_output_processing)
        return case_df, adw_df
    except FileNotFoundError as e:
        st.error(f"Critical file not found: {e}")
        return None, None

def get_info_for_step(case_info, step):
    """根据当前步骤返回应显示的信息"""
    info_to_show = {
        'step': 'HPI',
        'info': f"**History of Present Illness:**\n{case_info['hpi']}"
    }
    if step >= 2:
        info_to_show = {
            'step': 'PE',
            'info': f"**Physical Examination:**\n{case_info['pe']}"
        }
    if step >= 3:
        info_to_show = {
            'step': 'Lab Tests',
            'info': f"**Laboratory Tests:**\n{case_info['lab']}"
        }
    if step >= 4:
        info_to_show = {
            'step': 'Imaging Reports',
            'info': f"**Imaging Reports:**\n{case_info['imaging']}"
        }
    return info_to_show
    
def get_adw_info_for_step(adw_data, step):
    """根据当前步骤返回ADW的辅助信息"""
    if step == 1: return {
        'step': 'HPI Abnormalities',
        'info': f"{adw_data['ab_hpi']}"
    }
    if step == 2: return {
        'step': 'PE Abnormalities',
        'info': f"{adw_data['ab_pe']}"
    }
    if step == 3: return {
        'step': 'Lab Abnormalities',
        'info': f"{adw_data['ab_lab']}"
    }
    if step == 4: return {
        'step': 'Imaging Abnormalities',
        'info': f"{adw_data['ab_ima']}"
    }
    if step == 5: return [
        {'step': 'App Assessment', 'info': f"{adw_data['diag_app']}"},
        {'step': 'Cho Assessment', 'info': f"{adw_data['diag_cho']}"},
        {'step': 'Pan Assessment', 'info': f"{adw_data['diag_pan']}"},
        {'step': 'Div Assessment', 'info': f"{adw_data['diag_div']}"}
    ]
    if step == 6: return {
        'step': 'Final Review',
        'info': f"{adw_data['final_review']}"
    }
    return ""

# --- 主应用 ---
st.set_page_config(layout="wide")
st.title("Clinical Diagnostic Process Simulation")

# --- Session State 初始化 ---
if 'app_state' not in st.session_state:
    st.session_state.app_state = "login"
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
# ---

# --- 登录界面 ---
if st.session_state.app_state == "login":
    st.header("Welcome, Physician")
    rater_id_input = st.text_input("Please enter your assigned Rater ID to begin (e.g., 'Physician_A'):")
    # st.session_state.rater_id = st.text_input("Please enter your assigned Evaluator ID to begin:")
    # 选择经验水平
    # st.session_state.experience_level = st.radio(
    #     "Please select your experience level:",
    #     options=["Junior Physician (Resident or <3 years experience)", "Senior Physician (>5 years experience)"],
    #     key="experience_select"
    # )
    if st.button("Start or Resume Exam"):
        if rater_id_input:
            st.session_state.rater_id = rater_id_input.strip()
            if getattr(sys, 'frozen', False):
                # 如果是打包后的 .exe
                output_dir = os.path.dirname(sys.executable)
            else:
                # 如果是直接运行 .py 脚本
                output_dir = os.path.abspath(".")
            rater_output_file = os.path.join(output_dir, f"results_{st.session_state.rater_id}.csv")
            
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
                        
                        case_list_df, _ = load_data() # 只加载病例列表以找到索引
                        case_ids_in_order = case_list_df['hadm_id'].tolist()
                        
                        # 找到最后一个病例的索引
                        last_case_idx = case_ids_in_order.index(last_case_id)
                        
                        if last_step < 6:
                            # 从当前病例的下一步开始
                            start_case_idx = last_case_idx
                            start_step = last_step + 1
                        else:
                            # 从下一个新病例的第一步开始
                            start_case_idx = last_case_idx + 1
                            start_step = 1

                        st.info(f"Welcome back, {st.session_state.rater_id}! Resuming from Case {start_case_idx + 1}, Step {start_step}.")
                except (pd.errors.EmptyDataError, KeyError, ValueError):
                    # 文件为空或格式错误，从头开始
                    st.info(f"Welcome, {st.session_state.rater_id}! Starting a new session.")

        # 设置会话状态
        st.session_state.current_case_idx = start_case_idx
        st.session_state.current_step = start_step
        st.session_state.app_state = "evaluation"
        st.session_state.step_start_time = time.time()
        st.rerun()
    else:
        st.warning("Evaluator ID is required.")

# --- 评估主界面 ---
elif st.session_state.app_state == "evaluation":
    case_list_df, adw_results_df = load_data()
    if case_list_df is None:
        st.stop()

    if st.session_state.current_case_idx >= len(case_list_df):
        st.session_state.app_state = "done"
        st.rerun()

    # 获取当前病例信息
    current_case = case_list_df.iloc[st.session_state.current_case_idx]
    hadm_id = current_case['hadm_id']
    
    # 动态加载信息
    extractor = get_patient_info_extractor()
    full_case_info = extract_full_case_info(hadm_id, extractor)
    adw_case_data = adw_results_df.loc[hadm_id]

    # --- 布局 ---
    st.header(f"Case {st.session_state.current_case_idx + 1} of {len(case_list_df)} - Step {st.session_state.current_step} of 6")
    st.subheader(f"Case ID: `{hadm_id}` | Difficulty: `{current_case['case_difficulty']}`")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Information")
        # 逐步显示信息
        for step in range(1, st.session_state.current_step+1):
            if step < 5:
                info_for_this_step = get_info_for_step(full_case_info, step)
                with st.expander(f"{info_for_this_step['step']}"):
                    st.markdown(f"<div style='height: 500px; overflow-y: auto; border: 1px solid #ccc; padding: 10px;'>{info_for_this_step['info'].replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)

    with col2:
        st.subheader("AI Assistant (ADW) Analysis")
        # 逐步显示ADW分析
        for step in range(1, st.session_state.current_step+1):
            adw_info_for_this_step = get_adw_info_for_step(adw_case_data, step)
            if step == 5:
                for diag_step in adw_info_for_this_step:
                    with st.expander(f"{diag_step['step']}"):
                        st.markdown(f"<div style='height: 500px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; background-color: #f0f2f6;'>{diag_step['info'].replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
            else:
                with st.expander(f"{adw_info_for_this_step['step']}"):
                    st.markdown(f"<div style='height: 500px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; background-color: #f0f2f6;'>{adw_info_for_this_step['info'].replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("Your Assessment")

    # --- 收集医生输入 ---
    with st.form(key=f"form_{hadm_id}_{st.session_state.current_step}"):
        # 让医生从包含"Uncertain"的列表中选择
        diagnosis = st.radio(
            "Based on the information available **at this step**, what is your primary diagnosis?",
            options=DISEASE_OPTIONS,
            horizontal=True, # 水平布局更紧凑
            key=f"diag_{hadm_id}_{st.session_state.current_step}"
        )

        # 信心评分始终可选
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
                # --- 记录结果的逻辑保持不变 ---
                st.session_state.results.append({
                    'rater_id': st.session_state.rater_id,
                    'experience_level': "Junior", # st.session_state.experience_level.split(" ")[0], # 只取 "Junior" or "Senior"
                    'case_id': hadm_id,
                    'case_difficulty': current_case['case_difficulty'],
                    'ground_truth': ground_truth_disease,
                    'step': st.session_state.current_step,
                    'condition': 'AI-Aided', # 或 'Unaided'
                    'diagnosis_choice': diagnosis,
                    'confidence_score': confidence.split(" ")[0],
                    'time_taken_sec': round(time_taken, 2)
                })
                
                if getattr(sys, 'frozen', False):
                    output_dir = os.path.dirname(sys.executable)
                else:
                    output_dir = os.path.abspath(".")
                rater_output_file = os.path.join(output_dir, f"results_{st.session_state.rater_id}.csv")
                # 使用追加模式保存到该评估者的专属文件
                new_result_df = pd.DataFrame([st.session_state.results[-1]])
                new_result_df.to_csv(rater_output_file, mode='a', header=not os.path.exists(rater_output_file), index=False)
                
                # 更新状态以进入下一步/下一病例
                if st.session_state.current_step < 6:
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
    # ... (下载按钮逻辑)