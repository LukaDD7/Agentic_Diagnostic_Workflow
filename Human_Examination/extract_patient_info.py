import sys
import os
import pandas as pd

class Template_customized:
    def __init__(self, base_path_relative_to_exe, file_names, template=None, hadm_id=None):
        """
        base_path_relative_to_exe: 相对于打包后应用根目录的数据文件夹路径 (e.g., 'data').
                                  如果数据文件直接放在应用根目录，则为 '.'.
        file_names: 需要加载的数据文件名列表。
        """
        self.hadm_id = hadm_id
        self.template = template
        self.base_path_relative = base_path_relative_to_exe # 存储相对路径
        self.file_names = file_names
        self.data_folder_path = self._get_data_folder_path() # 计算数据文件夹的完整路径

        # 检查数据文件夹是否存在
        if not os.path.isdir(self.data_folder_path):
            print(f"Error: Data folder not found at {self.data_folder_path}")
            # 如果在开发模式下，尝试查找一个备用开发路径
            if not getattr(sys, 'frozen', False):
                dev_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.base_path_relative)
                if os.path.isdir(dev_path):
                    print(f"Using development fallback path: {dev_path}")
                    self.data_folder_path = dev_path
                else:
                    # 如果两种路径都找不到，则抛出错误
                    raise FileNotFoundError(f"Data folder not found: {self.data_folder_path} or {dev_path}")
            else:
                 raise FileNotFoundError(f"Data folder not found: {self.data_folder_path}")

        # 加载所有数据文件 (可选，也可以在需要时按需加载)
        # self.loaded_data = self._load_all_data()

    def _get_app_base_path(self):
        """
        获取应用程序的可执行文件所在的基路径。
        _MEIPASS 是 PyInstaller --onefile 模式下临时解压目录。
        对于 --onedir 模式, sys.executable 是 .exe 文件本身，其目录是基路径。
        """
        if getattr(sys, 'frozen', False):
            # Running in a PyInstaller bundle
            base_path = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        else:
            # Running as a normal Python script (during development)
            base_path = os.path.dirname(os.path.abspath(__file__))
        return base_path

    def _get_data_folder_path(self):
        """
        计算数据文件夹的完整路径。
        """
        app_base_path = self._get_app_base_path()
        # 根据 __init__ 中传入的相对路径来构建完整路径
        return os.path.join(app_base_path, self.base_path_relative)

    def _get_full_file_path(self, filename):
        """
        根据数据文件夹路径和文件名，获取文件的完整路径。
        """
        return os.path.join(self.data_folder_path, filename)

    # --- 你原来所有的读取 CSV 的方法都需要修改 ---
    # 示例：laboratory_test_mapping 方法
    def laboratory_test_mapping(self, hadm_id) -> str:
        """
        find all relavent lab_test of an hadm_id and convert it to sre format
        """
        # 修改读取 CSV 的路径
        try:
            lab_test_path = self._get_full_file_path('laboratory_tests.csv')
            test_mapping_path = self._get_full_file_path('lab_test_mapping.csv')

            # print(f"Loading lab_test from: {lab_test_path}") # Debugging
            # print(f"Loading test_mapping from: {test_mapping_path}") # Debugging

            lab_test = pd.read_csv(lab_test_path)
            test_mapping = pd.read_csv(test_mapping_path)
        except FileNotFoundError as e:
            print(f"Error loading data file: {e}")
            return "Error loading lab test data."
        except Exception as e:
            print(f"An error occurred during data loading: {e}")
            return "Error loading lab test data."

        id_tests = lab_test[lab_test['hadm_id'] == hadm_id]
        lab_tests = []

        for index, content in id_tests.iterrows():
            row = test_mapping[test_mapping['itemid'] == content['itemid']].copy()
            if row.empty: # Handle cases where itemid might not be in mapping
                continue
            row.drop(columns=['itemid','corresponding_ids'], inplace=True)
            row.reset_index(drop=True, inplace=True)
            content_in_row = ", ".join(f"{col}: {row[col].values[0]}" for col in row.columns)
            valuestr = content.drop(columns=['hadm_id','itemid']).copy().to_string(index=False, header=False)
            valuestr_con_label = ' -- '.join([valuestr,content_in_row])
            lab_tests.append(valuestr_con_label)

        multi_line_string = "\n".join(lab_tests)
        return multi_line_string

    # --- 复制其他方法，并修改其中读取 CSV 的路径 ---
    # 示例：extract_hpi 方法
    def extract_hpi(self, hadm_id) -> str:
        """
        extract history of present illness of hadm_id
        """
        try:
            hpi_path = self._get_full_file_path('history_of_present_illness.csv')
            hpi_df = pd.read_csv(hpi_path)
        except FileNotFoundError as e:
            print(f"Error loading data file: {e}")
            return "Error loading HPI data."
        except Exception as e:
            print(f"An error occurred during data loading: {e}")
            return "Error loading HPI data."

        # 确保 hadm_id 存在且 hpi 列存在
        if 'hadm_id' in hpi_df.columns and 'hpi' in hpi_df.columns:
            hpi_rows = hpi_df[hpi_df['hadm_id'] == hadm_id]
            if not hpi_rows.empty:
                return hpi_rows['hpi'].iloc[0]
            else:
                return f"No HPI found for hadm_id: {hadm_id}"
        else:
            return "Error: Missing 'hadm_id' or 'hpi' column in HPI file."

    # --- 复制并修改 laboratory_test_mapping_v2_llm, extract_microbiology, extract_pe, extract_rr ---
    # 确保所有 pd.read_csv() 都使用 self._get_full_file_path('your_file.csv')

    # 示例：extract_microbiology
    def extract_microbiology(self, hadm_id):
        """mapping microbiology test"""
        try:
            micro_test_path = self._get_full_file_path('microbiology.csv')
            test_mapping_path = self._get_full_file_path('lab_test_mapping.csv')

            micro_test = pd.read_csv(micro_test_path)
            test_mapping = pd.read_csv(test_mapping_path)
        except FileNotFoundError as e:
            print(f"Error loading data file: {e}")
            return "Error loading microbiology data."
        except Exception as e:
            print(f"An error occurred during data loading: {e}")
            return "Error loading microbiology data."

        id_tests = micro_test[micro_test['hadm_id'] == hadm_id]
        micro_result = []

        for index, content in id_tests.iterrows():
            row = test_mapping[test_mapping['itemid'] == content['test_itemid']].copy()
            if row.empty: # Handle missing itemid in mapping
                continue
            row.drop(columns=['itemid','corresponding_ids'], inplace=True)
            row.reset_index(drop=True, inplace=True)
            content_in_row = ", ".join(f"{col}: {row[col].values[0]}" for col in row.columns)
            valuestr = content['valuestr']
            valuestr_con_label = ' -- '.join([valuestr, content_in_row])
            micro_result.append(valuestr_con_label)

        formatted_micro = "\n".join(micro_result)
        return formatted_micro

    # 示例：extract_pe
    def extract_pe(self, hadm_id):
        try:
            pe_path = self._get_full_file_path('physical_examination.csv')
            pe_df = pd.read_csv(pe_path)
        except FileNotFoundError as e:
            print(f"Error loading data file: {e}")
            return "Error loading PE data."
        except Exception as e:
            print(f"An error occurred during data loading: {e}")
            return "Error loading PE data."

        pe = pe_df[pe_df['hadm_id'] == hadm_id]['pe']
        return pe.iloc[0] if not pe.empty else "No PE found for this hadm_id."

    # 示例：extract_rr
    def extract_rr(self, hadm_id) -> str:
        try:
            rr_path = self._get_full_file_path('radiology_reports.csv')
            rr_df = pd.read_csv(rr_path)
        except FileNotFoundError as e:
            print(f"Error loading data file: {e}")
            return "Error loading radiology reports."
        except Exception as e:
            print(f"An error occurred during data loading: {e}")
            return "Error loading radiology reports."

        rr = rr_df[rr_df['hadm_id'] == hadm_id]
        all_rr = []

        for index, content in rr.iterrows():
            content = content.drop(index=['hadm_id', 'note_id']).copy()
            content_in_row = ', '.join(f"{index}: {content[index]}" for index in content.index)
            all_rr.append(content_in_row)

        formatted_rr = '\n'.join(all_rr)
        return formatted_rr

    # 假设还有laboratory_test_mapping_v2_llm 方法，也需要这样修改：
    def laboratory_test_mapping_v2_llm(self, hadm_id) -> str:
        """
        find all relavent lab_test of an hadm_id and convert it to sre format
        """
        try:
            lab_test_path = self._get_full_file_path('laboratory_tests.csv')
            test_mapping_path = self._get_full_file_path('lab_test_mapping.csv')

            lab_test = pd.read_csv(lab_test_path)
            test_mapping = pd.read_csv(test_mapping_path)
        except FileNotFoundError as e:
            print(f"Error loading data file: {e}")
            return "Error loading lab test data v2."
        except Exception as e:
            print(f"An error occurred during data loading: {e}")
            return "Error loading lab test data v2."

        id_tests = lab_test[lab_test['hadm_id'] == hadm_id]
        lab_tests = []

        for index, content in id_tests.iterrows():
            map_row = test_mapping[test_mapping['itemid'] == content['itemid']]
            if map_row.empty: # Handle missing itemid in mapping
                continue
            label = map_row['label'].iloc[0]
            valuestr = content['valuestr']
            valuestr = valuestr + f" ({content['ref_range_lower']}, {content['ref_range_upper']})"
            valuestr_con_label = '  '.join([label, valuestr])
            lab_tests.append(valuestr_con_label)

        multi_line_string = "\n".join(lab_tests)
        return multi_line_string


# --- Streamlit App Code Snippet (in exam_app.py or where get_patient_info_extractor is defined) ---

# import extract_patient_info # Make sure this import works in your app

# def get_patient_info_extractor():
#     """初始化并缓存你的数据提取类实例"""
#     # Define the path to your data files RELATIVE to the application's root.
#     # If your data files are in a subfolder named 'data':
#     PATIENT_INFO_RELATIVE_PATH = 'data'
#     # If they are directly in the same folder as exam_app.py and extract_patient_info.py:
#     # PATIENT_INFO_RELATIVE_PATH = '.'
#
#     PATIENT_INFO_FILE_NAMES = [
#         'history_of_present_illness.csv',
#         'physical_examination.csv',
#         'laboratory_tests.csv',
#         'radiology_reports.csv',
#         'microbiology.csv', # Make sure this file name is also in the list if you use it
#         'lab_test_mapping.csv' # Also add mapping files if used
#     ]
#
#     try:
#         extractor = extract_patient_info.Template_customized(
#             base_path_relative_to_exe=PATIENT_INFO_RELATIVE_PATH,
#             file_names=PATIENT_INFO_FILE_NAMES
#         )
#         return extractor
#     except FileNotFoundError as e:
#         st.error(f"Error initializing data extractor: {e}")
#         return None
#     except Exception as e:
#         st.error(f"An unexpected error occurred during initialization: {e}")
#         return None