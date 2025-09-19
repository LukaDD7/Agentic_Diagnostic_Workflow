import pandas as pd
from torch.utils.data import Dataset

class Template_customized:
    def __init__(self, base_path, file_names, template=None, hadm_id=None):
        "Note that if hadm_id was missing, you can give it when calling fill_template methods"
        self.hadm_id = hadm_id
        self.template = template
        self.base_path = base_path
        self.file_names = file_names

    def laboratory_test_mapping(self,hadm_id) -> str:
        """
        find all relavent lab_test of an hadm_id and convert it to sre format
        """
        lab_test = pd.read_csv(f"{self.base_path}/laboratory_tests.csv")
        test_mapping = pd.read_csv(f"{self.base_path}/lab_test_mapping.csv")

        id_tests= lab_test[ lab_test['hadm_id'] == hadm_id ]
        lab_tests = []

        for index, content in id_tests.iterrows():
            # print(content.index)
            row = test_mapping[test_mapping['itemid'] == content['itemid']].copy()
            row.drop(columns=['itemid','corresponding_ids'], inplace=True)
            row.reset_index(drop=True, inplace=True)
            content_in_row = ", ".join(f"{col}: {row[col].values[0]}" for col in row.columns)
            # valuestr = content.drop(columns=['hadm_id','itemid']).copy().to_string(index=False, header=False)
            valuestr = ", ".join( f"{index}: {content[index]}" for index in content.index if index not in ['hadm_id', 'itemid'])
            valuestr_con_label = ' -- '.join([valuestr,content_in_row])
            lab_tests.append(valuestr_con_label)

        multi_line_string = "\n".join(lab_tests)
        return multi_line_string
    
    def laboratory_test_mapping_v2_llm(self,hadm_id) -> str:
        """
        find all relavent lab_test of an hadm_id and convert it to sre format
        """
        lab_test = pd.read_csv(f"{self.base_path}/laboratory_tests.csv")
        test_mapping = pd.read_csv(f"{self.base_path}/lab_test_mapping.csv")

        id_tests= lab_test[ lab_test['hadm_id'] == hadm_id ]
        lab_tests = []

        for index, content in id_tests.iterrows():
            # print(content.index)
            map = test_mapping[test_mapping['itemid'] == content['itemid']]
            # map.drop(columns=['itemid', 'corresponding_ids', 'count', 'category', 'fluid'], inplace=True)
            # map.reset_index(drop=True, inplace=True)
            # content_in_row = "  ".join(f"{col}: {map[col].values[0]}" for col in map.columns)
            label = map['label'].iloc[0]
            valuestr = content['valuestr']
            valuestr = valuestr + f" ({content['ref_range_lower']}, {content['ref_range_upper']})"
            valuestr_con_label = '  '.join([label, valuestr])
            lab_tests.append(valuestr_con_label)
        multi_line_string = "\n".join(lab_tests)
        return multi_line_string

    def extract_hpi(self, hadm_id) -> str:
        """
        extract history of present illness of hadm_id
        """
        hpi_df = pd.read_csv(f"{self.base_path}/history_of_present_illness.csv")
        hpi = hpi_df[hpi_df['hadm_id'] == hadm_id]['hpi'].values[0]
        return hpi

    def extract_microbiology(self, hadm_id):
        """mapping microbiology test"""
        micro_test = pd.read_csv(f"{self.base_path}/microbiology.csv")
        test_mapping = pd.read_csv(f"{self.base_path}/lab_test_mapping.csv")

        # concatenate all the lab_test hadm_id did
        id_tests= micro_test[ micro_test['hadm_id'] == hadm_id ]  # take any row that connect to hadm_id
        micro_result = []

        for index, content in id_tests.iterrows():
            row = test_mapping[test_mapping['itemid'] == content['test_itemid']].copy()
            row.drop(columns=['itemid','corresponding_ids'], inplace=True)
            row.reset_index(drop=True, inplace=True)
            content_in_row = ", ".join(f"{col}: {row[col].values[0]}" for col in row.columns)
            valuestr = content['valuestr']
            valuestr_con_label = ' -- '.join([valuestr, content_in_row])
            micro_result.append(valuestr_con_label)
        
        formatted_micro = "\n".join(micro_result)
        return formatted_micro

    def extract_pe(self, hadm_id):
        pe_df = pd.read_csv(f"{self.base_path}/physical_examination.csv")
        pe = pe_df[pe_df['hadm_id'] == hadm_id]['pe']
        return pe.iloc[0] if not pe.empty else None
    
    def extract_rr(self, hadm_id) -> str:
        rr_df = pd.read_csv(f"{self.base_path}/radiology_reports.csv")
        rr = rr_df[rr_df['hadm_id'] == hadm_id]
        all_rr = []

        # concat all radiology reports
        for index, content in rr.iterrows():
            content = content.drop(index=['hadm_id', 'note_id']).copy()
            content_in_row = ', '.join(f"{index}: {content[index]}" for index in content.index)
            all_rr.append(content_in_row)

        formatted_rr = '\n'.join(all_rr)
        return formatted_rr

class MedicalDataset(Dataset):
    def __init__(self, csv_file_path, subset=False, seed=42):
        """
        Args:
            csv_file_path(str): Path to the CSV file.
        """
        if subset:
            full_data = pd.read_parquet(csv_file_path)
            random_seed = seed
            samples_per_label = 20
            random_samples = full_data.groupby('diagnosis').apply(lambda x: x.sample(n=samples_per_label, random_state=random_seed))
            self.data = random_samples.reset_index(drop=True)
        else:
            self.data = pd.read_parquet(csv_file_path)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        """
        Fetch a sample (patient_info, diagnosis) given an index.
        """
        row = self.data.iloc[id]
        return {
            'patient_info': row['patient_info'],
            'diagnosis': row['diagnosis'],
            'hadm_id': row['hadm_id']
        }

