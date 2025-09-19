# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_submodules, copy_metadata
import sys

# Streamlit 应用主文件名
streamlit_app_main_file = "examination_wo_ADW_app.py"
entry_point_script = "run_streamlit_wo_ADW.py"

# 查找 Streamlit 库的路径，以包含所有必要文件
# 找到 Python 解释器，然后根据 Streamlit 的安装路径推断
# This part can be tricky and might need adjustment depending on your setup.
# The below is a common way to find streamlit's data files.
def find_streamlit_path():
    try:
        import streamlit
        # sys.prefix should point to the active Conda environment root
        return os.path.dirname(streamlit.__file__)
    except ImportError:
        print("Error: Streamlit not found. Please ensure it's installed in the active environment.")
        sys.exit(1)
    except Exception as e:
        print(f"Error finding Streamlit path: {e}")
        sys.exit(1)

streamlit_lib_path = find_streamlit_path()
if streamlit_lib_path is None:
    print("Error: Could not determine Streamlit installation path.")
    sys.exit(1)

a = Analysis(
    [entry_point_script], # The script that PyInstaller will compile
    # Get the directory of the spec file itself
    # In many cases, PyInstaller sets the current working directory to the spec file's directory.
    # So, os.getcwd() should point to your project root.
    # Or, we can get the spec file's directory by finding the script that runs pyinstaller.
    # The current script is __main__.py from PyInstaller.
    # Let's try to get the directory where pyinstaller.exe is running from.
    # Or more reliably, use the directory of the script that *calls* pyinstaller.
    # A simpler and often working way is to directly use os.getcwd() if pyinstaller is run from the project dir.

    # Try using os.getcwd() if you are running pyinstaller from the project root
    pathex=[
        os.getcwd(),  # This refers to the directory where you run the 'pyinstaller' command.
                      # Ensure you are in your project root (C:\Users\W11\Desktop\Human_Examination\) when you run PyInstaller.
        os.path.join(sys.prefix, 'Lib', 'site-packages') # Path to the site-packages of your ACTIVE Conda environment.
                                                       # sys.prefix should correctly point to E:\anaconda3\envs\pyins
                                                       # if your 'pyins' environment is activated.
    ],
    binaries=[],
    # 'datas' option to copy additional files/directories needed by the app.
    # This is crucial for Streamlit as it needs its internal files.
    # We'll try to include common Streamlit data directories.
    datas=[
        # ...
        (streamlit_app_main_file, "."),
        ("extract_patient_info.py", "."),
        # 你的所有 CSV 数据文件，都放到包内的 'data' 文件夹下
        ("data/pre_experiment_case_list.csv", "data"),
        ("data/results_ADW-own.csv", "data"),
        ("data/history_of_present_illness.csv", "data"), # Add all data files
        ("data/physical_examination.csv", "data"),
        ("data/laboratory_tests.csv", "data"),
        ("data/radiology_reports.csv", "data"),
        ("data/lab_test_mapping.csv", "data"),
        (os.path.join(streamlit_lib_path, 'static'), 'streamlit/static'),
        # 包含 Streamlit 的前端静态文件和元数据，这是必须的
        *copy_metadata('streamlit'), 
        # ... Streamlit static files ...
        # Include Streamlit's internal static files
            (os.path.join(streamlit_lib_path, 'static'), 'streamlit/static'),
            # Optionally try adding other Streamlit internal dirs if needed:
            # (os.path.join(streamlit_lib_path, 'lib'), 'streamlit/lib'),
            # (os.path.join(streamlit_lib_path, 'proto'), 'streamlit/proto'),
            # (os.path.join(streamlit_lib_path, 'scriptrunner'), 'streamlit/scriptrunner'),
            # (os.path.join(streamlit_lib_path, 'web'), 'streamlit/web'),
    ],
    hiddenimports=[
        'streamlit',
        'altair', # Streamlit 常用，最好加上
        'pyarrow', # Pandas 加速，有时需要
        'pandas',
        'numpy',
        'extract_patient_info',
        # 确保所有 streamlit 的子模块都被包含
        *collect_submodules('streamlit') 
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# For debugging if Streamlit path isn't found or data is missing
if streamlit_lib_path is None:
    print("Warning: Could not find Streamlit installation path. Data files might be missing.")
    # As a fallback, you might want to add common paths manually if you know them.
    # e.g., sys._getframe(0).f_code.co_filename is the path of the current script (run_streamlit.py)
    # This is where your app.py and run_streamlit.py reside.
    # Your app.py and its data files are usually included via pathex if they are in the same dir.
    # But Streamlit's internal data files are important.

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# For a single-file executable (.exe)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='streamlit_app_wo_ADW', # The name of your .exe file
    # --- Console Window Handling ---
    # console=True means a console window will appear when the .exe is run.
    # console=False means no console window will appear (GUI app).
    # For Streamlit, you usually want the console to see logs or errors, so keep it True.
    # However, if Streamlit opens a browser automatically, you might prefer False for a cleaner user experience.
    # Let's try False for a cleaner user experience, as Streamlit usually opens the browser.
    # If you get errors, change it back to True.
    console=True,
    # --- Icon ---
    # You can specify an icon for your .exe file.
    # Replace 'your_icon.ico' with the path to your .ico file.
    # The icon file must be in the .ico format.
    # --- Additional Settings ---
    # disable_windowed_traceback=True, # Hides traceback if an error occurs in GUI mode.
    # target_arch='x86_64', # Specify architecture if needed (usually auto-detected)
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='streamlit_app_wo_ADW',
)

# If you want a folder with all dependencies instead of a single .exe file:
# For a folder distribution, you would typically use:
# pyinstaller --onedir --name streamlit_app --collect-data streamlit --add-data "your_app.py;." --add-data "icon.ico;." --add-data "your_data.csv;." --windowed --icon="icon.ico" run_streamlit.py
# The .spec file is more robust for managing complex dependencies.