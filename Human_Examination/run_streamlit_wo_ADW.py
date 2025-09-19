import sys
import os
import streamlit.web.cli as stcli
import webbrowser
import time
import threading

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def open_browser():
    """
    Waits for the server to start and then opens the browser.
    This function will run in a separate thread.
    """
    print("Browser-opener thread started. Waiting for server...")
    # Give the server a few seconds to initialize
    time.sleep(4)
    print("Opening browser at http://localhost:8501")
    webbrowser.open("http://localhost:8501")

def run_streamlit():
    """
    Prepares and runs the Streamlit server.
    This function MUST be called from the main thread.
    """
    streamlit_app_file = resource_path("examination_wo_ADW_app.py")
    
    print(f"Streamlit app file found at: {streamlit_app_file}")
    
    sys.argv = [
        "streamlit",
        "run",
        streamlit_app_file,
        "--server.port", "8501",
        "--server.headless", "true",
        "--global.developmentMode", "false",
    ]
    
    print(f"Starting Streamlit with args: {sys.argv}")
    # This will start the server and block until the server is shut down.
    stcli.main()

if __name__ == "__main__":
    print("Application starting...")

    # --- THIS IS THE NEW LOGIC ---
    
    # 1. Start the browser-opening function in a separate, non-blocking thread.
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True  # Ensures thread exits when main program does
    browser_thread.start()

    # 2. Run the Streamlit server in the main thread.
    #    This is a blocking call and will keep the application alive.
    #    It can now correctly set up its signal handlers.
    run_streamlit()
    
    # This line will only be reached when the Streamlit server is shut down.
    print("Application has been closed.")



# import subprocess
# import sys
# import os
# import streamlit.web.cli as stcli
# import webbrowser
# import time

# # --- 辅助函数：处理打包后的路径 ---
# def resource_path(relative_path):
#     """ 获取资源的绝对路径，无论是开发环境还是PyInstaller打包后 """
#     try:
#         # PyInstaller 创建一个临时文件夹，并将路径存储在 _MEIPASS 中
#         base_path = sys._MEIPASS
#     except Exception:
#         # 在开发环境中，sys._MEIPASS 不存在
#         base_path = os.path.abspath(".")

#     return os.path.join(base_path, relative_path)

# # --- 主逻辑 ---
# # 1. 定位 Streamlit 应用主文件
# #    我们使用 resource_path 来确保在 .exe 中也能找到它
# streamlit_app_file = resource_path("examination_app.py")

# print(f"Checking for Streamlit app file: {streamlit_app_file}")
# if not os.path.exists(streamlit_app_file):
#     print(f"Error: Streamlit application file '{streamlit_app_file}' not found.")
#     # 在打包后的环境中，打印这个信息可能一闪而过，但对于调试很有用
#     time.sleep(10)
#     sys.exit(1)

# # 2. 构建 Streamlit 运行命令
# #    --server.headless=true 阻止 Streamlit 自己打开浏览器，由我们来控制
# command = [
#     sys.executable,
#     "-m", "streamlit", "run", streamlit_app_file,
#     "--server.port", "8501",
#     "--server.headless", "true"
# ]

# print(f"Executing command: {' '.join(command)}")

# # 3. 执行 Streamlit 命令
# try:
#     # 使用 Popen 启动 Streamlit 服务器，这是一个非阻塞调用
#     process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    
#     # 给服务器一点时间来启动
#     print("Streamlit server starting...")
#     time.sleep(5) 

#     # 4. 手动打开浏览器
#     webbrowser.open("http://localhost:8501")
#     print("Browser opened. The application is running.")
#     print("The console window will remain open. Close it to shut down the application.")

#     # 5. 等待子进程结束 (这是关键！)
#     #    主程序会在这里阻塞，直到 Streamlit 进程被关闭（例如用户关闭了控制台窗口）
#     #    这样可以防止主程序退出导致 Streamlit 服务器被终止
    
#     # 你可以实时打印输出来进行调试
#     # for line in iter(process.stdout.readline, ''):
#     #     sys.stdout.write(line)
    
#     process.wait()

# except FileNotFoundError:
#     print("Error: 'streamlit' command not found. Is Streamlit installed?")
#     time.sleep(10)
#     sys.exit(1)
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")
#     if hasattr(e, 'stderr'):
#         print("--- Stderr ---")
#         print(e.stderr)
#     time.sleep(10)
#     sys.exit(1)

# # 获取 Streamlit 应用文件的路径
# # 假设 Streamlit 应用的主文件名为 'your_app.py'
# streamlit_app_file = "examination_app.py" # <<< 修改这里为你实际的 Streamlit 应用主文件名

# # 检查 Streamlit 应用文件是否存在
# print(f"Checking for Streamlit app file: {os.path.abspath(streamlit_app_file)}")
# if not os.path.exists(streamlit_app_file):
#     print(f"Error: Streamlit application file '{streamlit_app_file}' not found.")
#     print("Please make sure 'run_streamlit.py' is in the same directory as your Streamlit app.")
#     sys.exit(1)

# # 构建 Streamlit 运行命令
# # sys.executable 会找到当前运行的 Python 解释器（在 PyInstaller 打包后就是那个被包含的解释器）
# command = [sys.executable, "-m", "streamlit", "run", streamlit_app_file,
#         "--server.address", "127.0.0.1",
#         "--server.port", "8501",
#         "--server.headless", "true"  # 关闭 Streamlit 自动打开浏览器
#     ]

# # 打印将要执行的命令（用于调试）
# print(f"Executing: {' '.join(command)}")

# # 执行 Streamlit 命令
# try:
#     # 使用 subprocess.run 来执行命令
#     # capture_output=True 可以捕获 stdout 和 stderr
#     # text=True (或 encoding='utf-8') 使输出为字符串
#     # check=True 会在命令返回非零退出码时抛出 CalledProcessError
#     process = subprocess.Popen(command)
#     print("Streamlit process finished successfully.")
#     print("--- Streamlit Output ---")
#     print(process.stdout)
#     print("------------------------")
#     time.sleep(5)
#     # ✅ 自动打开浏览器指向 Streamlit 默认地址
#     webbrowser.open("http://localhost:8501")
# except FileNotFoundError:
#     print("Error: 'streamlit' command not found. Make sure Streamlit is installed in the environment.")
#     sys.exit(1)
# except subprocess.CalledProcessError as e:
#     print(f"Error executing Streamlit command (return code {e.returncode}):")
#     print("--- Streamlit Stdout ---")
#     print(e.stdout)
#     print("--- Streamlit Stderr ---")
#     print(e.stderr)
#     sys.exit(1)
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")
#     sys.exit(1)