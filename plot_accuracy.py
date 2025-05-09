import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_accuracy(csv_file_path, output_dir):
    """
    讀取 CSV 檔案中的準確率數據並繪製圖表，然後儲存為 PNG 和 EPS 格式。

    參數:
    csv_file_path (str): CSV 檔案的路徑。
    output_dir (str): 圖表儲存的目錄。
    """
    try:
        # 讀取 CSV 檔案
        df = pd.read_csv(csv_file_path)

        # 檢查必要的欄位是否存在
        required_columns = ['checkpoint', 'test_accuracy', 'new_test_accuracy']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"錯誤：CSV 檔案 {csv_file_path} 中缺少以下欄位: {missing_columns}")
            print(f"可用的欄位有: {df.columns.tolist()}")
            print("請檢查檔案內容並確保所有必要的欄位都存在。")
            return

        epochs = df['checkpoint']
        accuracy1 = df['test_accuracy']
        accuracy2 = df['new_test_accuracy']

        # 繪製準確率圖表
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, accuracy1, marker='o', linestyle='-', label='test_accuracy')
        plt.plot(epochs, accuracy2, marker='x', linestyle='--', label='new_test_accuracy')
        
        plt.title('Accuracy vs. Epoch')
        plt.xlabel('Checkpoint (Epoch)')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)

        # 儲存為 PNG 格式
        png_path = os.path.join(output_dir, 'accuracy_plot.png')
        plt.savefig(png_path)
        print(f"準確率圖表已儲存為: {png_path}")

        # 儲存為 EPS 格式
        eps_path = os.path.join(output_dir, 'accuracy_plot.eps')
        plt.savefig(eps_path, format='eps')
        print(f"準確率圖表已儲存為: {eps_path}")

        plt.show()

    except FileNotFoundError:
        print(f"錯誤：找不到 CSV 檔案 {csv_file_path}")
    except Exception as e:
        print(f"繪製圖表時發生錯誤: {e}")

if __name__ == '__main__':
    # 設定 CSV 檔案路徑和輸出目錄
    csv_file = 'results/experiment_14/test_results.csv'
    output_directory = 'results/experiment_14/' # 圖表將儲存在與 CSV 相同的目錄

    # 呼叫函數繪製並儲存圖表
    plot_accuracy(csv_file, output_directory) 