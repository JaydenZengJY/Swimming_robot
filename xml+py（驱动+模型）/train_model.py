import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# 1. 从 model.py 复制 LSTM 网络结构
class LSTNetModified(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTNetModified, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# 2. 从 model.py 复制创建序列的函数
def create_sequences(input_data, output_data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(input_data) - sequence_length):
        seq = input_data[i:i + sequence_length]
        label = output_data[i + sequence_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# --- 主训练逻辑 ---
if __name__ == '__main__':
    # 3. 定义超参数 (应与 model.py 中的参数匹配)
    SEQUENCE_LENGTH = 16
    NUM_EPOCHS = 100
    DROPOUT_RATE = 0.21
    BATCH_SIZE = 64
    HIDDEN_SIZE = 60
    NUM_LAYERS = 2
    LEARNING_RATE = 0.001

    # 定义输入和输出列
    input_columns = ['water_speed', 'angle_l', 'angle_s', 'speed_l', 'speed_s']
    output_columns = ['TX/N*m', 'TY/N*m', 'TZ/N*m', 'FX/N', 'FY/N', 'FZ/N']
    all_columns = input_columns + output_columns

    # 4. 加载数据
    print("加载数据...")
    # 确保 training_data.csv 与此脚本在同一目录下
    try:
        df = pd.read_csv('training_data.csv')
    except FileNotFoundError:
        print("错误: 找不到 'training_data.csv'。请确保它和 train_model.py 在同一个文件夹里。")
        exit()
        
    df = df.dropna()

    # 5. 数据归一化和保存缩放器
    print("归一化数据并保存缩放器...")
    scalers = {}
    scaled_df = df.copy()
    
    # 准备保存缩放器的文件夹
    output_folder = f'sequence_length{SEQUENCE_LENGTH}_num_epochs{NUM_EPOCHS}_dropout_rate{DROPOUT_RATE}'
    os.makedirs(output_folder, exist_ok=True)

    for col in all_columns:
        scaler = MinMaxScaler()
        scaled_df[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler

        sanitized_col_name = col.replace('/', '_').replace('*', '_')
        scaler_filename = f"{sanitized_col_name}_scaler.pkl"
        
        # 保存缩放器
        joblib.dump(scaler, os.path.join(output_folder, scaler_filename))
        # ==========================================================
        
    print("缩放器已保存。")

    # 6. 创建序列数据
    print("创建序列数据...")
    X, y = create_sequences(scaled_df[input_columns].values, scaled_df[output_columns].values, SEQUENCE_LENGTH)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 7. 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = LSTNetModified(
        input_size=len(input_columns),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=len(output_columns),
        dropout_rate=DROPOUT_RATE
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 8. 训练循环
    print("开始训练...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.6f}')

    # 9. 保存训练好的模型
    model_path = os.path.join(output_folder, f'sequence_length{SEQUENCE_LENGTH}_num_epochs{NUM_EPOCHS}_dropout_rate{DROPOUT_RATE}_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\n训练完成！模型已保存到: {model_path}")
