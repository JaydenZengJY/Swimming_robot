import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import math
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
import joblib
import time  # 导入 time 模块
import matplotlib.pyplot as plt

# Define the LSTM Network structure
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

# Load models
def load_model(model_path, input_size, hidden_size, num_layers, output_size, dropout_rate, device):
    model = LSTNetModified(input_size, hidden_size, num_layers, output_size, dropout_rate).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Normalization and inverse normalization functions
def normalize_data(data, columns, scalers):
    normalized_data = data.copy()
    for column in columns:
        scaler = scalers[column]
        normalized_data[[column]] = scaler.transform(data[[column]])
    return normalized_data

def load_scalers(columns, filepath):
    scalers = {}
    for column in columns:
        sanitized_column = column.replace('/', '_')
        scalers[column] = joblib.load(f'{filepath}/{sanitized_column}_scaler.pkl')
    return scalers

def inverse_transform(data, scalers, columns):
    restored_data = data.copy()
    for column in columns:
        scaler = scalers[column]
        restored_data[[column]] = scaler.inverse_transform(data[[column]])
    return restored_data

def create_sequences(input_data, sequence_length):
    sequences = []
    for i in range(len(input_data) - sequence_length + 1):
        seq = input_data[i:i + sequence_length]
        sequences.append(seq)
    return np.array(sequences)

def calculate_force_contribution(r_sensor, r_body, rotation_axis, data, theta, device):
    torque_sensor = data[:3]
    force_sensor = data[3:]  
    torque_leg = torque_sensor + torch.cross(r_sensor - r_body, force_sensor, dim=0)
    force_leg = force_sensor 
    
    # 计算平移矩阵 T1 和 T2
    T1 = torch.eye(4, device=device)
    T1[:3, 3] = rotation_axis

    T2 = torch.eye(4, device=device)
    T2[:3, 3] = -rotation_axis

    # 计算旋转矩阵 Ry(theta)
    cos_theta = torch.cos(theta).item()  # Convert to Python scalar
    sin_theta = torch.sin(theta).item()  # Convert to Python scalar
    R_hip = torch.tensor([
        [cos_theta, 0, sin_theta, 0],
        [0, 1, 0, 0],
        [-sin_theta, 0, cos_theta, 0],
        [0, 0, 0, 1]
    ], device=device)

    # 计算齐次坐标
    A_h = torch.cat((r_body, torch.tensor([1.0], device=device)))

    # 计算旋转后的点 B_h
    B_h = T2 @ R_hip @ T1 @ A_h
    B = B_h[:3]

    # 计算旋转后的力
    force_Body = R_hip[:3, :3] @ force_leg

    # 计算旋转后的扭矩
    torque_rotated = R_hip[:3, :3] @ torque_leg

    # 计算 B 点在中性浮力点处的受力和扭矩
    r_B = B - torch.tensor([0, 0, 0], device=device)
    torque_Body = torque_rotated + torch.cross(r_B, force_Body, dim=0)

    return force_Body, torque_Body


def discretize_and_clip(x, steps, xl, xu):
    x_discrete = np.round(x / steps) * steps
    return np.clip(x_discrete, xl, xu)

# Optimization problem
class MyOptimizationProblem(ElementwiseProblem):
    def __init__(self, sequence_length, num_epochs, dropout_rate):
        self.sequence_length = sequence_length
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.eval_count = 0
        pi = math.pi
        super().__init__(n_var=15,  # 12维参数
                         n_obj=3,  # 一个目标
                         n_constr=0,  # 无约束
                         xl=np.array([170, 170, 170, 170, 180, 180, 180, 180, 30, 30, 30, 30, 0, 0, 0]),  # 参数下限
                         xu=np.array([230, 230, 230, 230, 260, 260, 260, 260, 60, 60, 60, 60, 2*pi, 2*pi, 2*pi]))  # 参数上限

    def _evaluate(self, x, out, *args, **kwargs):
        steps = np.array([5]*8 + [1]*4 + [0.1]*3)
        x_discrete = x
        self.eval_count += 1
        # Set control frequency and compute time step
        control_frequency = 65  # Hz
        time_step = 1 / control_frequency  # Time step in seconds

        # Model paths
        current_directory = os.path.dirname(os.path.abspath(__file__))
        first_folder = 'withweb'
        model_folder = f'sequence_length{self.sequence_length}_num_epochs{self.num_epochs}_dropout_rate{self.dropout_rate}'
        folder_path = os.path.join(current_directory, 'Training_results', first_folder, model_folder)
        model_path = f'{folder_path}/sequence_length{self.sequence_length}_num_epochs{self.num_epochs}_dropout_rate{self.dropout_rate}_model.pth'
        
        # Model parameters
        input_size = 5
        hidden_size = 60
        output_size = 6
        num_layers = 2
        dropout_rate = 0.21
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load each leg's model
        start_time = time.time()  # 记录加载模型的开始时间
        model_FL = load_model(model_path, input_size, hidden_size, num_layers, output_size, dropout_rate, device)
        model_FR = load_model(model_path, input_size, hidden_size, num_layers, output_size, dropout_rate, device)
        model_HL = load_model(model_path, input_size, hidden_size, num_layers, output_size, dropout_rate, device)
        model_HR = load_model(model_path, input_size, hidden_size, num_layers, output_size, dropout_rate, device)
        model_loading_time = time.time() - start_time  # 计算加载模型的时间

        # Load scalers for normalization
        start_time = time.time()  # 记录加载scalers的开始时间
        input_columns = ['water_speed', 'angle_l', 'angle_s', 'speed_l', 'speed_s']
        output_columns = ['TX/N*m', 'TY/N*m', 'TZ/N*m', 'FX/N', 'FY/N', 'FZ/N']
        scalers = load_scalers(input_columns + output_columns, folder_path)
        scalers_loading_time = time.time() - start_time  # 计算加载scalers的时间
        
        # Initialize tensors for the model inputs and outputs
        model_inputs = torch.zeros((4, self.sequence_length, input_size), device=device)  # 4 legs, sequence_length time points, input_size parameters each
        model_outputs = torch.zeros((4, output_size), device=device)  # 4 legs, output_size force/torque parameters

        # Initialize past angles, speeds, and water speed with zeros
        past_angles_speeds = torch.zeros((4, self.sequence_length, input_size), device=device)
        
        torque_z = torch.tensor(0.0, device=device)
        I_z = torch.tensor(0.047, device=device)
        theta_yaw = torch.tensor(0.0, device=device)  # 初始偏航角为0
        v_x = torch.tensor(0.0, device=device)
        v_y = torch.tensor(0.0, device=device)
        x_displacement = torch.tensor(0.0, device=device)
        y_displacement = torch.tensor(0.0, device=device)
        impulse_x = torch.tensor(0.0, device=device)
        impulse_y = torch.tensor(0.0, device=device)
        m = torch.tensor(3.0, device=device)
        distance = torch.tensor(100.0, device=device)
        yaw_torque = torch.tensor(0.0, device=device)
        point_d = torch.tensor([3.0, 3.0], device=device)
        # 初始化全局变量
        x_global = torch.tensor(0.0, device=device)
        y_global = torch.tensor(0.0, device=device)
        # Initialize time
        time_now = torch.tensor(0.0, device=device)

        # Calculating the objective function
        start_time = time.time()  # 记录计算目标函数的开始时间
        while (yaw_torque <0.2) & (time_now < 25) & (distance>=0.1):
            pi = math.pi
            Al_max = torch.tensor([280.0] * 4, device=device)
            Al_min = torch.tensor(x_discrete[0:4], device=device)
            As_max = torch.tensor(x_discrete[4:8], device=device)
            As_min = torch.tensor([100.0] * 4, device=device)
            Omeg = torch.tensor(x_discrete[8:12], device=device)
            Phase_diff = torch.tensor([0.0]*4, device=device)
            Phase_diff[1:] = torch.tensor(x_discrete[12:15], device=device)
            Omega = Omeg / 100.0
            Fai = torch.tensor(1.0 / 3.0, device=device)
            Al = (Al_max - Al_min) / 2.0
            As = (As_max - As_min) / 2.0
            MIDLE_L = Al + Al_min
            MIDLE_S = As + As_min
            # Compute angles and speeds for all legs at different time points
            for i in range(4):
                # Shift the past angles and speeds to the left to make room for the new time step
                past_angles_speeds[i, :-1] = past_angles_speeds[i, 1:].clone()
                
                # Calculate the new angles and speeds
                angle_l = (Al[i] * torch.sin(Omega[i] * 2.0 * pi * time_now + Phase_diff[i]) + MIDLE_L[i]).int()
                angle_s = (As[i] * torch.sin(Omega[i] * 2.0 * pi * time_now + pi * Fai+ Phase_diff[i]) + MIDLE_S[i]).int()
                speed_l = (torch.abs(Al[i] * Omega[i] * pi * torch.cos(Omega[i] * 2.0 * pi * time_now+ Phase_diff[i]))).int()
                speed_s = (torch.abs(As[i] * Omega[i] * pi * torch.cos(Omega[i] * 2.0 * pi * time_now + pi * Fai+ Phase_diff[i]))).int()
                # 计算角度差并应用机械角度限制
                angle_gap = angle_l - angle_s
                angle_s = torch.where(angle_gap <= 20, angle_l - 20, angle_s)
                angle_s = torch.where(angle_gap >= 130, angle_l - 130, angle_s)
                
                # Add the new time step data
                past_angles_speeds[i, -1, :] = torch.tensor([v_y.item(), angle_l.item(), angle_s.item(), speed_l.item(), speed_s.item()], device=device)

                # Normalize the input sequence
                past_data_df = pd.DataFrame(past_angles_speeds[i].cpu().numpy(), columns=input_columns)
                normalized_sequence = normalize_data(past_data_df, input_columns, scalers)
                model_inputs[i, :, :] = torch.tensor(normalized_sequence.values, device=device)
            # Get predictions (force and torque data) for each leg
            start_leg_pred_time = time.time()  # 记录每条腿预测的开始时间
            Nor_FL = model_FL(model_inputs[0].unsqueeze(0)).squeeze().cpu().detach().numpy()
            Nor_FR = model_FR(model_inputs[1].unsqueeze(0)).squeeze().cpu().detach().numpy()
            Nor_HL = model_HL(model_inputs[2].unsqueeze(0)).squeeze().cpu().detach().numpy()
            Nor_HR = model_HR(model_inputs[3].unsqueeze(0)).squeeze().cpu().detach().numpy()
            leg_pred_time = time.time() - start_leg_pred_time  # 计算每条腿预测的时间
            
            Data_FL = torch.tensor(inverse_transform(pd.DataFrame([Nor_FL], columns=output_columns), scalers, output_columns).values.squeeze(), device=device)
            Data_FR = torch.tensor(inverse_transform(pd.DataFrame([Nor_FR], columns=output_columns), scalers, output_columns).values.squeeze(), device=device)
            Data_HL = torch.tensor(inverse_transform(pd.DataFrame([Nor_HL], columns=output_columns), scalers, output_columns).values.squeeze(), device=device)
            Data_HR = torch.tensor(inverse_transform(pd.DataFrame([Nor_HR], columns=output_columns), scalers, output_columns).values.squeeze(), device=device)
            #各受力点位置
            
            # 传感器测量位置
            r_FL_sensor = torch.tensor([-0.070, 0.140, 0.053], device=device)
            r_FR_sensor = torch.tensor([0.070, 0.140, 0.053], device=device)
            r_HL_sensor = torch.tensor([-0.070, -0.060, 0.053], device=device)
            r_HR_sensor = torch.tensor([0.070, -0.060, 0.053], device=device)
            
            
            r_FL = torch.tensor([-0.070, 0.140, 0], device=device)
            r_FR = torch.tensor([0.070, 0.140, 0], device=device)
            r_HL = torch.tensor([-0.070, -0.060, 0], device=device)
            r_HR = torch.tensor([0.070, -0.060, 0], device=device)

            rotation_axis_L = torch.tensor([-0.04, 0, 0], device=device)
            rotation_axis_R = torch.tensor([0.04, 0, 0], device=device)

            # Right-hand rule
            rot_angle_FL = torch.tensor(30.0 * (torch.pi / 180.0), device=device)  # 45 degrees in radians
            rot_angle_RL = torch.tensor(-30.0 * (torch.pi / 180.0), device=device)  # -45 degrees in radians
            rot_angle_HL = torch.tensor(0 * (torch.pi / 180.0), device=device)  # 0 degrees in radians
            rot_angle_HR = torch.tensor(0 * (torch.pi / 180.0), device=device)  # 0 degrees in radians

            # Calculate total force and torque contributions from each leg
            
            force_Body_FL, torque_Body_FL = calculate_force_contribution(r_FL_sensor, r_FL, rotation_axis_L, Data_FL, rot_angle_FL, device)
            force_Body_FR, torque_Body_FR = calculate_force_contribution(r_FR_sensor, r_FR, rotation_axis_R, Data_FR, rot_angle_RL, device)
            force_Body_HL, torque_Body_HL = calculate_force_contribution(r_HL_sensor, r_HL, rotation_axis_L, Data_HL, rot_angle_HL, device)
            force_Body_HR, torque_Body_HR = calculate_force_contribution(r_HR_sensor, r_HR, rotation_axis_R, Data_HR, rot_angle_HR, device)
            
            total_force = force_Body_FL + force_Body_FR + force_Body_HL + force_Body_HR
            
            Force_y = -(total_force[1] + 12.15 * v_y**2) # Plus the drag force of body and adjust the direction of thrust force
            Force_x = total_force[0] - 37.8 * v_x**2
            total_force[1] = Force_y
            total_force[0] = Force_x

            total_torque = torque_Body_FL + torque_Body_FR + torque_Body_HL + torque_Body_HR
            
            # 提取Z轴扭矩
            yaw_torque = total_torque[2]

            # 计算偏航角速度，根据右手定则，向左yaw>0,向右yaw<0
            omega_z = yaw_torque / I_z

            # 更新偏航角
            theta_yaw += omega_z * time_step
            
            # Update displacement and velocity
            impulse_x += total_force[0] * time_step
            impulse_y += total_force[1] * time_step
            # 计算加速度
            a_x = total_force[0] / m
            a_y = total_force[1] / m

            # 更新速度
            v_x += a_x * time_step
            v_y += a_y * time_step
            y_displacement += v_y * time_step
            # 计算增量位移
            delta_x_local = v_x * time_step
            delta_y_local = v_y * time_step

            # 将增量位移转换到全局坐标系
            delta_x_global = delta_x_local * torch.cos(theta_yaw) - delta_y_local * torch.sin(theta_yaw)
            delta_y_global = delta_x_local * torch.sin(theta_yaw) + delta_y_local * torch.cos(theta_yaw)
            # 更新全局位移
            x_global += delta_x_global
            y_global += delta_y_global    
            
            distance = torch.sqrt((x_global - point_d[0])**2 + (y_global - point_d[1])**2)
            # Update time
            time_now += time_step
        
        objective_function_time = time.time() - start_time  # 计算目标函数的总时间
        # print(f'Loading models time: {model_loading_time:.4f} seconds')
        # print(f'Loading scalers time: {scalers_loading_time:.4f} seconds')
        # print(f'Leg prediction time per loop: {leg_pred_time:.4f} seconds')
        # print(f'Objective function calculation time: {objective_function_time:.4f} seconds')
                
        if distance.item() <= 0.15:
            print(f'Evaluation count: {self.eval_count}')            
            print('Solution:',x_discrete)
            print('Cost time:', time_now.item(), ', Displacement x:', x_global.item(), ', Displacement y:', y_global.item(), y_displacement.item())
            print('theta_yaw:', theta_yaw.item() * 180 / math.pi)
            print('V_y:', v_y.item(), ', Whole impulse y:', impulse_y.item())
            print('V_x:', v_x.item(), ', Whole impulse x:', impulse_x.item())
        # 目标函数：最大化 impulse_y 和 最小化 theta_yaw 的绝对值
        out["F"] = [distance.item(), abs(3 - x_global.item()),abs(theta_yaw.item() + 1.57)]
# Optimization Callback
# 自定义回调类
class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []
        self.all_solutions = []
    def notify(self, algorithm):
        self.data["best"].append(algorithm.pop.get("F").min())
        self.all_solutions.append((algorithm.pop.get("X"), algorithm.pop.get("F")))
    def save_to_csv(self, filename):
        all_solutions = np.concatenate([np.hstack((x, f)) for x, f in self.all_solutions])
        columns = [f"x{i}" for i in range(all_solutions.shape[1] - 3)] + ["f1", "f2", "f3"]
        df = pd.DataFrame(all_solutions, columns=columns)
        df.to_csv(filename, index=False)
        
# 定义一个函数来绘制帕累托前沿并保存为图片
def plot_pareto_front(F, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(F[:, 0], F[:, 1], F[:, 2], c='red', edgecolors='k')
    ax.set_xlabel('Objective 1: distance')
    ax.set_ylabel('Objective 2: x_axis_error')
    ax.set_zlabel('Objective 3: yaw_error')
    ax.set_title('Pareto Front')
    plt.grid(True)
    plt.savefig(filename)



# Define the optimization algorithm
algorithm = NSGA2(pop_size=150)

# Define the problem with specific parameters
sequence_length = 16
num_epochs = 100
dropout_rate = 0.21
callback=MyCallback()
problem = MyOptimizationProblem(sequence_length, num_epochs, dropout_rate)
# Run optimization
res = minimize(problem,
               algorithm,
               seed=1,
               termination=('n_gen', 30),
               verbose=True,
               save_history=True,
               callback=callback)

# Output results
print("Best solution found: %s" % res.X)
print("Function value: %s" % res.F)
if hasattr(res, 'CV') and res.CV is not None:
    print("Constraint violation: %s" % res.CV)
else:
    print("No constraint violations.")
# 查看回调中保存的数据
# 绘制收敛曲线
val = callback.data["best"]
plt.plot(np.arange(len(val)), val)
plt.xlabel('Generation')
plt.ylabel('Best Objective Value')
plt.title('Convergence Plot')

current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'Optimisation_results/Fturn35_results/whole_turnR_para/int')
if not os.path.exists(file_path):
    os.makedirs(file_path)

plt.savefig(f'{file_path}/Convergence.png')


# 绘制帕累托前沿


# 保存结果到 CSV 文件
callback.save_to_csv(f'{file_path}/optimization_results.csv')

F = res.F
plot_pareto_front(F, f'{file_path}/Pareto_front.png')