import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import pandas as pd  # 导入 pandas 库用于数据处理

def quaternion_to_euler(w, x, y, z):
    """
    将四元数转换为欧拉角 (roll, pitch, yaw)
    使用ZYX顺序：先绕Z轴旋转(yaw)，然后绕Y轴旋转(pitch)，最后绕X轴旋转(roll)
    """
    # ... (此函数保持不变) ...
    r11 = 1 - 2*(y**2 + z**2)
    r12 = 2*(x*y - z*w)
    r13 = 2*(x*z + y*w)
    r21 = 2*(x*y + z*w)
    r22 = 1 - 2*(x**2 + z**2)
    r23 = 2*(y*z - x*w)
    r31 = 2*(x*z - y*w)
    r32 = 2*(y*z + x*w)
    r33 = 1 - 2*(x**2 + y**2)
    
    yaw = np.arctan2(r21, r11)
    pitch = np.arcsin(-r31)
    roll = np.arctan2(r32, r33)
    
    return roll, pitch, yaw

def get_robot_orientation(model, data):
    """
    获取机器人的当前姿态角 (roll, pitch, yaw)
    """
    # ... (此函数保持不变) ...
    free_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'floating_base')
    if free_joint_id == -1:
        return 0, 0, 0
    
    qpos_start = model.jnt_qposadr[free_joint_id]
    quat = data.qpos[qpos_start + 3:qpos_start + 7]
    w, x, y, z = quat
    
    roll, pitch, yaw = quaternion_to_euler(w, x, y, z)
    return roll, pitch, yaw

def print_orientation_info(roll, pitch, yaw, simulation_time):
    """
    打印姿态角信息
    """
    # ... (此函数保持不变) ...
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    
    print(f"时间: {simulation_time:.1f}s - 姿态角 - Roll: {roll_deg:6.1f}°, Pitch: {pitch_deg:6.1f}°, Yaw: {yaw_deg:6.1f}°")

def control_multiple_joints():
    # 确保XML文件名与你的文件系统匹配
    xml_path = r'D:\NEW\meshes\Untitled-2.xml'

    if not os.path.exists(xml_path):
        print(f"错误: 找不到文件 {xml_path}")
        return
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        print(f"成功加载模型: {xml_path}")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return

    # 这里我们仍然将重力设置为0，因为我们主要关注水动力
    model.opt.gravity = (0, 0, 0) 

    # 获取所有需要控制的关节ID
    joint_ids = {
        'joint2': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint2'),
        'joint3': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint3'),
        'joint4': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint4'),
        'joint5': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint5'),
        'joint7': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint7'),
        'joint8': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint8'),
        'joint9': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint9'),
        'joint10': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint10'),
        'joint12': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint12'),
        'joint13': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint13'),
        'joint14': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint14'),
        'joint15': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint15'),
        'joint17': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint17'),
        'joint18': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint18'),
        'joint19': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint19'),
        'joint20': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint20'),
    }
    
    # 检查是否有未找到的关节
    missing_joints = [name for name, idx in joint_ids.items() if idx == -1]
    if missing_joints:
        print(f"警告: 未找到以下关节: {missing_joints}")
        return

    # 定义运动参数
    amplitude = 0.5   
    frequency = 1.0    
    phases = {
        'joint2': -np.pi/2.2, 'joint3': -np.pi/0.5, 'joint4': -np.pi/7.05, 'joint5': -np.pi/1.33,
        'joint7': -np.pi/3, 'joint8': -np.pi/6, 'joint9': -np.pi/14, 'joint10': np.pi/0.7,
        'joint12': -np.pi/3, 'joint13': -np.pi/6, 'joint14': -np.pi/14, 'joint15': np.pi/0.7,
        'joint17': -np.pi/3, 'joint18': -np.pi/6, 'joint19': -np.pi/6, 'joint20': -np.pi/13
    }
    
    # 定义圆周运动和姿态扰动参数
    circle_radius, linear_speed = 0.8, 0.3
    angular_speed = linear_speed / circle_radius  
    roll_amplitude, pitch_amplitude = 0.15, 0.1
    roll_frequency, pitch_frequency = 0.4, 0.3
    
    print("\n开始模拟并收集数据...")
    print("按ESC或关闭窗口以结束模拟并保存数据")

    # ==========================================================
    # 1. 初始化一个空列表，用于存储每个时间步的数据记录
    training_data_records = []
    # ==========================================================
    
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 5.0
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -20
            
            start_time = time.time()
            last_print_time = 0
            
            while viewer.is_running():
                simulation_time = time.time() - start_time

                # === 机器人运动控制部分 (与你原代码一致) ===
                angle = angular_speed * simulation_time
                circle_x = circle_radius * np.cos(angle)
                circle_y = circle_radius * np.sin(angle)
                velocity_x = -linear_speed * np.sin(angle)
                velocity_y = linear_speed * np.cos(angle)
                orientation_angle = np.arctan2(velocity_y, velocity_x)
                
                roll_perturbation = roll_amplitude * np.sin(2 * np.pi * roll_frequency * simulation_time)
                pitch_perturbation = pitch_amplitude * np.sin(2 * np.pi * pitch_frequency * simulation_time + np.pi/4)
                
                total_roll = roll_perturbation + 0.02 * np.sin(2 * np.pi * 0.8 * simulation_time + np.pi/3)
                total_pitch = pitch_perturbation + 0.015 * np.sin(2 * np.pi * 1.2 * simulation_time + np.pi/6)

                free_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'floating_base')
                if free_joint_id != -1:
                    qpos_start = model.jnt_qposadr[free_joint_id]
                    qvel_start = model.jnt_dofadr[free_joint_id]

                    data.qpos[qpos_start:qpos_start+3] = [circle_x, circle_y, 0.4]
                    
                    cy, sy = np.cos(orientation_angle * 0.5), np.sin(orientation_angle * 0.5)
                    cp, sp = np.cos(total_pitch * 0.5), np.sin(total_pitch * 0.5)
                    cr, sr = np.cos(total_roll * 0.5), np.sin(total_roll * 0.5)
                    w, x, y, z = cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy
                    data.qpos[qpos_start+3:qpos_start+7] = [w, x, y, z]

                    data.qvel[qvel_start:qvel_start+3] = [velocity_x, velocity_y, 0]
                    data.qvel[qvel_start+5] = angular_speed

                # 计算并设置所有腿部关节的目标位置
                joint_positions = {}
                for name, jnt_id in joint_ids.items():
                    pos = amplitude * np.sin(2 * np.pi * frequency * simulation_time + phases[name])
                    joint_positions[name] = pos
                    data.qpos[model.jnt_qposadr[jnt_id]] = pos

                # 执行一步物理模拟
                mujoco.mj_step(model, data)

                # ==========================================================
                # 2. 在每一步模拟后，提取并记录数据
                # ----------------------------------------------------------
                # 获取输入特征 (Input Features)
                # 'water_speed' 定义为机器人身体在世界坐标系Y轴上的速度
                body_y_velocity = data.qvel[qvel_start + 1]
                
                # 我们以左前腿为例来记录数据。你可以为四条腿都记录，然后合并数据。
                # 这里的 'angle_l' 和 'angle_s' 对应模型中的两个主要驱动关节。
                # 我们将 joint2 映射为 'l'，joint3 映射为 's'。
                target_angle_l = joint_positions['joint2'] # 目标角度
                target_angle_s = joint_positions['joint3'] 

                # 实际关节速度
                actual_speed_l = data.qvel[model.jnt_dofadr[joint_ids['joint2']]]
                actual_speed_s = data.qvel[model.jnt_dofadr[joint_ids['joint3']]]
                
                # 获取输出标签 (Output Labels) - 来自传感器
                # 你的XML中为四条腿都定义了传感器，我们这里记录左前腿的
                fl_force = data.sensor('FL_force_sensor').data.copy()
                fl_torque = data.sensor('FL_torque_sensor').data.copy()

                # 创建一个字典来存储当前时间步的所有数据
                record = {
                    'time': simulation_time,
                    'water_speed': body_y_velocity,
                    'angle_l': target_angle_l,
                    'angle_s': target_angle_s,
                    'speed_l': actual_speed_l,
                    'speed_s': actual_speed_s,
                    'FX/N': fl_force[0],
                    'FY/N': fl_force[1],
                    'FZ/N': fl_force[2],
                    'TX/N*m': fl_torque[0],
                    'TY/N*m': fl_torque[1],
                    'TZ/N*m': fl_torque[2]
                }
                # 将记录添加到我们的数据列表中
                training_data_records.append(record)
                # ==========================================================

                # === 打印信息部分 (与你原代码一致) ===
                if simulation_time - last_print_time >= 0.5:
                    roll, pitch, yaw = get_robot_orientation(model, data)
                    print("-" * 50)
                    print(f"时间: {simulation_time:.1f}s - 位置: ({circle_x:.2f}, {circle_y:.2f})")
                    print_orientation_info(roll, pitch, yaw, simulation_time)
                    last_print_time = simulation_time

                # 同步查看器并等待
                viewer.sync()
                # time.sleep(model.opt.timestep) # 建议在数据收集时注释掉，以加快速度

    except Exception as e:
        print(f"模拟过程中出错: {e}")
    finally:
        # ==========================================================
        # 3. 模拟结束后，无论是否出错，都尝试保存数据
        # ==========================================================
        print("\n模拟结束。")
        if training_data_records:
            print(f"已收集 {len(training_data_records)} 条数据记录，正在保存到文件...")
            # 使用 pandas 创建 DataFrame
            df = pd.DataFrame(training_data_records)
            
            # 定义输出文件名
            output_filename = "training_data.csv"
            
            # 保存为 CSV 文件，不包含索引列
            df.to_csv(output_filename, index=False)
            
            print(f"数据已成功保存到: {os.path.join(os.getcwd(), output_filename)}")
        else:
            print("警告: 未收集到任何数据。")

if __name__ == "__main__":
    print("四足机器人运动模拟与数据收集脚本")
    print("=" * 50)
    control_multiple_joints()

