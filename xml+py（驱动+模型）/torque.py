import mujoco
import mujoco.viewer
import numpy as np
import time
import os

def control_multiple_joints():
    xml_path = r'D:\NEW\meshes\New.xml'

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

    model.opt.gravity = (0, 0, 0)

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
        'joint20': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint20')
    }
    
    print("关节ID列表:")
    for name, idx in joint_ids.items():
        print(f"  {name}: {idx}")

    missing_joints = [name for name, idx in joint_ids.items() if idx == -1]
    if missing_joints:
        print(f"警告: 未找到以下关节: {missing_joints}")
        print("可用关节列表:")
        for i in range(model.njnt):
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jnt_name:
                print(f"  {jnt_name}")
        return

    amplitude = 0.5 
    frequency = 1.0    

    phases = {
        'joint2': -np.pi/3,       
        'joint3': -np.pi/3,
        'joint4': -np.pi/2,       
        'joint5': -np.pi/3,          
        'joint7': np.pi/3,   
        'joint8': -np.pi/3,              
        'joint9': -np.pi/3,       
        'joint10': np.pi/3,       
        'joint12': np.pi/3,
        'joint13': -np.pi/3,
        'joint14': -np.pi/3,
        'joint15': np.pi/3,
        'joint17': -np.pi/3,
        'joint18': np.pi/3,
        'joint19': np.pi/3,
        'joint20': -np.pi/3
    }

    print("\n请选择要控制的转动轴:")
    print("1. X轴转动 (绕X轴旋转)")
    print("2. Y轴转动 (绕Y轴旋转)")
    print("3. Z轴转动 (绕Z轴旋转)")
    print("4. 组合转动 (多轴同时转动)")
    
    try:
        axis_choice = int(input("选择转动轴 (1-4，默认1): ") or 1)
    except ValueError:
        axis_choice = 1

    torque_x = 0.0
    torque_y = 0.0
    torque_z = 0.0

    if axis_choice == 1:  # X轴转动
        try:
            target_angle = float(input("请输入X轴目标转动角度(度，默认90): ") or 90)
            torque_x = float(input("请输入X轴扭矩大小(默认0.1): ") or 0.1)
            print(f"将绕X轴转动 {target_angle} 度，扭矩大小: {torque_x}")
        except ValueError:
            target_angle = 90
            torque_x = 0.1
            print("输入无效，使用默认值")
            
    elif axis_choice == 2:  # Y轴转动
        try:
            target_angle = float(input("请输入Y轴目标转动角度(度，默认90): ") or 90)
            torque_y = float(input("请输入Y轴扭矩大小(默认0.1): ") or 0.1)
            print(f"将绕Y轴转动 {target_angle} 度，扭矩大小: {torque_y}")
        except ValueError:
            target_angle = 90
            torque_y = 0.1
            print("输入无效，使用默认值")
            
    elif axis_choice == 3:  # Z轴转动
        try:
            target_angle = float(input("请输入Z轴目标转动角度(度，默认90): ") or 90)
            torque_z = float(input("请输入Z轴扭矩大小(默认0.1): ") or 0.1)
            print(f"将绕Z轴转动 {target_angle} 度，扭矩大小: {torque_z}")
        except ValueError:
            target_angle = 90
            torque_z = 0.1
            print("输入无效，使用默认值")
            
    elif axis_choice == 4:  # 组合转动
        print("请分别设置各轴的转动参数:")
        try:
            target_angle_x = float(input("X轴目标转动角度(度，默认0): ") or 0)
            torque_x = float(input("X轴扭矩大小(默认0): ") or 0)
            
            target_angle_y = float(input("Y轴目标转动角度(度，默认0): ") or 0)
            torque_y = float(input("Y轴扭矩大小(默认0): ") or 0)
            
            target_angle_z = float(input("Z轴目标转动角度(度，默认90): ") or 90)
            torque_z = float(input("Z轴扭矩大小(默认0.1): ") or 0.1)
            
            print(f"组合转动设置: X轴{target_angle_x}度, Y轴{target_angle_y}度, Z轴{target_angle_z}度")
        except ValueError:
            target_angle_x, target_angle_y, target_angle_z = 0, 0, 90
            torque_x, torque_y, torque_z = 0, 0, 0.1
            print("输入无效，使用默认值")

    print("\n请选择扭矩模式:")
    print("1. 恒定扭矩")
    print("2. 角度控制模式 (达到目标角度后停止)")
    print("3. 正弦变化扭矩")
    
    try:
        torque_mode = int(input("选择模式 (1-3，默认1): ") or 1)
    except ValueError:
        torque_mode = 1
 
    if torque_mode == 3:
        try:
            torque_frequency = float(input("正弦频率 (Hz，默认0.5): ") or 0.5)
            torque_amplitude_x = float(input("X轴扭矩幅度 (默认0.1): ") or 0.1)
            torque_amplitude_y = float(input("Y轴扭矩幅度 (默认0.1): ") or 0.1)
            torque_amplitude_z = float(input("Z轴扭矩幅度 (默认0.1): ") or 0.1)
        except ValueError:
            torque_frequency = 0.5
            torque_amplitude_x = torque_amplitude_y = torque_amplitude_z = 0.1
    
    print("\n开始模拟...")
    print(f"控制关节: {list(joint_ids.keys())}")
    print(f"扭矩模式: {['恒定', '角度控制', '正弦'][torque_mode-1]}")
    print(f"施加扭矩 - X: {torque_x}, Y: {torque_y}, Z: {torque_z}")
    print("按ESC退出查看器")
    
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 5.0
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -20
            
            start_time = time.time()
            simulation_time = 0
            target_reached = False

            free_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'floating_base')
            if free_joint_id != -1:
                qpos_start = model.jnt_qposadr[free_joint_id]
                w0, xq0, yq0, zq0 = data.qpos[qpos_start+3:qpos_start+7]
                initial_quat = np.array([w0, xq0, yq0, zq0])
            
            while viewer.is_running():
                simulation_time = time.time() - start_time

                leg_time = simulation_time

                base_sin_motion = {}
                for name in joint_ids.keys():
                    base_sin = amplitude * np.sin(2 * np.pi * frequency * leg_time + phases[name])
                    base_sin_motion[name] = base_sin
 
                joint_positions = {}
                for name in joint_ids.keys():
                    pos = base_sin_motion[name]
                    joint_positions[name] = pos
                    
                    jnt_id = joint_ids[name]
                    qpos_adr = model.jnt_qposadr[jnt_id] 
                    data.qpos[qpos_adr] = pos

                current_torque_x = torque_x
                current_torque_y = torque_y
                current_torque_z = torque_z

                if torque_mode == 2 and not target_reached:
                    if free_joint_id != -1:
                        qpos_start = model.jnt_qposadr[free_joint_id]
                        w, xq, yq, zq = data.qpos[qpos_start+3:qpos_start+7]
                        current_quat = np.array([w, xq, yq, zq])

                        sinr_cosp = 2 * (w * xq + yq * zq)
                        cosr_cosp = 1 - 2 * (xq * xq + yq * yq)
                        roll = np.arctan2(sinr_cosp, cosr_cosp)
                        
                        sinp = 2 * (w * yq - zq * xq)
                        pitch = np.arcsin(sinp)
                        
                        siny_cosp = 2 * (w * zq + xq * yq)
                        cosy_cosp = 1 - 2 * (yq * yq + zq * zq)
                        yaw = np.arctan2(siny_cosp, cosy_cosp)

                        if axis_choice == 1:  # X轴
                            angle_error = np.degrees(roll) - target_angle
                        elif axis_choice == 2:  # Y轴
                            angle_error = np.degrees(pitch) - target_angle
                        elif axis_choice == 3:  # Z轴
                            angle_error = np.degrees(yaw) - target_angle
                        else: 
                            angle_error_x = np.degrees(roll) - target_angle_x
                            angle_error_y = np.degrees(pitch) - target_angle_y
                            angle_error_z = np.degrees(yaw) - target_angle_z
                        
                        kp = 0.01  
                        if axis_choice == 4:  
                            current_torque_x = -kp * angle_error_x if torque_x != 0 else 0
                            current_torque_y = -kp * angle_error_y if torque_y != 0 else 0
                            current_torque_z = -kp * angle_error_z if torque_z != 0 else 0

                            if (abs(angle_error_x) < 5 and abs(angle_error_y) < 5 and abs(angle_error_z) < 5):
                                target_reached = True
                                print("所有目标角度已达到!")
                        else:
                            if axis_choice == 1:
                                current_torque_x = -kp * angle_error
                            elif axis_choice == 2:
                                current_torque_y = -kp * angle_error
                            elif axis_choice == 3:
                                current_torque_z = -kp * angle_error
    
                            if abs(angle_error) < 5:
                                target_reached = True
                                print(f"目标角度已达到! 当前误差: {angle_error:.2f}度")

                if torque_mode == 2 and target_reached:
                    current_torque_x = current_torque_y = current_torque_z = 0

                elif torque_mode == 3:
                    current_torque_x = torque_x + torque_amplitude_x * np.sin(2 * np.pi * torque_frequency * simulation_time)
                    current_torque_y = torque_y + torque_amplitude_y * np.sin(2 * np.pi * torque_frequency * simulation_time + np.pi/3)
                    current_torque_z = torque_z + torque_amplitude_z * np.sin(2 * np.pi * torque_frequency * simulation_time + 2*np.pi/3)

                free_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'floating_base')
                if free_joint_id != -1:
                    qvel_start = model.jnt_dofadr[free_joint_id]

                    data.qfrc_applied[qvel_start + 3] = current_torque_x   
                    data.qfrc_applied[qvel_start + 4] = current_torque_y  
                    data.qfrc_applied[qvel_start + 5] = current_torque_z   

                mujoco.mj_step(model, data)

                if int(simulation_time * 2.0) != int((simulation_time - model.opt.timestep) * 2.0):
                    free_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'floating_base')
                    if free_joint_id != -1:
                        qpos_start = model.jnt_qposadr[free_joint_id]
                        x, y, z = data.qpos[qpos_start:qpos_start+3]
                        w, xq, yq, zq = data.qpos[qpos_start+3:qpos_start+7]

                        sinr_cosp = 2 * (w * xq + yq * zq)
                        cosr_cosp = 1 - 2 * (xq * xq + yq * yq)
                        roll = np.arctan2(sinr_cosp, cosr_cosp)
                        
                        sinp = 2 * (w * yq - zq * xq)
                        pitch = np.arcsin(sinp)
                        
                        siny_cosp = 2 * (w * zq + xq * yq)
                        cosy_cosp = 1 - 2 * (yq * yq + zq * zq)
                        yaw = np.arctan2(siny_cosp, cosy_cosp)
                        
                        torque_info = f"当前扭矩 - X: {current_torque_x:.3f}, Y: {current_torque_y:.3f}, Z: {current_torque_z:.3f}"
                        angle_info = f"当前角度 - 滚转: {np.degrees(roll):.1f}°, 俯仰: {np.degrees(pitch):.1f}°, 偏航: {np.degrees(yaw):.1f}°"
                        
                        print(f"时间: {simulation_time:.1f}s - {angle_info} - {torque_info}")

                        if axis_choice == 1:
                            print(f"目标: 滚转 {target_angle}°")
                        elif axis_choice == 2:
                            print(f"目标: 俯仰 {target_angle}°")
                        elif axis_choice == 3:
                            print(f"目标: 偏航 {target_angle}°")
                        elif axis_choice == 4:
                            print(f"目标: 滚转 {target_angle_x}°, 俯仰 {target_angle_y}°, 偏航 {target_angle_z}°")

                viewer.sync()
                time.sleep(model.opt.timestep)
                    
    except Exception as e:
        print(f"模拟过程中出错: {e}")

if __name__ == "__main__":
    print("四足机器人角度控制演示")
    print("=" * 50)
    print("机器人腿部保持正弦运动")
    print("施加扭矩使机器人在指定轴上转动到目标角度")
    print("=" * 50)

    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")
    
    xml_path = r'D:\NEW\meshes\New.xml'
    if not os.path.exists(xml_path):
        print(f"未找到模型文件: {xml_path}")
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            xml_path = filedialog.askopenfilename(
                title="选择MuJoCo模型文件",
                filetypes=[("XML files", "*.xml"), ("All files", "*.*")]
            )
            if not xml_path:
                print("未选择文件，退出程序")
                exit()
        except:
            print("请手动修改代码中的xml_path变量")
            exit()
    
    control_multiple_joints()