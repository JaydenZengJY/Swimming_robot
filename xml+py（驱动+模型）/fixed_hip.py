import mujoco
import mujoco.viewer
import numpy as np
import time
import os

def control_specific_hip_joint():
    xml_path = r'D:\NEW\meshes\Untitled-1.xml'

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

    # 定义四个髋关节
    hip_joints = ['joint1', 'joint6', 'joint11', 'joint16']

    print("\n请指定要控制的髋关节和角度:")
    print(f"可选的髋关节: {', '.join(hip_joints)}")
    
    while True:
        joint_name = input("请输入关节名称 (例如: joint1): ").strip()
        if joint_name in hip_joints:
            break
        else:
            print(f"错误: 请输入有效的髋关节名称 ({', '.join(hip_joints)})")
    
    while True:
        try:
            angle_deg = float(input("请输入向上运动的角度 (度): "))
            break
        except ValueError:
            print("错误: 请输入有效的数字")
    
    # 将角度转换为弧度
    angle_rad = -np.radians(angle_deg)
    print(f"\n设置: {joint_name} 将向上运动 {angle_deg}° ({angle_rad:.3f} 弧度)")
    print("其他三个髋关节将保持不动")
    print("非髋关节的腿部关节保持原有正弦运动")

    joint_ids = {}
    for i in range(1, 21):
        joint_name_str = f'joint{i}'
        joint_ids[joint_name_str] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name_str)
    
    print("\n关节ID列表:")
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

    if joint_ids[joint_name] == -1:
        print(f"错误: 未找到关节 {joint_name}")
        return

    amplitude = 0.5   
    frequency = 1.0    

    phases = {
        'joint1': 0,         
        'joint2': -np.pi/3,       
        'joint3': -np.pi/3,
        'joint4': -np.pi/2,       
        'joint5': -np.pi/3,          
        'joint6': np.pi,       
        'joint7': np.pi/3,   
        'joint8': -np.pi/3,              
        'joint9': -np.pi/3,       
        'joint10': np.pi/3,       
        'joint11': 0,          
        'joint12': np.pi/3,
        'joint13': -np.pi/3,
        'joint14': -np.pi/3,
        'joint15': np.pi/3,
        'joint16': np.pi,      
        'joint17': -np.pi/3,
        'joint18': np.pi/3,
        'joint19': np.pi/3,
        'joint20': -np.pi/3
    }
    
    print("\n开始模拟...")
    print(f"目标关节: {joint_name} 将向上运动 {angle_deg}°")
    print("其他三个髋关节将保持不动")
    print("非髋关节的腿部关节保持原有正弦运动")
    print("按ESC退出查看器")
    
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 5.0
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -20
            
            start_time = time.time()
            
            while viewer.is_running():
                simulation_time = time.time() - start_time

                for name, jnt_id in joint_ids.items():
                    if jnt_id != -1:
                        if name == joint_name:
                            pos = angle_rad
                        elif name in hip_joints:
                            pos = 0.0
                        else:
                            pos = amplitude * np.sin(2 * np.pi * frequency * simulation_time + phases[name])

                        qpos_adr = model.jnt_qposadr[jnt_id]
                        data.qpos[qpos_adr] = pos

                mujoco.mj_step(model, data)

                if int(simulation_time * 2.0) != int((simulation_time - model.opt.timestep) * 2.0):
                    hip1_pos = data.qpos[model.jnt_qposadr[joint_ids['joint1']]]
                    hip6_pos = data.qpos[model.jnt_qposadr[joint_ids['joint6']]]
                    hip11_pos = data.qpos[model.jnt_qposadr[joint_ids['joint11']]]
                    hip16_pos = data.qpos[model.jnt_qposadr[joint_ids['joint16']]]
                    
                    print(f"时间: {simulation_time:.1f}s - "
                          f"髋关节角度: J1:{np.degrees(hip1_pos):.1f}° "
                          f"J6:{np.degrees(hip6_pos):.1f}° "
                          f"J11:{np.degrees(hip11_pos):.1f}° "
                          f"J16:{np.degrees(hip16_pos):.1f}°")

                viewer.sync()
                time.sleep(model.opt.timestep)
                    
    except Exception as e:
        print(f"模拟过程中出错: {e}")

if __name__ == "__main__":
    print("四足机器人指定髋关节运动演示")
    print("=" * 50)
    print("您可以指定任意一个髋关节(joint1,6,11,16)向上动任意角度")
    print("其他三个髋关节将保持不动")
    print("非髋关节的腿部关节保持原有正弦运动")
    print("=" * 50)

    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")
    
    xml_path = r'D:\NEW\meshes\Untitled-1.xml'
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
    
    control_specific_hip_joint()