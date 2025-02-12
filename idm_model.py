import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 定义模型的参数
params = {
    'alpha': 0.1,  # 认知的自我反馈系数
    'beta': 0.05,  # 家庭环境的影响
    'gamma': 0.03,  # 社会环境的影响
    'delta': 0.02,  # 生物学因素的影响
    'theta': 0.01,  # 环境与认知的交织系数
}

# 定义微分方程系统
def model(y, t, params):
    cognition, family_env, society, biology = y
    
    # 认知的变化
    dcognition_dt = -params['alpha'] * cognition + params['beta'] * family_env + params['gamma'] * society
    
    # 家庭环境的变化
    dfamily_env_dt = -params['beta'] * family_env + params['theta'] * cognition
    
    # 社会环境的变化
    dsociety_dt = -params['gamma'] * society + params['theta'] * cognition
    
    # 生物学因素的变化
    dbiology_dt = -params['delta'] * biology + params['theta'] * (family_env + society)
    
    return [dcognition_dt, dfamily_env_dt, dsociety_dt, dbiology_dt]

# 初始条件
initial_conditions = [0.5, 0.5, 0.5, 0.5]  # 假设初始值为0.5（可以根据实际情况调整）

# 时间点
time = np.linspace(0, 100, 1000)

# 解微分方程
solution = odeint(model, initial_conditions, time, args=(params,))

# 提取每个变量的解决方案
cognition = solution[:, 0]
family_env = solution[:, 1]
society = solution[:, 2]
biology = solution[:, 3]

# 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(time, cognition, label="Cognition", color='r')
plt.plot(time, family_env, label="Family Environment", color='g')
plt.plot(time, society, label="Society", color='b')
plt.plot(time, biology, label="Biology", color='purple')
plt.xlabel("Time")
plt.ylabel("Values")
plt.title("Dynamics of Cognition, Family Environment, Society, and Biology")
plt.legend()
plt.grid(True)
plt.show()
