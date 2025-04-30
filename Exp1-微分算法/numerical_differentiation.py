import numpy as np
import matplotlib.pyplot as plt
from sympy import tanh, symbols, diff, lambdify

def f(x):
    """计算函数值 f(x) = 1 + 0.5*tanh(2x)
    
    参数：
        x: 标量或numpy数组，输入值
    
    返回：
        标量或numpy数组，函数值
    """
    return 1 + 0.5 * np.tanh(2 * x)

def get_analytical_derivative():
    """使用sympy获取解析导数函数
    
    返回：
        可调用函数，用于计算导数值
    """
    x = symbols('x')
    func = 1 + 0.5 * tanh(2 * x)
    dfunc = diff(func, x)  # 求导
    return lambdify(x, dfunc)  # 转换为可调用函数

def calculate_central_difference(x, f):
    """使用中心差分法计算数值导数
    
    参数：
        x: numpy数组，要计算导数的点
        f: 可调用函数，要求导的函数
    
    返回：
        numpy数组，x[1:-1]处的导数值
    """
    dx = x[1] - x[0]
    return (f(x[2:]) - f(x[:-2])) / (2 * dx)

def richardson_derivative_all_orders(x, f, h, max_order=3):
    """使用Richardson外推法计算不同阶数的导数值
    
    参数：
        x: 标量，要计算导数的点
        f: 可调用函数，要求导的函数
        h: 浮点数，初始步长
        max_order: 整数，最大外推阶数
    
    返回：
        列表，不同阶数计算的导数值
    """
    derivatives = []
    for order in range(max_order + 1):
        derivative = (f(x + h / (2**order)) - f(x - h / (2**order))) / (2 * h / (2**order))
        derivatives.append(derivative)
    return derivatives

def create_comparison_plot(x, x_central, dy_central, dy_richardson, df_analytical):
    """创建对比图，展示导数计算结果和误差分析
    
    参数：
        x: numpy数组，所有x坐标点
        x_central: numpy数组，中心差分法使用的x坐标点
        dy_central: numpy数组，中心差分法计算的导数值
        dy_richardson: numpy数组，Richardson方法计算的导数值
        df_analytical: 可调用函数，解析导数函数
    """
    # 创建四个子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
    
    # 1. 导数对比图
    ax1.plot(x_central, dy_central, label='Central Difference', marker='o')
    ax1.plot(x, df_analytical(x), label='Analytical Derivative', linestyle='--')
    ax1.set_title('Derivative Comparison')
    ax1.legend()
    
    # 2. 误差分析图（对数坐标）
    errors_central = np.abs(dy_central - df_analytical(x_central))
    ax2.plot(x_central, errors_central, label='Central Difference Error', marker='o')
    ax2.set_yscale('log')  # 对数坐标
    ax2.set_title('Error Analysis (Log Scale)')
    ax2.legend()
    
    # 3. Richardson外推不同阶数误差对比图（对数坐标）
    errors_richardson = [np.abs(dy_richardson[:, i] - df_analytical(x)) for i in range(dy_richardson.shape[1])]
    for i, error in enumerate(errors_richardson):
        ax3.plot(x, error, label=f'Order {i}')
    ax3.set_yscale('log')
    ax3.set_title('Richardson Extrapolation Errors')
    ax3.legend()
    
    # 4. 步长敏感性分析图（双对数坐标）
    step_sizes = [0.1 / (2**i) for i in range(1, 10)]
    errors_h = []
    for h in step_sizes:
        derivative_h = (f(x[1]) - f(x[0])) / h
        errors_h.append(np.abs(derivative_h - df_analytical(x[1])))
    ax4.loglog(step_sizes, errors_h, marker='o')
    ax4.set_title('Step Size Sensitivity (Log-Log)')
    ax4.set_xlabel('Step Size (h)')
    ax4.set_ylabel('Error')
    
    plt.tight_layout()
    plt.show()

def main():
    """运行数值微分实验的主函数"""
    # 设置实验参数
    x = np.linspace(-2, 2, 100)
    h = 0.1

    # 获取解析导数函数
    df_analytical = get_analytical_derivative()
    
    # 计算中心差分导数
    dy_central = calculate_central_difference(x, f)
    
    # 计算Richardson外推导数
    dy_richardson = np.array([richardson_derivative_all_orders(x0, f, h) for x0 in x])

    # 绘制结果对比图
    create_comparison_plot(x, x[1:-1], dy_central, dy_richardson, df_analytical)

if __name__ == '__main__':
    main()
