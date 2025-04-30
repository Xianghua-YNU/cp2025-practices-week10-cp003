import numpy as np

# 定义被积函数：f(x) = x^4 - 2x + 1
def f(x):
    return x**4 - 2*x + 1

# 梯形法则数值积分实现
def trapezoidal(f, a, b, N):
    h = (b - a) / N  # 计算步长
    x = np.linspace(a, b, N+1)  # 将区间[a,b]等分为N个子区间，生成N+1个节点
    fx = f(x)  # 计算所有节点处的函数值
    # 梯形公式：积分 ≈ h*(首尾项均值 + 中间项和)
    integral = h * (0.5 * fx[0] + np.sum(fx[1:-1]) + 0.5 * fx[-1])
    return integral

# Simpson法则数值积分实现
def simpson(f, a, b, N):
    if N % 2 != 0:
        # Simpson法则必须使用偶数个子区间，否则无法正确应用公式
        raise ValueError("Simpson 法则要求 N 必须为偶数")
    h = (b - a) / N  # 计算步长
    x = np.linspace(a, b, N+1)  # 生成等距采样点(N+1个点对应N个子区间)
    fx = f(x)  # 获取所有采样点的函数值
    
    # 计算奇数索引项的和(对应每个二次曲线区间中间点的函数值)
    # 索引范围：[1, 3, 5,..., N-1] (共N/2个点)
    odd_sum = np.sum(fx[1:N:2])
    
    # 计算偶数索引项的和(对应相邻二次曲线区间接合点的函数值)
    # 索引范围：[2, 4, 6,..., N-2] (共(N/2 - 1)个点)
    even_sum = np.sum(fx[2:N:2])
    
    # 应用Simpson公式：积分 ≈ h/3 * [起点值 + 4*奇数和 + 2*偶数和 + 终点值]
    integral = h / 3 * (fx[0] + 4 * odd_sum + 2 * even_sum + fx[N])
    return integral

def main():
    a, b = 0, 2  # 积分区间
    exact_integral = 4.4  # 精确积分值
    
    # 分别用N=100和N=1000测试两种积分方法
    for N in [100, 1000]:
        # 计算梯形法则结果
        trapezoidal_result = trapezoidal(f, a, b, N)
        # 计算Simpson法则结果
        simpson_result = simpson(f, a, b, N)
        
        # 计算相对误差（绝对值形式）
        trapezoidal_error = abs(trapezoidal_result - exact_integral) / exact_integral
        simpson_error = abs(simpson_result - exact_integral) / exact_integral
        
        # 格式化输出结果
        print(f"N = {N}")
        # 保留8位小数，误差用科学计数法显示2位有效数字
        print(f"梯形法则结果: {trapezoidal_result:.8f}, 相对误差: {trapezoidal_error:.2e}")
        print(f"Simpson法则结果: {simpson_result:.8f}, 相对误差: {simpson_error:.2e}")
        print("-" * 40)  # 分隔线

if __name__ == '__main__':
    main()
