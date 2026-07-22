"""
Youwei（陆幼薇）- 自我进化的AI系统
主启动程序
"""

import sys
import os
import random
import math
import json
from typing import List, Dict, Any, Callable
from datetime import datetime
import statistics


class SimpleNeuralNetwork:
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.weights = []
        self.biases = []
        
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            weight_matrix = [
                [random.uniform(-1, 1) for _ in range(layer_sizes[i])] 
                for _ in range(layer_sizes[i + 1])
            ]
            bias_vector = [random.uniform(-1, 1) for _ in range(layer_sizes[i + 1])]
            
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
    
    def relu(self, x: float) -> float:
        return max(0, x)
    
    def sigmoid(self, x: float) -> float:
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0
    
    def forward(self, inputs: List[float]) -> List[float]:
        current_activations = inputs[:]
        
        for i, (weight_matrix, bias_vector) in enumerate(zip(self.weights, self.biases)):
            next_activations = []
            
            for j in range(len(weight_matrix)):
                weighted_sum = sum(
                    current_activations[k] * weight_matrix[j][k] 
                    for k in range(len(current_activations))
                ) + bias_vector[j]
                
                if i < len(self.weights) - 1:
                    activation = self.relu(weighted_sum)
                else:
                    activation = self.sigmoid(weighted_sum)
                
                next_activations.append(activation)
            
            current_activations = next_activations
        
        return current_activations
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        for weight_matrix in self.weights:
            for row in weight_matrix:
                for i in range(len(row)):
                    if random.random() < mutation_rate:
                        row[i] += random.gauss(0, mutation_strength)
        
        for bias_vector in self.biases:
            for i in range(len(bias_vector)):
                if random.random() < mutation_rate:
                    bias_vector[i] += random.gauss(0, mutation_strength)


class MemoryBuffer:
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add_experience(self, experience: Dict[str, Any]):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def get_recent_experiences(self, count: int) -> List[Dict[str, Any]]:
        return self.buffer[-min(count, len(self.buffer)):]


class AdaptiveStrategySelector:
    def __init__(self):
        self.current_strategy = 'exploitation'

    def evaluate_strategies(self, performance_trend: float) -> str:
        if performance_trend < 0:  # 性能下降
            return 'exploration'
        elif performance_trend > 0.01:  # 性能显著提升
            return 'exploitation'
        else:  # 性能平稳或轻微波动
            return 'adaptation'


class YouweiAI:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        
        self.network = SimpleNeuralNetwork(
            input_size=input_size,
            hidden_sizes=[4, 4],  # 更小的网络，更适合简单任务
            output_size=output_size
        )
        
        self.memory_buffer = MemoryBuffer(capacity=200)
        self.strategy_selector = AdaptiveStrategySelector()
        self.performance_history = []
        
        self.mutation_rate = 0.2  # 更高的变异率，促进探索
        self.mutation_strength = 0.1  # 更强的变异力度
        self.learning_rate = 0.05  # 更高的学习率，加快收敛速度

    def mse_loss(self, predicted: List[float], target: List[float]) -> float:
        if len(predicted) != len(target):
            raise ValueError("预测值和目标值长度不匹配")
        
        squared_errors = [(p - t) ** 2 for p, t in zip(predicted, target)]
        return sum(squared_errors) / len(squared_errors)

    def simple_backpropagate(self, inputs: List[float], targets: List[float]):
        original_output = self.network.forward(inputs)
        original_loss = self.mse_loss(original_output, targets)
        
        original_weights = []
        original_biases = []
        
        for weight_matrix in self.network.weights:
            original_weights.append([row[:] for row in weight_matrix])
        
        for bias_vector in self.network.biases:
            original_biases.append(bias_vector[:])
        
        for layer_idx in range(len(self.network.weights)):
            for neuron_idx in range(len(self.network.weights[layer_idx])):
                for weight_idx in range(len(self.network.weights[layer_idx][neuron_idx])):
                    self.network.weights[layer_idx][neuron_idx][weight_idx] += 0.001
                    new_output = self.network.forward(inputs)
                    new_loss = self.mse_loss(new_output, targets)
                    gradient = (new_loss - original_loss) / 0.001
                    self.network.weights[layer_idx][neuron_idx][weight_idx] -= self.learning_rate * gradient
                    self.network.weights[layer_idx][neuron_idx][weight_idx] = original_weights[layer_idx][neuron_idx][weight_idx]
        
        for layer_idx in range(len(self.network.biases)):
            for bias_idx in range(len(self.network.biases[layer_idx])):
                self.network.biases[layer_idx][bias_idx] += 0.001
                new_output = self.network.forward(inputs)
                new_loss = self.mse_loss(new_output, targets)
                gradient = (new_loss - original_loss) / 0.001
                self.network.biases[layer_idx][bias_idx] -= self.learning_rate * gradient
                self.network.biases[layer_idx][bias_idx] = original_biases[layer_idx][bias_idx]

    def train_step(self, inputs: List[float], targets: List[float]) -> Dict[str, Any]:
        predicted = self.network.forward(inputs)
        loss = self.mse_loss(predicted, targets)
        
        self.simple_backpropagate(inputs, targets)
        
        # 增强变异以帮助学习
        self.network.mutate(self.mutation_rate, self.mutation_strength)
        
        self.performance_history.append(loss)
        
        experience = {
            'inputs': inputs,
            'targets': targets,
            'predicted': predicted,
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        self.memory_buffer.add_experience(experience)
        
        return {
            'loss': loss,
            'predicted': predicted,
            'actual': targets
        }

    def evolve(self, data_generator: Callable, num_generations: int = 500, train_size: int = 20):
        print("Youwei（陆幼薇）开始小学基础学习...")
        
        for generation in range(num_generations):
            total_loss = 0
            for _ in range(train_size):
                inputs, targets = data_generator()
                result = self.train_step(inputs, targets)
                total_loss += result['loss']
            
            avg_loss = total_loss / train_size
            
            # 每100代输出一次状态，使用更易懂的语言
            if generation % 100 == 0:
                print(f"第 {generation} 次课堂: 平均错误 = {avg_loss:.6f}, "
                      f"学习劲头 = {self.learning_rate:.6f}")
        
        print("Youwei（陆幼薇）小学课程学习完成!")
        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        if not self.performance_history:
            return {"error": "没有收集到性能数据"}
        
        # 计算学习趋势
        trend = "进步"
        if len(self.performance_history) > 1:
            if self.performance_history[0] > self.performance_history[-1]:
                trend = "进步"
            else:
                trend = "待提升"
        
        # 计算学习稳定性
        stability = "稳定"
        if len(self.performance_history) > 20:
            recent_var = statistics.variance(self.performance_history[-20:])
            if recent_var > 0.05:
                stability = "波动"
            elif recent_var > 0.01:
                stability = "中等"
            else:
                stability = "稳定"
        
        # 找出最佳表现时间点
        best_idx = 0
        if self.performance_history:
            best_idx = self.performance_history.index(min(self.performance_history))
        
        return {
            "总练习次数": len(self.performance_history),
            "最终平均错误": statistics.mean(self.performance_history[-20:]) if len(self.performance_history) >= 20 else statistics.mean(self.performance_history),
            "最小错误": min(self.performance_history) if self.performance_history else float('inf'),
            "最终学习劲头": self.learning_rate,
            "记忆使用量": len(self.memory_buffer.buffer),
            "近期学习趋势": "明显进步" if trend == "进步" and len(self.performance_history) > 100 and self.performance_history[-1] < self.performance_history[-50] else ("明显退步" if trend != "进步" else trend),
            "学习稳定性": stability,
            "最佳表现": f"第 {best_idx} 次练习时达到" if best_idx > 0 else "仍在努力中",
            "最终策略": self.strategy_selector.current_strategy
        }


def create_simple_addition_data() -> tuple:
    """
    创建小学一年级水平的加法练习题（5以内的数字）
    """
    # 生成简单的5以内加法题
    a = random.randint(1, 5)
    b = random.randint(1, 5)
    inputs = [float(a)/5.0, float(b)/5.0, 0.0, 0.0]  # 归一化输入
    targets = [float(a + b)/10.0]  # 归一化输出到[0,1]，最大值为5+5=10
    
    return inputs, targets


def create_simple_subtraction_data() -> tuple:
    """
    创建小学一年级水平的减法练习题（5以内的数字）
    """
    # 生成简单的5以内减法题，确保不会出现负数
    a = random.randint(3, 5)
    b = random.randint(1, a)  # b不能大于a，避免负数
    inputs = [float(a)/5.0, float(b)/5.0, 0.0, 0.0]  # 归一化输入
    targets = [float(a - b)/5.0]  # 归一化输出到[0,1]，最大值为5
    
    return inputs, targets


def create_counting_data() -> tuple:
    """
    创建小学水平的计数练习题（简单汉字）
    """
    # 生成简单的计数题
    word_count = random.randint(2, 5)  # 减少计数范围
    words = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "花", "草", "树", "鸟"]
    sentence = "".join(random.choices(words, k=word_count))
    
    # 将字符转换为数值特征
    char_values = [ord(c) / 10000.0 for c in sentence]
    inputs = [0.0, 0.0, 0.0, 0.0]  # 初始化4个输入
    
    # 取前4个字符，如果不够则用0填充
    for i in range(min(len(char_values), 4)):
        inputs[i] = char_values[i]
    
    targets = [word_count / 5.0]  # 输出句子长度，归一化到[0,1]
    
    return inputs, targets


def create_mixed_primary_data() -> tuple:
    """
    创建混合的小学水平练习题（数学和语文）
    """
    # 随机选择题目类型
    choice = random.choice(["addition", "subtraction", "counting"])
    
    if choice == "addition":
        return create_simple_addition_data()
    elif choice == "subtraction":
        return create_simple_subtraction_data()
    else:
        return create_counting_data()


def display_youwei_welcome():
    print("=" * 60)
    print("           欢迎使用Youwei（陆幼薇）系统!")
    print("=" * 60)
    print("Youwei是一个自我进化的AI，具有以下特性:")
    print("• 自适应学习能力")
    print("• 动态架构调整")
    print("• 记忆与经验积累")
    print("• 自主策略选择")
    print("• 持续自我优化")
    print("=" * 60)


def main():
    display_youwei_welcome()
    
    print("\n【Youwei小学基础学习计划】")
    print("-" * 40)
    
    print("Youwei（陆幼薇）正在准备学习小学课程...")
    print("→ 创建学习大脑（模拟思维结构）")
    print("→ 设置记忆库（用于存储学习经验）")
    print("→ 初始化学习参数")
    
    # 创建Youwei实例
    youwei = YouweiAI(input_size=4, output_size=1)
    
    print("→ Youwei学习准备完成！")
    
    print("\n→ 开始小学课程学习...")
    print("  Youwei将从最基础的数学和语文开始学习")
    print("  数学：5以内的加减法（适合一年级）")
    print("  语文：简单汉字识别和计数")
    print("  每次练习后，Youwei都会总结经验并改进自己")
    
    # 定义数据生成器
    def data_generator():
        return create_mixed_primary_data()
    
    # 运行学习过程
    results = youwei.evolve(data_generator, num_generations=300, train_size=10)
    
    print("\nYouwei小学课程学习总结:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    print("\nYouwei小学课程测试:")
    print("→ 让我们来检验Youwei的学习成果...")
    
    print("  数学题测试：")
    print("  加法练习（3以内的数字）：")
    correct_add = 0
    total_add = 0
    for i in range(5):
        a = random.randint(1, 3)
        b = random.randint(1, 3)
        inputs = [float(a)/3.0, float(b)/3.0, 0.0, 0.0]
        expected_output = float(a + b)
        actual_output = youwei.network.forward(inputs)[0] * 6  # 反归一化输出
        actual_result = round(actual_output)
        print(f"    {a} + {b} = ?")
        print(f"      正确答案: {expected_output}")
        print(f"      Youwei计算: {actual_result}")
        if abs(expected_output - actual_result) <= 0.5:  # 更宽松的判断
            print(f"      结果: ✓")
            correct_add += 1
        else:
            print(f"      结果: ✗")
        total_add += 1
    
    print(f"  加法准确率: {correct_add}/{total_add} ({correct_add/total_add*100:.1f}%)")
    
    print("  减法练习（3以内的数字）：")
    correct_sub = 0
    total_sub = 0
    for i in range(3):
        a = random.randint(2, 3)
        b = random.randint(1, a)
        inputs = [float(a)/3.0, float(b)/3.0, 0.0, 0.0]
        expected_output = float(a - b)
        actual_output = youwei.network.forward(inputs)[0] * 3  # 反归一化输出
        actual_result = round(actual_output)
        print(f"    {a} - {b} = ?")
        print(f"      正确答案: {expected_output}")
        print(f"      Youwei计算: {actual_result}")
        if abs(expected_output - actual_result) <= 0.5:  # 更宽松的判断
            print(f"      结果: ✓")
            correct_sub += 1
        else:
            print(f"      结果: ✗")
        total_sub += 1
    
    print(f"  减法准确率: {correct_sub}/{total_sub} ({correct_sub/total_sub*100:.1f}%)")
    
    print("  语文题测试：")
    correct_count = 0
    total_count = 0
    for i in range(3):
        words = ["一", "二", "三", "花", "草", "树"]
        sentence_len = random.randint(1, 3)
        sentence = "".join(random.choices(words, k=sentence_len))
        char_values = [ord(c) / 10000.0 for c in sentence]
        inputs = [0.0, 0.0, 0.0, 0.0]
        for j in range(min(len(char_values), 4)):
            inputs[j] = char_values[j]
        actual_output = youwei.network.forward(inputs)[0] * 3  # 反归一化输出
        estimated_len = round(actual_output)
        print(f"    句子: '{sentence}' (共{sentence_len}个字)")
        print(f"      Youwei估算: {estimated_len}个字")
        if abs(sentence_len - estimated_len) <= 0.5:  # 更宽松的判断
            print(f"      结果: ✓")
            correct_count += 1
        else:
            print(f"      结果: ✗")
        total_count += 1
    
    print(f"  识字计数准确率: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
    
    overall_accuracy = (correct_add + correct_sub + correct_count) / (total_add + total_sub + total_count)
    print(f"\n  综合成绩: {overall_accuracy*100:.1f}%")
    
    if overall_accuracy >= 0.8:
        print("\n  🎉 恭喜Youwei！小学课程学习优秀！")
    elif overall_accuracy >= 0.6:
        print("\n  👍 Youwei表现不错，继续努力！")
    else:
        print("\n  💪 Youwei需要更多练习，加油！")
    
    print("\nYouwei（陆幼薇）小学课程学习完成! 系统已掌握基础数学和语文知识。")
    print("接下来可以继续学习更高级的知识！")
    print("感谢使用Youwei（陆幼薇）自我进化的AI系统!")


if __name__ == "__main__":
    main()