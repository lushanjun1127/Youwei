"""
Youwei（陆幼薇）- 自我进化的AI系统
主启动程序
"""

import random
import math
from typing import List, Dict, Any, Callable
from datetime import datetime
import statistics
import threading


try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
except ImportError:
    print("警告：tkinter模块不可用，将使用命令行界面")
    tk = None


class SimpleNeuralNetwork:
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        # 添加网络结构限制
        self.input_size = input_size
        self.output_size = output_size
        
        # 限制网络结构大小
        max_layers = 5  # 最多5层
        max_neurons_per_layer = 20  # 每层最多20个神经元
        adjusted_hidden_sizes = []
        for size in hidden_sizes:
            adjusted_hidden_sizes.append(min(size, max_neurons_per_layer))
        # 限制层数
        adjusted_hidden_sizes = adjusted_hidden_sizes[:max_layers]
        self.hidden_sizes = adjusted_hidden_sizes
        
        self.weights = []
        self.biases = []
        
        layer_sizes = [input_size] + self.hidden_sizes + [output_size]
        
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
    
    def add_layer(self, position: int = -1):
        """动态添加一层神经网络"""
        if len(self.hidden_sizes) >= 5:  # 达到最大层数限制
            return False
            
        if position == -1:
            position = len(self.hidden_sizes)  # 在最后添加
        
        # 确定新层的大小（中间值或现有层的大小）
        new_layer_size = min(10, max(self.input_size, self.output_size, max(self.hidden_sizes) if self.hidden_sizes else 4))
        
        # 更新hidden_sizes
        self.hidden_sizes.insert(position, new_layer_size)
        
        # 重建网络结构
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        new_weights = []
        new_biases = []
        
        for i in range(len(layer_sizes) - 1):
            weight_matrix = [
                [random.uniform(-1, 1) for _ in range(layer_sizes[i])] 
                for _ in range(layer_sizes[i + 1])
            ]
            bias_vector = [random.uniform(-1, 1) for _ in range(layer_sizes[i + 1])]
            
            new_weights.append(weight_matrix)
            new_biases.append(bias_vector)
        
        self.weights = new_weights
        self.biases = new_biases
        
        return True
    
    def remove_layer(self, position: int = -1):
        """动态移除一层神经网络"""
        if len(self.hidden_sizes) <= 1:  # 至少保留一层
            return False
            
        if position == -1:
            position = len(self.hidden_sizes) - 1  # 移除最后一层
        
        if position < 0 or position >= len(self.hidden_sizes):
            return False
        
        # 更新hidden_sizes
        del self.hidden_sizes[position]
        
        # 重建网络结构
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        new_weights = []
        new_biases = []
        
        for i in range(len(layer_sizes) - 1):
            weight_matrix = [
                [random.uniform(-1, 1) for _ in range(layer_sizes[i])] 
                for _ in range(layer_sizes[i + 1])
            ]
            bias_vector = [random.uniform(-1, 1) for _ in range(layer_sizes[i + 1])]
            
            new_weights.append(weight_matrix)
            new_biases.append(bias_vector)
        
        self.weights = new_weights
        self.biases = new_biases
        
        return True


class MemoryBuffer:
    def __init__(self, capacity: int = 100):
        # 限制记忆容量
        self.capacity = min(capacity, 500)  # 最大记忆容量限制
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

    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """随机采样一批经验用于经验回放"""
        if len(self.buffer) == 0:
            return []
        sample_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, sample_size)


class AdaptiveStrategySelector:
    def __init__(self):
        self.current_strategy = 'exploitation'
        self.strategy_params = {
            'exploration': {'learning_rate': 0.15, 'mutation_rate': 0.2, 'explore_new_arch': True},
            'exploitation': {'learning_rate': 0.05, 'mutation_rate': 0.05, 'explore_new_arch': False},
            'adaptation': {'learning_rate': 0.1, 'mutation_rate': 0.1, 'explore_new_arch': True}
        }

    def evaluate_strategies(self, performance_trend: float) -> str:
        if performance_trend < 0:  # 性能下降
            return 'exploration'
        elif performance_trend > 0.01:  # 性能显著提升
            return 'exploitation'
        else:  # 性能平稳或轻微波动
            return 'adaptation'
    
    def get_strategy_params(self, strategy: str) -> Dict[str, Any]:
        """获取策略对应的参数"""
        return self.strategy_params.get(strategy, self.strategy_params['exploitation'])


class SafetyGovernance:
    """安全治理类 - 实现Youwei系统安全边界和控制机制"""
    def __init__(self):
        # 固化安全边界
        self.fixed_boundaries = {
            'max_generations': 1000,  # 最大进化代数
            'max_memory_usage': 500,  # 最大记忆容量
            'resource_budget': 10000,  # 资源预算
            'max_network_layers': 5,  # 最大网络层数
            'max_neurons_per_layer': 20,  # 每层最大神经元数
        }
        self.current_resource_usage = 0
        self.is_emergency_stop = False  # 紧急停止标志

    def check_resource_limit(self, increment: int = 1) -> bool:
        """检查资源使用限制"""
        if self.current_resource_usage + increment > self.fixed_boundaries['resource_budget']:
            print("⚠️  Youwei（陆幼薇）系统警告：资源预算即将超限，正在调整学习策略...")
            return False
        self.current_resource_usage += increment
        return True

    def emergency_stop(self):
        """紧急停止机制"""
        print("🚨 Youwei（陆幼薇）系统已被紧急停止！")
        self.is_emergency_stop = True

    def reset_system(self):
        """重置系统资源配额"""
        self.current_resource_usage = 0
        self.is_emergency_stop = False
        print("🔄 Youwei（陆幼薇）系统已重置资源配额")


class SelfEvolutionManager:
    """自进化管理器 - 实现完整的自进化循环"""
    def __init__(self, youwei_ai):
        self.youwei_ai = youwei_ai
        self.evolution_history = []
        self.best_performance = float('inf')  # 最佳性能（最低损失）
        self.best_config = None  # 最佳配置
    
    def evaluate_results(self, test_data_generator: Callable, num_tests: int = 10) -> Dict[str, Any]:
        """评价Youwei的表现"""
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for _ in range(num_tests):
            inputs, targets = test_data_generator()
            predicted = self.youwei_ai.network.forward(inputs)
            
            # 计算损失
            loss = self.youwei_ai.mse_loss(predicted, targets)
            total_loss += loss
            
            # 计算准确预测数
            for p, t in zip(predicted, targets):
                if abs(p - t) < 0.1:  # 容忍度为0.1
                    correct_predictions += 1
                total_predictions += 1
        
        avg_loss = total_loss / num_tests
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        evaluation_result = {
            'average_loss': avg_loss,
            'accuracy': accuracy,
            'total_tests': num_tests,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions
        }
        
        print(f"🔍 Youwei（陆幼薇）能力检测：准确率 {accuracy*100:.1f}%，平均错误 {avg_loss:.6f}")
        return evaluation_result
    
    def discover_defects(self, evaluation_result: Dict[str, Any]) -> List[str]:
        """发现Youwei的缺陷"""
        defects = []
        
        if evaluation_result['accuracy'] < 0.7:  # 准确率低于70%
            defects.append("整体准确率偏低")
        if evaluation_result['average_loss'] > 0.2:  # 损失过高
            defects.append("预测误差过大")
        if len(self.youwei_ai.network.hidden_sizes) == 1 and evaluation_result['accuracy'] < 0.8:
            defects.append("网络结构可能过于简单")
        if self.youwei_ai.learning_rate > 0.1 and evaluation_result['accuracy'] < 0.7:
            defects.append("学习率可能过高导致不稳定")
        
        if defects:
            print(f"🔍 Youwei（陆幼薇）发现缺陷：{', '.join(defects)}")
        else:
            print("✅ Youwei（陆幼薇）未发现明显缺陷")
        
        return defects
    
    def modify_strategy(self, defects: List[str]):
        """根据发现的缺陷修改策略"""
        if not defects:
            return  # 没有缺陷则不修改
        
        print("🔧 Youwei（陆幼薇）正在尝试新方法...")
        
        for defect in defects:
            if "准确率偏低" in defect or "预测误差过大" in defect:
                # 增加探索，尝试不同的学习策略
                self.youwei_ai.strategy_selector.current_strategy = 'exploration'
                strategy_params = self.youwei_ai.strategy_selector.get_strategy_params('exploration')
                self.youwei_ai.learning_rate = strategy_params['learning_rate']
                self.youwei_ai.mutation_rate = strategy_params['mutation_rate']
                print("🔄 Youwei（陆幼薇）正在尝试新节奏...")
            
            if "网络结构可能过于简单" in defect:
                # 尝试增加网络复杂度
                success = self.youwei_ai.network.add_layer()
                if success:
                    print(f"🧠 Youwei（陆幼薇）大脑多了一个小房间，当前结构: {[self.youwei_ai.network.input_size] + self.youwei_ai.network.hidden_sizes + [self.youwei_ai.network.output_size]}")
            
            if "学习率可能过高" in defect:
                # 降低学习率
                self.youwei_ai.learning_rate *= 0.8
                print(f"⚖️ Youwei（陆幼薇）调整了学习劲头，当前值: {self.youwei_ai.learning_rate:.6f}")
    
    def test_new_version(self, test_data_generator: Callable) -> float:
        """测试新版本的性能"""
        # 临时保存当前状态
        temp_network_state = {
            'weights': [w[:] for w in self.youwei_ai.network.weights],
            'biases': [b[:] for b in self.youwei_ai.network.biases],
            'hidden_sizes': self.youwei_ai.network.hidden_sizes[:]
        }
        
        # 测试性能
        evaluation = self.evaluate_results(test_data_generator, num_tests=20)
        new_performance = evaluation['average_loss']
        
        # 恢复状态（因为我们只是测试）
        self.youwei_ai.network.weights = temp_network_state['weights']
        self.youwei_ai.network.biases = temp_network_state['biases']
        self.youwei_ai.network.hidden_sizes = temp_network_state['hidden_sizes']
        
        return new_performance
    
    def retain_better_version(self, test_data_generator: Callable, threshold_improvement: float = 0.05):
        """保留更优版本"""
        current_evaluation = self.evaluate_results(test_data_generator, num_tests=20)
        current_performance = current_evaluation['average_loss']
        
        # 如果当前性能比历史最佳好一定阈值，则保留为最佳
        if current_performance < self.best_performance - threshold_improvement:
            print(f"🏆 Youwei（陆幼薇）取得进步！当前错误: {current_performance:.6f}，历史最佳: {self.best_performance:.6f}")
            self.best_performance = current_performance
            # 保存当前配置为最佳
            self.best_config = {
                'network_structure': [self.youwei_ai.network.input_size] + self.youwei_ai.network.hidden_sizes + [self.youwei_ai.network.output_size],
                'learning_rate': self.youwei_ai.learning_rate,
                'mutation_rate': self.youwei_ai.mutation_rate
            }
            print(f"💾 Youwei（陆幼薇）保存了当前优秀配置: {self.best_config}")
        elif current_performance < self.best_performance:
            # 性能有所改善但未超过阈值
            self.best_performance = current_performance
            print(f"📈 Youwei（陆幼薇）正在进步！当前错误: {current_performance:.6f}")
        else:
            # 性能没有改善，考虑恢复到最佳配置
            print(f"📊 Youwei（陆幼薇）当前错误: {current_performance:.6f}，历史最佳: {self.best_performance:.6f}")


class YouweiAI:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        
        self.network = SimpleNeuralNetwork(
            input_size=input_size,
            hidden_sizes=[6, 6],  # 适中的网络
            output_size=output_size
        )
        
        self.memory_buffer = MemoryBuffer(capacity=200)
        self.strategy_selector = AdaptiveStrategySelector()
        self.performance_history = []
        self.safety_governance = SafetyGovernance()  # 添加安全治理机制
        
        # 初始化参数，这些将根据策略动态调整
        self.learning_rate = 0.1
        self.mutation_rate = 0.1
        self.mutation_strength = 0.05
        
        # 添加自进化管理器
        self.self_evolution_manager = SelfEvolutionManager(self)

    def mse_loss(self, predicted: List[float], target: List[float]) -> float:
        if len(predicted) != len(target):
            raise ValueError("预测值和目标值长度不匹配")
        
        squared_errors = [(p - t) ** 2 for p, t in zip(predicted, target)]
        return sum(squared_errors) / len(squared_errors)

    def compute_gradients(self, inputs: List[float], targets: List[float]) -> tuple:
        """计算真正的梯度，提高训练效率"""
        # 前向传播
        activations = [inputs]
        zs = []  # 存储每一层的加权输入
        
        for i, (weight_matrix, bias_vector) in enumerate(zip(self.network.weights, self.network.biases)):
            z = []
            a = []
            
            for j in range(len(weight_matrix)):
                weighted_sum = sum(
                    activations[-1][k] * weight_matrix[j][k] 
                    for k in range(len(activations[-1]))
                ) + bias_vector[j]
                
                z.append(weighted_sum)
                
                if i < len(self.network.weights) - 1:
                    activation = self.network.relu(weighted_sum)
                else:
                    activation = self.network.sigmoid(weighted_sum)
                
                a.append(activation)
            
            zs.append(z)
            activations.append(a)
        
        # 反向传播
        # 输出层误差
        output_error = [2 * (a - t) for a, t in zip(activations[-1], targets)]
        
        # 计算输出层梯度
        output_delta = []
        for i, (a, error) in enumerate(zip(activations[-1], output_error)):
            sigmoid_derivative = a * (1 - a)  # sigmoid导数
            output_delta.append(error * sigmoid_derivative)
        
        deltas = [output_delta]
        
        # 计算隐藏层梯度
        for l in range(len(self.network.weights) - 2, -1, -1):
            prev_delta = deltas[-1]
            layer_delta = []
            
            for i in range(len(activations[l+1])):
                error = sum(
                    prev_delta[j] * self.network.weights[l+1][j][i] 
                    for j in range(len(prev_delta))
                )
                
                if l < len(self.network.weights) - 2:  # 隐藏层使用ReLU导数
                    relu_derivative = 1 if zs[l][i] > 0 else 0
                    layer_delta.append(error * relu_derivative)
                else:  # 倒数第二层使用ReLU导数
                    relu_derivative = 1 if zs[l][i] > 0 else 0
                    layer_delta.append(error * relu_derivative)
            
            deltas.append(layer_delta)
        
        deltas.reverse()  # 使顺序与权重匹配
        
        # 计算权重和偏置梯度
        weight_gradients = []
        bias_gradients = []
        
        for l in range(len(self.network.weights)):
            w_grad = []
            b_grad = []
            
            for i in range(len(self.network.weights[l])):
                row_grad = []
                for j in range(len(self.network.weights[l][i])):
                    grad = deltas[l][i] * activations[l][j]
                    row_grad.append(grad)
                w_grad.append(row_grad)
                
                b_grad.append(deltas[l][i])
            
            weight_gradients.append(w_grad)
            bias_gradients.append(b_grad)
        
        return weight_gradients, bias_gradients

    def train_step(self, inputs: List[float], targets: List[float], use_mutation: bool = False) -> Dict[str, Any]:
        # 检查资源限制
        if not self.safety_governance.check_resource_limit(1):
            return {'loss': float('inf'), 'predicted': [], 'actual': targets}
        
        predicted = self.network.forward(inputs)
        loss = self.mse_loss(predicted, targets)
        
        # 使用真正的梯度计算
        weight_gradients, bias_gradients = self.compute_gradients(inputs, targets)
        
        # 更新权重
        for l in range(len(self.network.weights)):
            for i in range(len(self.network.weights[l])):
                for j in range(len(self.network.weights[l][i])):
                    self.network.weights[l][i][j] -= self.learning_rate * weight_gradients[l][i][j]
        
        # 更新偏置
        for l in range(len(self.network.biases)):
            for i in range(len(self.network.biases[l])):
                self.network.biases[l][i] -= self.learning_rate * bias_gradients[l][i]
        
        # 根据策略决定是否应用突变
        if use_mutation:
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

    def replay_experience(self, batch_size: int = 10):
        """经验回放机制，从记忆中随机抽取经验进行再训练"""
        experiences = self.memory_buffer.sample_batch(batch_size)
        if not experiences:
            return
        
        for exp in experiences:
            # 使用当前学习参数重新训练
            self.train_step(exp['inputs'], exp['targets'])

    def adjust_strategy(self):
        """根据性能趋势调整策略和参数"""
        if len(self.performance_history) < 10:
            return  # 数据不足
        
        # 计算最近的性能趋势
        recent_performance = self.performance_history[-10:]
        avg_recent = sum(recent_performance[-5:]) / 5
        avg_prev = sum(recent_performance[:5]) / 5
        trend = avg_recent - avg_prev
        
        # 选择策略
        strategy = self.strategy_selector.evaluate_strategies(trend)
        params = self.strategy_selector.get_strategy_params(strategy)
        
        # 更新参数
        self.learning_rate = params['learning_rate']
        self.mutation_rate = params['mutation_rate']
        
        # 根据策略决定是否探索新架构
        if params['explore_new_arch'] and random.random() < 0.1:  # 10%概率探索新架构
            if random.random() < 0.5:
                # 尝试添加层
                success = self.network.add_layer()
                if success:
                    print(f"🔄 Youwei（陆幼薇）正在尝试添加神经网络层，当前结构: {[self.network.input_size] + self.network.hidden_sizes + [self.network.output_size]}")
            else:
                # 尝试移除层
                success = self.network.remove_layer()
                if success:
                    print(f"🔄 Youwei（陆幼薇）正在尝试移除神经网络层，当前结构: {[self.network.input_size] + self.network.hidden_sizes + [self.network.output_size]}")

    def self_evolution_cycle(self, data_generator: Callable):
        """执行一个完整的自进化循环"""
        print("🔄 Youwei（陆幼薇）开始自进化循环...")
        
        # 1. 任务输入（由data_generator提供）
        # 2. 模型执行（在evolve方法中已完成）
        
        # 3. 结果评价
        evaluation_result = self.self_evolution_manager.evaluate_results(data_generator)
        
        # 4. 发现缺陷
        defects = self.self_evolution_manager.discover_defects(evaluation_result)
        
        # 5. 修改自身策略/参数
        self.self_evolution_manager.modify_strategy(defects)
        
        # 6. 测试新版本（在retain_better_version中完成）
        # 7. 保留更优版本
        self.self_evolution_manager.retain_better_version(data_generator)
        
        print("✅ Youwei（陆幼薇）自进化循环完成！")

    def evolve(self, data_generator: Callable, num_generations: int = 500, train_size: int = 20, callback=None):
        """进化过程，支持回调函数用于GUI更新"""
        print("Youwei（陆幼薇）开始自我进化...")
        print("🔒 系统安全边界已设定，资源使用开始监控...")
        
        # 限制进化代数
        max_generations = min(num_generations, self.safety_governance.fixed_boundaries['max_generations'])
        
        for generation in range(max_generations):
            # 检查紧急停止
            if self.safety_governance.is_emergency_stop:
                print("🛑 Youwei（陆幼薇）系统已响应紧急停止命令，进化过程终止！")
                break
            
            # 调整策略
            self.adjust_strategy()
            
            total_loss = 0
            for _ in range(train_size):
                inputs, targets = data_generator()
                # 根据当前策略决定是否使用突变
                use_mutation = random.random() < self.mutation_rate
                result = self.train_step(inputs, targets, use_mutation)
                if result['loss'] == float('inf'):  # 资源超限
                    continue
                total_loss += result['loss']
            
            avg_loss = total_loss / train_size
            
            # 每100代进行一次经验回放和自进化循环
            if generation % 100 == 0:
                print(f"第 {generation} 代进化: 平均错误 = {avg_loss:.6f}, "
                      f"学习劲头 = {self.learning_rate:.6f}, "
                      f"资源使用率 = {self.safety_governance.current_resource_usage}/{self.safety_governance.fixed_boundaries['resource_budget']}")
                self.replay_experience(20)  # 回放经验
                # 执行自进化循环
                self.self_evolution_cycle(data_generator)
            
            # 每50代进行一次经验回放
            if generation % 50 == 0 and generation > 0:
                self.replay_experience(10)
            
            # 如果提供了回调函数，更新GUI
            if callback:
                callback(generation, avg_loss, self.learning_rate, self.safety_governance.current_resource_usage)
                time.sleep(0.01)  # 短暂延迟以避免GUI阻塞
        
        print("Youwei（陆幼薇）自我进化完成!")
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
            "总训练次数": len(self.performance_history),
            "最终平均错误": statistics.mean(self.performance_history[-20:]) if len(self.performance_history) >= 20 else statistics.mean(self.performance_history),
            "最小错误": min(self.performance_history) if self.performance_history else float('inf'),
            "最终学习劲头": self.learning_rate,
            "记忆使用量": len(self.memory_buffer.buffer),
            "近期学习趋势": "明显进步" if trend == "进步" and len(self.performance_history) > 100 and self.performance_history[-1] < self.performance_history[-50] else ("明显退步" if trend != "进步" else trend),
            "学习稳定性": stability,
            "最佳表现": f"第 {best_idx} 次训练时达到" if best_idx > 0 else "仍在努力中",
            "最终策略": self.strategy_selector.current_strategy,
            "最终网络结构": [self.network.input_size] + self.network.hidden_sizes + [self.network.output_size],
            "最佳历史性能": self.self_evolution_manager.best_performance,
            "最佳配置": self.self_evolution_manager.best_config,
            "资源使用量": self.safety_governance.current_resource_usage,
            "系统状态": "正常运行" if not self.safety_governance.is_emergency_stop else "已紧急停止"
        }


def create_generic_data() -> tuple:
    """
    创建通用的训练数据，用于测试网络的基本学习能力
    """
    # 生成随机输入和目标值
    inputs = [random.random() for _ in range(4)]  # 4维输入
    # 目标是输入值的加权平均，使问题更有趣
    weights = [random.uniform(0.5, 1.5) for _ in range(4)]
    weighted_sum = sum(input_val * weight for input_val, weight in zip(inputs, weights))
    targets = [weighted_sum / 4]  # 目标是加权平均值
    
    return inputs, targets


class YouweiGUI:
    """Youwei（陆幼薇）系统的GUI界面"""
    def __init__(self):
        if tk is None:
            return
            
        self.root = tk.Tk()
        self.root.title("Youwei（陆幼薇）- 自我进化的AI系统")
        self.root.geometry("800x600")
        
        # 创建控制面板
        self.create_control_panel()
        
        # 创建状态显示区域
        self.create_status_area()
        
        # 创建图表显示区域
        self.create_chart_area()
        
        # 创建日志显示区域
        self.create_log_area()
        
        # 初始化YouweiAI实例
        self.youwei = None
        self.is_running = False
        self.training_thread = None
        
        # 训练进度变量
        self.progress_var = tk.DoubleVar()
        
        # 添加标签页
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 控制面板标签页
        self.control_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.control_frame, text="控制面板")
        
        # 添加控制面板元素到标签页
        self.setup_control_frame()
        
        # 状态面板标签页
        self.status_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.status_frame, text="系统状态")
        
        # 添加状态面板元素到标签页
        self.setup_status_frame()
    
    def create_control_panel(self):
        """创建控制面板"""
        if tk is None:
            return
            
        control_frame = ttk.LabelFrame(self.root, text="控制面板", padding=(10, 5))
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 进度条
        self.progress_label = ttk.Label(control_frame, text="进化进度: 0%")
        self.progress_label.pack(side=tk.TOP, anchor=tk.W)
        
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # 控制按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.start_btn = ttk.Button(button_frame, text="开始进化", command=self.start_evolution)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="停止进化", command=self.stop_evolution, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = ttk.Button(button_frame, text="重置系统", command=self.reset_system)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
    
    def create_status_area(self):
        """创建状态显示区域"""
        if tk is None:
            return
            
        status_frame = ttk.LabelFrame(self.root, text="系统状态", padding=(10, 5))
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 创建状态标签
        self.status_labels = {}
        statuses = [
            ("当前代数:", "generation"),
            ("平均错误:", "loss"),
            ("学习劲头:", "learning_rate"),
            ("资源使用:", "resources"),
            ("网络结构:", "network_structure"),
            ("系统状态:", "system_status")
        ]
        
        for i, (label_text, key) in enumerate(statuses):
            row = i // 2
            col = i % 2
            frame = ttk.Frame(status_frame)
            frame.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            
            ttk.Label(frame, text=label_text).pack(side=tk.LEFT)
            self.status_labels[key] = ttk.Label(frame, text="-")
            self.status_labels[key].pack(side=tk.LEFT, padx=(5, 0))
    
    def create_chart_area(self):
        """创建图表显示区域"""
        if tk is None:
            return
            
        chart_frame = ttk.LabelFrame(self.root, text="进化曲线", padding=(10, 5))
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 这里可以添加图表显示，由于tkinter不直接支持绘图，我们使用文本显示
        self.chart_text = tk.Text(chart_frame, height=10, state=tk.DISABLED)
        self.chart_text.pack(fill=tk.BOTH, expand=True)
    
    def create_log_area(self):
        """创建日志显示区域"""
        if tk is None:
            return
            
        log_frame = ttk.LabelFrame(self.root, text="系统日志", padding=(10, 5))
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def setup_control_frame(self):
        """设置控制面板标签页内容"""
        if tk is None:
            return
            
        # 参数设置
        params_frame = ttk.LabelFrame(self.control_frame, text="参数设置", padding=(10, 5))
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 进化代数
        ttk.Label(params_frame, text="进化代数:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.generations_var = tk.StringVar(value="500")
        ttk.Entry(params_frame, textvariable=self.generations_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 训练批次大小
        ttk.Label(params_frame, text="训练批次:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.batch_size_var = tk.StringVar(value="15")
        ttk.Entry(params_frame, textvariable=self.batch_size_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
    
    def setup_status_frame(self):
        """设置状态面板标签页内容"""
        if tk is None:
            return
            
        # 详细状态信息
        detail_frame = ttk.LabelFrame(self.status_frame, text="详细状态", padding=(10, 5))
        detail_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.detail_text = tk.Text(detail_frame, state=tk.DISABLED)
        self.detail_text.pack(fill=tk.BOTH, expand=True)
    
    def start_evolution(self):
        """开始进化"""
        if tk is None:
            print("开始Youwei进化...")
            return
            
        if self.is_running:
            return
            
        # 初始化YouweiAI
        self.youwei = YouweiAI(input_size=4, output_size=1)
        self.is_running = True
        
        # 更新按钮状态
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # 启动进化线程
        self.training_thread = threading.Thread(target=self.run_evolution)
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def run_evolution(self):
        """运行进化过程"""
        if tk is None:
            return
            
        generations = int(self.generations_var.get())
        batch_size = int(self.batch_size_var.get())
        
        def update_callback(generation, loss, learning_rate, resources):
            """更新GUI的回调函数"""
            self.root.after(0, self.update_gui, generation, loss, learning_rate, resources)
        
        try:
            results = self.youwei.evolve(create_generic_data, num_generations=generations, train_size=batch_size, callback=update_callback)
            self.root.after(0, self.on_evolution_complete, results)
        except Exception as e:
            self.root.after(0, self.on_evolution_error, str(e))
    
    def update_gui(self, generation, loss, learning_rate, resources):
        """更新GUI显示"""
        if tk is None:
            return
            
        # 更新进度条
        total_generations = int(self.generations_var.get())
        progress = (generation / total_generations) * 100 if total_generations > 0 else 0
        self.progress_var.set(progress)
        self.progress_label.config(text=f"进化进度: {progress:.1f}% (第 {generation} 代)")
        
        # 更新状态标签
        self.status_labels["generation"].config(text=str(generation))
        self.status_labels["loss"].config(text=f"{loss:.6f}")
        self.status_labels["learning_rate"].config(text=f"{learning_rate:.6f}")
        self.status_labels["resources"].config(text=f"{resources}/10000")
        
        if self.youwei:
            self.status_labels["network_structure"].config(
                text=str([self.youwei.network.input_size] + self.youwei.network.hidden_sizes + [self.youwei.network.output_size])
            )
            self.status_labels["system_status"].config(
                text="运行中" if not self.youwei.safety_governance.is_emergency_stop else "已停止"
            )
        
        # 添加日志
        self.log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] 第 {generation} 代: 错误={loss:.6f}, 学习劲头={learning_rate:.6f}\n")
        self.log_text.see(tk.END)
    
    def on_evolution_complete(self, results):
        """进化完成时的处理"""
        if tk is None:
            print("Youwei进化完成!")
            return
            
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        # 显示结果
        self.log_text.insert(tk.END, "\nYouwei（陆幼薇）进化完成!\n")
        for key, value in results.items():
            self.log_text.insert(tk.END, f"  {key}: {value}\n")
        self.log_text.see(tk.END)
        
        # 显示详细状态
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete(1.0, tk.END)
        for key, value in results.items():
            self.detail_text.insert(tk.END, f"{key}: {value}\n")
        self.detail_text.config(state=tk.DISABLED)
    
    def on_evolution_error(self, error_msg):
        """进化出错时的处理"""
        if tk is None:
            print(f"Youwei进化出错: {error_msg}")
            return
            
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        messagebox.showerror("进化错误", f"Youwei（陆幼薇）进化过程中出现错误:\n{error_msg}")
    
    def stop_evolution(self):
        """停止进化"""
        if tk is None:
            print("停止Youwei进化...")
            return
            
        if self.youwei:
            self.youwei.safety_governance.emergency_stop()
        
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
    
    def reset_system(self):
        """重置系统"""
        if tk is None:
            print("重置Youwei系统...")
            return
            
        if self.youwei:
            self.youwei.safety_governance.reset_system()
            self.log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Youwei（陆幼薇）系统已重置\n")
            self.log_text.see(tk.END)
    
    def run(self):
        """运行GUI"""
        if tk is None:
            print("=" * 60)
            print("           欢迎使用Youwei（陆幼薇）系统!")
            print("=" * 60)
            print("Youwei是一个自我进化的AI，具有以下特性:")
            print("• 自适应学习能力")
            print("• 动态架构调整（受安全边界约束）")
            print("• 记忆与经验积累（支持经验回放）")
            print("• 自主策略选择（根据性能调整参数）")
            print("• 完整自进化循环（评价→发现缺陷→改进→测试→保留优版本）")
            print("🔒 系统已启用安全治理机制")
            print("=" * 60)
            
            print("\n【Youwei自我进化计划】")
            print("-" * 40)
            
            print("Youwei（陆幼薇）正在启动自我进化进程...")
            print("→ 创建学习大脑（模拟思维结构）")
            print("→ 设置记忆库（用于存储学习经验，容量受限）")
            print("→ 初始化学习参数")
            print("→ 启动安全治理机制（资源监控、紧急停止等）")
            print("→ 初始化自进化管理器（评价、缺陷发现、改进、版本选择）")
            
            # 创建Youwei实例
            youwei = YouweiAI(input_size=4, output_size=1)
            
            print("→ Youwei自我进化准备完成！")
            
            print("\n→ 开始自我进化...")
            print("  Youwei将通过通用数据进行自我训练")
            print("  每次训练后，Youwei都会总结经验并改进自己")
            print("  🔒 系统将持续监控资源使用情况，确保安全运行")
            print("  🔄 Youwei将根据学习情况动态调整策略和网络结构")
            print("  🔄 Youwei将执行完整的自进化循环（评价→发现缺陷→改进→测试→保留优版本）")
            
            # 定义通用数据生成器
            def data_generator():
                return create_generic_data()
            
            # 运行进化过程
            results = youwei.evolve(data_generator, num_generations=500, train_size=15)
            
            print("\nYouwei自我进化总结:")
            for key, value in results.items():
                print(f"  {key}: {value}")
            
            print("\nYouwei自我进化测试:")
            print("→ 测试Youwei的学习成果...")
            
            test_correct = 0
            test_total = 0
            for i in range(10):
                inputs, targets = create_generic_data()
                predicted = youwei.network.forward(inputs)
                
                for p, t in zip(predicted, targets):
                    if abs(p - t) < 0.1:  # 容忍度为0.1
                        test_correct += 1
                    test_total += 1
            
            accuracy = test_correct / test_total if test_total > 0 else 0
            print(f"  测试准确率: {test_correct}/{test_total} ({accuracy*100:.1f}%)")
            
            if accuracy >= 0.8:
                print("\n  🎉 恭喜Youwei！进化效果优秀！")
            elif accuracy >= 0.6:
                print("\n  👍 Youwei表现不错，继续努力！")
            else:
                print("\n  💪 Youwei需要更多练习，加油！")
            
            print("\nYouwei（陆幼薇）自我进化完成!")
            print("接下来可以继续学习更高级的知识！")
            print("🔒 安全治理机制将持续守护Youwei的成长之路！")
            print("🔄 自进化循环将帮助Youwei持续改进自己！")
            print("感谢使用Youwei（陆幼薇）自我进化的AI系统!")
        else:
            self.root.mainloop()


def main():
    app = YouweiGUI()
    app.run()


if __name__ == "__main__":
    main()