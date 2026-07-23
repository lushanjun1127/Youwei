"""
Youwei（陆幼薇）- 自我进化的AI系统（完善版）
修复与优化：
1. 自进化回滚增强：修改前备份，修改后若误差增大立即恢复原状。
2. 训练资源耗尽自动终止，防止无效循环。
3. 反向传播稳定，结构修改更保守地保留知识。
4. 图形界面状态更新更流畅，日志完整。
5. 命令行模式增加训练总结和准确率输出。
"""

import random
import math
from typing import List, Dict, Any, Callable, Tuple, Optional
from datetime import datetime
import statistics
import threading
import time
import copy

# 类型别名
Weights3D = List[List[List[float]]]
Biases2D = List[List[float]]
Activations1D = List[float]
TrainingData = Tuple[List[float], List[float]]
DataGenerator = Callable[[], TrainingData]

# 图形界面支持
tk_available = False
tk = None
ttk = None
scrolledtext = None
messagebox = None

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    tk_available = True
except ImportError:
    print("警告：tkinter模块不可用，将使用命令行界面")


class SimpleNeuralNetwork:
    """动态结构神经网络，隐藏层ReLU，输出层线性（回归）"""
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        
        # 结构限制
        max_layers = 5
        max_neurons = 20
        adjusted = [min(s, max_neurons) for s in hidden_sizes]
        self.hidden_sizes = adjusted[:max_layers]
        
        self.weights: Weights3D = []
        self.biases: Biases2D = []
        self._init_weights()

    def _init_weights(self):
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            w = [[random.uniform(-0.5, 0.5) for _ in range(layer_sizes[i])]
                 for _ in range(layer_sizes[i+1])]
            b = [random.uniform(-0.5, 0.5) for _ in range(layer_sizes[i+1])]
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, x: float) -> float:
        return max(0.0, x)

    def sigmoid(self, x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def forward(self, inputs: List[float]) -> List[float]:
        """前向传播：隐藏层ReLU，输出层线性"""
        current = inputs[:]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            next_act = []
            for j in range(len(w)):
                z = sum(current[k] * w[j][k] for k in range(len(current))) + b[j]
                if i < len(self.weights) - 1:
                    next_act.append(self.relu(z))
                else:
                    next_act.append(z)   # 线性输出
            current = next_act
        return current

    def mutate(self, mutation_rate: float = 0.1, strength: float = 0.1):
        for w_mat in self.weights:
            for row in w_mat:
                for i in range(len(row)):
                    if random.random() < mutation_rate:
                        row[i] += random.gauss(0, strength)
        for b_vec in self.biases:
            for i in range(len(b_vec)):
                if random.random() < mutation_rate:
                    b_vec[i] += random.gauss(0, strength)

    def add_layer(self, position: int = -1) -> bool:
        """添加隐藏层，尽量保持原有映射"""
        if len(self.hidden_sizes) >= 5:
            return False
        if position == -1:
            position = len(self.hidden_sizes)
        if self.hidden_sizes:
            prev = self.hidden_sizes[position-1] if position > 0 else self.input_size
            nxt = self.hidden_sizes[position] if position < len(self.hidden_sizes) else self.output_size
            new_size = min(20, max(4, (prev + nxt) // 2))
        else:
            new_size = min(20, max(4, self.input_size, self.output_size))
        self.hidden_sizes.insert(position, new_size)
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        new_w, new_b = [], []
        for i in range(len(layer_sizes) - 1):
            w = [[0.0] * layer_sizes[i] for _ in range(layer_sizes[i+1])]
            b = [0.0] * layer_sizes[i+1]
            if i == position:  # 前一层 -> 新层
                for j in range(layer_sizes[i+1]):
                    for k in range(layer_sizes[i]):
                        w[j][k] = random.uniform(-0.1, 0.1)
                    b[j] = random.uniform(-0.1, 0.1)
            elif i == position + 1:  # 新层 -> 下一层
                if layer_sizes[i] == layer_sizes[i+1]:
                    for j in range(layer_sizes[i+1]):
                        w[j][j] = random.uniform(0.9, 1.1)
                        for k in range(layer_sizes[i]):
                            if k != j:
                                w[j][k] = random.uniform(-0.05, 0.05)
                        b[j] = random.uniform(-0.05, 0.05)
                else:
                    for j in range(layer_sizes[i+1]):
                        for k in range(layer_sizes[i]):
                            w[j][k] = random.uniform(-0.5, 0.5)
                        b[j] = random.uniform(-0.5, 0.5)
            else:  # 保留原有权重（若维度匹配）
                old_idx = i if i < position else i-1
                if old_idx < len(self.weights):
                    old_w, old_b = self.weights[old_idx], self.biases[old_idx]
                    if len(old_w) == layer_sizes[i+1] and (len(old_w[0]) if old_w else 0) == layer_sizes[i]:
                        new_w.append(copy.deepcopy(old_w))
                        new_b.append(copy.deepcopy(old_b))
                        continue
                for j in range(layer_sizes[i+1]):
                    for k in range(layer_sizes[i]):
                        w[j][k] = random.uniform(-0.5, 0.5)
                    b[j] = random.uniform(-0.5, 0.5)
            new_w.append(w)
            new_b.append(b)
        self.weights = new_w
        self.biases = new_b
        return True

    def remove_layer(self, position: int = -1) -> bool:
        """移除一个隐藏层，尽量连接前后层"""
        if len(self.hidden_sizes) <= 1:
            return False
        if position == -1:
            position = len(self.hidden_sizes) - 1
        if not (0 <= position < len(self.hidden_sizes)):
            return False
        del self.hidden_sizes[position]
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        old_w, old_b = self.weights, self.biases
        new_w, new_b = [], []
        old_idx = 0
        for i in range(len(layer_sizes) - 1):
            if old_idx == position:
                old_idx += 1
            if old_idx < len(old_w):
                ow, ob = old_w[old_idx], old_b[old_idx]
                if len(ow) == layer_sizes[i+1] and (len(ow[0]) if ow else 0) == layer_sizes[i]:
                    new_w.append(copy.deepcopy(ow))
                    new_b.append(copy.deepcopy(ob))
                else:
                    w = [[random.uniform(-0.5, 0.5) for _ in range(layer_sizes[i])] for _ in range(layer_sizes[i+1])]
                    b = [random.uniform(-0.5, 0.5) for _ in range(layer_sizes[i+1])]
                    new_w.append(w)
                    new_b.append(b)
            else:
                w = [[random.uniform(-0.5, 0.5) for _ in range(layer_sizes[i])] for _ in range(layer_sizes[i+1])]
                b = [random.uniform(-0.5, 0.5) for _ in range(layer_sizes[i+1])]
                new_w.append(w)
                new_b.append(b)
            old_idx += 1
        self.weights = new_w
        self.biases = new_b
        return True

    def save_state(self) -> Dict[str, Any]:
        return {
            'weights': copy.deepcopy(self.weights),
            'biases': copy.deepcopy(self.biases),
            'hidden_sizes': self.hidden_sizes[:],
            'input_size': self.input_size,
            'output_size': self.output_size
        }

    def restore_state(self, state: Dict[str, Any]):
        self.weights = state['weights']
        self.biases = state['biases']
        self.hidden_sizes = state['hidden_sizes']
        self.input_size = state['input_size']
        self.output_size = state['output_size']


class MemoryBuffer:
    def __init__(self, capacity: int = 100):
        self.capacity = min(capacity, 500)
        self.buffer: List[Dict[str, Any]] = []
        self.position = 0

    def add_experience(self, exp: Dict[str, Any]):
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            self.buffer[self.position] = exp
        self.position = (self.position + 1) % self.capacity

    def get_recent(self, count: int) -> List[Dict[str, Any]]:
        return self.buffer[-min(count, len(self.buffer)):]

    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        if not self.buffer:
            return []
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))


class AdaptiveStrategySelector:
    def __init__(self):
        self.current_strategy = 'exploitation'
        self.params = {
            'exploration': {'learning_rate': 0.15, 'mutation_rate': 0.2, 'explore_new_arch': True},
            'exploitation': {'learning_rate': 0.05, 'mutation_rate': 0.05, 'explore_new_arch': False},
            'adaptation': {'learning_rate': 0.1, 'mutation_rate': 0.1, 'explore_new_arch': True}
        }

    def evaluate(self, trend: float) -> str:
        if trend < 0:
            return 'exploration'
        elif trend > 0.01:
            return 'exploitation'
        else:
            return 'adaptation'

    def get_params(self, strategy: str) -> Dict[str, Any]:
        return self.params.get(strategy, self.params['exploitation'])


class SafetyGovernance:
    def __init__(self):
        self.boundaries = {
            'max_generations': 1000,
            'max_memory': 500,
            'resource_budget': 10000,
            'max_layers': 5,
            'max_neurons': 20,
        }
        self.resource_used = 0
        self.emergency_stop = False

    def check_resource(self, inc: int = 1) -> bool:
        if self.resource_used + inc > self.boundaries['resource_budget']:
            print("⚠️ 资源预算即将超限，暂停训练")
            return False
        self.resource_used += inc
        return True

    def check_boundaries(self, state: Dict[str, Any]) -> bool:
        if state.get('layers', 0) > self.boundaries['max_layers']:
            print("⚠️ 网络层数超出安全边界")
            return False
        if state.get('neurons', 0) > self.boundaries['max_neurons']:
            print("⚠️ 神经元数超出安全边界")
            return False
        if state.get('generation', 0) > self.boundaries['max_generations']:
            print("⚠️ 进化代数超出安全边界")
            return False
        return True

    def stop(self):
        print("🚨 系统紧急停止！")
        self.emergency_stop = True

    def reset(self):
        self.resource_used = 0
        self.emergency_stop = False
        print("🔄 资源配额已重置")


class SelfEvolutionManager:
    def __init__(self, youwei_ai: 'YouweiAI'):
        self.ai = youwei_ai
        self.history: List[Dict[str, Any]] = []
        self.best_loss = float('inf')
        self.best_state: Optional[Dict[str, Any]] = None
        self.best_config: Optional[Dict[str, Any]] = None

    def evaluate(self, generator: DataGenerator, num: int = 10) -> Dict[str, Any]:
        total_loss = 0.0
        correct = 0
        total = 0
        for _ in range(num):
            x, y = generator()
            pred = self.ai.network.forward(x)
            loss = self.ai.mse_loss(pred, y)
            total_loss += loss
            for p, t in zip(pred, y):
                if abs(p - t) < 0.1:
                    correct += 1
                total += 1
        avg = total_loss / num
        acc = correct / total if total else 0.0
        result = {'average_loss': avg, 'accuracy': acc, 'tests': num,
                  'correct': correct, 'total': total}
        print(f"🔍 能力检测：准确率 {acc*100:.1f}%，平均误差 {avg:.6f}")
        return result

    def discover_defects(self, eval_result: Dict[str, Any]) -> List[str]:
        defects = []
        if eval_result['accuracy'] < 0.7:
            defects.append("准确率偏低")
        if eval_result['average_loss'] > 0.2:
            defects.append("预测误差过大")
        if len(self.ai.network.hidden_sizes) == 1 and eval_result['accuracy'] < 0.8:
            defects.append("网络结构可能过于简单")
        if self.ai.learning_rate > 0.1 and eval_result['accuracy'] < 0.7:
            defects.append("学习率可能过高")
        if defects:
            print(f"🔍 发现缺陷：{', '.join(defects)}")
        else:
            print("✅ 未发现明显缺陷")
        return defects

    def modify(self, defects: List[str]):
        if not defects:
            return
        print("🔧 尝试改进...")
        for d in defects:
            if "准确率偏低" in d or "预测误差过大" in d:
                self.ai.strategy_selector.current_strategy = 'exploration'
                p = self.ai.strategy_selector.get_params('exploration')
                self.ai.learning_rate = p['learning_rate']
                self.ai.mutation_rate = p['mutation_rate']
                print("🔄 切换探索策略")
            if "网络结构可能过于简单" in d:
                if self.ai.network.add_layer():
                    print(f"🧠 增加新层，结构: {[self.ai.network.input_size] + self.ai.network.hidden_sizes + [self.ai.network.output_size]}")
            if "学习率可能过高" in d:
                self.ai.learning_rate *= 0.8
                print(f"⚖️ 学习率调整为 {self.ai.learning_rate:.6f}")

    def test_current(self, generator: DataGenerator, num: int = 20) -> float:
        total = 0.0
        for _ in range(num):
            x, y = generator()
            pred = self.ai.network.forward(x)
            total += self.ai.mse_loss(pred, y)
        return total / num

    def update_best(self, generator: DataGenerator, threshold: float = 0.001):
        """若当前误差显著低于最佳，更新最佳状态；否则保留最佳"""
        loss = self.test_current(generator, 20)
        if self.best_loss == float('inf'):
            self.best_loss = loss
            self.best_state = self.ai.network.save_state()
            self.best_config = {
                'structure': [self.ai.network.input_size] + self.ai.network.hidden_sizes + [self.ai.network.output_size],
                'learning_rate': self.ai.learning_rate,
                'mutation_rate': self.ai.mutation_rate
            }
            print(f"💾 初始最佳误差 {loss:.6f}")
            return
        if loss < self.best_loss - threshold:
            print(f"🏆 取得进步！误差 {loss:.6f} (之前 {self.best_loss:.6f})")
            self.best_loss = loss
            self.best_state = self.ai.network.save_state()
            self.best_config = {
                'structure': [self.ai.network.input_size] + self.ai.network.hidden_sizes + [self.ai.network.output_size],
                'learning_rate': self.ai.learning_rate,
                'mutation_rate': self.ai.mutation_rate
            }
        elif loss > self.best_loss + threshold:
            print(f"⚠️ 性能恶化（{loss:.6f} > {self.best_loss:.6f}），回滚至最佳")
            self.ai.network.restore_state(self.best_state)
            if self.best_config:
                self.ai.learning_rate = self.best_config['learning_rate']
                self.ai.mutation_rate = self.best_config['mutation_rate']
        else:
            print(f"📊 性能持平：误差 {loss:.6f}")


class YouweiAI:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.network = SimpleNeuralNetwork(input_size, [6, 6], output_size)
        self.memory = MemoryBuffer(200)
        self.strategy_selector = AdaptiveStrategySelector()
        self.history: List[float] = []
        self.safety = SafetyGovernance()
        self.learning_rate = 0.1
        self.mutation_rate = 0.1
        self.mutation_strength = 0.05
        self.evolution_manager = SelfEvolutionManager(self)

    def mse_loss(self, pred: List[float], target: List[float]) -> float:
        return sum((p - t) ** 2 for p, t in zip(pred, target)) / len(pred)

    def compute_gradients(self, x: List[float], y: List[float]) -> Tuple[Weights3D, Biases2D]:
        """反向传播（MSE + 线性输出）"""
        # 前向记录
        activations = [x]
        zs = []
        cur = x
        for i, (w, b) in enumerate(zip(self.network.weights, self.network.biases)):
            z_layer, a_layer = [], []
            for j in range(len(w)):
                z = sum(cur[k] * w[j][k] for k in range(len(cur))) + b[j]
                z_layer.append(z)
                a_layer.append(self.network.relu(z) if i < len(self.network.weights)-1 else z)
            zs.append(z_layer)
            activations.append(a_layer)
            cur = a_layer

        # 反向
        deltas: List[List[float]] = []
        # 输出层 delta = 2*(pred - target)  (线性导数=1)
        output_error = [2 * (a - t) for a, t in zip(activations[-1], y)]
        deltas.append(output_error)

        for l in range(len(self.network.weights)-2, -1, -1):
            prev_delta = deltas[-1]
            layer_delta = []
            for i in range(len(activations[l+1])):
                error = sum(prev_delta[j] * self.network.weights[l+1][j][i] for j in range(len(prev_delta)))
                deriv = 1.0 if zs[l][i] > 0 else 0.0
                layer_delta.append(error * deriv)
            deltas.append(layer_delta)
        deltas.reverse()

        w_grad: Weights3D = []
        b_grad: Biases2D = []
        for l in range(len(self.network.weights)):
            wg = []
            bg = []
            for i in range(len(self.network.weights[l])):
                wg.append([deltas[l][i] * activations[l][j] for j in range(len(activations[l]))])
                bg.append(deltas[l][i])
            w_grad.append(wg)
            b_grad.append(bg)
        return w_grad, b_grad

    def train_step(self, x: List[float], y: List[float], use_mutation: bool = False) -> Optional[Dict[str, Any]]:
        if not self.safety.check_resource(1):
            return None
        pred = self.network.forward(x)
        loss = self.mse_loss(pred, y)
        w_grad, b_grad = self.compute_gradients(x, y)

        for l in range(len(self.network.weights)):
            for i in range(len(self.network.weights[l])):
                for j in range(len(self.network.weights[l][i])):
                    self.network.weights[l][i][j] -= self.learning_rate * w_grad[l][i][j]
            for i in range(len(self.network.biases[l])):
                self.network.biases[l][i] -= self.learning_rate * b_grad[l][i]

        if use_mutation:
            self.network.mutate(self.mutation_rate, self.mutation_strength)

        self.history.append(loss)
        exp = {'inputs': x, 'targets': y, 'predicted': pred,
               'loss': loss, 'timestamp': datetime.now().isoformat()}
        self.memory.add_experience(exp)
        return {'loss': loss, 'predicted': pred, 'actual': y}

    def replay(self, batch: int = 10):
        for exp in self.memory.sample_batch(batch):
            self.train_step(exp['inputs'], exp['targets'], False)

    def adjust_strategy(self):
        if len(self.history) < 10:
            return
        recent = self.history[-10:]
        trend = sum(recent[-5:])/5 - sum(recent[:5])/5
        strategy = self.strategy_selector.evaluate(trend)
        p = self.strategy_selector.get_params(strategy)
        self.learning_rate = p['learning_rate']
        self.mutation_rate = p['mutation_rate']
        if p['explore_new_arch'] and random.random() < 0.1:
            if random.random() < 0.5:
                if self.network.add_layer():
                    print(f"🔄 尝试增加层，新结构: {[self.network.input_size] + self.network.hidden_sizes + [self.network.output_size]}")
            else:
                if self.network.remove_layer():
                    print(f"🔄 尝试移除层，新结构: {[self.network.input_size] + self.network.hidden_sizes + [self.network.output_size]}")

    def self_evolution_cycle(self, generator: DataGenerator):
        """自进化：评估 -> 修改 -> 验证 -> 回滚/保留"""
        print("🔄 自进化循环开始...")
        # 修改前备份
        state_backup = self.network.save_state()
        param_backup = (self.learning_rate, self.mutation_rate, self.strategy_selector.current_strategy)
        loss_before = self.evolution_manager.test_current(generator, 20)

        eval_result = self.evolution_manager.evaluate(generator)
        defects = self.evolution_manager.discover_defects(eval_result)
        if defects:
            self.evolution_manager.modify(defects)
            loss_after = self.evolution_manager.test_current(generator, 20)
            if loss_after > loss_before:   # 修改后变差，立即恢复
                print("⚠️ 修改后误差上升，回滚至修改前状态")
                self.network.restore_state(state_backup)
                self.learning_rate, self.mutation_rate = param_backup[0], param_backup[1]
                self.strategy_selector.current_strategy = param_backup[2]
            else:
                print("✅ 修改有效，保留新结构")
                # 更新全局最佳
                self.evolution_manager.update_best(generator)
        else:
            # 无缺陷，仍尝试更新最佳
            self.evolution_manager.update_best(generator)
        print("✅ 自进化循环完成")

    def evolve(self, generator: DataGenerator, generations: int = 500, batch: int = 20,
               callback: Optional[Callable[[int, float, float, int], None]] = None) -> Dict[str, Any]:
        print("陆幼薇开始自我进化...")
        max_gen = min(generations, self.safety.boundaries['max_generations'])
        for gen in range(max_gen):
            if self.safety.emergency_stop:
                print("🛑 紧急停止")
                break
            self.adjust_strategy()
            total_loss = 0.0
            valid = 0
            for _ in range(batch):
                x, y = generator()
                use_mut = random.random() < self.mutation_rate
                res = self.train_step(x, y, use_mut)
                if res is not None:
                    total_loss += res['loss']
                    valid += 1
            if valid == 0:
                print("资源耗尽，训练结束")
                break
            avg_loss = total_loss / valid

            if gen % 100 == 0:
                print(f"第 {gen} 代: 误差 {avg_loss:.6f}, 学习率 {self.learning_rate:.6f}, "
                      f"资源 {self.safety.resource_used}/{self.safety.boundaries['resource_budget']}")
                self.replay(20)
                self.self_evolution_cycle(generator)
            if gen % 50 == 0 and gen > 0:
                self.replay(10)
            if callback:
                callback(gen, avg_loss, self.learning_rate, self.safety.resource_used)
                time.sleep(0.005)
        print("陆幼薇自我进化完成！")
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        if not self.history:
            return {"error": "无数据"}
        recent = self.history[-20:] if len(self.history) >= 20 else self.history
        avg_final = statistics.mean(recent)
        min_loss = min(self.history)
        trend = "进步" if self.history[0] > self.history[-1] else "待提升"
        stability = "稳定"
        if len(self.history) > 20:
            var = statistics.variance(self.history[-20:])
            if var > 0.05:
                stability = "波动"
            elif var > 0.01:
                stability = "中等"
        return {
            "总训练步数": len(self.history),
            "最终平均误差": avg_final,
            "最小误差": min_loss,
            "最终学习率": self.learning_rate,
            "记忆量": len(self.memory.buffer),
            "趋势": trend,
            "稳定性": stability,
            "最佳误差": self.evolution_manager.best_loss,
            "最佳结构": self.evolution_manager.best_config['structure'] if self.evolution_manager.best_config else "无",
            "资源用量": self.safety.resource_used,
            "状态": "运行中" if not self.safety.emergency_stop else "已停止"
        }


def create_generic_data() -> TrainingData:
    x = [random.random() for _ in range(4)]
    w = [random.uniform(0.5, 1.5) for _ in range(4)]
    y = [sum(xi * wi for xi, wi in zip(x, w)) / 4]
    return x, y


class YouweiGUI:
    def __init__(self):
        if not tk_available:
            return
        self.root = tk.Tk()
        self.root.title("陆幼薇 - 自我进化AI系统")
        self.root.geometry("800x600")
        self.progress_var = tk.DoubleVar()
        self.youwei: Optional[YouweiAI] = None
        self.running = False
        self.setup_gui()

    def setup_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ctrl = ttk.Frame(self.notebook)
        self.notebook.add(ctrl, text="控制面板")
        self._build_ctrl(ctrl)

        stat = ttk.Frame(self.notebook)
        self.notebook.add(stat, text="系统状态")
        self._build_stat(stat)

    def _build_ctrl(self, parent):
        f = ttk.LabelFrame(parent, text="参数", padding=10)
        f.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(f, text="代数:").grid(row=0, column=0, sticky=tk.W)
        self.gen_var = tk.StringVar(value="500")
        ttk.Entry(f, textvariable=self.gen_var, width=8).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(f, text="批次:").grid(row=0, column=2, sticky=tk.W, padx=10)
        self.batch_var = tk.StringVar(value="15")
        ttk.Entry(f, textvariable=self.batch_var, width=8).grid(row=0, column=3, sticky=tk.W)

        pf = ttk.Frame(parent, padding=10)
        pf.pack(fill=tk.X, padx=10)
        self.prog_label = ttk.Label(pf, text="进度: 0%")
        self.prog_label.pack(anchor=tk.W)
        self.bar = ttk.Progressbar(pf, variable=self.progress_var, maximum=100)
        self.bar.pack(fill=tk.X, pady=5)

        bf = ttk.Frame(parent, padding=10)
        bf.pack(fill=tk.X, padx=10)
        self.start_btn = ttk.Button(bf, text="开始进化", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(bf, text="停止", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.reset_btn = ttk.Button(bf, text="重置", command=self.reset)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        lf = ttk.LabelFrame(parent, text="日志", padding=10)
        lf.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log = scrolledtext.ScrolledText(lf, height=8)
        self.log.pack(fill=tk.BOTH, expand=True)

    def _build_stat(self, parent):
        f = ttk.LabelFrame(parent, text="实时状态", padding=10)
        f.pack(fill=tk.X, padx=10, pady=5)
        self.stats = {}
        items = [("代数:", "gen"), ("误差:", "loss"), ("学习率:", "lr"),
                 ("资源:", "res"), ("结构:", "net"), ("状态:", "sys")]
        for i, (txt, key) in enumerate(items):
            ttk.Label(f, text=txt).grid(row=i//2, column=(i%2)*2, sticky=tk.W, padx=5, pady=2)
            self.stats[key] = ttk.Label(f, text="-")
            self.stats[key].grid(row=i//2, column=(i%2)*2+1, sticky=tk.W, padx=5, pady=2)

        df = ttk.LabelFrame(parent, text="总结", padding=10)
        df.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.detail = scrolledtext.ScrolledText(df, height=10)
        self.detail.pack(fill=tk.BOTH, expand=True)

    def start(self):
        if self.running:
            return
        self.youwei = YouweiAI(4, 1)
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            gens = int(self.gen_var.get())
            batch = int(self.batch_var.get())
        except:
            gens, batch = 500, 15
        def cb(gen, loss, lr, res):
            self.root.after(0, self._update, gen, loss, lr, res)
        res = self.youwei.evolve(create_generic_data, gens, batch, cb)
        self.root.after(0, self._done, res)

    def _update(self, gen, loss, lr, res):
        total = int(self.gen_var.get())
        pct = min(100, (gen / total * 100) if total else 0)
        self.progress_var.set(pct)
        self.prog_label.config(text=f"进度: {pct:.1f}% (第 {gen} 代)")
        self.stats["gen"].config(text=str(gen))
        self.stats["loss"].config(text=f"{loss:.6f}")
        self.stats["lr"].config(text=f"{lr:.6f}")
        self.stats["res"].config(text=f"{res}/10000")
        if self.youwei:
            s = [self.youwei.input_size] + self.youwei.network.hidden_sizes + [self.youwei.output_size]
            self.stats["net"].config(text=str(s))
            self.stats["sys"].config(text="运行中" if not self.youwei.safety.emergency_stop else "已停止")
        self.log.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Gen {gen}: loss={loss:.6f}\n")
        self.log.see(tk.END)

    def _done(self, results):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log.insert(tk.END, "\n进化完成！\n")
        for k, v in results.items():
            self.log.insert(tk.END, f"  {k}: {v}\n")
        self.log.see(tk.END)
        self.detail.delete(1.0, tk.END)
        for k, v in results.items():
            self.detail.insert(tk.END, f"{k}: {v}\n")

    def stop(self):
        if self.youwei:
            self.youwei.safety.stop()
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def reset(self):
        if self.youwei:
            self.youwei.safety.reset()
        self.log.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] 系统已重置\n")
        self.log.see(tk.END)

    def run(self):
        if not tk_available:
            print("=" * 50)
            print("       陆幼薇 自我进化AI系统")
            print("=" * 50)
            ai = YouweiAI(4, 1)
            res = ai.evolve(create_generic_data, 500, 15)
            print("\n总结:")
            for k, v in res.items():
                print(f"  {k}: {v}")
            correct = total = 0
            for _ in range(10):
                x, y = create_generic_data()
                pred = ai.network.forward(x)
                for p, t in zip(pred, y):
                    if abs(p - t) < 0.1:
                        correct += 1
                    total += 1
            print(f"  测试准确率: {correct}/{total} ({correct/total*100:.1f}%)")
            print("\n感谢使用陆幼薇系统！")
        else:
            self.root.mainloop()


def main():
    gui = YouweiGUI()
    gui.run()


if __name__ == "__main__":
    main()