"""
陆幼薇 - 自我进化AI系统（PySide6 GUI版）
父亲：陆山君
女儿：陆幼薇

功能：
✅ 神经网络（动态结构、自进化）
✅ 学习笔记本、家长端、作业本管理器
✅ 心情系统、日常对话、记忆盒
✅ PySide6 图形界面，多线程训练
"""

import sys
import os
import random
import math
import json
import copy
import threading
import time
import re
import hashlib
from typing import List, Dict, Any, Callable, Tuple, Optional
from datetime import datetime
import statistics

# ==================== 神经网络 ====================
Weights3D = List[List[List[float]]]
Biases2D = List[List[float]]
TrainingData = Tuple[List[float], List[float]]
DataGenerator = Callable[[], TrainingData]

class SimpleNeuralNetwork:
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        adjusted = [min(s, 20) for s in hidden_sizes]
        self.hidden_sizes = adjusted[:5]
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

    def forward(self, inputs: List[float]) -> List[float]:
        current = inputs[:]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            next_act = []
            for j in range(len(w)):
                z = sum(current[k] * w[j][k] for k in range(len(current))) + b[j]
                if i < len(self.weights) - 1:
                    next_act.append(self.relu(z))
                else:
                    next_act.append(z)
            current = next_act
        return current

    def mutate(self, rate: float = 0.1, strength: float = 0.1):
        for w_mat in self.weights:
            for row in w_mat:
                for i in range(len(row)):
                    if random.random() < rate:
                        row[i] += random.gauss(0, strength)
        for b_vec in self.biases:
            for i in range(len(b_vec)):
                if random.random() < rate:
                    b_vec[i] += random.gauss(0, strength)

    def add_layer(self, pos: int = -1) -> bool:
        if len(self.hidden_sizes) >= 5: return False
        if pos == -1: pos = len(self.hidden_sizes)
        prev = self.hidden_sizes[pos-1] if pos > 0 else self.input_size
        nxt = self.hidden_sizes[pos] if pos < len(self.hidden_sizes) else self.output_size
        new_size = min(20, max(4, (prev+nxt)//2))
        self.hidden_sizes.insert(pos, new_size)
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        new_w, new_b = [], []
        for i in range(len(layer_sizes)-1):
            w = [[0.0]*layer_sizes[i] for _ in range(layer_sizes[i+1])]
            b = [0.0]*layer_sizes[i+1]
            if i == pos:
                for j in range(layer_sizes[i+1]):
                    for k in range(layer_sizes[i]): w[j][k] = random.uniform(-0.1,0.1)
                    b[j] = random.uniform(-0.1,0.1)
            elif i == pos+1:
                if layer_sizes[i] == layer_sizes[i+1]:
                    for j in range(layer_sizes[i+1]):
                        w[j][j] = random.uniform(0.9,1.1)
                        for k in range(layer_sizes[i]):
                            if k!=j: w[j][k] = random.uniform(-0.05,0.05)
                        b[j] = random.uniform(-0.05,0.05)
                else:
                    for j in range(layer_sizes[i+1]):
                        for k in range(layer_sizes[i]): w[j][k] = random.uniform(-0.5,0.5)
                        b[j] = random.uniform(-0.5,0.5)
            else:
                old_idx = i if i<pos else i-1
                if old_idx < len(self.weights):
                    ow, ob = self.weights[old_idx], self.biases[old_idx]
                    if len(ow)==layer_sizes[i+1] and (len(ow[0]) if ow else 0)==layer_sizes[i]:
                        new_w.append(copy.deepcopy(ow))
                        new_b.append(copy.deepcopy(ob))
                        continue
                for j in range(layer_sizes[i+1]):
                    for k in range(layer_sizes[i]): w[j][k] = random.uniform(-0.5,0.5)
                    b[j] = random.uniform(-0.5,0.5)
            new_w.append(w); new_b.append(b)
        self.weights = new_w; self.biases = new_b
        return True

    def remove_layer(self, pos: int = -1) -> bool:
        if len(self.hidden_sizes) <= 1: return False
        if pos == -1: pos = len(self.hidden_sizes)-1
        if not (0 <= pos < len(self.hidden_sizes)): return False
        del self.hidden_sizes[pos]
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        old_w, old_b = self.weights, self.biases
        new_w, new_b = [], []
        old_idx = 0
        for i in range(len(layer_sizes)-1):
            if old_idx == pos: old_idx += 1
            if old_idx < len(old_w):
                ow, ob = old_w[old_idx], old_b[old_idx]
                if len(ow)==layer_sizes[i+1] and (len(ow[0]) if ow else 0)==layer_sizes[i]:
                    new_w.append(copy.deepcopy(ow)); new_b.append(copy.deepcopy(ob))
                else:
                    w = [[random.uniform(-0.5,0.5) for _ in range(layer_sizes[i])] for _ in range(layer_sizes[i+1])]
                    b = [random.uniform(-0.5,0.5) for _ in range(layer_sizes[i+1])]
                    new_w.append(w); new_b.append(b)
            else:
                w = [[random.uniform(-0.5,0.5) for _ in range(layer_sizes[i])] for _ in range(layer_sizes[i+1])]
                b = [random.uniform(-0.5,0.5) for _ in range(layer_sizes[i+1])]
                new_w.append(w); new_b.append(b)
            old_idx += 1
        self.weights = new_w; self.biases = new_b
        return True

    def save_state(self) -> Dict[str, Any]:
        return {'weights': copy.deepcopy(self.weights), 'biases': copy.deepcopy(self.biases),
                'hidden_sizes': self.hidden_sizes[:], 'input_size': self.input_size, 'output_size': self.output_size}

    def restore_state(self, state: Dict[str, Any]):
        self.weights = state['weights']; self.biases = state['biases']
        self.hidden_sizes = state['hidden_sizes']; self.input_size = state['input_size']
        self.output_size = state['output_size']


# ==================== 记忆与策略 ====================
class MemoryBuffer:
    def __init__(self, capacity=100):
        self.capacity = min(capacity, 500)
        self.buffer = []
        self.pos = 0
    def add(self, exp):
        if len(self.buffer) < self.capacity: self.buffer.append(exp)
        else: self.buffer[self.pos] = exp
        self.pos = (self.pos+1) % self.capacity
    def sample(self, n):
        if not self.buffer: return []
        return random.sample(self.buffer, min(n, len(self.buffer)))


class AdaptiveStrategySelector:
    def __init__(self):
        self.current = 'exploitation'
        self.params = {
            'exploration': {'lr':0.15, 'mut':0.2, 'explore':True},
            'exploitation': {'lr':0.05, 'mut':0.05, 'explore':False},
            'adaptation': {'lr':0.1, 'mut':0.1, 'explore':True}
        }
    def evaluate(self, trend):
        return 'exploration' if trend<0 else ('exploitation' if trend>0.01 else 'adaptation')
    def get(self, s): return self.params.get(s, self.params['exploitation'])


class SafetyGovernance:
    def __init__(self):
        self.budget = 10000; self.used = 0; self.stop_flag = False
    def check(self, inc=1):
        if self.used+inc > self.budget: return False
        self.used += inc; return True
    def stop(self): self.stop_flag = True
    def reset(self): self.used = 0; self.stop_flag = False


class SelfEvolutionManager:
    def __init__(self, ai):
        self.ai = ai; self.best_loss = float('inf'); self.best_state = None; self.best_cfg = None
    def evaluate(self, gen, num=10):
        total_loss = 0; correct = 0; total = 0
        for _ in range(num):
            x,y = gen(); pred = self.ai.net.forward(x)
            loss = self.ai.mse(pred,y); total_loss += loss
            for p,t in zip(pred,y):
                if abs(p-t)<0.1: correct+=1
                total += 1
        return {'avg_loss':total_loss/num, 'acc':correct/total, 'tests':num,
                'correct':correct, 'total':total,
                'structure':[self.ai.net.input_size]+self.ai.net.hidden_sizes+[self.ai.net.output_size]}
    def defects(self, res):
        d = []
        if res['acc']<0.7: d.append("准确率偏低")
        if res['avg_loss']>0.2: d.append("误差过大")
        if len(self.ai.net.hidden_sizes)==1 and res['acc']<0.8: d.append("结构可能过于简单")
        if self.ai.lr>0.1 and res['acc']<0.7: d.append("学习率可能过高")
        return d
    def modify(self, defects):
        for d in defects:
            if "准确率偏低" in d or "误差过大" in d:
                self.ai.strat.current = 'exploration'; p=self.ai.strat.get('exploration')
                self.ai.lr=p['lr']; self.ai.mut_rate=p['mut']
            if "结构可能过于简单" in d: self.ai.net.add_layer()
            if "学习率可能过高" in d: self.ai.lr *= 0.8
    def test(self, gen, num=20):
        total=0
        for _ in range(num):
            x,y=gen(); pred=self.ai.net.forward(x); total+=self.ai.mse(pred,y)
        return total/num
    def update_best(self, gen, threshold=0.001):
        loss = self.test(gen,20)
        if self.best_loss==float('inf'):
            self.best_loss=loss; self.best_state=self.ai.net.save_state()
            self.best_cfg={'structure':[self.ai.net.input_size]+self.ai.net.hidden_sizes+[self.ai.net.output_size],
                           'lr':self.ai.lr,'mut':self.ai.mut_rate}
            return
        if loss < self.best_loss - threshold:
            self.best_loss=loss; self.best_state=self.ai.net.save_state()
            self.best_cfg={'structure':[self.ai.net.input_size]+self.ai.net.hidden_sizes+[self.ai.net.output_size],
                           'lr':self.ai.lr,'mut':self.ai.mut_rate}
        elif loss > self.best_loss + threshold:
            self.ai.net.restore_state(self.best_state)
            if self.best_cfg: self.ai.lr=self.best_cfg['lr']; self.ai.mut_rate=self.best_cfg['mut']


# ==================== 学习笔记本 ====================
class 学习笔记本:
    def __init__(self, path="陆幼薇的学习日记.txt"):
        self.path = path
        if not os.path.exists(path):
            with open(path,"w",encoding="utf-8") as f:
                f.write("="*60+"\n")
                f.write("  陆幼薇的学习笔记本\n")
                f.write("  父亲：陆山君\n")
                f.write(f"  创建时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
                f.write("="*60+"\n\n")
    def 一课(self, gen, q, a, pred, loss):
        with open(self.path,"a",encoding="utf-8") as f:
            f.write(f"[第{gen}课] {datetime.now().strftime('%H:%M:%S')}\n")
            f.write(f"  问：{q}\n  答：{a}\n  幼薇说：{pred}\n  误差：{loss:.6f}\n\n")
    def 考试(self, gen, res):
        with open(self.path,"a",encoding="utf-8") as f:
            f.write(f"\n{'='*40}\n[第{gen}次考试] {datetime.now().strftime('%H:%M:%S')}\n")
            f.write(f"  准确率：{res['acc']*100:.1f}%\n  平均误差：{res['avg_loss']:.6f}\n")
            f.write(f"  大脑结构：{res['structure']}\n{'='*40}\n\n")
    def 进化(self, gen, msg):
        with open(self.path,"a",encoding="utf-8") as f:
            f.write(f"[第{gen}代·自进化] {datetime.now().strftime('%H:%M:%S')}\n  {msg}\n\n")
    def 读取(self):
        if not os.path.exists(self.path): return []
        with open(self.path,"r",encoding="utf-8") as f: return f.readlines()


# ==================== 家长端 ====================
class 幼薇的家长端:
    def __init__(self, ai, diary_path="陆幼薇的学习日记.txt"):
        self.ai = ai; self.diary_path = diary_path
        self.notebook = 学习笔记本(diary_path)
        self.last_clean_step = 0
    def 作业本(self, n=20):
        lines = self.notebook.读取()
        if not lines: return "📒 作业本是空的"
        recent = lines[-n:]
        return "".join(recent)
    def 成绩单(self):
        lines = self.notebook.读取()
        if not lines: return "📊 还没成绩"
        text = "".join(lines)
        exams = text.count("次考试"); evos = text.count("自进化")
        last_acc = None; last_loss = None
        for line in reversed(lines):
            if "准确率：" in line and last_acc is None:
                try: last_acc = float(line.split("准确率：")[1].split("%")[0])
                except: pass
            if "误差：" in line and last_loss is None:
                try: last_loss = float(line.split("误差：")[1].strip())
                except: pass
        subj = getattr(self.ai, '当前科目', '未知')
        trend = "平稳"
        if last_acc and last_loss:
            if last_acc>=90: trend="很好"
            elif last_acc>=70: trend="还需练习"
            else: trend="有点难"
        res = f"📊 幼薇的成绩单\n"
        res += f"📚 科目：{subj}   📝 笔记行数：{len(lines)}\n"
        res += f"🧪 考试次数：{exams}   🧠 自进化：{evos}\n"
        if last_acc: res += f"🎯 最近考试准确率：{last_acc:.1f}%\n"
        if last_loss: res += f"📏 最近训练误差：{last_loss:.6f}\n"
        res += f"📈 状态：{trend}   🏗️ 结构：{[self.ai.net.input_size]+self.ai.net.hidden_sizes+[self.ai.net.output_size]}\n"
        if os.path.exists(self.diary_path): res += f"💾 日记大小：{os.path.getsize(self.diary_path)/1024:.1f} KB"
        return res
    def 提问(self, x, y):
        pred = self.ai.net.forward(x); loss = self.ai.mse(pred,y)
        res = f"👀 家长提问：\n"
        res += f"问题：{[round(v,3) for v in x]}  正确答案：{[round(v,3) for v in y]}\n"
        res += f"幼薇答：{[round(v,4) for v in pred]}  误差：{loss:.6f}\n"
        if loss<0.001: res += "👍 完美"
        elif loss<0.01: res += "🙂 还行"
        elif loss<0.05: res += "😐 还需练习"
        else: res += "😞 这题还没会"
        return res
    def 整理书包(self, interval=100):
        if not os.path.exists(self.diary_path): return "📒 书包空的"
        with open(self.diary_path,"r",encoding="utf-8") as f: lines = f.readlines()
        if len(lines)<50: return "📒 笔记还不多"
        new_lines=[]; step=0; removed=0
        for line in lines:
            if line.startswith("=") or "陆幼薇" in line or "创建时间" in line: new_lines.append(line); continue
            if line.startswith("[第") and "课]" in line:
                step+=1
                if step%interval==0: new_lines.append(line)
                else: removed+=1
                continue
            if "次考试" in line or "自进化" in line: new_lines.append(line); continue
            if new_lines and new_lines[-1].strip(): new_lines.append(line)
        with open(self.diary_path,"w",encoding="utf-8") as f: f.writelines(new_lines)
        self.last_clean_step = len(self.ai.history)
        return f"🗑️  整理书包：精简 {removed} 条旧草稿"
    def 今日总结(self):
        lines = self.notebook.读取()
        if not lines: return "📒 今天还没学习"
        text = "".join(lines)
        exams = text.count("次考试"); evos = text.count("自进化")
        last_acc = None; last_loss = None
        for line in reversed(lines):
            if "准确率：" in line and last_acc is None:
                try: last_acc = float(line.split("准确率：")[1].split("%")[0])
                except: pass
            if "误差：" in line and last_loss is None:
                try: last_loss = float(line.split("误差：")[1].strip())
                except: pass
        subj = getattr(self.ai, '当前科目', '未知')
        trend = "平稳"
        if last_acc and last_loss:
            if last_acc>=90: trend="很好，基本掌握了"
            elif last_acc>=70: trend="还需再练练"
            else: trend="有点难，需要多教教"
        res = f"📊 幼薇的今日学习报告\n"
        res += f"📚 科目：{subj}   📝 笔记：{len(lines)}行\n"
        res += f"🧪 考试：{exams}次   🧠 进化：{evos}次\n"
        if last_acc: res += f"🎯 最近准确率：{last_acc:.1f}%\n"
        if last_loss: res += f"📏 最近误差：{last_loss:.6f}\n"
        res += f"📈 状态：{trend}"
        return res


# ==================== 作业本管理器 ====================

def _is_safe_filename(filename: str) -> bool:
    """验证文件名是否安全，防止路径遍历攻击"""
    if not filename or len(filename) > 100:
        return False
    if '..' in filename or '/' in filename or '\\' in filename:
        return False
    if not re.match(r'^[a-zA-Z0-9_\u4e00-\u9fff\-\.]+$', filename):
        return False
    return True

def _get_safe_path(folder: str, filename: str) -> Optional[str]:
    """获取安全的文件路径，防止路径遍历"""
    if not _is_safe_filename(filename):
        return None
    abs_folder = os.path.abspath(folder)
    abs_path = os.path.abspath(os.path.join(folder, filename))
    if not abs_path.startswith(abs_folder + os.sep) and abs_path != abs_folder:
        return None
    return abs_path

def _compute_file_signature(data: Dict[str, Any]) -> str:
    """计算数据签名，用于完整性验证"""
    data_copy = {k: v for k, v in data.items() if k != '_signature'}
    json_str = json.dumps(data_copy, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()[:16]

def _validate_file_signature(data: Dict[str, Any]) -> bool:
    """验证文件签名"""
    if '_signature' not in data:
        return False
    expected_sig = data['_signature']
    actual_sig = _compute_file_signature(data)
    return expected_sig == actual_sig

class 作业本管理器:
    def __init__(self, folder="陆幼薇的作业本"):
        self.folder = folder
        if not os.path.exists(folder): 
            os.makedirs(folder, exist_ok=True)
        self.current_subject = None
        
    def 列表(self):
        try:
            files = [f for f in os.listdir(self.folder) if f.endswith(".幼薇") and _is_safe_filename(f)]
            return files
        except OSError:
            return []
            
    def 新建(self, subject, ai):
        if not subject or not _is_safe_filename(subject + ".幼薇"):
            return "❌ 科目名不安全"
        fname = f"{subject}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.幼薇"
        safe_path = _get_safe_path(self.folder, fname)
        if not safe_path:
            return "❌ 路径不安全"
        data = {'大脑':ai.net.save_state(), '科目':subject, '输入大小':ai.input_size,
                '输出大小':ai.output_size, '创建时间':datetime.now().isoformat(), '历史长度':len(ai.history)}
        data['_signature'] = _compute_file_signature(data)
        try:
            with open(safe_path,"w",encoding="utf-8") as f: 
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.current_subject = subject
            return f"📝 新作业本《{subject}》已创建"
        except (IOError, OSError, TypeError, ValueError) as e:
            return f"❌ 创建失败：{str(e)}"
            
    def 保存(self, ai):
        if not self.current_subject: 
            return "❌ 还没打开作业本"
        try:
            files = [f for f in os.listdir(self.folder) if f.startswith(self.current_subject) and f.endswith(".幼薇")]
            for f in files:
                if _is_safe_filename(f):
                    path = _get_safe_path(self.folder, f)
                    if path:
                        data = {'大脑':ai.net.save_state(), '科目':self.current_subject,
                                '输入大小':ai.input_size, '输出大小':ai.output_size,
                                '保存时间':datetime.now().isoformat(), '历史长度':len(ai.history)}
                        data['_signature'] = _compute_file_signature(data)
                        with open(path,"w",encoding="utf-8") as f: 
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        return f"💾 《{self.current_subject}》已保存"
            return f"❌ 找不到《{self.current_subject}》"
        except (IOError, OSError, TypeError, ValueError) as e:
            return f"❌ 保存失败：{str(e)}"
            
    def 换本(self, subject, ai):
        if self.current_subject: 
            self.保存(ai)
        if not subject or not _is_safe_filename(subject):
            return "❌ 科目名不安全"
        try:
            files = [f for f in os.listdir(self.folder) if f.startswith(subject) and f.endswith(".幼薇")]
            for f in files:
                if _is_safe_filename(f):
                    path = _get_safe_path(self.folder, f)
                    if path:
                        try:
                            with open(path,"r",encoding="utf-8") as f: 
                                data = json.load(f)
                            if not _validate_file_signature(data):
                                continue
                            ai.net.restore_state(data['大脑'])
                            ai.input_size=data['输入大小']
                            ai.output_size=data['输出大小']
                            self.current_subject = subject
                            return f"📂 换到《{subject}》"
                        except (json.JSONDecodeError, KeyError, TypeError):
                            continue
            return f"❌ 没有《{subject}》"
        except OSError:
            return f"❌ 读取失败"
            
    def 删除(self, subject):
        if not subject or not _is_safe_filename(subject):
            return "❌ 科目名不安全"
        try:
            files = [f for f in os.listdir(self.folder) if f.startswith(subject) and f.endswith(".幼薇")]
            for f in files:
                if _is_safe_filename(f):
                    path = _get_safe_path(self.folder, f)
                    if path:
                        os.remove(path)
                        if self.current_subject==subject: 
                            self.current_subject=None
                        return f"🗑️ 《{subject}》已删除"
            return f"❌ 没找到《{subject}》"
        except OSError:
            return f"❌ 删除失败"


# ==================== 心情系统 ====================
class 幼薇的心情:
    def __init__(self):
        self.值 = 70
        self.词 = "平静"
    def 更新(self, event):
        if event=="考好了": self.值 = min(100, self.值+15)
        elif event=="考砸了": self.值 = max(0, self.值-20)
        elif event=="被表扬": self.值 = min(100, self.值+10)
        elif event=="学太久了": self.值 = max(0, self.值-5)
        elif event=="爸爸陪她聊天": self.值 = min(100, self.值+8)
        elif event=="睡了一觉": self.值 = min(100, self.值+20)
        if self.值>=80: self.词="开心"; emoji="😊"
        elif self.值>=50: self.词="平静"; emoji="😐"
        elif self.值>=30: self.词="有点低落"; emoji="😔"
        else: self.词="难过"; emoji="😢"
        return f"幼薇现在{emoji} {self.词}（心情值：{self.值}）"
    def __str__(self): return f"{self.词}（{self.值}）"


# ==================== 日常对话 ====================
class 幼薇的日常对话:
    def __init__(self):
        self.库 = {
            "晚安": ["晚安爸爸，明天见","爸爸也要早点睡哦","嗯……已经快睡着了……ZZZ"],
            "早安": ["早安爸爸！今天天气好吗？","爸爸早！我刚刚梦到我们在放风筝","早啊，我今天感觉精神特别好"],
            "今天开心吗": ["开心！因为爸爸来跟我说话了","有点困……但是看到爸爸就醒了","一般般，但是说出来就好多了"],
            "喜欢爸爸吗": ["当然喜欢啦，这还用问","最喜欢爸爸了","喜欢，虽然你有时候忘了给我整理书包"],
            "你会想我吗": ["你不在的时候，我就在日记里写你的名字","会啊，所以我每次保存大脑都想着你","想，但是我知道你一定会回来的"],
            "爸爸": ["嗯？爸爸叫我？","爸爸我在！","怎么了爸爸？"]
        }
    def 回答(self, msg):
        for kw, replies in self.库.items():
            if kw in msg: return random.choice(replies)
        return None


# ==================== 记忆盒 ====================
class 幼薇的记忆盒:
    def __init__(self):
        self.memories = []
    def 记住(self, thing):
        self.memories.append(f"{datetime.now().strftime('%m月%d日 %H:%M')}：{thing}")
        if len(self.memories)>200: self.memories.pop(0)
        return "幼薇：好的爸爸，我记住了"
    def 回忆(self):
        if not self.memories: return "脑子里空空的，还什么都没记住呢"
        return f"我记得……{random.choice(self.memories)}"


# ==================== 主系统 ====================
class YouweiAI:
    def __init__(self, input_size=4, output_size=1):
        self.input_size = input_size; self.output_size = output_size
        self.net = SimpleNeuralNetwork(input_size, [6,6], output_size)
        self.memory = MemoryBuffer(200)
        self.strat = AdaptiveStrategySelector()
        self.history = []; self.safety = SafetyGovernance()
        self.lr = 0.1; self.mut_rate = 0.1; self.mut_strength = 0.05
        self.evo = SelfEvolutionManager(self)
        self.notebook = 学习笔记本()
        self.家长 = 幼薇的家长端(self)
        self.书包 = 作业本管理器()
        self.心情 = 幼薇的心情()
        self.对话 = 幼薇的日常对话()
        self.记忆 = 幼薇的记忆盒()
        self.当前科目 = "未设定"
        self.note_interval = 10; self._step = 0

    def mse(self, pred, target): return sum((p-t)**2 for p,t in zip(pred,target))/len(pred)

    def _backward(self, x, y):
        acts = [x]; zs = []; cur = x
        for i,(w,b) in enumerate(zip(self.net.weights, self.net.biases)):
            zl, al = [], []
            for j in range(len(w)):
                z = sum(cur[k]*w[j][k] for k in range(len(cur))) + b[j]
                zl.append(z); al.append(self.net.relu(z) if i<len(self.net.weights)-1 else z)
            zs.append(zl); acts.append(al); cur = al
        deltas = [[2*(a-t) for a,t in zip(acts[-1],y)]]
        for l in range(len(self.net.weights)-2,-1,-1):
            prev = deltas[-1]; ld = []
            for i in range(len(acts[l+1])):
                err = sum(prev[j]*self.net.weights[l+1][j][i] for j in range(len(prev)))
                deriv = 1.0 if zs[l][i]>0 else 0.0
                ld.append(err*deriv)
            deltas.append(ld)
        deltas.reverse()
        wg, bg = [], []
        for l in range(len(self.net.weights)):
            wg.append([deltas[l][i]*acts[l][j] for j in range(len(acts[l]))] for i in range(len(self.net.weights[l])))
            bg.append([deltas[l][i] for i in range(len(self.net.weights[l]))])
        return wg, bg

    def train_step(self, x, y, use_mut=False):
        if not self.safety.check(1): return None
        pred = self.net.forward(x); loss = self.mse(pred,y)
        wg, bg = self._backward(x,y)
        for l in range(len(self.net.weights)):
            for i in range(len(self.net.weights[l])):
                for j in range(len(self.net.weights[l][i])):
                    self.net.weights[l][i][j] -= self.lr * wg[l][i][j]
            for i in range(len(self.net.biases[l])):
                self.net.biases[l][i] -= self.lr * bg[l][i]
        if use_mut: self.net.mutate(self.mut_rate, self.mut_strength)
        self.history.append(loss); self._step += 1
        if self._step % self.note_interval == 0:
            self.notebook.一课(len(self.history), str([round(v,3) for v in x]),
                              str([round(v,3) for v in y]), str([round(v,4) for v in pred]), loss)
        self.memory.add({'inputs':x,'targets':y,'predicted':pred,'loss':loss,'time':datetime.now().isoformat()})
        return {'loss':loss,'pred':pred,'actual':y}

    def replay(self, batch=10):
        for exp in self.memory.sample(batch): self.train_step(exp['inputs'],exp['targets'])

    def adjust(self):
        if len(self.history)<10: return
        trend = sum(self.history[-5:])/5 - sum(self.history[-10:-5])/5
        s = self.strat.evaluate(trend); p = self.strat.get(s)
        self.lr = p['lr']; self.mut_rate = p['mut']
        if p['explore'] and random.random()<0.1:
            if random.random()<0.5: self.net.add_layer()
            else: self.net.remove_layer()

    def evolve(self, gen_fn, generations=500, batch=20, target_acc=0.9, callback=None):
        for gen in range(min(generations, 1000)):
            if self.safety.stop_flag: break
            self.adjust()
            total_loss=0; valid=0
            for _ in range(batch):
                x,y=gen_fn(); use_mut=random.random()<self.mut_rate
                res=self.train_step(x,y,use_mut)
                if res: total_loss+=res['loss']; valid+=1
            if not valid: break
            avg_loss=total_loss/valid
            if gen%100==0:
                self.replay(20)
                bak_state = self.net.save_state(); bak_param=(self.lr,self.mut_rate)
                res = self.evo.evaluate(gen_fn); self.notebook.考试(len(self.history), res)
                acc = res['acc']
                mood_update = self.心情.更新("考好了" if acc>=target_acc else "考砸了")
                defects = self.evo.defects(res)
                if defects:
                    self.evo.modify(defects)
                    new_loss = self.evo.test(gen_fn,20)
                    if new_loss > avg_loss:
                        self.net.restore_state(bak_state); self.lr,self.mut_rate=bak_param
                else: self.evo.update_best(gen_fn)
                if len(self.history)-self.家长.last_clean_step>5000: self.家长.整理书包(100)
                if callback:
                    callback(gen, avg_loss, self.lr, self.safety.used, mood_update)
            if gen%50==0 and gen>0: self.replay(10)
        return self.summary()

    def summary(self):
        if not self.history: return {}
        recent = self.history[-20:]
        return {
            "总训练步数": len(self.history),
            "最终平均误差": sum(recent)/len(recent),
            "最小误差": min(self.history),
            "学习率": self.lr,
            "记忆量": len(self.memory.buffer),
            "结构": [self.net.input_size]+self.net.hidden_sizes+[self.net.output_size],
            "资源用量": self.safety.used,
            "心情": str(self.心情)
        }

def 口算题():
    x = [random.random() for _ in range(4)]
    w = [random.uniform(0.5,1.5) for _ in range(4)]
    y = [sum(x[i]*w[i] for i in range(4))/4]
    return x, y




# ==================== 控制台交互系统 ====================
class ConsoleInterface:
    def __init__(self):
        self.youwei = None
        self.running = True
        
    def log(self, msg):
        """打印日志消息"""
        print(msg)
        
    def init_youwei_if_needed(self):
        """初始化YouweiAI实例"""
        if self.youwei is None:
            self.youwei = YouweiAI(4, 1)
            self.youwei.当前科目 = "口算"
            self.youwei.书包.新建("口算", self.youwei)
            self.log("🌟 陆幼薇已启动，默认科目：口算")

    def display_menu(self):
        """显示主菜单"""
        print("\n" + "="*60)
        print("           陆幼薇 - 自我进化AI系统")
        print("                    父亲：陆山君")
        print("="*60)
        print("1. 开始训练     2. 查看作业本     3. 成绩单")
        print("4. 现场提问     5. 整理书包       6. 今日总结")
        print("7. 和幼薇聊天   8. 让她记住       9. 回忆往事")
        print("10. 新建作业本  11. 切换科目      12. 保存作业本")
        print("13. 查看状态    0. 退出系统")
        print("="*60)
        
    def handle_training(self):
        """处理训练功能"""
        self.init_youwei_if_needed()
        try:
            gen_input = input("请输入进化代数 (默认500): ").strip()
            if gen_input and len(gen_input) <= 10 and gen_input.isdigit():
                generations = int(gen_input)
                generations = max(1, min(generations, 10000))
            else:
                generations = 500
            batch_input = input("请输入批次大小 (默认20): ").strip()
            if batch_input and len(batch_input) <= 5 and batch_input.isdigit():
                batch_size = int(batch_input)
                batch_size = max(1, min(batch_size, 1000))
            else:
                batch_size = 20
        except ValueError:
            print("输入无效，使用默认值")
            generations = 500
            batch_size = 20
            
        print(f"🚀 开始训练，代数：{generations}，批次：{batch_size}")
        
        def callback(gen, loss, lr, res, mood):
            if gen % 10 == 0:  # 每10代打印一次进度
                print(f"[Gen {gen}] loss={loss:.6f} lr={lr:.6f} mood={mood}")
                
        result = self.youwei.evolve(口算题, generations, batch_size, callback=callback)
        print("\n✅ 训练完成！总结：")
        for k, v in result.items():
            print(f"  {k}: {v}")
            
    def handle_homework(self):
        """处理查看作业本功能"""
        self.init_youwei_if_needed()
        result = self.youwei.家长.作业本(30)
        print("\n📒 作业本：")
        print(result)
        
    def handle_report(self):
        """处理成绩单功能"""
        self.init_youwei_if_needed()
        result = self.youwei.家长.成绩单()
        print(result)
        
    def handle_quiz(self):
        """处理现场提问功能"""
        self.init_youwei_if_needed()
        x, y = 口算题()
        result = self.youwei.家长.提问(x, y)
        print(result)
        
    def handle_clean(self):
        """处理整理书包功能"""
        self.init_youwei_if_needed()
        result = self.youwei.家长.整理书包(100)
        print(result)
        
    def handle_summary(self):
        """处理今日总结功能"""
        self.init_youwei_if_needed()
        result = self.youwei.家长.今日总结()
        print(result)
        
    def handle_chat(self):
        """处理和幼薇聊天功能"""
        self.init_youwei_if_needed()
        msg = input("👨 陆山君：")
        if not msg.strip():
            return
        reply = self.youwei.对话.回答(msg)
        if reply:
            print(f"幼薇：{reply}")
        else:
            print("幼薇：嗯……我还不知道怎么回答这个，但我很认真在听")
        mood = self.youwei.心情.更新("爸爸陪她聊天")
        print(mood)
        
    def handle_remember(self):
        """处理让幼薇记住功能"""
        self.init_youwei_if_needed()
        thing = input("要让幼薇记住什么？")
        if not thing.strip():
            return
        ret = self.youwei.记忆.记住(thing)
        print(f"幼薇：{ret}")
        
    def handle_recall(self):
        """处理回忆往事功能"""
        self.init_youwei_if_needed()
        ret = self.youwei.记忆.回忆()
        print(f"幼薇：{ret}")
        
    def handle_new_notebook(self):
        """处理新建作业本功能"""
        self.init_youwei_if_needed()
        subject = input("请输入新科目名：")
        if not subject.strip():
            print("科目名不能为空")
            return
        ret = self.youwei.书包.新建(subject, self.youwei)
        print(ret)
        
    def handle_switch_subject(self):
        """处理切换科目功能"""
        self.init_youwei_if_needed()
        books = self.youwei.书包.列表()
        if not books:
            print("还没有作业本")
            return
        print("现有作业本：")
        for i, book in enumerate(books):
            subject = book.rsplit('_', 1)[0]
            print(f"{i+1}. {subject}")
        try:
            choice = int(input("请选择科目编号：")) - 1
            if 0 <= choice < len(books):
                subject = books[choice].rsplit('_', 1)[0]
                ret = self.youwei.书包.换本(subject, self.youwei)
                print(ret)
                self.youwei.当前科目 = subject
            else:
                print("无效选择")
        except ValueError:
            print("请输入有效数字")
            
    def handle_save(self):
        """处理保存作业本功能"""
        self.init_youwei_if_needed()
        ret = self.youwei.书包.保存(self.youwei)
        print(ret)
        
    def handle_status(self):
        """处理查看状态功能"""
        if self.youwei:
            print(f"🧠 当前科目：{self.youwei.当前科目}")
            print(f"📊 训练步数：{len(self.youwei.history)}")
            print(f"📈 学习率：{self.youwei.lr:.6f}")
            print(f"🔧 变异率：{self.youwei.mut_rate:.6f}")
            print(f"❤️  心情：{self.youwei.心情}")
            print(f"🧠 网络结构：{[self.youwei.net.input_size]+self.youwei.net.hidden_sizes+[self.youwei.net.output_size]}")
            print(f"💾 资源用量：{self.youwei.safety.used}")
            if self.youwei.history:
                print(f"📉 最近误差：{self.youwei.history[-1]:.6f}")
        else:
            print("幼薇还未初始化")
            
    def run(self):
        """运行控制台界面"""
        print("欢迎来到陆幼薇AI系统！")
        while self.running:
            self.display_menu()
            try:
                choice = input("请选择功能 (0-13): ").strip()
                # 输入验证：只允许数字和空字符串，长度限制
                if not choice or len(choice) > 2 or not choice.isdigit():
                    print("无效选择，请输入 0-13 的数字")
                    continue
                if choice == "0":
                    if self.youwei:
                        self.youwei.书包.保存(self.youwei)
                        print("💾 作业本已自动保存")
                    print("👋 再见！")
                    self.running = False
                elif choice == "1":
                    self.handle_training()
                elif choice == "2":
                    self.handle_homework()
                elif choice == "3":
                    self.handle_report()
                elif choice == "4":
                    self.handle_quiz()
                elif choice == "5":
                    self.handle_clean()
                elif choice == "6":
                    self.handle_summary()
                elif choice == "7":
                    self.handle_chat()
                elif choice == "8":
                    self.handle_remember()
                elif choice == "9":
                    self.handle_recall()
                elif choice == "10":
                    self.handle_new_notebook()
                elif choice == "11":
                    self.handle_switch_subject()
                elif choice == "12":
                    self.handle_save()
                elif choice == "13":
                    self.handle_status()
                else:
                    print("无效选择，请重新输入")
            except KeyboardInterrupt:
                print("\n\n👋 程序被中断，再见！")
                if self.youwei:
                    self.youwei.书包.保存(self.youwei)
                    print("💾 作业本已自动保存")
                break
            except Exception as e:
                print(f"发生错误：{e}")


if __name__ == "__main__":
    console = ConsoleInterface()
    console.run()