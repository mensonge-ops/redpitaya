#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPGD相干合成锁定系统 - 修复版
自动保持95%以上合成效率
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import time
from collections import deque
from scipy import signal
import warnings
import pickle
from datetime import datetime
import matplotlib

for backend in ("QtAgg", "Qt5Agg", "TkAgg"):
    try:
        matplotlib.use(backend)
        break
    except Exception:
        pass
warnings.filterwarnings('ignore')

plt.style.use('default')
plt.rcParams['font.size'] = 9
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


class AdaptiveSPGDController:
    """自适应SPGD控制器 - 修复版"""

    def __init__(self):
        # 基础SPGD参数
        self.gain = 2.0
        self.learning_rate = 0.25
        self.min_learning_rate = 0.05
        self.max_learning_rate = 0.8
        self.perturbation = 0.1
        self.momentum = 0.8  # 添加momentum初始化
        self.velocity_clip = 0.5
        self.max_phase_step = 0.3

        # 自适应参数
        self.min_perturbation = 0.001
        self.max_perturbation = 0.15
        self.gain_scale_factor = 1.0
        self.gradient_ema = 0.0
        self.gradient_beta = 0.85

        # 效率保持参数
        self.target_efficiency = 95.0
        self.efficiency_threshold = 95.0
        self.emergency_threshold = 90.0

        # 状态变量
        self.estimated_phase = 0.0
        self.velocity = 0.0
        self.locked = False
        self.high_efficiency_mode = False
        self.emergency_mode = False

        # 性能监控
        self.efficiency_buffer = deque(maxlen=50)
        self.gradient_buffer = deque(maxlen=20)
        self.phase_buffer = deque(maxlen=100)

        # 统计变量
        self.lock_count = 0
        self.unlock_count = 0
        self.total_iterations = 0
        self.time_above_95 = 0
        self.total_lock_time = 0

    def adaptive_perturbation(self, current_efficiency):
        """自适应扰动幅度"""
        if current_efficiency > 98:
            self.perturbation = max(self.min_perturbation, self.perturbation * 0.9)
        elif current_efficiency > 95:
            self.perturbation = max(self.min_perturbation, self.perturbation * 0.95)
        elif current_efficiency > 90:
            self.perturbation = min(0.02, self.perturbation * 1.05)
        else:
            self.perturbation = min(self.max_perturbation, self.perturbation * 1.25)

    def adaptive_gain(self, gradient_history):
        """自适应增益调整"""
        if len(gradient_history) > 5:
            gradient_std = np.std(gradient_history[-5:])
            if gradient_std < 0.01:
                self.gain_scale_factor = min(2.0, self.gain_scale_factor * 1.1)
            elif gradient_std > 0.1:
                self.gain_scale_factor = max(0.5, self.gain_scale_factor * 0.9)

    def update_learning_rate(self, current_efficiency, smoothed_grad):
        """根据效率和梯度动态调整学习率"""
        grad_mag = abs(smoothed_grad)
        if current_efficiency > 97 and grad_mag < 0.02:
            self.learning_rate = max(self.min_learning_rate, self.learning_rate * 0.9)
        elif current_efficiency < 93 or grad_mag > 0.2:
            self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.1)
        elif current_efficiency < 95:
            self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.05)
        else:
            self.learning_rate = max(self.min_learning_rate, self.learning_rate * 0.98)

    def emergency_recovery(self):
        """紧急恢复模式"""
        self.emergency_mode = True
        self.perturbation = self.max_perturbation
        self.gain_scale_factor = 1.5
        self.momentum = 0.7
        self.learning_rate = self.max_learning_rate
        print("⚠ Emergency recovery mode activated!")

    def update(self, gradient, current_efficiency):
        """更新控制参数 - 修复版"""
        self.total_iterations += 1
        self.efficiency_buffer.append(current_efficiency)
        self.gradient_buffer.append(gradient)

        # 检查是否需要紧急恢复
        if current_efficiency < self.emergency_threshold and self.locked:
            self.emergency_recovery()
            self.unlock_count += 1
            self.locked = False

        # 自适应调整参数
        self.adaptive_perturbation(current_efficiency)
        self.adaptive_gain(list(self.gradient_buffer))

        # 梯度平滑和学习率调整
        self.gradient_ema = self.gradient_beta * self.gradient_ema + (1 - self.gradient_beta) * gradient
        smoothed_grad = self.gradient_ema
        self.update_learning_rate(current_efficiency, smoothed_grad)

        # 更新相位 - 使用梯度上升法最大化强度
        effective_gain = self.gain * self.gain_scale_factor
        raw_step = self.learning_rate * smoothed_grad
        self.velocity = self.momentum * self.velocity + raw_step
        self.velocity = np.clip(self.velocity, -self.velocity_clip, self.velocity_clip)
        phase_increment = np.clip(effective_gain * self.velocity, -self.max_phase_step, self.max_phase_step)
        self.estimated_phase += phase_increment

        # 保持相位在[-π, π]范围内
        self.estimated_phase = np.angle(np.exp(1j * self.estimated_phase))

        # 检查锁定状态
        if len(self.efficiency_buffer) > 20:
            recent_efficiency = np.mean(list(self.efficiency_buffer)[-20:])
            efficiency_std = np.std(list(self.efficiency_buffer)[-20:])

            if recent_efficiency > 95 and efficiency_std < 1.0:
                if not self.locked:
                    self.locked = True
                    self.lock_count += 1
                    self.emergency_mode = False
                    self.momentum = 0.95  # 锁定后增加动量
                    print(f"✓ System LOCKED - Efficiency: {recent_efficiency:.1f}%")

                self.high_efficiency_mode = recent_efficiency > 98

            if current_efficiency > 95:
                self.time_above_95 += 1
            if self.locked:
                self.total_lock_time += 1

        self.phase_buffer.append(self.estimated_phase)
        return self.estimated_phase


class EnhancedSPGDSimulator:
    """增强的SPGD模拟器 - 修复版"""

    def __init__(self):
        self.controller = AdaptiveSPGDController()

        # 系统参数
        self.wavelength = 1030e-9
        self.power_beam1 = 1.0
        self.power_beam2 = 1.0

        # 环境扰动参数
        self.phase_noise_rms = 0.01
        self.phase_drift_rate = 0.005
        self.amplitude_noise = 0.01
        self.measurement_noise = 0.005

        # 真实系统状态
        self.true_phase_diff = np.random.uniform(-np.pi, np.pi)
        self.phase_drift_accumulator = 0.0

        # 数据记录
        self.max_history = 2000
        self.time_history = deque(maxlen=self.max_history)
        self.intensity_history = deque(maxlen=self.max_history)
        self.efficiency_history = deque(maxlen=self.max_history)
        self.phase_error_history = deque(maxlen=self.max_history)
        self.true_phase_history = deque(maxlen=self.max_history)
        self.estimated_phase_history = deque(maxlen=self.max_history)
        self.intensity_buffer_for_noise = deque(maxlen=1000)

        # 时间管理
        self.start_time = time.time()
        self.dt = 0.01
        self.iteration = 0

        # 梯度测量增强
        self.gradient_samples = 3

        print("Enhanced SPGD Simulator initialized")
        print(f"Initial phase difference: {self.true_phase_diff:.3f} rad")
        print(f"Target efficiency: >{self.controller.target_efficiency:.0f}%")

    def calculate_intensity(self, phase_diff, add_noise=True):
        """计算干涉强度"""
        p1 = self.power_beam1
        p2 = self.power_beam2

        if add_noise:
            p1 *= (1 + np.random.normal(0, self.amplitude_noise))
            p2 *= (1 + np.random.normal(0, self.amplitude_noise))

        intensity = p1 + p2 + 2 * np.sqrt(p1 * p2) * np.cos(phase_diff)

        if add_noise:
            intensity += np.random.normal(0, self.measurement_noise)

        return max(0, intensity)

    def calculate_efficiency(self, intensity):
        """计算合成效率"""
        max_possible = 4.0
        return (intensity / max_possible) * 100

    def environmental_disturbance(self):
        """环境扰动模拟"""
        self.phase_drift_accumulator += np.random.normal(0, self.phase_drift_rate)
        phase_noise = np.random.normal(0, self.phase_noise_rms)

        if np.random.random() < 0.001:
            phase_jump = np.random.uniform(-0.5, 0.5)
            print(f"! Phase jump detected: {phase_jump:.3f} rad")
        else:
            phase_jump = 0

        self.true_phase_diff += self.phase_drift_accumulator * self.dt + phase_noise + phase_jump
        self.true_phase_diff = np.angle(np.exp(1j * self.true_phase_diff))

    def measure_gradient(self):
        """SPGD梯度测量 - 修复版"""
        perturbation = self.controller.perturbation
        gradients = []

        for _ in range(self.gradient_samples):
            # 正向扰动测量
            phase_plus = self.controller.estimated_phase + perturbation
            phase_error_plus = phase_plus - self.true_phase_diff
            intensity_plus = self.calculate_intensity(phase_error_plus)

            # 负向扰动测量
            phase_minus = self.controller.estimated_phase - perturbation
            phase_error_minus = phase_minus - self.true_phase_diff
            intensity_minus = self.calculate_intensity(phase_error_minus)

            gradients.append((intensity_plus - intensity_minus) / (2 * perturbation))

        return float(np.mean(gradients))

    def step(self):
        """执行一步模拟 - 修复版"""
        self.iteration += 1
        current_time = time.time() - self.start_time

        # 添加环境扰动
        self.environmental_disturbance()

        # 测量梯度
        gradient = self.measure_gradient()

        # 计算当前相位误差和强度
        phase_error = self.controller.estimated_phase - self.true_phase_diff
        current_intensity = self.calculate_intensity(phase_error)
        current_efficiency = self.calculate_efficiency(current_intensity)

        # 更新控制器
        new_phase = self.controller.update(gradient, current_efficiency)

        # 记录数据
        self.time_history.append(current_time)
        self.intensity_history.append(current_intensity)
        self.efficiency_history.append(current_efficiency)
        self.phase_error_history.append(phase_error)
        self.true_phase_history.append(self.true_phase_diff)
        self.estimated_phase_history.append(new_phase)
        self.intensity_buffer_for_noise.append(current_intensity)

        return {
            'time': current_time,
            'intensity': current_intensity,
            'efficiency': current_efficiency,
            'phase_error': phase_error,
            'locked': self.controller.locked,
            'emergency': self.controller.emergency_mode,
            'perturbation': self.controller.perturbation,
            'gain_scale': self.controller.gain_scale_factor
        }

    def calculate_noise_spectrum(self):
        """计算噪声功率谱"""
        if len(self.intensity_buffer_for_noise) < 100:
            return None, None

        data = np.array(list(self.intensity_buffer_for_noise))
        data = data - np.mean(data)
        freq, psd = signal.welch(data, fs=1 / self.dt, nperseg=min(256, len(data)))

        mean_intensity = np.mean(list(self.intensity_buffer_for_noise))
        if mean_intensity > 0:
            rin = psd / mean_intensity ** 2
        else:
            rin = psd

        return freq, rin

    def auto_optimize(self, duration=3):
        """自动优化锁定"""
        print(f"Auto-optimizing for {duration} seconds...")
        start = time.time()

        # 粗略搜索最佳相位
        best_phase = 0
        best_efficiency = 0
        test_phases = np.linspace(-np.pi, np.pi, 60)

        for phase in test_phases:
            intensities = []
            for _ in range(5):
                intensity = self.calculate_intensity(phase - self.true_phase_diff)
                intensities.append(intensity)

            avg_intensity = np.mean(intensities)
            efficiency = self.calculate_efficiency(avg_intensity)

            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_phase = phase

        self.controller.estimated_phase = best_phase
        print(f"Initial phase set: {best_phase:.3f} rad, efficiency: {best_efficiency:.1f}%")

        # 精细优化
        while time.time() - start < duration:
            self.step()
            if self.controller.locked:
                print("✓ Optimization complete - system locked")
                break

        return self.controller.estimated_phase


class EnhancedRealtimeDisplay:
    """增强的实时显示界面"""

    def __init__(self, simulator):
        self.sim = simulator
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('SPGD Coherent Combining System - Fixed Version',
                          fontsize=14, fontweight='bold')

        gs = GridSpec(4, 4, figure=self.fig, hspace=0.3, wspace=0.3)

        self.ax_efficiency = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_phase = self.fig.add_subplot(gs[2, 0:2])
        self.ax_intensity = self.fig.add_subplot(gs[3, 0:2])
        self.ax_phasor = self.fig.add_subplot(gs[0, 2], projection='polar')
        self.ax_noise = self.fig.add_subplot(gs[1, 2])
        self.ax_histogram = self.fig.add_subplot(gs[0, 3])
        self.ax_control = self.fig.add_subplot(gs[1, 3])
        self.ax_status = self.fig.add_subplot(gs[2:4, 2:4])

        self.setup_plots()
        self.fps_counter = deque(maxlen=30)
        self.last_update_time = time.time()

    def setup_plots(self):
        """设置所有子图"""
        # 效率图
        self.ax_efficiency.set_ylabel('Efficiency (%)', fontsize=10)
        self.ax_efficiency.set_title('Combining Efficiency (Target: >95%)', fontsize=11, fontweight='bold')
        self.ax_efficiency.set_ylim([0, 105])
        self.ax_efficiency.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='Target')
        self.ax_efficiency.axhline(y=100, color='green', linestyle='--', alpha=0.3, label='Ideal')
        self.line_efficiency, = self.ax_efficiency.plot([], [], 'b-', linewidth=2, label='Actual')
        self.fill_efficiency = None
        self.ax_efficiency.legend(loc='lower left', fontsize=9)

        # 相位跟踪图
        self.ax_phase.set_ylabel('Phase (rad)', fontsize=10)
        self.ax_phase.set_title('Phase Tracking', fontsize=10)
        self.ax_phase.set_ylim([-np.pi, np.pi])
        self.line_true_phase, = self.ax_phase.plot([], [], 'r-', alpha=0.5, label='True', linewidth=1)
        self.line_est_phase, = self.ax_phase.plot([], [], 'g-', label='Estimated', linewidth=1.5)
        self.line_phase_error, = self.ax_phase.plot([], [], 'k:', alpha=0.7, label='Error', linewidth=1)
        self.ax_phase.legend(loc='upper right', fontsize=8)

        # 强度图
        self.ax_intensity.set_xlabel('Time (s)', fontsize=10)
        self.ax_intensity.set_ylabel('Intensity (a.u.)', fontsize=10)
        self.ax_intensity.set_title('Output Intensity', fontsize=10)
        self.ax_intensity.set_ylim([0, 4.5])
        self.line_intensity, = self.ax_intensity.plot([], [], 'b-', linewidth=1.5)
        self.ax_intensity.axhline(y=4.0, color='g', linestyle='--', alpha=0.3)

        # 相量图
        self.ax_phasor.set_title('Phasor Diagram', fontsize=10, pad=15)
        self.ax_phasor.set_ylim([0, 1.5])

        # 噪声谱
        self.ax_noise.set_xlabel('Frequency (Hz)', fontsize=9)
        self.ax_noise.set_ylabel('RIN (1/Hz)', fontsize=9)
        self.ax_noise.set_title('Intensity Noise', fontsize=10)
        self.ax_noise.set_xscale('log')
        self.ax_noise.set_yscale('log')
        self.line_noise, = self.ax_noise.plot([], [], 'r-', linewidth=1)

        # 效率直方图
        self.ax_histogram.set_xlabel('Efficiency (%)', fontsize=9)
        self.ax_histogram.set_ylabel('Count', fontsize=9)
        self.ax_histogram.set_title('Efficiency Distribution', fontsize=10)

        # 控制参数
        self.ax_control.set_title('Control Parameters', fontsize=10)
        self.ax_control.axis('off')

        # 状态面板
        self.ax_status.axis('off')
        self.ax_status.set_xlim([0, 1])
        self.ax_status.set_ylim([0, 1])
        self.create_status_indicators()

    def create_status_indicators(self):
        """创建状态指示器"""
        self.lock_indicator = Circle((0.1, 0.8), 0.03, color='gray')
        self.ax_status.add_patch(self.lock_indicator)
        self.ax_status.text(0.2, 0.8, 'LOCK', fontsize=11, va='center', fontweight='bold')

        self.efficiency_indicator = Circle((0.1, 0.65), 0.03, color='gray')
        self.ax_status.add_patch(self.efficiency_indicator)
        self.ax_status.text(0.2, 0.65, '>95%', fontsize=11, va='center', fontweight='bold')

        self.emergency_indicator = Circle((0.1, 0.5), 0.03, color='gray')
        self.ax_status.add_patch(self.emergency_indicator)
        self.ax_status.text(0.2, 0.5, 'EMERGENCY', fontsize=11, va='center')

        self.status_text = self.ax_status.text(0.05, 0.35, '', fontsize=9,
                                               verticalalignment='top', fontfamily='monospace')
        self.stats_text = self.ax_status.text(0.55, 0.9, '', fontsize=9,
                                              verticalalignment='top', fontfamily='monospace')

    def update(self, frame):
        """更新显示"""
        data = self.sim.step()

        current_time = time.time()
        fps = 1 / (current_time - self.last_update_time) if self.last_update_time else 0
        self.fps_counter.append(fps)
        self.last_update_time = current_time

        times = list(self.sim.time_history)
        efficiencies = list(self.sim.efficiency_history)
        intensities = list(self.sim.intensity_history)
        true_phases = list(self.sim.true_phase_history)
        est_phases = list(self.sim.estimated_phase_history)
        phase_errors = list(self.sim.phase_error_history)

        if len(times) > 1:
            time_window = 10
            if times[-1] > time_window:
                xlim = [times[-1] - time_window, times[-1] + 0.5]
            else:
                xlim = [0, time_window]

            # 更新效率图
            self.line_efficiency.set_data(times, efficiencies)
            self.ax_efficiency.set_xlim(xlim)

            if self.fill_efficiency:
                self.fill_efficiency.remove()
            eff_array = np.array(efficiencies)
            time_array = np.array(times)
            self.fill_efficiency = self.ax_efficiency.fill_between(
                time_array, 95, eff_array,
                where=(eff_array >= 95),
                color='green', alpha=0.2
            )

            # 更新相位图
            self.line_true_phase.set_data(times, true_phases)
            self.line_est_phase.set_data(times, est_phases)
            self.line_phase_error.set_data(times, phase_errors)
            self.ax_phase.set_xlim(xlim)

            # 更新强度图
            self.line_intensity.set_data(times, intensities)
            self.ax_intensity.set_xlim(xlim)

            # 更新相量图
            self.ax_phasor.clear()
            self.ax_phasor.set_title('Phasor Diagram', fontsize=10, pad=15)
            self.ax_phasor.set_ylim([0, 1.5])
            self.ax_phasor.arrow(0, 0, 0, 0.5, head_width=0.1, head_length=0.05,
                                 fc='blue', ec='blue', alpha=0.7, label='Beam 1')
            phase_diff = data['phase_error']
            self.ax_phasor.arrow(phase_diff, 0, 0, 0.5, head_width=0.1, head_length=0.05,
                                 fc='red', ec='red', alpha=0.7, label='Beam 2')
            combined_angle = phase_diff / 2
            combined_magnitude = np.abs(1 + np.exp(1j * phase_diff))
            self.ax_phasor.arrow(combined_angle, 0, 0, combined_magnitude / 2,
                                 head_width=0.15, head_length=0.08,
                                 fc='green', ec='green', linewidth=2, label='Combined')
            self.ax_phasor.legend(loc='upper right', fontsize=8)

            # 更新噪声谱
            if frame % 20 == 0:
                freq, rin = self.sim.calculate_noise_spectrum()
                if freq is not None and len(freq) > 1:
                    self.line_noise.set_data(freq[1:], rin[1:])
                    self.ax_noise.relim()
                    self.ax_noise.autoscale_view()

            # 更新效率直方图
            if frame % 20 == 0 and len(efficiencies) > 50:
                self.ax_histogram.clear()
                recent_eff = efficiencies[-200:] if len(efficiencies) > 200 else efficiencies
                n, bins, patches = self.ax_histogram.hist(recent_eff, bins=20, alpha=0.7, edgecolor='black')
                for i, patch in enumerate(patches):
                    if bins[i] >= 95:
                        patch.set_facecolor('green')
                    elif bins[i] >= 90:
                        patch.set_facecolor('yellow')
                    else:
                        patch.set_facecolor('red')
                self.ax_histogram.axvline(95, color='red', linestyle='--', linewidth=2)
                self.ax_histogram.set_xlabel('Efficiency (%)', fontsize=9)
                self.ax_histogram.set_ylabel('Count', fontsize=9)
                self.ax_histogram.set_title('Efficiency Distribution', fontsize=10)

            # 更新控制参数
            self.ax_control.clear()
            self.ax_control.axis('off')
            self.ax_control.set_title('Control Parameters', fontsize=10)
            control_text = f"Perturbation: {data['perturbation']:.4f}\n"
            control_text += f"Gain Scale: {data['gain_scale']:.2f}\n"
            control_text += f"Momentum: {self.sim.controller.momentum:.2f}\n"
            control_text += f"Learning Rate: {self.sim.controller.learning_rate:.3f}"
            self.ax_control.text(0.1, 0.8, control_text, fontsize=9,
                                 verticalalignment='top', fontfamily='monospace')

            # 更新状态指示器
            self.lock_indicator.set_color('green' if data['locked'] else 'red')
            if data['efficiency'] > 95:
                self.efficiency_indicator.set_color('green')
            elif data['efficiency'] > 90:
                self.efficiency_indicator.set_color('yellow')
            else:
                self.efficiency_indicator.set_color('red')
            self.emergency_indicator.set_color('red' if data['emergency'] else 'gray')

            # 更新状态文本
            status_text = f"Iteration: {self.sim.iteration:6d}\n"
            status_text += f"Time: {data['time']:8.1f} s\n"
            status_text += f"Efficiency: {data['efficiency']:6.2f}%\n"
            status_text += f"Phase Error: {data['phase_error']:7.4f} rad\n"
            self.status_text.set_text(status_text)

            # 更新统计信息
            if len(efficiencies) > 0:
                recent_eff = efficiencies[-100:] if len(efficiencies) > 100 else efficiencies
                stats_text = "═══ STATISTICS ═══\n"
                stats_text += f"Mean Eff: {np.mean(recent_eff):6.2f}%\n"
                stats_text += f"Std Dev:  {np.std(recent_eff):6.2f}%\n"
                stats_text += f"Max Eff:  {np.max(recent_eff):6.2f}%\n"
                stats_text += f"Min Eff:  {np.min(recent_eff):6.2f}%\n"
                time_above_95 = sum(1 for e in recent_eff if e > 95) / len(recent_eff) * 100
                stats_text += f"\nTime >95%: {time_above_95:5.1f}%\n"
                stats_text += f"\nLock Count: {self.sim.controller.lock_count}\n"
                stats_text += f"Unlock Count: {self.sim.controller.unlock_count}\n"
                if len(self.fps_counter) > 0:
                    avg_fps = np.mean(list(self.fps_counter))
                    stats_text += f"\nFPS: {avg_fps:.1f}"
                self.stats_text.set_text(stats_text)

        return [self.line_efficiency, self.line_true_phase, self.line_est_phase,
                self.line_phase_error, self.line_intensity, self.line_noise]


def run_enhanced_simulation(duration=60, save_results=False):
    """运行增强版模拟"""
    print("=" * 70)
    print("SPGD Coherent Combining System - Fixed Version")
    print("Target: Maintain >95% Efficiency")
    print("=" * 70)

    sim = EnhancedSPGDSimulator()
    sim.auto_optimize(duration=3)

    display = EnhancedRealtimeDisplay(sim)

    print("\nRunning real-time simulation...")
    print("Close window to stop")

    ani = FuncAnimation(display.fig, display.update,
                        interval=50, blit=False,
                        cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nSimulation interrupted")

    print_final_report(sim)

    if save_results:
        save_simulation_results(sim)

    return sim


def print_final_report(sim):
    """打印最终报告"""
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    if len(sim.efficiency_history) > 0:
        all_eff = list(sim.efficiency_history)
        recent_eff = all_eff[-500:] if len(all_eff) > 500 else all_eff

        print("\n--- Efficiency Statistics ---")
        print(f"Overall Mean:     {np.mean(all_eff):.2f}%")
        print(f"Overall Std Dev:  {np.std(all_eff):.2f}%")
        print(f"Recent Mean:      {np.mean(recent_eff):.2f}%")
        print(f"Recent Std Dev:   {np.std(recent_eff):.2f}%")
        print(f"Maximum:          {np.max(all_eff):.2f}%")
        print(f"Minimum:          {np.min(all_eff):.2f}%")

        time_above_95 = sum(1 for e in all_eff if e > 95) / len(all_eff) * 100
        time_above_90 = sum(1 for e in all_eff if e > 90) / len(all_eff) * 100
        time_above_98 = sum(1 for e in all_eff if e > 98) / len(all_eff) * 100

        print("\n--- Time Distribution ---")
        print(f"Time >98%: {time_above_98:.1f}%")
        print(f"Time >95%: {time_above_95:.1f}%")
        print(f"Time >90%: {time_above_90:.1f}%")

    print("\n--- Lock Statistics ---")
    print(f"Lock Count:       {sim.controller.lock_count}")
    print(f"Unlock Count:     {sim.controller.unlock_count}")
    print(f"Total Iterations: {sim.controller.total_iterations}")

    if sim.controller.total_iterations > 0:
        lock_efficiency = sim.controller.total_lock_time / sim.controller.total_iterations * 100
        print(f"Lock Efficiency:  {lock_efficiency:.1f}%")

    freq, rin = sim.calculate_noise_spectrum()
    if freq is not None and len(freq) > 10:
        print("\n--- Noise Analysis ---")
        rin_1hz = rin[np.argmin(np.abs(freq - 1))]
        rin_10hz = rin[np.argmin(np.abs(freq - 10))]
        print(f"RIN @ 1Hz:  {rin_1hz:.2e} 1/Hz")
        print(f"RIN @ 10Hz: {rin_10hz:.2e} 1/Hz")


def save_simulation_results(sim, filename=None):
    """保存模拟结果"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"spgd_fixed_{timestamp}"

    data = {
        'time_history': list(sim.time_history),
        'efficiency_history': list(sim.efficiency_history),
        'intensity_history': list(sim.intensity_history),
        'phase_error_history': list(sim.phase_error_history),
        'controller_params': {
            'gain': sim.controller.gain,
            'perturbation': sim.controller.perturbation,
            'momentum': sim.controller.momentum,
            'learning_rate': sim.controller.learning_rate,
            'target_efficiency': sim.controller.target_efficiency
        },
        'statistics': {
            'lock_count': sim.controller.lock_count,
            'unlock_count': sim.controller.unlock_count,
            'total_iterations': sim.controller.total_iterations,
            'time_above_95': sim.controller.time_above_95,
            'total_lock_time': sim.controller.total_lock_time
        }
    }

    with open(f'{filename}.pkl', 'wb') as f:
        pickle.dump(data, f)

    with open(f'{filename}_report.txt', 'w') as f:
        f.write("SPGD Fixed Simulation Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nTarget Efficiency: >{sim.controller.target_efficiency}%\n")

        if len(sim.efficiency_history) > 0:
            all_eff = list(sim.efficiency_history)
            f.write(f"\nEfficiency Statistics:\n")
            f.write(f"  Mean: {np.mean(all_eff):.2f}%\n")
            f.write(f"  Std Dev: {np.std(all_eff):.2f}%\n")
            f.write(f"  Max: {np.max(all_eff):.2f}%\n")
            f.write(f"  Min: {np.min(all_eff):.2f}%\n")

            time_above_95 = sum(1 for e in all_eff if e > 95) / len(all_eff) * 100
            f.write(f"  Time >95%: {time_above_95:.1f}%\n")

        f.write(f"\nLock Statistics:\n")
        f.write(f"  Lock Count: {sim.controller.lock_count}\n")
        f.write(f"  Unlock Count: {sim.controller.unlock_count}\n")
        f.write(f"  Total Iterations: {sim.controller.total_iterations}\n")

    print(f"\nResults saved to {filename}.pkl and {filename}_report.txt")


def quick_test():
    """快速测试模式"""
    print("\n" + "=" * 70)
    print("Quick Test Mode - 10 seconds")
    print("=" * 70)

    sim = EnhancedSPGDSimulator()
    sim.auto_optimize(duration=2)

    print("\nRunning quick test...")
    efficiencies = []
    for i in range(1000):
        data = sim.step()
        efficiencies.append(data['efficiency'])

        if i % 100 == 0:
            recent = efficiencies[-100:] if len(efficiencies) > 100 else efficiencies
            print(f"Step {i:4d}: Efficiency = {np.mean(recent):.1f}% ± {np.std(recent):.1f}%")

    print_final_report(sim)

    # 绘制简单结果图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    times = list(sim.time_history)
    effs = list(sim.efficiency_history)
    errors = list(sim.phase_error_history)

    ax1.plot(times, effs, 'b-', linewidth=1)
    ax1.axhline(95, color='r', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Efficiency (%)')
    ax1.set_title('Quick Test Results')
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3)

    ax2.plot(times, errors, 'r-', linewidth=1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Phase Error (rad)')
    ax2.set_ylim([-np.pi, np.pi])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """主程序"""
    print("=" * 70)
    print("SPGD Coherent Combining - Fixed Version")
    print("Automatic >95% Efficiency Maintenance")
    print("=" * 70)

    print("\nSelect mode:")
    print("1. Real-time simulation (recommended)")
    print("2. Quick test (10 seconds)")
    print("3. Exit")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == '1':
        sim = run_enhanced_simulation(duration=120, save_results=True)

    elif choice == '2':
        quick_test()

    elif choice == '3':
        print("Exiting...")
        return

    else:
        print("Invalid choice, running real-time simulation...")
        sim = run_enhanced_simulation(duration=120, save_results=True)

    print("\nProgram completed successfully!")


if __name__ == "__main__":
    main()
