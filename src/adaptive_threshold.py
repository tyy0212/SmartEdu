# -*- coding: utf-8 -*-
"""
自适应阈值管理器
基于用户反馈动态调整语义相似度阈值，支持个性化阈值配置。
"""

import numpy as np
from datetime import datetime
from typing import List, Optional, Deque
from collections import deque
from ..config.settings import settings
from .utils.logger import get_logger

logger = get_logger(__name__)


class AdaptiveThresholdManager:
    """自适应阈值管理器，基于滑动窗口平均更新阈值"""

    def __init__(
        self,
        initial_threshold: Optional[float] = None,
        learning_rate: Optional[float] = None,
        window_size: Optional[int] = None
    ):
        """
        初始化自适应阈值管理器

        Args:
            initial_threshold: 初始相似度阈值，如果为None则从配置读取
            learning_rate: 学习率，控制阈值更新幅度
            window_size: 滑动窗口大小，用于计算平均阈值
        """
        self.initial_threshold = initial_threshold or settings.initial_similarity_threshold
        self.current_threshold = self.initial_threshold
        self.learning_rate = learning_rate or settings.adaptive_learning_rate
        self.window_size = window_size or settings.adaptive_window_size

        # 反馈历史：存储(反馈值, 时间戳)元组
        self.feedback_history: Deque[float] = deque(maxlen=self.window_size)

        # 阈值历史：存储阈值调整记录
        self.threshold_history: List[float] = [self.current_threshold]

        logger.info(
            f"初始化自适应阈值管理器: 初始阈值={self.current_threshold:.3f}, "
            f"学习率={self.learning_rate}, 窗口大小={self.window_size}"
        )

    def update_threshold(self, feedback: float) -> float:
        """
        基于反馈更新相似度阈值

        Args:
            feedback: 反馈值 (-1, 0, 1)
                -1: 误判（相似度太高，产生了错误关联）
                0: 无反馈/中性
                1: 正确检测（相似度阈值合适，产生了有用关联）

        Returns:
            更新后的阈值
        """
        # 验证反馈值范围
        if feedback not in [-1, 0, 1]:
            logger.warning(f"无效反馈值: {feedback}，应使用-1, 0, 1")
            return self.current_threshold

        # 记录反馈
        self.feedback_history.append(feedback)

        # 如果自适应阈值未启用，返回当前阈值
        if not settings.adaptive_threshold_enabled:
            logger.debug("自适应阈值功能已禁用，保持当前阈值")
            return self.current_threshold

        # 计算窗口内的平均反馈
        if not self.feedback_history:
            avg_feedback = 0.0
        else:
            avg_feedback = sum(self.feedback_history) / len(self.feedback_history)

        # 基于平均反馈调整阈值
        adjustment = self._calculate_adjustment(avg_feedback)
        new_threshold = self.current_threshold + adjustment

        # 限制阈值范围 (0.1 到 0.9)
        new_threshold = max(0.1, min(0.9, new_threshold))

        # 记录调整
        old_threshold = self.current_threshold
        self.current_threshold = new_threshold
        self.threshold_history.append(new_threshold)

        logger.info(
            f"阈值更新: {old_threshold:.3f} -> {new_threshold:.3f} "
            f"(反馈={feedback}, 平均反馈={avg_feedback:.3f}, 调整={adjustment:.3f})"
        )

        return new_threshold

    def _calculate_adjustment(self, avg_feedback: float) -> float:
        """
        计算阈值调整量

        Args:
            avg_feedback: 窗口内的平均反馈值

        Returns:
            阈值调整量
        """
        # 调整逻辑：
        # - 如果平均反馈为正（>0），说明阈值合适或偏低，可适当降低阈值以捕获更多相关节点
        # - 如果平均反馈为负（<0），说明阈值太高导致误判，应提高阈值减少误判
        # - 如果平均反馈为0，保持阈值稳定

        # 使用sigmoid形状的调整函数，使调整更平滑
        # 当avg_feedback接近±1时，调整幅度接近±learning_rate
        # 当avg_feedback接近0时，调整幅度接近0

        # 简单的线性调整
        adjustment = avg_feedback * self.learning_rate

        # 可选：添加动量项，基于最近调整的方向
        if len(self.threshold_history) > 1:
            recent_trend = self.threshold_history[-1] - self.threshold_history[-2]
            momentum = recent_trend * 0.1  # 动量系数
            adjustment += momentum

        return adjustment

    def get_current_threshold(self) -> float:
        """获取当前阈值"""
        return self.current_threshold

    def get_feedback_stats(self) -> dict:
        """获取反馈统计信息"""
        if not self.feedback_history:
            return {
                "total_feedbacks": 0,
                "avg_feedback": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0
            }

        total = len(self.feedback_history)
        positive = sum(1 for f in self.feedback_history if f > 0)
        negative = sum(1 for f in self.feedback_history if f < 0)
        neutral = total - positive - negative

        return {
            "total_feedbacks": total,
            "avg_feedback": sum(self.feedback_history) / total,
            "positive_ratio": positive / total if total > 0 else 0.0,
            "negative_ratio": negative / total if total > 0 else 0.0,
            "neutral_ratio": neutral / total if total > 0 else 0.0,
            "window_size": self.window_size,
            "current_threshold": self.current_threshold
        }

    def reset(self, new_initial_threshold: Optional[float] = None):
        """重置管理器到初始状态"""
        if new_initial_threshold is not None:
            self.initial_threshold = new_initial_threshold

        self.current_threshold = self.initial_threshold
        self.feedback_history.clear()
        self.threshold_history = [self.current_threshold]

        logger.info(f"重置自适应阈值管理器，新初始阈值: {self.current_threshold:.3f}")

    def save_state(self) -> dict:
        """保存管理器状态（用于持久化）"""
        return {
            "initial_threshold": self.initial_threshold,
            "current_threshold": self.current_threshold,
            "feedback_history": list(self.feedback_history),
            "threshold_history": self.threshold_history,
            "learning_rate": self.learning_rate,
            "window_size": self.window_size
        }

    def load_state(self, state: dict):
        """从保存的状态恢复管理器"""
        self.initial_threshold = state.get("initial_threshold", self.initial_threshold)
        self.current_threshold = state.get("current_threshold", self.current_threshold)
        self.learning_rate = state.get("learning_rate", self.learning_rate)
        self.window_size = state.get("window_size", self.window_size)

        feedback_history = state.get("feedback_history", [])
        self.feedback_history = deque(feedback_history, maxlen=self.window_size)

        self.threshold_history = state.get("threshold_history", [self.current_threshold])

        logger.info(f"加载自适应阈值管理器状态，当前阈值: {self.current_threshold:.3f}")