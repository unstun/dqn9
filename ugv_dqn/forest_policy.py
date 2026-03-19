"""训练和推理共享的 forest 动作选择流水线。

V7: 通过在训练和推理中运行*完全相同*的门控逻辑
（贪心 Q → 可行性检查 → top-k → prog_mask → 回退）
消除训练/推理不一致问题。唯一的区别是 ``explore`` 标志，
训练时启用 epsilon-greedy 探索。
"""

from __future__ import annotations

import numpy as np
import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ugv_dqn.agents import DQNFamilyAgent
    from ugv_dqn.env import UGVBicycleEnv


def forest_select_action(
    env: UGVBicycleEnv,
    agent: DQNFamilyAgent,
    obs: np.ndarray,
    *,
    episode: int,
    explore: bool,
    horizon_steps: int = 15,
    topk: int = 10,
    min_od_m: float = 0.0,
    min_progress_m: float = 1e-4,
    training_mode: bool = False,
) -> int:
    """UGVBicycleEnv (forest) 环境的统一动作选择。

    流水线（训练和推理完全一致）：
        1. 计算 Q 值
        2. 若 *explore* 且 epsilon 触发 → 从可行动作 mask 中随机选择
        3. 贪心 a0 = argmax Q
        4. 若 a0 可行 → 返回 a0
        5. Top-k 搜索：尝试 Q 值次优的动作检查可行性
        6. prog_mask 回退：在可行动作上做 masked argmax Q
        7. 最终兜底：环境启发式短 rollout 回退

    V9: ``training_mode=True`` 仅使用碰撞安全检查（min_progress_m=0），
    让 Q 网络从奖励中学习前进，而非依赖 mask 过滤。
    """
    adm_h = max(1, int(horizon_steps))
    topk_k = max(1, int(topk))
    min_od = float(min_od_m)
    # V9: 训练时仅强制碰撞安全；让 Q 网络自行学习前进
    min_prog = 0.0 if bool(training_mode) else float(min_progress_m)

    # --- epsilon 探索（仅训练时） ---------------------------------
    if explore and (agent._rng.random() < agent.epsilon(episode)):
        mask = env.admissible_action_mask(
            horizon_steps=adm_h,
            min_od_m=min_od,
            min_progress_m=min_prog,
            fallback_to_safe=True,
        )
        idxs = np.nonzero(mask)[0]
        if idxs.size == 0:
            return int(agent._rng.integers(0, agent._n_actions))
        return int(agent._rng.choice(idxs))

    # --- 贪心路径（训练和推理共享） -----------------------------
    with torch.no_grad():
        x = torch.from_numpy(agent._prep_obs(obs)).to(agent.device)
        q = agent.q(x.unsqueeze(0)).squeeze(0)

    a0 = int(torch.argmax(q).item())

    # 第4步：对贪心动作进行可行性检查
    if bool(env.is_action_admissible(int(a0), horizon_steps=adm_h, min_od_m=min_od, min_progress_m=min_prog)):
        return int(a0)

    # 第5步：top-k 搜索
    kk = int(min(topk_k, int(q.numel())))
    topk_indices = torch.topk(q, k=kk, dim=0).indices.detach().cpu().numpy()
    for cand in topk_indices.tolist():
        cand_i = int(cand)
        if cand_i == int(a0):
            continue
        if bool(env.is_action_admissible(cand_i, horizon_steps=adm_h, min_od_m=min_od, min_progress_m=min_prog)):
            return int(cand_i)

    # 第6步：prog_mask 回退（masked argmax Q）
    prog_mask = env.admissible_action_mask(
        horizon_steps=adm_h,
        min_od_m=min_od,
        min_progress_m=min_prog,
        fallback_to_safe=False,
    )
    if bool(prog_mask.any()):
        q_masked = q.clone()
        q_masked[torch.from_numpy(~prog_mask).to(q.device)] = torch.finfo(q_masked.dtype).min
        return int(torch.argmax(q_masked).item())

    # 第7步：启发式兜底
    return int(env._fallback_action_short_rollout(horizon_steps=adm_h, min_od_m=min_od))


def forest_compute_next_mask(
    env: UGVBicycleEnv,
    *,
    horizon_steps: int = 15,
    min_od_m: float = 0.0,
    min_progress_m: float = 1e-4,
) -> np.ndarray:
    """计算*下一个*状态的可行动作 mask（用于 replay buffer 的 TD target）。"""
    return env.admissible_action_mask(
        horizon_steps=max(1, int(horizon_steps)),
        min_od_m=float(min_od_m),
        min_progress_m=float(min_progress_m),
        fallback_to_safe=True,
    )
