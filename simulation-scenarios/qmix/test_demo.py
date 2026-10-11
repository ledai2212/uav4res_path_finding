"""Chạy thử 1 episode với checkpoint đã train (epsilon thấp, có vẽ pygame).

Chạy từ thư mục gốc dự án:
    python simulation-scenarios/qmix/test_demo.py
    python simulation-scenarios/qmix/test_demo.py --checkpoint best_ep4713_steps1323_9117.pt --epsilon 0.05

Mặc định tự chọn file best_*.pt có điểm cao nhất; nếu chưa có thì dùng qmix_model.pt.
"""
import time
from dataclasses import dataclass
from typing import Optional

import torch
import tyro

from qmix import (
    Args, build_env, build_networks, find_best_checkpoint,
    load_checkpoint, run_episode, Drawer,
)


@dataclass
class TestArgs:
    checkpoint: Optional[str] = None
    """đường dẫn checkpoint; None = best_*.pt điểm cao nhất, không có thì qmix_model.pt"""
    epsilon: float = 0.05
    """xác suất hành động ngẫu nhiên khi test (đủ nhỏ để khôn, đủ lớn để không kẹt)"""
    render_fps: int = 45
    hold_seconds: float = 5.0
    """giữ cửa sổ thêm vài giây sau khi chạy xong"""


def run_test():
    targs = tyro.cli(TestArgs)

    args = Args()
    args.use_clusters = True
    args.use_energy = True
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = build_env(args)
    agent_net, target_agent, mixer, target_mixer = build_networks(env, args, device)

    ckpt_path = targs.checkpoint or find_best_checkpoint(args.model_path)
    print(f"Nạp dữ liệu từ {ckpt_path}...")
    load_checkpoint(ckpt_path, agent_net, mixer, target_agent, target_mixer, device)

    drawer = Drawer("qmix")

    start_time = time.time()
    episode, ep_reward, coverage = run_episode(
        env, agent_net, epsilon=targs.epsilon, device=device, n_agents=args.num_uavs,
        drawer=drawer, render_fps=targs.render_fps)
    flight_duration = time.time() - start_time

    print("\n" + "=" * 50)
    print("KẾT QUẢ ĐÁNH GIÁ (TESTING)")
    print("=" * 50)
    print(f"Checkpoint        : {ckpt_path}")
    print(f"Điểm số (Return)  : {ep_reward:.2f}")
    print(f"Tỷ lệ quét (Cov)  : {coverage:.2%}")
    print(f"Số bước di chuyển : {env.steps} steps")
    print(f"Thời gian mô phỏng: {flight_duration:.2f} giây")
    print("=" * 50)

    time.sleep(targs.hold_seconds)


if __name__ == "__main__":
    run_test()
