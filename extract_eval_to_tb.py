"""
从 GRPO TensorBoard event 文件里提取 eval 数据点（与 train 共用 step 号的"重复条目"），
写到一个独立目录，方便 TensorBoard 当成第二条 run 显示。

用法:
    python extract_eval_to_tb.py \
        --src train_logs/grpo-426 \
        --dst train_logs/grpo-426-eval

随后启动 TB（指向上层目录即可同时看到两条曲线）:
    tensorboard --logdir train_logs --port 6006
"""

import argparse
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


def extract(src_dir: str, dst_dir: str):
    files = sorted(glob.glob(os.path.join(src_dir, "events.out.tfevents.*")))
    if not files:
        raise FileNotFoundError(f"no event files in {src_dir}")

    os.makedirs(dst_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=dst_dir)

    n_eval_points = 0
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            seen = {}
            for e in ea.Scalars(tag):
                # 同 step 第二次出现 = eval（train 已经先写过）
                if e.step in seen and seen[e.step] != e.value:
                    writer.add_scalar(f"eval/{tag}", e.value, global_step=e.step)
                    n_eval_points += 1
                seen[e.step] = e.value

    writer.flush()
    writer.close()
    print(f"[done] wrote {n_eval_points} eval points to {dst_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="train_logs/grpo-426",
                   help="原始 train 日志目录")
    p.add_argument("--dst", default="train_logs/grpo-426-eval",
                   help="提取出的 eval 日志输出目录")
    args = p.parse_args()
    extract(args.src, args.dst)


if __name__ == "__main__":
    main()
