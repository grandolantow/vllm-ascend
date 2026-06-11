#!/usr/bin/env python3
"""Analyze hot KV cache hit rate from dumped topk tensors."""

from __future__ import annotations

import argparse
import csv
import json
import os

from vllm_ascend.attention.hot_kv_cache import (
    HotKVCacheSimulator,
    LRUKVCacheSimulator,
    load_topk_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_path", help="Path to hot_kv_cache_config.dump_path .pt file")
    parser.add_argument("--buffer-size", type=int, required=True, help="Resident KV buffer size to simulate")
    parser.add_argument(
        "--policy",
        choices=["hot-kv", "lru"],
        default="hot-kv",
        help="Offline cache policy to simulate. Default keeps existing Hot KV behavior.",
    )
    parser.add_argument(
        "--topk-length",
        type=int,
        default=2048,
        help="Number of leading topk positions to replay from each dump row. Default: 2048.",
    )
    parser.add_argument("--recent-window", type=int, default=32)
    parser.add_argument("--ema-beta", type=float, default=0.9)
    parser.add_argument("--recent-weight", type=float, default=1.0)
    parser.add_argument("--ema-weight", type=float, default=0.5)
    parser.add_argument("--age-weight", type=float, default=0.01)
    parser.add_argument(
        "--candidate-size",
        type=int,
        default=256,
        help=(
            "Raw HotKVCacheConfig candidate_size. The simulator applies the "
            "online rule min(buffer_size, max(candidate_size, topk_width * 2)) "
            "and uses online-style candidate-window slot selection."
        ),
    )
    parser.add_argument("--out-dir", default="", help="Optional directory for CSV/JSON outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_topk_records(args.dump_path)
    if args.policy == "lru":
        simulator = LRUKVCacheSimulator(
            buffer_size=args.buffer_size,
            topk_length=args.topk_length,
        )
    else:
        simulator = HotKVCacheSimulator(
            buffer_size=args.buffer_size,
            recent_window=args.recent_window,
            ema_beta=args.ema_beta,
            recent_weight=args.recent_weight,
            ema_weight=args.ema_weight,
            age_weight=args.age_weight,
            candidate_size=args.candidate_size,
            topk_length=args.topk_length,
        )
    stats = simulator.run(records)
    stats["config"] = {
        "policy": args.policy,
        "buffer_size": args.buffer_size,
        "topk_length": args.topk_length,
        "recent_window": args.recent_window if args.policy == "hot-kv" else None,
        "ema_beta": args.ema_beta if args.policy == "hot-kv" else None,
        "recent_weight": args.recent_weight if args.policy == "hot-kv" else None,
        "ema_weight": args.ema_weight if args.policy == "hot-kv" else None,
        "age_weight": args.age_weight if args.policy == "hot-kv" else None,
        "candidate_size": args.candidate_size if args.policy == "hot-kv" else None,
    }

    print("GLOBAL")
    print(_format_row("ALL", stats["global"]))
    print()
    print("BY_LAYER")
    for layer, row in sorted(stats["by_layer"].items(), key=lambda item: item[0]):
        print(_format_row(layer, row))

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "hot_kv_summary.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.out_dir, "hot_kv_by_layer.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["layer", "requests", "needed", "hits", "misses", "hit_rate"])
            writer.writeheader()
            for layer, row in sorted(stats["by_layer"].items(), key=lambda item: item[0]):
                writer.writerow({"layer": layer, **row})


def _format_row(name: str, row: dict) -> str:
    return (
        f"{name}: requests={row['requests']} needed={row['needed']} "
        f"hits={row['hits']} misses={row['misses']} hit_rate={row['hit_rate']:.6f}"
    )


if __name__ == "__main__":
    main()
