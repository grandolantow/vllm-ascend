import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from vllm_ascend.attention.cpu_cache_miss_topk import (  # noqa: E402
    make_cpu_cache_miss_topk_workspace,
    update_topk_indices_cpu,
)


def _run_numpy_bridge():
    req_ids = np.array([3], dtype=np.int64)
    last_req_ids = req_ids.copy()
    old = np.array(
        [[3, 7, 15, 22, -1, -1]],
        dtype=np.int64,
    )
    new = np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int32)
    workspace = make_cpu_cache_miss_topk_workspace(topk=6, max_token=128)

    update_topk_indices_cpu(req_ids, last_req_ids, old, new, workspace)

    print("new", new.tolist())
    print("old", old.tolist())


def _run_torch_bridge():
    req_ids = torch.tensor([3], dtype=torch.int64)
    last_req_ids = torch.tensor([3], dtype=torch.int64)
    old = torch.tensor(
        [[3, 7, 15, 22, -1, -1]],
        dtype=torch.int64,
    )
    new = torch.tensor([[5, 7, 18, 22, 30, 33]], dtype=torch.int32)

    req_ids_cpu = np.ascontiguousarray(req_ids.numpy(), dtype=np.int64)
    last_req_ids_cpu = np.ascontiguousarray(last_req_ids.numpy(), dtype=np.int64)
    old_cpu = np.ascontiguousarray(old.numpy(), dtype=np.int64)
    new_cpu = np.ascontiguousarray(new.numpy(), dtype=np.int32)
    workspace = make_cpu_cache_miss_topk_workspace(topk=6, max_token=128)

    update_topk_indices_cpu(req_ids_cpu, last_req_ids_cpu, old_cpu, new_cpu,
                            workspace)

    last_req_ids.copy_(torch.from_numpy(last_req_ids_cpu))
    old.copy_(torch.from_numpy(old_cpu))
    new.copy_(torch.from_numpy(new_cpu))

    print("new", new.tolist())
    print("old", old.tolist())


def main():
    if torch is None:
        _run_numpy_bridge()
    else:
        _run_torch_bridge()


if __name__ == "__main__":
    main()
