from __future__ import annotations

from typing import Any

import torch

from vllm_ascend import envs


def _iter_connector_names(kv_transfer_config: Any) -> list[str]:
    connector_names: list[str] = []
    connector_name = getattr(kv_transfer_config, "kv_connector", "")
    if connector_name:
        connector_names.append(str(connector_name))

    extra_config = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    connectors = extra_config.get("connectors", [])
    if isinstance(connectors, dict):
        connectors = connectors.values()

    for connector in connectors:
        if isinstance(connector, dict):
            nested_name = connector.get("kv_connector", "")
        else:
            nested_name = getattr(connector, "kv_connector", "")
        if nested_name:
            connector_names.append(str(nested_name))

    return connector_names


def kv_transfer_uses_mooncake(kv_transfer_config: Any) -> bool:
    if kv_transfer_config is None:
        return False
    return any("mooncake" in name.lower() for name in _iter_connector_names(kv_transfer_config))


def should_use_dsa_pd_mooncake_cpu_kv(
    vllm_config: Any,
    *,
    use_sparse: bool,
    is_kv_consumer: bool,
    enable_cpu_kv_store: bool,
    dsa_sparse_attention_mode: str | None = None,
) -> bool:
    if not envs.VLLM_ASCEND_DSA_PD_MOONCAKE_CPU_KV:
        return False
    if dsa_sparse_attention_mode == "baseline":
        return False
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    return bool(
        use_sparse
        and is_kv_consumer
        and enable_cpu_kv_store
        and kv_transfer_config is not None
        and getattr(kv_transfer_config, "is_kv_consumer", False)
        and kv_transfer_uses_mooncake(kv_transfer_config)
    )


def empty_swapped_memory(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    try:
        import torch_npu
    except ImportError as exc:
        raise RuntimeError(
            "DSA PD Mooncake CPU KV requires torch_npu.empty_with_swapped_memory."
        ) from exc

    empty_with_swapped_memory = getattr(torch_npu, "empty_with_swapped_memory", None)
    if empty_with_swapped_memory is None:
        raise RuntimeError(
            "DSA PD Mooncake CPU KV requires torch_npu.empty_with_swapped_memory."
        )
    return empty_with_swapped_memory(shape, dtype=dtype, device=device)
