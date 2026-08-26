#include "ops.h"
#include <torch/library.h>

// Delivery surface for the vllm-ascend integration: the four V2-aligned
// entries only. (The evolution tree additionally registers internal/legacy
// entries — gmm_situ_vcv, gmm_situ_vcv_dev, grouped_matmul_situ_quant_tl —
// which are intentionally NOT shipped here.)
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("gmm_situ_quant(Tensor x, Tensor weight, Tensor weightScale, Tensor? weightAssistMatrix, Tensor? bias, Tensor xScale, Tensor? smoothScale, Tensor groupList, int dequantMode, int dequantDtype, int quantMode, int groupListType, int[]? tuningConfigOptional, float beta, float linearBeta) -> (Tensor output, Tensor outputScale)");
    m.def("gmm_situ_quant.list(Tensor x, Tensor[] weight, Tensor[] weightScale, Tensor[]? weightAssistMatrix, Tensor? bias, Tensor xScale, Tensor? smoothScale, Tensor groupList, int dequantMode, int dequantDtype, int quantMode, int groupListType, int[]? tuningConfigOptional, float beta, float linearBeta) -> (Tensor output, Tensor outputScale)");
    m.def("gmm_situ_quant_weight_nz(Tensor x, Tensor weight, Tensor weightScale, Tensor? weightAssistMatrix, Tensor? bias, Tensor xScale, Tensor? smoothScale, Tensor groupList, int dequantMode, int dequantDtype, int quantMode, int groupListType, int[]? tuningConfigOptional, float beta, float linearBeta) -> (Tensor output, Tensor outputScale)");
    m.def("gmm_situ_quant_weight_nz.list(Tensor x, Tensor[] weight, Tensor[] weightScale, Tensor[]? weightAssistMatrix, Tensor? bias, Tensor xScale, Tensor? smoothScale, Tensor groupList, int dequantMode, int dequantDtype, int quantMode, int groupListType, int[]? tuningConfigOptional, float beta, float linearBeta) -> (Tensor output, Tensor outputScale)");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("gmm_situ_quant", TORCH_FN(ascend_kernel::gmm_situ_quant_v2_nd));
    m.impl("gmm_situ_quant.list", TORCH_FN(ascend_kernel::gmm_situ_quant_v2_nd_list));
    m.impl("gmm_situ_quant_weight_nz", TORCH_FN(ascend_kernel::gmm_situ_quant_v2_nz));
    m.impl("gmm_situ_quant_weight_nz.list", TORCH_FN(ascend_kernel::gmm_situ_quant_v2_nz_list));
}
