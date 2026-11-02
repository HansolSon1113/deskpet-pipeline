#pragma once
#include "ggml.h"

// Parameters stay FP16 on their owning GPU. ggml matmul accumulates into F32;
// Metal's ADD requires identical operand types, so convert the small bias on
// the same graph/backend before adding it. No host inference path is involved.
inline ggml_tensor *deskpet_head_linear(ggml_context *ctx,ggml_tensor *input,
                                       ggml_tensor *weight,ggml_tensor *bias) {
    auto output=ggml_mul_mat(ctx,weight,input);
    if(!bias)return output;
    auto typedBias=bias->type==output->type?bias:ggml_cast(ctx,bias,output->type);
    return ggml_add(ctx,output,typedBias);
}
