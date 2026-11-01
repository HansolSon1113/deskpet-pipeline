# Guard the pinned llama backend BEFORE executing a scheduled graph. The library
# still needs CPU metadata/sampling support, but numerical model ops cannot fall
# back to it. This patch is applied only to the ignored dependency checkout.
set(file "${LLAMA_SOURCE}/src/llama-context.cpp")
file(READ "${file}" source)
set(needle "    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);")
# Reapply our own guard when this file changes; do not leave an older variant
# installed in an already configured local checkout.
string(FIND "${source}" "    // DESKPET_STRICT_GPU" previous)
if(NOT previous EQUAL -1)
  string(FIND "${source}" "${needle}" end)
  if(end LESS previous)
    message(FATAL_ERROR "Cannot locate the end of the previous strict GPU guard")
  endif()
  string(SUBSTRING "${source}" 0 ${previous} before)
  string(SUBSTRING "${source}" ${end} -1 after)
  set(source "${before}${after}")
endif()
  string(FIND "${source}" "${needle}" found)
  if(found EQUAL -1)
    message(FATAL_ERROR "Pinned llama graph execution point changed; strict GPU guard needs review")
  endif()
  set(guard [=[
    // DESKPET_STRICT_GPU: fail before scheduling any numerical CPU model op.
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        auto * node = ggml_graph_node(gf, i);
        // Forward-select retains inactive embedding branches in the graph.
        // They are allocated for topology reuse but never execute.
        if (!(node->flags & GGML_TENSOR_FLAG_COMPUTE)) continue;
        if (node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) continue;
        auto * backend = ggml_backend_sched_get_tensor_backend(sched.get(), node);
        if (!backend) return GGML_STATUS_FAILED;
        const auto type = ggml_backend_dev_type(ggml_backend_get_device(backend));
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            LLAMA_LOG_ERROR("GPU_OPERATION_UNSUPPORTED: %s\n", ggml_op_name(node->op));
            return GGML_STATUS_FAILED;
        }
    }
]=])
  string(REPLACE "${needle}" "${guard}\n${needle}" source "${source}")
  file(WRITE "${file}" "${source}")
