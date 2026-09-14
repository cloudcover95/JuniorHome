# Vulkan compute vs JuniorOS

Compute shader = vkCreateComputePipelines + vkCmdDispatch + SPIR-V. Invocations / subgroups / shared memory. Cooperative matrix and integer-dot help INT8; they are not 1.58 trit ISA.

Apple: Darwin uses Metal (llama.cpp + MLX), not Vulkan. MoltenVK / KosmicKrisp are translation. Asahi Honeykrisp is conformant Vulkan 1.3 on M1 including compute; M3 GPU still weak. llama.cpp already has a Vulkan backend for GGUF offload on iGPU.

JuniorHome: T0 stays CPU winsor. Vulkan is a T3-class probe (`rails/linux/vulkan.py`) — ready only if vulkaninfo exists. No shader.spv in tree.
