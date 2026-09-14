# Vulkan compute vs this suite

Asahi has a conformant Vulkan 1.3+ stack on M1/M2; M3 GPU still weak. Mesa 26.2 + LunarG 1.4 SDK exist on Linux/macOS (KosmicKrisp = Vulkan-on-Metal). Blender Cycles has no Vulkan backend on purpose.

JuniorOS action: `rails.linux.vulkan_probe` looks for vulkaninfo / libvulkan. `dispatch: false`. Trit pack stays C/Python on CPU. No SPIR-V in-tree.
