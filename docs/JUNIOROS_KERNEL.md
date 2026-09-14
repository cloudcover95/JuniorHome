# JuniorOS kernel architecture

There is no cloudcover95 vmlinuz. JuniorOS is an overlay + userspace rail on a vendor kernel (Debian/Alpine/Pi/Asahi).

```
hardware (cpu | mlx-if-import | cuda-if-torch)
  vendor kernel (namespaces, seccomp, cgroups — kconfig.junior fragment)
    overlay (os-release.junior, bitnetd loopback, i2sd)
      JuniorHome os_route
        JuniorLLM handshake / T0–T2 / OSai goldens
```

What looks like a kernel module is not: AbsMean and I2_S live in userspace (`junior_bitnet`, `rails/linux/i2s_pack.c`). MLX is not a Kconfig option. CUDA is the NVIDIA module on that vendor kernel.

Do not merge kconfig.junior into a tree from CI. Operator copies the fragment onto their own build.
