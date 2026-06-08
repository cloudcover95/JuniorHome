# JuniorHome

**Production Development Update**

- Removed dependency on GitHub Actions free tier (rate limits + paid tokens avoided).
- Added `Makefile` for easy local development, testing, linting, and building across **Linux, macOS, and Windows**.
- Packaging improved for clean `pip install -e .` experience on all platforms.
- Core modules remain focused on sovereign, efficient, black-box design with BitNet/ternary foundations.

**Local Development (all platforms)**
```bash
make install
test
make lint
make format
make build
```

The ecosystem is being hardened for real production use without reliance on external CI services.