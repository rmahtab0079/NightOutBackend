# Backend launch-gate testing

Full guide (client + server): see `NightOutClient/TESTING.md`.

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt

# Required before App Store / Play Store submit:
make test-latency-local
# Expect: cache-hit p95 << 50ms for restaurants + activities

# Optional HTTP suite (server must be running):
make run            # terminal A
make test-latency   # terminal B
```
