# Backend launch-gate tests

See also: `../Makefile` targets `test-latency-local` and `test-latency`,
and the client doc `NightOutClient/TESTING.md`.

```bash
pip install -r requirements-dev.txt
make test-latency-local   # cache-hit p95 < 50ms (in-process)
make run                  # other terminal
make test-latency         # same asserts over HTTP
```
