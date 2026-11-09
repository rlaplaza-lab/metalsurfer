"""Process-local flags set from ``pytest_configure`` before test modules import."""

# True until pytest_configure narrows the session. Importing MLIP marks outside
# pytest (or with an empty marker expression) still probes the stack.
mlip_probe_enabled = True
