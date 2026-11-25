# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""av_fd_longmem: full-duplex long-horizon audio-visual memory benchmark.

A client-side harness around a deployed vLLM-Omni duplex server. Queries arrive
on the input stream after a controlled lag of filler; replies are judged and
evidence residency is read from the server's Stage-0 eviction trace.

Design: the av_fd_longmem plan (author's notes); trace format in
``vllm_omni/engine/duplex/tick_trace.py``.
External dependencies: none at import time.
"""
