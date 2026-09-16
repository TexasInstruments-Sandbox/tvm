# Debugging DSP Test Failures

## DSP_KEEP_TEMP=1

Preserves the workspace directory (e.g. `/tmp/dsp_test_conv2d_dsp_20260505_143027/`):
```
workspace/
├── lib0.c           <- Generated C source
├── weights.bin      <- Serialized parameters
├── build-c7x_host/  <- Host emulation build
│   └── cg_dsp      <- Executable
└── build-c7x_dload/ <- DLOAD build
    ├── lib0.out     <- DLOAD ELF
    └── lib0.asm     <- Assembly (with -k flag)
```

## Inspect Generated Code

```bash
DSP_KEEP_TEMP=1 pytest --rootdir=. dsp-tests/test_conv2d_dsp.py -v \
    --dsp-mode=c7x_host -k test_conv2d
# Then inspect /tmp/dsp_test_conv2d_dsp_*/lib0.c
```

## DSP Trace Output

Always use `c7x_compute trace` on the board to read DSP printf/trace output:
```bash
ssh root@am67a c7x_compute trace
```

Do NOT `cat` the remoteproc trace file directly (e.g. `/sys/kernel/debug/remoteproc/remoteprocN/trace0`) — the remoteproc index can change after a reboot, making the path unreliable. `c7x_compute trace` discovers the correct device via the device tree address (`7e000000.dsp`).

## Detecting a DSP Hang Mid-Test

If a `c7x_dload` test produces no output for more than ~60 seconds after the
build/deploy step, the DSP has likely hung (page fault, infinite loop, or
exception).  Do NOT wait for the pytest timeout — check immediately:

```bash
ssh root@am67a c7x_compute trace
```

Signs of a hard hang in the trace:
- `Page fault exception` / `IERR=` lines
- `Exception at 0x...` with garbage `IEAR` address
- SA counter dumps (`SA0CNTR0=...`) followed by no further progress

When a hang is confirmed (substitute `--board beagley-ai`/`BOARD_HOSTNAME=beagley-ai`
and the board's own hostname if you're not on am67a):
1. Kill the pytest process (Ctrl-C or `TaskStop`)
2. Try `deploy-c7x.sh --restart`; if `EBUSY` errors persist after 10 retries, do `ssh root@am67a reboot` and wait 30 s for the board to come back
3. After recovery, redeploy firmware: `deploy-c7x.sh dsp/build/c7x_compute.out && deploy-c7x.sh --restart`
4. Run only the failing test in isolation (not the full suite) to avoid re-hanging

## Common Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Numerical mismatch (small) | fp_reassoc reordering | Add `--fp_reassoc=off` to cl7x flags |
| Numerical mismatch (large) | Quantization error compounding | Check per-layer accuracy |
| Timeout on c7x_dload | Model too large / firmware hang | Check `c7x_compute status`, try smaller timeout |
| "No response from DSP" | Firmware crashed | `deploy-c7x.sh --restart` or reboot board |
| Build failure (cl7x) | Missing TI_CGT_C7000_PATH | Export env var |
| Linker: unresolved symbol | New runtime API not in firmware exports | Add to `dyn_loader.c`, rebuild firmware |
| `c7x: INFER failed: status=-11 return_value=-1` with a clean, specific `ERROR: OOM in DDR pool: requested X, free Y / Z` message | Genuine DDR heap exhaustion, hit inside `cg_main_dsp` (the OOM log is unconditional) | Grow `DDR_C7X_1_LOCAL_HEAP` in `linker_c75_freertos.cmd` if MMU-mapped headroom exists (check `mmu_armv8_r13` in `c75ss0.syscfg`), or reduce peak workspace |
| Same `status=-11 return_value=-1`, but **zero printf/profile output at all** (not even `TVMPrintLayerProfile`'s unconditional "Total: X cycles" line), even with `-profile-layers` compiled in and the symbol resolving | Not diagnosed by the OOM path — `cg_main_dsp` returns `-1` from somewhere else with no logging. See "Debugging a Silent `-1` Failure" below before assuming it's DMA or a compute kernel bug — a real instance of this traced all the way to the DSP-side response containing a correct nonzero `printf_size`, meaning the bug was in the ARM client (`c7x_compute_client.cpp`, `sync_output_from_device()`) not reading/syncing it, not on the DSP side at all |

## Debugging a Silent `-1` Failure (no printf, no crash)

`status=-11 return_value=-1` is a **generic** code — `cg_main_dsp` returns
whatever nonzero value the first failing kernel call inside it returned, and
the firmware just forwards it. When there's also zero printf/profile output,
narrow it down in this order rather than guessing:

1. **Confirm `-profile-layers` is actually compiled in.** Not every test file
   wires this through `get_target_string(..., profile_layers=True)` +
   `compile_and_run_dsp(..., profile_layers=True)` — e.g. as of this writing
   `quantized/test_quantized_torchvision.py`'s `_run_test()` hardcodes
   `use_cpp_api=True` only and has no `profile_layers` parameter at all, so
   passing `--profile`/`--profile-layers` on the pytest CLI silently does
   nothing for that file. If a test file doesn't support it, write a small
   standalone script that calls the same `cl_torchvision.py` /
   `pt2e_utils.e2e_quantize_and_import` / `dsp_utils.compile_and_run_dsp`
   helpers directly with `profile_layers=True` explicitly.
2. **Rule out DMA** by enabling `TVM_DMA_DEBUG` (see
   `relax-c7x:firmware` DMA Debug Tracing) and reading `deploy-c7x.sh
   --trace`. If every `[DMA] copy`/`wait` line shows `q=0` and `wait:
   completed`, DMA is not the cause — don't keep chasing it.
3. **Check the general printf/profile mechanism isn't itself broken** by
   running the SAME diagnostic against a model that's known to pass. If a
   full, correct profile dump comes back for the passing model, the
   mechanism is fine in general and the failing model's issue is specific to
   it (different code path, different memory footprint, etc.) — don't waste
   time on "is the whole profiling pipeline broken" once this is ruled out.
4. **Add direct trace-buffer probes around the print step in
   `compute_service.c`**, not more printf-based probes (printf is exactly
   the channel under suspicion). Add a small `shm_printf_debug_dump(tag)`
   helper to `shm_printf.c`/`.h` (logs `g_hdr->magic`/`wr_index`/`buf_size`
   via `DebugP_log`, which goes to the trace buffer, a separate channel from
   the shm printf buffer) and call it: before `shm_printf_reset()`, after
   `cg_main_dsp` returns, and after `print_profile()` runs. This tells you
   exactly which hop drops the data:
   - `wr_index` still 0 after `print_profile()` returns → the profile
     printer itself isn't writing (dig into codegen/`TVMPrintLayerProfile`).
   - `wr_index` nonzero after `print_profile()`, and `resp->printf_size` set
     to a matching sane value (add one more `DebugP_log` right after
     `resp->printf_size = shm_printf_finish();`) → the DSP side is correct
     and the bug is in the **ARM client**. Check `sync_output_from_device()`
     in `c7x_compute_client.cpp` — specifically whether its sync range/size
     is unconditional or derived from a field (like `num_outputs`) that gets
     zeroed on the failure path, which would leave the client's view of
     `result_buf` stale for the printf region specifically on failures.
5. Remember `_tvm_layer_count`/`_tvm_layer_names[]` are not
   `dyn_loader_query_symbol()`-queryable (plain statics, not DLOAD-exported)
   — don't spend time trying to peek at them from firmware code directly.

Revert all TEMP diagnostic code (`TVM_DMA_DEBUG` define,
`shm_printf_debug_dump` calls, any `[DIAG]` `DebugP_log` lines) once the
root cause is found — none of it is meant to ship enabled.

## Hardware Rules (c7x_dload)

- **NEVER run in background** — single DSP core, conflicts cause firmware hangs
- **NEVER run concurrent sessions** — requires board reboot to recover
- **Always sequential** — one pytest session at a time on hardware

## Firmware Recovery (escalation order)

1. `deploy-c7x.sh --stop && deploy-c7x.sh --start` (add `--board beagley-ai` if applicable)
2. `ssh root@<board-host> reboot` (am67a, or beagley-ai)
3. Network power cycle via the same PDU, different outlet per board:
   - am67a: `wget --no-proxy --http-user=admin --http-password="" "http://10.219.15.103/outlet.cgi?outlet=1&command=3"`
   - beagley-ai: `wget --no-proxy --http-user=admin --http-password="" "http://10.219.15.103/outlet.cgi?outlet=2&command=3"`

**Don't loop on steps 1-2 if you see this signature — go straight to step
3.** A wedged DSP core lets SSH/step-2's reboot succeed normally (Linux side
is fine) while the DSP itself never re-attaches: `remoteproc0` (`7e000000.dsp`)
sits in `state=running` with no `/dev/rpmsg_ctrl*` ever appearing, `dmesg`
shows repeated `k3_rproc_stop: timedout waiting for rproc completion event` /
`can't stop rproc: -16`, and `c7x_compute status` hangs for minutes instead
of connecting or failing fast even right after a fresh reboot. It's a real
DSP hang, not a "wait longer after reboot" timing issue -- the power cycle
in step 3 reliably clears it because it resets state a soft `reboot` leaves
alone.
