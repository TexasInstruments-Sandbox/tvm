# C75x DSP Debug Scripts

Scripts for loading and running programs on the J722S C75x DSP via JTAG.

## Quick Start

### One-shot Mode (Simple)

Load and run a single program:

```bash
./scripts/run_on_c75x.sh hello_world_standalone/build/c75_hello_world.out
```

This loads the FPGA image each time (~12 seconds per run).

### Persistent Server Mode (Fast)

For iterative development, use the persistent debug server to avoid FPGA reload:

```bash
# 1. Start the debug server (once per session)
./scripts/c75x_server_start.sh --background

# 2. Load and run programs (fast, ~5 seconds each)
./scripts/c75x_load.sh hello_world_standalone/build/c75_hello_world.out
./scripts/c75x_load.sh another_program.out

# 3. Stop the server when done
./scripts/c75x_server_stop.sh
```

## Performance Comparison

| Method | Total Time | Notes |
|--------|------------|-------|
| One-shot mode | ~12 sec | Loads FPGA image every run |
| Persistent server | ~5 sec | Skips FPGA load after first run |

The persistent server saves ~7 seconds per program load by keeping the JTAG connection and FPGA state alive between runs.

## Scripts Reference

### One-shot Scripts

| Script | Description |
|--------|-------------|
| `run_on_c75x.sh` | Shell wrapper for one-shot program execution |
| `load_and_run_c75x.js` | DSS script for one-shot load and run |

### Persistent Server Scripts

| Script | Description |
|--------|-------------|
| `c75x_server_start.sh` | Start the persistent debug server |
| `c75x_load.sh` | Load and run a program on the running server |
| `c75x_server_stop.sh` | Stop the debug server |
| `debug_server_start.js` | DSS script for persistent server |
| `debug_server_load.js` | DSS script to send load commands |
| `debug_server_stop.js` | DSS script to stop the server |

### Configuration Files

| File | Description |
|------|-------------|
| `J722S_560v2.ccxml` | Target configuration for XDS560v2 emulator |

## Options

### c75x_server_start.sh

```
--background    Run server in background (recommended)
```

### c75x_load.sh

```
--timeout <ms>  Timeout waiting for program completion (default: 60000)
```

### load_and_run_c75x.js

```
--timeout <ms>     Execution timeout in milliseconds (default: 60000)
--output <file>    File to capture console output
--no-wait          Don't wait for program to halt
--reset            Reset target before loading
--force            Force connect (recover from fault state)
--ccxml <file>     CCXML configuration file
--core <pattern>   Core name pattern (default: .*C75X_0)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CCS_ROOT` | `/home/a0323430/ti/ccs2040/ccs` | CCS installation directory |

## How the Persistent Server Works

The persistent server uses file-based IPC in `/tmp/c75x_debug/`:

```
/tmp/c75x_debug/
├── status.txt    # Server status: ready, busy, stopping, error
├── command.txt   # Program path to load (written by client)
├── result.txt    # Execution result (written by server)
├── server.pid    # Server process ID
├── server.log    # Server output (when run with --background)
└── stop.txt      # Create this file to stop the server
```

The server maintains the JTAG connection and FPGA state, so subsequent program loads skip the ~8 second FPGA initialization.

## Troubleshooting

### Server won't start

Check if another instance is running:
```bash
cat /tmp/c75x_debug/status.txt
cat /tmp/c75x_debug/server.pid
```

Clean up stale files if needed:
```bash
rm -rf /tmp/c75x_debug
```

### Target stuck in fault state

Use the `--force` flag with the one-shot script, or power cycle the EVM:
```bash
./scripts/run_on_c75x.sh program.out --force
```

### View server logs

```bash
tail -f /tmp/c75x_debug/server.log
```
