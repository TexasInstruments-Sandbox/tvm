# Hardware Debug Scripts

Scripts for loading and running programs on the J722S C75x DSP via JTAG.
Located at `src/runtime/ti_dsp/scripts/`.

## Quick Start

Load and run a single program:

```bash
./scripts/run_on_c75x.sh hello_world_standalone/build/c75_hello_world.out
```

This loads the FPGA image each time (~12 seconds per run).

## Scripts Reference

| Script | Description |
|--------|-------------|
| `run_on_c75x.sh` | Shell wrapper for program execution |
| `load_and_run_c75x.js` | DSS script for load and run |

### Configuration Files

| File | Description |
|------|-------------|
| `J722S_560v2.ccxml` | Target configuration for XDS560v2 emulator |

## Options

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
| `CCS_ROOT` | `/opt/ti/ccs2040/ccs` | CCS installation directory |

## Troubleshooting

### Target stuck in fault state

Use the `--force` flag, or power cycle the EVM:
```bash
./scripts/run_on_c75x.sh program.out --force
```
