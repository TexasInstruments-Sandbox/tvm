# skills-c7x

Claude Code plugin marketplace providing TVM Relax C7x DSP skills for
the full compilation-to-deployment stack on TI AM67A (J722S). The skills
describe the fork rebased on TVM 0.27.

## Skills

| Skill | Scope |
|-------|-------|
| `relax-c7x:build` | Building TVM core, runtime, firmware, DLOAD modules, env setup |
| `relax-c7x:cstatic` | c_static_lib backend, code generation, target options, pipeline pass order |
| `relax-c7x:dsp-runtime` | DSP runtime library, memory pools, Model API |
| `relax-c7x:firmware` | c7x_compute firmware, DLOAD, DMA, IPC, memory layout, deploy |
| `relax-c7x:dsp-ops` | Writing DSP operator kernels, DMA tiling, quantization math |
| `relax-c7x:tidl-offload` | TIDL subgraph partitioning, import, bridge generation |
| `relax-c7x:mmalib-offload` | MMALIB QDQ fusion, int16 offload, MMA wrappers |
| `relax-c7x:testing` | Test authoring, pytest fixtures, profiling, cycle counts |
| `relax-c7x:model-workflow` | End-to-end recipe: export → quantize → offload → compile → verify |
| `relax-c7x:relax-passes` | Writing Relax/TIR passes: DFPattern, PyExprMutator, call_extern, pipeline wiring |
| `relax-c7x:tvm-027-api` | TVM 0.23 -> 0.27 API changes: Relax/TIR Python, target strings, CodeGenC/C++ |

## Directory Structure

```
skills-c7x/                          <- marketplace root
├── README.md
└── relax-c7x/                      <- plugin
    ├── .claude-plugin/
    │   └── plugin.json                 <- plugin metadata (name, version, author)
    └── skills/
        ├── build/SKILL.md              <- Full build: TVM, runtime, firmware, deploy
        ├── cstatic/SKILL.md            <- Relax -> TIR -> C -> DLOAD ELF
        ├── dsp-runtime/SKILL.md        <- Runtime library, memory, Model API
        ├── firmware/SKILL.md           <- c7x_compute, DLOAD linker, IPC
        ├── dsp-ops/SKILL.md            <- Operator kernels, DMA tiling, quant math
        ├── tidl-offload/SKILL.md       <- TIDL subgraph partitioning + MMA
        ├── mmalib-offload/SKILL.md     <- MMALIB direct integration + QDQ
        ├── testing/                    <- Test infrastructure
        │   ├── SKILL.md               <- Test authoring, fixtures, profiling
        │   └── references/
        │       ├── debugging.md       <- DSP_KEEP_TEMP, failure modes, recovery
        │       └── smollm-e2e.md      <- SmolLM compile/deploy/board pipeline
        ├── model-workflow/SKILL.md     <- New model decision tree + recipes
        ├── relax-passes/SKILL.md      <- Writing Relax/TIR compiler passes
        └── tvm-027-api/SKILL.md       <- TVM 0.23 -> 0.27 API changes
```

## Install

`skills-c7x/` is a directory of the TVM repository, so a checkout of the
repo already contains it.

```bash
# Register this directory as a plugin marketplace
claude plugin marketplace add /path/to/tvm/skills-c7x

# Install the plugin
claude plugin install relax-c7x@skills-c7x

# Activate (no restart needed)
/reload-plugins
```

## Usage

Skills trigger automatically based on context (e.g. mentioning
"c_static_lib", "firmware", "MMALIB"). They can also be invoked
explicitly:

```
/relax-c7x:cstatic
/relax-c7x:firmware
/relax-c7x:model-workflow
```

## Updating

The skills change with the code, so they update when your TVM checkout
does (`git pull`, a rebase or a branch switch). Reload them in Claude Code:

```
/reload-plugins
```

No reinstall needed — the plugin is installed from a local directory,
so `/reload-plugins` picks up all changes (new skills, edits, removals)
in the current session.

If the plugin was renamed or `plugin.json` changed structurally,
reinstall:
```bash
/plugin install relax-c7x@skills-c7x
/reload-plugins
```
