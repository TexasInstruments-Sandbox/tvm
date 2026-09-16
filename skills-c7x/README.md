# skills-c7x

Claude Code plugin marketplace providing TVM Relax C7x DSP skills for
the full compilation-to-deployment stack on TI AM67A (J722S).

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

## Relationship to c7x-optimizer

The `c7x-optimizer` plugin covers low-level kernel micro-optimization
(ISA intrinsics, streaming engine, software pipelining, compiler
tuning). These skills sit one layer above -- system integration,
compilation pipeline, deployment, and offload strategies.

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
        └── relax-passes/SKILL.md      <- Writing Relax/TIR compiler passes
```

## Install

```bash
# Register this directory as a plugin marketplace
claude plugin marketplace add /path/to/skills-c7x

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

Pull the latest changes from the repository and reload:

```bash
cd /path/to/skills-c7x
git pull

# In Claude Code:
/reload-plugins
```

No reinstall needed — the plugin is installed from a local directory,
so `git pull` + `/reload-plugins` picks up all changes (new skills,
edits, removals) in the current session.

If the plugin was renamed or `plugin.json` changed structurally,
reinstall:
```bash
/plugin install relax-c7x@skills-c7x
/reload-plugins
```
