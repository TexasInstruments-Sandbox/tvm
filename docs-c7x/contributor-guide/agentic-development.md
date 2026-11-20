# Agentic Development

The repository carries two sets of instructions for AI coding agents:
`AGENTS.md` at the repository root, and the skills in `skills-c7x/`. Both
are plain Markdown, written for agents and readable by people.

## AGENTS.md

`AGENTS.md` holds what applies to every change in the repository: how to
build and test, conventions such as where MMALIB passes are registered and
how commit messages are written. Agents that support the
`AGENTS.md` convention read it at the start of every session.

## Skills

A skill is a Markdown reference for one subsystem, in a `SKILL.md` file
whose frontmatter `description` says when it applies. An agent loads a
skill only when its task matches that description, so the firmware
reference, for example, enters the agent's context only for firmware work.
`AGENTS.md` lists the skills and tells agents to read the matching one
before working in an area.

The skills live in `skills-c7x/relax-c7x/skills/`:

| Skill | Covers | Related pages |
|-------|--------|---------------|
| `build` | TVM core, DSP runtime, firmware, ARM client and DLOAD builds; SDK paths and `--board`/`--ddr`; build failures | [Getting Started](../user-guide/getting-started.md), [Deploying Firmware](../user-guide/deploying-firmware.md) |
| `cstatic` | The `c_static_lib` backend: code generation, target options, weight serialization, pipeline pass order | [C Static Lib Backend](backend/c-static-lib.md), [CodeGenCStaticLib Internals](backend/codegen-cstatic.md) |
| `dsp-runtime` | The DSP runtime library: Model API, memory pools, allocator, KV cache | [Runtime Internals](dsp-runtime/internals.md) |
| `firmware` | `c7x_compute`: DLOAD, IPC, DMA, memory layout, deployment | [Firmware Architecture](firmware/architecture.md), [Firmware Design Deep-Dive](firmware/design-deep-dive.md) |
| `dsp-ops` | Operator kernels, DMA tiling, quantization math, new `call_extern` kernels | [te.extern Lowering](backend/te-extern-lowering.md) |
| `tidl-offload` | TIDL subgraph partitioning, import and bridge generation | [TIDL Integration](backend/tidl-integration.md) |
| `mmalib-offload` | MMALIB QDQ fusion passes, int16 offload, MMALIB wrappers | [MMALIB Integration](backend/mmalib-integration.md) |
| `testing` | Writing and running tests: fixtures, markers, profiling, debugging failures | [Testing Overview](testing/overview.md), [DSP Suite](testing/dsp-suite.md) |
| `model-workflow` | Bringing up a new model: offload strategy, quantization approach, export, first inference | [Quantization](../user-guide/quantization.md), [Compilation](../user-guide/compilation.md) |
| `relax-passes` | Writing Relax and TIR passes: DFPattern, mutators, `call_extern`, pipeline wiring | [Authoring Passes](backend/authoring-passes.md), [Debugging Passes](backend/debugging-passes.md) |
| `tvm-027-api` | API changes from TVM 0.23 to 0.27, for porting or reading older code | — |

### Installing

The skills are packaged as a Claude Code plugin, `relax-c7x`, in a local
plugin marketplace at `skills-c7x/`:

```bash
claude plugin marketplace add /path/to/tvm/skills-c7x
claude plugin install relax-c7x@skills-c7x
```

Then run `/reload-plugins` in Claude Code. The plugin is installed from
your checkout, so after a pull or branch switch `/reload-plugins` picks up
the skills as they are on that branch.

Skills load on their own when a task matches their description. To load
one explicitly, invoke it by name, for example `/relax-c7x:firmware`.
Agents without plugin support can read a `SKILL.md` directly; `AGENTS.md`
points them there.

## Keeping Docs and Skills in Sync

`docs-c7x/` is the source of truth. A skill condenses the matching pages
for an agent and adds procedural detail, such as commands and known failure
modes, but it can fall behind the code. When a change alters behavior that
a page above describes, update the page and the matching skill in the same
commit.

When adding or changing a skill:

- Keep it to one subsystem. Refer to another skill by name
  (`relax-c7x:build`) instead of repeating its content.
- Write the `description` to say when to use the skill and what it is not
  for. Agents choose skills by their descriptions alone.
- List a new skill in `skills-c7x/README.md`, in the skills list in
  `AGENTS.md`, and in the table on this page.
