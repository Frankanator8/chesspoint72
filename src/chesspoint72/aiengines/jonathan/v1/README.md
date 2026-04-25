# Calix

An agent-driven UCI chess engine. Calix doesn't ship a fixed feature set —
at startup (and at every `go` command in non-blind modes) a deterministic
**Module Selector Agent** inspects the registry of available modules, the
current position, and the clock, and decides which capabilities to switch
on at what aggressiveness.

The engine is self-contained inside this folder. Everything outside it is
either an upstream chess primitive (board, search, pruning algorithms,
evaluator) consumed through a small adapter, or a `# @capability:` tag
the registry scanner reads.

## Layout

| File | Role |
| --- | --- |
| `agent.py` | `AgentContext`, `EngineConfig`, `select_modules`, `build_context` |
| `registry.py` | Walks `src/chesspoint72`, returns `ModuleDescriptor[]` |
| `policies.py` | Local stub + captures-first move-ordering policies |
| `main.py` | CLI flag, factory, `CalixController` (UCI loop) |
| `modules/` | Reserved directory; the agent **does not** create stubs here |

## Running

```bash
# UCI stdio (default mode = aware)
python -m chesspoint72.aiengines.jonathan.v1.main --agent-mode aware
```

Available CLI flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--agent-mode {blind,aware,autonomous}` | `aware` | Picks the AgentContext preset |
| `--depth N` | `4` | Default max search depth |
| `--time S` | `5.0` | Default time budget per move (seconds) |

## The three modes

Each mode is a fixed AgentContext preset. The dial that varies between them
is *what the agent gets to see*.

### Blind (`--agent-mode blind`)
- `mode = "minimal"`, no FEN, no clock, no permission to extend.
- Resolves to **raw α-β** with stub eval, stub ordering, no pruning.
- Useful as a known-good baseline and for legality regression testing.

### Aware (`--agent-mode aware`)
- `mode = "standard"`, sees FEN + clock, no permission to extend.
- Material evaluator, captures-first ordering, futility + LMR always on.
- NMP and razoring switch on only when the clock is generous (≥ 30s).
- Endgame FENs disable NMP regardless of the clock (zugzwang protection).
- Tactical FENs lower razoring margins and bump quiescence depth.

### Autonomous (`--agent-mode autonomous`)
- `mode = "full"`, sees FEN + clock, **`can_add_modules = True`**.
- HCE evaluator (all modules), captures-first ordering, every pruning
  technique on with aggressive razoring margins and deeper quiescence.
- All endgame and tactical position adjustments still apply on top.

> **Important:** the `can_add_modules` flag is carried for API parity but
> the agent will *never* synthesise, scaffold, or register new modules at
> runtime. Module additions happen only through human-authored source code.

## How the registry works

`registry.scan_modules()` walks the package tree and looks for files whose
first 40 lines contain one or more `# @capability: <name>` tags. Each
discovered module becomes a `ModuleDescriptor` carrying its name, path,
list of capability strings, and any keyword defaults parsed from its
`default_*_config` factory. Both the agent and the tests consume this
list directly — there is no hardcoded module catalogue.

## Tests

```
pytest tests/test_calix_registry.py \
       tests/test_calix_agent_modes.py \
       tests/test_calix_integration.py
```
