"""Calix — agent-driven chess engine.

The package layout:
    agent.py    — Module Selector Agent: AgentContext -> EngineConfig.
    registry.py — scanner that walks src/ and builds ModuleDescriptor[].
    modes.py    — three named info-mode presets (blind / aware / autonomous).
    main.py     — CLI + UCI entrypoint.
    modules/    — destination for stubs synthesized in autonomous mode.
"""
