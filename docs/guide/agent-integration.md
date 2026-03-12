# Agent Integration

Squeez works with any LLM coding agent that can run shell commands.

## Claude Code

Add this to your project's `CLAUDE.md` (or `~/.claude/CLAUDE.md` for global):

```
Always when you invoke a shell command, pipe it through `squeez` and tell exactly what you want to know.

Examples:
- `bun test 2>&1 | squeez "did the tests pass?"`
- `git log --oneline -50 | squeez "find the commit that broke CSRF"`
- `cat src/auth/middleware.py | squeez "find the referer validation logic"`

Do NOT use squeez when:
- You need exact, uncompressed output (e.g. writing a patch)
- The command is interactive
```

## Codex CLI

Add equivalent instructions to your Codex configuration or system prompt.

## OpenCode

Add to your OpenCode instructions file.

## General pattern

The integration pattern is the same for any agent:

1. Install squeez and configure a backend (local model or API)
2. Instruct the agent to pipe shell output through `squeez "what to look for"`
3. The agent gets compressed output with only the relevant evidence block

## When NOT to use squeez

- **Writing patches** — the agent needs exact file contents
- **Interactive commands** — squeez reads from stdin
- **Short output** — no point compressing 5 lines
- **Debugging squeez itself** — avoid infinite loops
