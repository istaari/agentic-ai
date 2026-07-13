# Claude Code Cheatsheet

## Installation

```bash
curl -fsSL https://claude.ai/install.sh | bash   # macOS / Linux / WSL
irm https://claude.ai/install.ps1 | iex           # Windows PowerShell
brew install --cask claude-code                   # Homebrew
winget install Anthropic.ClaudeCode               # WinGet
```

---

## CLI — Starting a Session

```bash
claude                              # interactive session
claude "prompt"                     # start with a prompt
claude -p "prompt"                  # print mode: run & exit (CI/CD)
claude -c                           # continue last session
claude -r "name"                    # resume named session
claude -n "name"                    # name the session
claude -w                           # run in isolated git worktree
claude --from-pr <#>                # resume a PR-linked session
```

### Common Flags

```bash
--model sonnet|opus|haiku|fable
--effort low|medium|high|xhigh|max
--permission-mode default|acceptEdits|plan|auto|bypassPermissions
--add-dir <path>                    # grant access outside CWD
--allowedTools "Edit,Read"          # whitelist tools
--disallowedTools "Bash(rm:*)"      # block tools/patterns
--tools "Bash,Edit"                 # restrict to these tools
--max-turns <n>                     # cap agentic iterations
--max-budget-usd <n>                # cap spend
--output-format text|json|stream-json
--system-prompt "..."               # override system prompt
--chrome                            # enable browser automation
--verbose / --debug                 # detailed / debug logging
--dangerously-skip-permissions      # auto-accept everything (risky)
```

### Piping

```bash
tail -200 app.log | claude -p "find errors"
git diff main     | claude -p "review for bugs"
cat file.ts       | claude -p "explain this"
```

---

## CLI Subcommands

```bash
claude update                       # update to latest version
claude doctor                       # setup diagnostics
claude --version                    # show version

# auth
claude auth login | logout | status

# config
claude config list                  # show all settings
claude config get <key>
claude config set <key> <value>
claude config add <key> <value>     # append to an array setting

# MCP servers
claude mcp add <name> <command>     # register a server
claude mcp list                     # list configured servers

# background agents
claude agents                       # list running sessions
claude attach <id>                  # reattach
claude logs <id>                    # view output
claude stop <id>                    # stop

claude project purge [path]         # delete all local state for a project
```

---

## Session Management

| Command | Description |
|---------|-------------|
| `/clear` | Fresh conversation, keep project memory |
| `/compact [hint]` | Summarize context to free space |
| `/resume [session]` | Resume a previous session |
| `/rewind` | Roll back to a checkpoint (or `Esc Esc`) |
| `/branch [name]` | Branch the conversation |
| `/rename <name>` | Rename the session |
| `/context` | Visualize context usage |
| `/export [file]` | Export conversation as text |
| `/teleport` | Pull a web session into the terminal |

---

## Model & Configuration

| Command | Description |
|---------|-------------|
| `/model [model]` | Switch model (picker if no arg) |
| `/effort [level]` | Set reasoning effort |
| `/fast` | Toggle fast mode (faster, higher cost) |
| `/config` | Open settings UI |
| `/permissions` | View / update tool permissions |
| `/keybindings` | Edit `~/.claude/keybindings.json` |
| `/vim` | Toggle vim editing mode |
| `/terminal-setup` | Configure multiline input |

---

## Development

| Command | Description |
|---------|-------------|
| `/init` | Generate `CLAUDE.md` for the project |
| `/memory` | View / edit memory files |
| `/plan [description]` | Enter plan mode (read-only) |
| `/agents` | Manage sub-agents |
| `/mcp` | Manage / check MCP servers |
| `/run` | Launch the project app |
| `/verify` | Verify a change works end-to-end |
| `/debug [description]` | Systematic troubleshooting |

---

## Code Review

| Command | Description |
|---------|-------------|
| `/code-review [level] [--fix]` | Review diff for bugs & cleanups |
| `/simplify` | Cleanup-only review, applies fixes |
| `/security-review` | Scan changes for vulnerabilities |
| `/review [PR#]` | Review a GitHub PR |
| `/diff` | Interactive diff viewer |
| `/pr_comments` | View GitHub PR feedback |
| `/install-github-app` | Set up automated PR reviews |

---

## Work & Background

| Command | Description |
|---------|-------------|
| `/tasks` | Monitor background tasks / agents |
| `/background "prompt"` | Detach session as background agent |
| `/loop [interval] [prompt]` | Run a prompt on a schedule |
| `/batch <instruction>` | Large-scale parallel refactor |
| `/goal [condition]` | Work autonomously until a condition is met |
| `/workflows` | Monitor dynamic workflows |

---

## Utilities

| Command | Description |
|---------|-------------|
| `/help` | List all commands |
| `/status` | Version, model, account |
| `/usage` · `/cost` · `/stats` | Token usage & cost |
| `/insights` | Usage analytics report |
| `/copy [N]` | Copy last response(s) to clipboard |
| `/doctor` | Health diagnostics |
| `/btw <question>` | Side question (no history impact) |
| `/exit` | End session (or `Ctrl+D`) |

---

## Input Syntax

```
@path/to/file          Attach file/dir to context (autocomplete after @)
@agent-name            Invoke a custom sub-agent
!npm test              Shell mode: run command, add output to context
/command               Slash command or skill
```

---

## Permission Modes (`Shift+Tab` cycles)

| Mode | Editing | Execution |
|------|---------|-----------|
| `default` | asks | asks |
| `acceptEdits` | auto | asks |
| `plan` | ❌ read-only | ❌ read-only |
| `auto` | classifier decides | classifier decides |
| `bypassPermissions` | auto | auto (CI/CD only) |

---

## `.claude/` Directory Structure

### Project level — `.claude/` (in your repo)

```
.claude/
├── CLAUDE.md            # project memory & conventions
├── settings.json        # shared team settings + hooks (commit)
├── settings.local.json  # personal overrides (gitignore)
├── commands/            # custom slash commands (*.md)
├── skills/              # skills — folders each with SKILL.md
├── agents/              # custom sub-agents (*.md)
├── hooks/               # event scripts (*.sh / *.ps1)
└── workflows/           # dynamic workflow scripts (*.js)
```

### Global level — `~/.claude/` (all projects)

```
~/.claude/
├── CLAUDE.md            # global personal memory
├── settings.json        # global settings
├── keybindings.json     # custom shortcuts
├── commands/            # personal global commands
└── skills/              # personal global skills
```

### Settings precedence (highest wins)

```
1. /etc/claude-code/managed-settings.json   # enterprise-managed
2. .claude/settings.local.json              # project personal
3. .claude/settings.json                    # project shared
4. ~/.claude/settings.json                  # user global
```

---

## The Big 5 — Extension System

| Extension | Location | What it does |
|-----------|----------|--------------|
| **CLAUDE.md** | `.claude/CLAUDE.md` | Project memory: style, architecture, commands. Made via `/init`. |
| **Slash commands** | `.claude/commands/*.md` | Filename → command (`review.md` → `/review`). |
| **Skills** | `.claude/skills/<name>/SKILL.md` | Auto-invoked knowledge modules. |
| **Sub-agents** | `.claude/agents/*.md` | Specialized isolated Claude instances (`@name`). |
| **MCP servers** | `claude mcp add …` | Connect external tools/data sources. |

### Custom command / agent frontmatter

```markdown
---
name: my-agent
description: Use when <trigger condition>
model: sonnet
tools: Read, Write, Edit, Bash
---
Instructions here…   ($ARGUMENTS / $1 $2 for command args)
```

---

## Hooks — Event Automation

Scripts in `.claude/hooks/` triggered on events. Exit `0` = continue, `2` = block.

| Event | Fires |
|-------|-------|
| `PreToolUse` / `PostToolUse` | before / after a tool runs |
| `UserPromptSubmit` | before a prompt is processed |
| `SessionStart` / `SessionEnd` | session begins / ends |
| `Stop` | response completes |
| `PreCompact` | before context compression |
| `Notification` | when a notification is sent |

Configure in `settings.json` under a `"hooks"` key, or use `/hooks` to inspect.
