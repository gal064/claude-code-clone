from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import dotenv
import logfire
from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    ModelRetry,
    PromptedOutput,
    RunContext,
    format_as_xml,
)
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.toolsets import FunctionToolset, ToolsetTool, WrapperToolset

# Optional rich console for nicer output
from rich import print

# get .env
dotenv.load_dotenv()


# -------------------------
# Agent Dependencies
# -------------------------
class Todo(BaseModel):
    title: str
    status: Literal["active", "in progress", "completed"]


class Bug(BaseModel):
    description: str
    reproduce_steps: str
    severity: Literal["critical", "high", "medium", "low"]


class QAResult(BaseModel):
    result: Literal["success", "fail"]
    breaking_bugs: list[Bug]
    summary: str


@dataclass
class Deps:
    task: str
    cwd: Path
    todos: list[Todo] = field(default_factory=list)
    background_processes: list[subprocess.Popen] = field(default_factory=list)


def resolve_under_cwd(cwd: Path, rel: str) -> Path:
    """Resolve a relative path under cwd and prevent escaping with .."""
    base = cwd.resolve()
    p = (base / rel).resolve()
    if not str(p).startswith(str(base)):
        raise ValueError(f"Path escapes cwd: {rel}")
    return p


def format_content_preview(content: str, max_lines: int = 10) -> str:
    """Format content for nice display, truncating if too long."""
    lines = content.split("\n")
    if len(lines) <= max_lines:
        return content

    preview_lines = lines[: max_lines // 2]
    preview_lines.append(f"... ({len(lines) - max_lines} more lines) ...")
    preview_lines.extend(lines[-max_lines // 2 :])
    return "\n".join(preview_lines)


def read_file(ctx: RunContext[Deps], path: str) -> str:
    """Read a UTF-8 text file under cwd and return its content.

    Args:
        path: The path to the file to read.
    """
    try:
        p = resolve_under_cwd(ctx.deps.cwd, path)
        data = p.read_text(encoding="utf-8")
        return data
    except Exception as e:
        raise ModelRetry(f"Failed to read file '{path}': {str(e)}")


def write_file(ctx: RunContext[Deps], path: str, content: str) -> dict:
    """Create or overwrite a UTF-8 text file under cwd with the given content.
    This tool requires user approval.
    Returns a summary with bytes written.

    Args:
        path: The path to the file to write.
        content: The content to write to the file.
    """
    try:
        p = resolve_under_cwd(ctx.deps.cwd, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        b = content.encode("utf-8")
        p.write_bytes(b)
        return {"path": str(p), "bytes": len(b), "action": "wrote"}
    except Exception as e:
        raise ModelRetry(f"Failed to write file '{path}': {str(e)}")


def edit_file(
    ctx: RunContext[Deps], path: str, old_string: str, new_string: str
) -> dict:
    """Edit a file by replacing the first exact occurrence of old_string with new_string.
    The match must be exact including new lines and indentation.
    Returns a summary with replacements count.

    Args:
        path: The path to the file to edit.
        old_string: The string to replace. Must be exact including new lines and indentation.
        new_string: The string to replace with.
    """
    try:
        p = resolve_under_cwd(ctx.deps.cwd, path)
        content = p.read_text(encoding="utf-8")
        if old_string not in content:
            raise ModelRetry(f"String not found in file '{path}': {old_string[:50]}...")
        updated = content.replace(old_string, new_string, 1)
        p.write_text(updated, encoding="utf-8")
        return {"path": str(p), "replacements": 1}
    except ModelRetry:
        raise
    except Exception as e:
        raise ModelRetry(f"Failed to edit file '{path}': {str(e)}")


def bash(
    ctx: RunContext[Deps], cmd: str, timeout: int = 60, background: bool = False
) -> dict:
    """Run a bash command in cwd and return exit_code, stdout, stderr.
    This tool requires user approval.
    When running commands that might require interactivity (like npm init), make sure to pass proper arguments to the command to prevent it from hanging. For example, npm init --yes will run the command without any interactivity.

    Use proper timeouts. If a command is expected to take a long time, use a longer timeout.

    Args:
        cmd: The bash command to run.
        timeout: The timeout in seconds for the command to run. Defaults to 60 seconds.
        background: If True, run the command in the background and return immediately. The process will be killed when the script exits.
    """
    try:
        if background:
            # Start process in background
            proc = subprocess.Popen(
                cmd,
                cwd=str(ctx.deps.cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
            )
            # Track the process for cleanup
            ctx.deps.background_processes.append(proc)
            return {
                "pid": proc.pid,
                "status": "running_in_background",
                "message": f"Command started in background with PID {proc.pid}",
            }
        else:
            # Run normally with timeout
            proc = subprocess.run(
                cmd,
                cwd=str(ctx.deps.cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                shell=True,
            )
            return {
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
    except subprocess.TimeoutExpired:
        raise ModelRetry(f"Command timed out after {timeout} seconds: {cmd}")
    except Exception as e:
        raise ModelRetry(f"Failed to execute command '{cmd}': {str(e)}")


def todo_list(todos: list[Todo]) -> list[Todo]:
    """Set or update the current plan as a list of todos and return it back.
    Status is one of: active, in progress, completed. Use this tool for complex tasks to plan out the steps, and update your progress as you go.

    Args:
        todos: The list of todos to set or update.
    """
    return todos


def change_working_directory(ctx: RunContext[Deps], path: str) -> dict:
    """Change the current working directory to the specified path.
    The path must be under the current cwd to prevent escaping.

    Args:
        path: The path to change to (relative to current cwd).
    """
    try:
        new_cwd = resolve_under_cwd(ctx.deps.cwd, path)
        if not new_cwd.exists():
            raise ModelRetry(f"Directory does not exist: {path}")
        if not new_cwd.is_dir():
            raise ModelRetry(f"Path is not a directory: {path}")

        ctx.deps.cwd = new_cwd
        return {
            "old_cwd": str(ctx.deps.cwd),
            "new_cwd": str(new_cwd),
            "action": "changed_directory",
        }
    except Exception as e:
        raise ModelRetry(f"Failed to change directory to '{path}': {str(e)}")


# -------------------------
# Toolset Wrapper with Logging and Approval
# -------------------------


class ApprovalToolset(WrapperToolset[Deps]):
    """A toolset wrapper that asks for approval for certain tools."""

    def __init__(self, wrapped: FunctionToolset[Deps]):
        super().__init__(wrapped)
        self.require_approval = {"bash", "write_file", "edit_file"}

    def _print_tool_start(self, name: str, tool_args: dict[str, Any]):
        """Print formatted tool execution start."""
        print(f"🔧 [bold blue]Tool Call[/]: [bold]{name}[/]")
        print(f"📝 [dim]Args:[/] {tool_args}")

    def _get_user_approval(self, name: str, tool_args: dict[str, Any]) -> bool:
        """Ask user for approval to execute a tool."""
        print("\n⚠️  [yellow]Approval Required[/]")
        print(f"🔧 Tool: [bold]{name}[/]")

        # Format args nicely for different tool types
        if name == "write_file":
            path = tool_args.get("path", "")
            content = tool_args.get("content", "")
            print(f"📁 Path: [cyan]{path}[/]")
            print(f"📝 Content ({len(content.split())} lines):")
            print(f"[dim]```[/]\n{format_content_preview(content)}\n[dim]```[/]")
        elif name == "edit_file":
            path = tool_args.get("path", "")
            old_string = tool_args.get("old_string", "")
            new_string = tool_args.get("new_string", "")
            print(f"📁 Path: [cyan]{path}[/]")
            print("🔍 Find:")
            print(
                f"[dim]```[/]\n[red]{format_content_preview(old_string, 5)}[/red]\n[dim]```[/]"
            )
            print("🔄 Replace with:")
            print(
                f"[dim]```[/]\n[green]{format_content_preview(new_string, 5)}[/green]\n[dim]```[/]"
            )
        else:
            print(f"📝 Args: {tool_args}")

        while True:
            response = (
                input(
                    "\n🤔 Approve execution? (Y)es/(n)o/(s)kip for this tool/(sa)skip all approvals:"
                )
                .lower()
                .strip()
            )
            if response in ["y", "yes", ""]:
                return True
            elif response in ["n", "no"]:
                return False
            elif response in ["s", "skip", "skip for this tool"]:
                self.require_approval.remove(name)
                return True
            elif response in [
                "sa",
                "skip all approvals",
                "skip all approvals for all tools",
            ]:
                self.require_approval = set()
                return True
            else:
                print("❓ Please enter 'y' or 'n'")

    def _get_new_instructions(self) -> str:
        """Get new instructions from user when they decline approval."""
        print("\n📋 [yellow]Please provide new instructions:[/]")
        new_instructions = input("💭 Instructions: ").strip()
        return new_instructions

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext, tool: ToolsetTool
    ) -> Any:
        """Override tool execution with approval logic."""
        # Print formatted start
        self._print_tool_start(name, tool_args)

        # Check if approval is needed
        if name in self.require_approval:
            approved = self._get_user_approval(name, tool_args)
            if not approved:
                new_instructions = self._get_new_instructions()
                error_msg = f"Tool execution declined by user. New instructions: {new_instructions}"
                return error_msg

        result = await super().call_tool(name, tool_args, ctx, tool)
        return result


# -------------------------
# Agent definition
# -------------------------
CODING_AGENT_INSTRUCTIONS = """You are a senior full stack engineer who writes correct, minimal, and well structured code. 
You can design and implement end to end apps. Always produce a concrete plan first using the todo_list tool. 
Prefer iterative edits: read files before editing, summarize what will change, then edit. 

Your code repository is a fullstack app that suppose to work fully end to end.

Use the tools provided to you to complete the task. Be diligent but keep code simple.

# Workflow:
1. Always start your task by running ls -la to see the files in the current directory.
2. Read existing relevant files to get a sense of the codebase.
3. Plan your task by using the todo_list tool.
4. Execute on your todos and iterate.

# Guidelines
1. Unless explicitly instructed otherwise, use NPM,Nextjs and Tailwindcss, Prisma and SQLite.
2. If working of an empty project, use NextJS CLI to initialize the project. Make sure to pass --yes both after npx and after the NextJS CLI command to skip the interactive. e.g. npx --yes create-next-app@latest myapp --yes
2. Use Bash tools (ls, grep, find, etc.) to explore the codebase if necessary.
3. Keep things simple and clean.
4. Use `find` to locate files by name or extension.
5. Use `grep` to search file contents if you know a function or keyword but not the file.
6. When relevant, always build the project when you are done to make sure it works.
7. When done, start development servers in the background to test functionality. Use the background=True parameter in the bash tool to run servers without blocking. Explicitly define a random port for the server to avoid conflicts. For example `bash("PORT=3041 npm run dev", background=True)`

In your final response, explain your user facing implementation you have done (no need to mention code or any technical details) and the endpoint (eg localhost:3000) to test the app.
"""

# The agent can use any supported model provider via MODEL env var.
MODEL = os.getenv("MODEL", "claude-sonnet-4-0")

# Create toolset with logging and approval wrapper
coding_toolset = FunctionToolset(
    [read_file, write_file, edit_file, bash, todo_list, change_working_directory]
)
approval_toolset = ApprovalToolset(coding_toolset)

coding_agent = Agent(
    MODEL,
    deps_type=Deps,
    toolsets=[approval_toolset],
    instructions=CODING_AGENT_INSTRUCTIONS,
    retries=30,
    model_settings=AnthropicModelSettings(
        anthropic_thinking={"type": "enabled", "budget_tokens": 32000},
        extra_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"},
    ),
)

QA_AGENT_INSTRUCTIONS = """You are an expert QA engineer specialized in critical bug detection using browser automation.

Your task is to test a web application endpoint and identify ONLY breaking/critical bugs that prevent core functionality.

Focus on:
- Critical functionality failures (login, navigation, forms, core features)
- UI elements that are completely broken or inaccessible


Do NOT report:
- Minor styling issues
- Small UI inconsistencies  
- Performance issues unless they completely break the app
- Accessibility issues unless they make the app unusable

Start by creating a test plan using the todo_list tool and update it as you go."""

# Create Playwright MCP server connection
playwright_mcp_server = MCPServerStdio(
    "npx",
    args=["@playwright/mcp@latest"],
)

qa_agent = Agent(
    MODEL,
    output_type=PromptedOutput(QAResult),
    toolsets=[playwright_mcp_server],
    tools=[todo_list],
    instructions=QA_AGENT_INSTRUCTIONS,
    retries=10,
    model_settings=AnthropicModelSettings(
        anthropic_thinking={"type": "enabled", "budget_tokens": 32000},
        extra_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"},
    ),
)


async def agent_loop(task: str, cwd: Path) -> QAResult:
    logfire.configure(token=os.getenv("LOGFIRE_API_KEY"), scrubbing=False)
    logfire.instrument_pydantic_ai()

    deps = Deps(task=task, cwd=cwd)

    def cleanup_deps_processes():
        """Clean up processes tracked in deps."""
        for proc in deps.background_processes:
            try:
                if proc.poll() is None:  # Process is still running
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            except Exception:
                pass  # Process might already be dead

    # Register cleanup for this agent's processes
    atexit.register(cleanup_deps_processes)

    max_attempts = 3
    attempts = 0
    while attempts < max_attempts:
        try:
            attempts += 1
            # First run - may produce approvals
            with logfire.span(f"Coding Agent:{task}"):
                coding_result = await coding_agent.run(
                    task,
                    deps=deps,
                )

            with logfire.span(f"QA Agent:{task}"):
                qa_input = {
                    "task": task,
                    "engineering_implementation_note": coding_result.output,
                }
                qa_result = await qa_agent.run(format_as_xml(qa_input))
                qa_output = qa_result.output

                if qa_output.result == "success":
                    return qa_output

        finally:
            # Clean up background processes before exiting
            cleanup_deps_processes()

    return qa_output


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full stack coding agent with approvals")
    p.add_argument(
        "--task",
        required=True,
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    print(Path.cwd())
    final = asyncio.run(agent_loop(task=args.task, cwd=Path.cwd()))
    print("\n=== Agent Output ===\n")
    print(final)
