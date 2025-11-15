#!/usr/bin/env python3
"""AgentLisp: A program-guided chatbot interpreter."""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import NoReturn, Any, cast

from anthropic import Anthropic
from anthropic.types import MessageParam, ToolUseBlock, TextBlock, ToolParam
from parser import parse_program, ParseError
from eval import (
    State,
    Computing,
    Interop,
    Done,
    SysCall,
    ReadCall,
    WriteCall,
    TellCall,
    AskCall,
    create_initial_state,
    step_with_syscall,
)


# Tool definition for the /run command
RUN_TOOL: ToolParam = {
    "name": "run",
    "description": "Advance the AgentLisp program state by one or more steps. Use this to execute the program and observe its behavior.",
    "input_schema": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "integer",
                "description": "Number of steps to execute (default: 1, use -1 for run until blocked)",
                "default": 1,
            }
        },
    },
}


class ChatbotSession:
    """Manages the chatbot session for an AgentLisp program."""

    def __init__(self, program_path: str, api_key: str) -> None:
        """Initialize the chatbot session."""
        self.program_path = program_path
        self.client = Anthropic(api_key=api_key)
        self.conversation: list[MessageParam] = []
        self.model = "claude-sonnet-4-20250514"

        # Parse the program
        with open(program_path, "r") as f:
            self.program_text = f.read()

        try:
            self.program = parse_program(self.program_text)
        except ParseError as e:
            print(f"Parse error: {e}", file=sys.stderr)
            sys.exit(1)

        # Create initial state
        self.state: State | None = create_initial_state(self.program)
        self.step_count = 0

    def get_state_description(self) -> str:
        """Get a human-readable description of the current state."""
        if self.state is None:
            return "Program terminated unexpectedly"
        elif isinstance(self.state, Done):
            return f"Program completed with result: {self.state.value}"
        elif isinstance(self.state, Interop):
            syscall = self.state.syscall
            if isinstance(syscall, ReadCall):
                return f"Waiting for user input (will bind to variable '{syscall.var}')"
            elif isinstance(syscall, WriteCall):
                return f"Writing to output: {repr(syscall.text)}"
            elif isinstance(syscall, TellCall):
                return f"Adding to conversation: {repr(syscall.text)}"
            elif isinstance(syscall, AskCall):
                return f"Asking LLM: {repr(syscall.question)} (will bind to variable '{syscall.var}')"
            return "Waiting for system call"
        elif isinstance(self.state, Computing):
            return f"Computing (step {self.step_count})"
        return "Unknown state"

    def handle_syscall_interactively(self, syscall: SysCall) -> str:
        """Handle a system call, potentially prompting the user."""
        if isinstance(syscall, ReadCall):
            # Read input from user
            print(f"\n[Program requests input for '{syscall.var}']")
            try:
                user_input = input("> ")
                return user_input
            except EOFError:
                return ""

        elif isinstance(syscall, WriteCall):
            # Write output to stdout
            print(f"\n[Program output]: {syscall.text}")
            return ""

        elif isinstance(syscall, TellCall):
            # Append to LLM conversation
            print(f"\n[Program adds to conversation]: {syscall.text}")
            msg: MessageParam = {"role": "user", "content": syscall.text}
            self.conversation.append(msg)
            return ""

        elif isinstance(syscall, AskCall):
            # Ask the LLM and get response
            print(f"\n[Program asks LLM]: {syscall.question}")
            msg_ask: MessageParam = {"role": "user", "content": syscall.question}
            self.conversation.append(msg_ask)

            # Make LLM call without tools (this is a program-initiated call)
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                messages=self.conversation,
            )

            # Extract text from response
            response_text = ""
            for block in response.content:
                if block.type == "text":
                    response_text += block.text

            # Add assistant's response to conversation
            msg_response: MessageParam = {"role": "assistant", "content": response_text}
            self.conversation.append(msg_response)

            print(f"[LLM responds]: {response_text}")
            return response_text

        return ""

    def execute_run_tool(self, steps: int = 1) -> str:
        """Execute the /run tool: advance the program state."""
        if self.state is None:
            return "Error: Program has terminated unexpectedly"

        if isinstance(self.state, Done):
            return f"Program is already complete with result: {self.state.value}"

        output_lines: list[str] = []
        steps_executed = 0

        # Run until we've executed the requested steps or need to stop
        while self.state is not None and (steps == -1 or steps_executed < steps):
            if isinstance(self.state, Done):
                output_lines.append(f"Program completed with result: {self.state.value}")
                break

            # Check if we're waiting on interop
            if isinstance(self.state, Interop):
                syscall = self.state.syscall
                output_lines.append(f"Step {self.step_count}: {self.get_state_description()}")

                # Handle the system call
                result = self.handle_syscall_interactively(syscall)

                # Create a syscall handler that returns this result
                def handler(sc: SysCall) -> str:
                    return result

                # Step with the handler
                self.state = step_with_syscall(self.state, handler)
                self.step_count += 1
                steps_executed += 1
            else:
                # Regular computation step
                output_lines.append(f"Step {self.step_count}: Computing...")
                self.state = step_with_syscall(self.state, None)
                self.step_count += 1
                steps_executed += 1

        # Add final state
        if self.state is not None:
            output_lines.append(f"Current state: {self.get_state_description()}")

        return "\n".join(output_lines)

    def handle_user_command(self, message: str) -> str | None:
        """Handle user commands like /run. Returns None if not a command."""
        message = message.strip()
        if message.startswith("/run"):
            parts = message.split()
            steps = 1
            if len(parts) > 1:
                try:
                    steps = int(parts[1])
                except ValueError:
                    return "Error: /run expects an integer argument"
            return self.execute_run_tool(steps)
        return None

    def run(self) -> NoReturn:
        """Run the chatbot REPL."""
        print(f"AgentLisp Chatbot: Loaded {self.program_path}")
        print("=" * 60)
        print("You are chatting with Claude, who can help you run the program.")
        print("Commands:")
        print("  /run [steps]  - Run the program (steps: number or -1 for until blocked)")
        print("  /quit         - Exit the chatbot")
        print("=" * 60)
        print(f"Initial state: {self.get_state_description()}")
        print()

        while True:
            # Read user input
            try:
                user_input = input("You: ").strip()
            except EOFError:
                print("\nGoodbye!")
                sys.exit(0)

            if not user_input:
                continue

            # Check for quit command
            if user_input == "/quit":
                print("Goodbye!")
                sys.exit(0)

            # Check for user commands
            command_result = self.handle_user_command(user_input)
            if command_result is not None:
                print(command_result)
                print()
                continue

            # Regular chat message - send to LLM with tools
            msg_user: MessageParam = {"role": "user", "content": user_input}
            self.conversation.append(msg_user)

            # Call LLM with tool support
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                messages=self.conversation,
                tools=[RUN_TOOL],
            )

            # Process response and handle tool calls
            assistant_content: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []

            for block in response.content:
                if isinstance(block, TextBlock):
                    if block.text:
                        print(f"\nAssistant: {block.text}")
                    assistant_content.append({"type": "text", "text": block.text})

                elif isinstance(block, ToolUseBlock):
                    print(f"\n[Assistant calls /run tool with input: {block.input}]")
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

                    # Execute the tool
                    steps_input = block.input.get("steps", 1)
                    steps = int(steps_input) if isinstance(steps_input, (int, float, str)) else 1
                    tool_output = self.execute_run_tool(steps)
                    print(tool_output)

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": tool_output,
                        }
                    )

            # Add assistant message to conversation
            msg_assistant: MessageParam = cast(
                MessageParam,
                {
                    "role": "assistant",
                    "content": assistant_content,
                },
            )
            self.conversation.append(msg_assistant)

            # If there were tool calls, add tool results and get next response
            if tool_results:
                msg_tool: MessageParam = cast(
                    MessageParam, {"role": "user", "content": tool_results}
                )
                self.conversation.append(msg_tool)

                # Get assistant's response after tool execution
                follow_up = self.client.messages.create(
                    model=self.model,
                    max_tokens=8192,
                    messages=self.conversation,
                    tools=[RUN_TOOL],
                )

                # Process follow-up (could have more tool calls)
                follow_up_content: list[dict[str, Any]] = []
                for block in follow_up.content:
                    if isinstance(block, TextBlock):
                        if block.text:
                            print(f"\nAssistant: {block.text}")
                        follow_up_content.append({"type": "text", "text": block.text})

                msg_follow_up: MessageParam = cast(
                    MessageParam,
                    {
                        "role": "assistant",
                        "content": follow_up_content,
                    },
                )
                self.conversation.append(msg_follow_up)

            print()


def main() -> NoReturn:
    """Main entry point for the AgentLisp interpreter."""
    parser = argparse.ArgumentParser(
        prog="agentlisp",
        description="AgentLisp: A program-guided chatbot interpreter",
        epilog="Example: agentlisp chatbot.alisp"
    )
    parser.add_argument(
        "program",
        type=str,
        help="Path to the .alisp program file to execute"
    )

    args = parser.parse_args()
    program_path = args.program

    # Check if file exists
    if not Path(program_path).exists():
        print(f"Error: File not found: {program_path}", file=sys.stderr)
        sys.exit(1)

    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    # Create and run session
    session = ChatbotSession(program_path, api_key)
    session.run()


if __name__ == "__main__":
    main()
