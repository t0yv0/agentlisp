# AgentLisp - A Program-Guided Chatbot

AgentLisp is a Lisp interpreter where program execution is driven by a chatbot interface. Both you and Claude can advance the program state using a `/run` tool.

## Features

- **Small-step evaluator**: Programs execute one step at a time
- **LLM integration**: Programs can call `(ask ...)` to query Claude
- **Interactive I/O**: Programs can read from stdin and write to stdout
- **Tool-based execution**: Both user and Claude can advance the program using `/run`

## Installation

```bash
# Install dependencies
uv sync

# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-key-here"
```

## Usage

```bash
./agentlisp.py <program.alisp>
```

### Chatbot Commands

- `/run [steps]` - Run the program for N steps (default: 1)
- `/run -1` - Run until the program blocks on I/O or completes
- `/quit` - Exit the chatbot
- Any other text - Chat with Claude about the program

### Example Session

```bash
$ ./agentlisp.py test.alisp
AgentLisp Chatbot: Loaded test.alisp
============================================================
You are chatting with Claude, who can help you run the program.
Commands:
  /run [steps]  - Run the program (steps: number or -1 for until blocked)
  /quit         - Exit the chatbot
============================================================
Initial state: Computing (step 0)

You: /run -1
Step 0: Computing...
Step 1: Computing...
Step 2: Writing to output: 'Hello from AgentLisp!'
[Program output]: Hello from AgentLisp!
...
Program completed with result:
Current state: Program completed with result:

You: /quit
Goodbye!
```

## Language Syntax

### Function Definitions

```lisp
(defun function-name (arg1 arg2 arg3)
  body-expr)

(defun main ()
  body-expr)
```

Every program must have a `main` function that takes no arguments.

### Expressions

- **Variables**: `some-var`
- **Integers**: `123`
- **Strings**: `"foobar"`
- **Function calls**: `(func-to-call arg1 arg2 arg3)`
- **Conditionals**: `(if condition then-expr else-expr)`
- **Let bindings**:
  ```lisp
  (let ((var1 expr1)
        (var2 expr2))
    body-expr)
  ```

### Primitive Forms

- `(write expr)` - Writes text to output, evaluates to `""`
- `(read)` - Reads text from input
- `(tell expr)` - Appends message to LLM conversation, evaluates to `""`
- `(ask expr)` - Asks LLM a question, evaluates to the response

## Example Programs

### test.alisp - Simple Hello World

```lisp
(defun main ()
  (let ((greeting "Hello from AgentLisp!"))
    (write greeting)))
```

### chatbot.alisp - LLM Greeting

```lisp
(defun greet-user ()
  (let ((intro "You are a helpful assistant. Greet the user warmly."))
    (let ((response (ask intro)))
      (write response))))

(defun main ()
  (greet-user))
```

### interactive.alisp - Full Interactive Demo

```lisp
(defun get-user-name ()
  (let ((prompt "What is your name?"))
    (let ((_ (write prompt)))
      (read))))

(defun greet-user (name)
  (let ((greeting-prompt "You are a friendly assistant. The user's name is: "))
    (let ((full-prompt (ask (write greeting-prompt))))
      (ask name))))

(defun main ()
  (let ((name (get-user-name)))
    (let ((greeting (greet-user name)))
      (write greeting))))
```

## Architecture

- **ast.py** - Typed AST definitions
- **parser.py** - Parser from text to AST
- **eval.py** - Small-step evaluator with system calls
- **agentlisp.py** - Chatbot REPL with tool support

## Type Checking

The project uses strict mypy type checking:

```bash
uv run mypy *.py
```

## How It Works

1. The chatbot loads your `.alisp` program and creates an initial state
2. You can chat with Claude about the program or use `/run` to execute it
3. Claude can also call the `run` tool to advance the program
4. When the program calls primitives like `(read)`, `(write)`, `(tell)`, or `(ask)`, the chatbot handles the I/O
5. The program continues until it completes or blocks on I/O

This creates a unique experience where the program execution is guided by conversation!
