Build a uv-based typed Python project.
Type-checking with mypy.

# ast.py

Define typed AST with key type Expr.


# parser.py

Parse text into structures from `ast.py`.

The language syntax like this. There are N function definitions with a special one called main.

```
(defun function-name (arg1 arg2 arg3)
  body-expr)

(defun main ()
  body-expr)
```

Expressions look like this:

```
some-var
123
"foobar"
(func-to-call arg1 arg2 arg3)
(if expr expr1 expr2)
(let ((some-var expr1)
      (another-var expr2))
  expr)
```

There are some primitive forms:

```
(write expr) ;; writes text to the output and evals to ""
(read)       ;; reads text from the input and evals it
(tell expr)  ;; evalutes expr and appends it as a prompt to the LLM conversation; evaluates to ""
(ask expr)   ;; evalutes expr and poses it as a question to the LLM; evalutes to the LLM response
```

# eval.py

Build a purely functional small-step evaluator.

The State type combines these variants:

  Computing (Env, Expr)
  Interop (SysCall, State)

Where SysCall variants are these:

  Read Var
  Write string
  Tell string
  Ask (Var, string)

The evaluator gives the step function:

  next :: (SysCall -> string)? -> State -> State?

The Computing state may make progress on its own using the normal rules;

If waiting on a system call it needs the caller to interpret it to make progress.


# entry-point

Make an agentlisp script that takes a path to a some.alisp program and executes it.

The result is a CLI or REPL that is a "program-guided chat bot".

Read ANTHROPIC_API_KEY to authenticate. Use ~anthropic~ SDK.

The evaluation rules are as follows:

- there is always a State, starting from evaluating main()
- there is a chat between agent and the user
- user messages and agent messages are added to the conversation
- everything is sent to the LLM
- LLM and user can both call /run tool that runs the evaluator to advance the state
- Perform interop with the user (stdin/stdout) and/or LLM per each SysCall
