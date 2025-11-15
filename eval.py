"""Small-step evaluator for the Lisp interpreter."""

from dataclasses import dataclass
from typing import Union, Callable
from lisp_ast import (
    Expr,
    IntLiteral,
    StringLiteral,
    FunctionCall,
    IfExpr,
    Variable,
    LetBinding,
    LetExpr,
    WriteExpr,
    ReadExpr,
    TellExpr,
    AskExpr,
    FunctionDef,
    Program,
)


# Value types that expressions can evaluate to
Value = Union[int, str]


@dataclass(frozen=True)
class Env:
    """Environment mapping variable names to values."""

    bindings: dict[str, Value]
    functions: dict[str, FunctionDef]

    def lookup(self, name: str) -> Value:
        """Look up a variable in the environment."""
        if name not in self.bindings:
            raise RuntimeError(f"Undefined variable: {name}")
        return self.bindings[name]

    def extend(self, name: str, value: Value) -> "Env":
        """Create a new environment with an additional binding."""
        new_bindings = self.bindings.copy()
        new_bindings[name] = value
        return Env(new_bindings, self.functions)

    def extend_many(self, names: list[str], values: list[Value]) -> "Env":
        """Create a new environment with multiple bindings."""
        new_bindings = self.bindings.copy()
        for name, value in zip(names, values):
            new_bindings[name] = value
        return Env(new_bindings, self.functions)

    def get_function(self, name: str) -> FunctionDef:
        """Look up a function definition."""
        if name not in self.functions:
            raise RuntimeError(f"Undefined function: {name}")
        return self.functions[name]


@dataclass(frozen=True)
class ReadCall:
    """System call to read input and bind to a variable."""

    var: str


@dataclass(frozen=True)
class WriteCall:
    """System call to write output."""

    text: str


@dataclass(frozen=True)
class TellCall:
    """System call to append to LLM conversation."""

    text: str


@dataclass(frozen=True)
class AskCall:
    """System call to ask LLM and bind response to a variable."""

    var: str
    question: str


# System call variants
SysCall = Union[ReadCall, WriteCall, TellCall, AskCall]


# Evaluation contexts - represent "holes" in expressions where evaluation is happening
@dataclass(frozen=True)
class IfContext:
    """Context for evaluating an if expression's condition."""

    then_expr: Expr
    else_expr: Expr


@dataclass(frozen=True)
class LetContext:
    """Context for evaluating a let binding's value."""

    var_name: str
    remaining_bindings: list[LetBinding]
    body: Expr


@dataclass(frozen=True)
class WriteContext:
    """Context for evaluating a write expression's argument."""

    pass


@dataclass(frozen=True)
class TellContext:
    """Context for evaluating a tell expression's argument."""

    pass


@dataclass(frozen=True)
class AskContext:
    """Context for evaluating an ask expression's argument."""

    pass


@dataclass(frozen=True)
class FunctionCallContext:
    """Context for evaluating function call arguments."""

    func_name: str
    evaluated_args: list[Value]  # Arguments evaluated so far
    remaining_args: list[Expr]  # Arguments still to evaluate


# Evaluation context variants
Context = Union[
    IfContext,
    LetContext,
    WriteContext,
    TellContext,
    AskContext,
    FunctionCallContext,
]


@dataclass(frozen=True)
class Computing:
    """State representing ongoing computation."""

    env: Env
    expr: Expr
    contexts: list[Context]  # Stack of evaluation contexts


@dataclass(frozen=True)
class Interop:
    """State representing a system call waiting to be handled."""

    syscall: SysCall
    continuation: "State"


@dataclass(frozen=True)
class Done:
    """State representing completed computation."""

    value: Value


# State variants
State = Union[Computing, Interop, Done]


def expr_to_value(expr: Expr) -> Value | None:
    """Convert an expression to a value if it's a literal."""
    if isinstance(expr, IntLiteral):
        return expr.value
    if isinstance(expr, StringLiteral):
        return expr.value
    return None


def is_truthy(value: Value) -> bool:
    """Determine if a value is truthy (non-zero, non-empty string)."""
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value != ""
    return False


def apply_context(env: Env, ctx: Context, value: Value, contexts: list[Context]) -> State:
    """Apply a value to an evaluation context, continuing computation."""
    if isinstance(ctx, IfContext):
        # Condition evaluated, choose branch
        if is_truthy(value):
            return Computing(env, ctx.then_expr, contexts)
        else:
            return Computing(env, ctx.else_expr, contexts)

    if isinstance(ctx, LetContext):
        # Binding value evaluated, extend environment
        new_env = env.extend(ctx.var_name, value)

        if len(ctx.remaining_bindings) == 0:
            # No more bindings, evaluate body
            return Computing(new_env, ctx.body, contexts)
        else:
            # More bindings to process
            next_binding = ctx.remaining_bindings[0]
            remaining_bindings_new = ctx.remaining_bindings[1:]
            let_ctx: Context = LetContext(next_binding.name, remaining_bindings_new, ctx.body)
            return Computing(new_env, next_binding.value, [let_ctx] + contexts)

    if isinstance(ctx, WriteContext):
        # Argument evaluated, perform system call
        continuation = Computing(env, StringLiteral(""), contexts)
        return Interop(WriteCall(str(value)), continuation)

    if isinstance(ctx, TellContext):
        # Argument evaluated, perform system call
        continuation = Computing(env, StringLiteral(""), contexts)
        return Interop(TellCall(str(value)), continuation)

    if isinstance(ctx, AskContext):
        # Question evaluated, perform system call
        temp_var = "__ask_result__"
        # After syscall, we need to look up the variable and continue
        continuation = Computing(
            env.extend(temp_var, ""), Variable(temp_var), contexts
        )
        return Interop(AskCall(temp_var, str(value)), continuation)

    if isinstance(ctx, FunctionCallContext):
        # One argument evaluated, check if more to go
        evaluated = ctx.evaluated_args + [value]

        if len(ctx.remaining_args) == 0:
            # All arguments evaluated, perform call
            func_def = env.get_function(ctx.func_name)

            if len(evaluated) != len(func_def.params):
                raise RuntimeError(
                    f"Function {ctx.func_name} expects {len(func_def.params)} "
                    f"arguments, got {len(evaluated)}"
                )

            new_env = env.extend_many(func_def.params, evaluated)
            return Computing(new_env, func_def.body, contexts)
        else:
            # More arguments to evaluate
            next_arg = ctx.remaining_args[0]
            remaining_args_new = ctx.remaining_args[1:]
            call_ctx: Context = FunctionCallContext(ctx.func_name, evaluated, remaining_args_new)
            return Computing(env, next_arg, [call_ctx] + contexts)

    raise RuntimeError(f"Unknown context type: {type(ctx)}")


def step(state: State) -> State | None:
    """
    Perform one step of evaluation.

    Returns the next state, or None if no further progress can be made
    without external input (should not happen in well-formed programs).
    """
    if isinstance(state, Done):
        # Computation is complete
        return None

    if isinstance(state, Interop):
        # Cannot make progress without handling the system call
        # This should be handled by the caller using step_with_syscall
        return None

    if isinstance(state, Computing):
        env = state.env
        expr = state.expr
        contexts = state.contexts

        # Check if expression is already a value
        value = expr_to_value(expr)
        if value is not None:
            # Expression is a value
            if len(contexts) == 0:
                # No contexts, we're done
                return Done(value)
            else:
                # Pop context and apply value
                ctx = contexts[0]
                remaining_contexts = contexts[1:]
                return apply_context(env, ctx, value, remaining_contexts)

        # Expression is not a value, decompose it

        # Variable reference
        if isinstance(expr, Variable):
            val = env.lookup(expr.name)
            if len(contexts) == 0:
                return Done(val)
            else:
                ctx = contexts[0]
                remaining_contexts = contexts[1:]
                return apply_context(env, ctx, val, remaining_contexts)

        # If expression - evaluate condition
        if isinstance(expr, IfExpr):
            if_ctx: Context = IfContext(expr.then_expr, expr.else_expr)
            return Computing(env, expr.condition, [if_ctx] + contexts)

        # Let expression - evaluate first binding
        if isinstance(expr, LetExpr):
            if len(expr.bindings) == 0:
                # No bindings, evaluate body
                return Computing(env, expr.body, contexts)

            first_binding = expr.bindings[0]
            remaining_bindings = expr.bindings[1:]
            let_ctx: Context = LetContext(first_binding.name, remaining_bindings, expr.body)
            return Computing(env, first_binding.value, [let_ctx] + contexts)

        # Write primitive - evaluate argument
        if isinstance(expr, WriteExpr):
            write_ctx: Context = WriteContext()
            return Computing(env, expr.expr, [write_ctx] + contexts)

        # Read primitive - perform system call immediately
        if isinstance(expr, ReadExpr):
            temp_var = "__read_result__"
            continuation = Computing(
                env.extend(temp_var, ""), Variable(temp_var), contexts
            )
            return Interop(ReadCall(temp_var), continuation)

        # Tell primitive - evaluate argument
        if isinstance(expr, TellExpr):
            tell_ctx: Context = TellContext()
            return Computing(env, expr.expr, [tell_ctx] + contexts)

        # Ask primitive - evaluate argument
        if isinstance(expr, AskExpr):
            ask_ctx: Context = AskContext()
            return Computing(env, expr.expr, [ask_ctx] + contexts)

        # Function call - evaluate arguments left to right
        if isinstance(expr, FunctionCall):
            if len(expr.args) == 0:
                # No arguments, call immediately
                func_def = env.get_function(expr.func_name)

                if len(func_def.params) != 0:
                    raise RuntimeError(
                        f"Function {expr.func_name} expects {len(func_def.params)} "
                        f"arguments, got 0"
                    )

                new_env = Env(env.bindings, env.functions)
                return Computing(new_env, func_def.body, contexts)
            else:
                # Evaluate first argument
                first_arg = expr.args[0]
                remaining_args = expr.args[1:]
                call_ctx: Context = FunctionCallContext(expr.func_name, [], remaining_args)
                return Computing(env, first_arg, [call_ctx] + contexts)

    return None


def step_with_syscall(
    state: State, syscall_handler: Callable[[SysCall], str] | None
) -> State | None:
    """
    Perform one step of evaluation, handling system calls if provided.

    Args:
        state: The current evaluation state
        syscall_handler: Optional function to handle system calls and return results

    Returns:
        The next state, or None if evaluation is complete
    """
    if isinstance(state, Interop):
        if syscall_handler is None:
            # Cannot make progress without handler
            return None

        # Handle the system call
        result = syscall_handler(state.syscall)

        # Update continuation with result
        if isinstance(state.syscall, ReadCall):
            # Read returns the input value - extend environment
            if isinstance(state.continuation, Computing):
                new_env = state.continuation.env.extend(state.syscall.var, result)
                return Computing(
                    new_env, state.continuation.expr, state.continuation.contexts
                )
            return state.continuation

        if isinstance(state.syscall, AskCall):
            # Ask returns the LLM response - extend environment
            if isinstance(state.continuation, Computing):
                new_env = state.continuation.env.extend(state.syscall.var, result)
                return Computing(
                    new_env, state.continuation.expr, state.continuation.contexts
                )
            return state.continuation

        # Write and Tell just continue with empty string result
        return state.continuation

    # Not an interop state, use regular step
    return step(state)


def create_initial_state(program: Program) -> State:
    """Create the initial evaluation state from a program."""
    # Build function environment
    func_map = {func.name: func for func in program.functions}

    # Get main function
    main_func = program.get_main()
    if main_func is None:
        raise RuntimeError("Program must have a main function")

    if len(main_func.params) != 0:
        raise RuntimeError("Main function must take no parameters")

    # Create initial environment and computing state with empty context stack
    env = Env({}, func_map)
    return Computing(env, main_func.body, [])
