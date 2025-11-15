"""Tests for the small-step evaluator."""

import unittest
from typing import Callable

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
    Env,
    create_initial_state,
    step,
    step_with_syscall,
)
from parser import parse_program


class TestBasicEvaluation(unittest.TestCase):
    """Test basic expression evaluation."""

    def test_integer_literal(self) -> None:
        """Test that integer literals evaluate to themselves."""
        func = FunctionDef("main", [], IntLiteral(42))
        program = Program([func])
        state = create_initial_state(program)

        # Should step to Done with value 42
        next_state = step(state)
        self.assertIsInstance(next_state, Done)
        assert isinstance(next_state, Done)
        self.assertEqual(next_state.value, 42)

    def test_string_literal(self) -> None:
        """Test that string literals evaluate to themselves."""
        func = FunctionDef("main", [], StringLiteral("hello"))
        program = Program([func])
        state = create_initial_state(program)

        next_state = step(state)
        self.assertIsInstance(next_state, Done)
        assert isinstance(next_state, Done)
        self.assertEqual(next_state.value, "hello")

    def test_variable_lookup(self) -> None:
        """Test variable lookup in let expressions."""
        # (let ((x 10)) x)
        let_expr = LetExpr([LetBinding("x", IntLiteral(10))], Variable("x"))
        func = FunctionDef("main", [], let_expr)
        program = Program([func])
        state: State | None = create_initial_state(program)

        # Run until done
        while state is not None and not isinstance(state, Done):
            state = step(state)

        self.assertIsInstance(state, Done)
        assert isinstance(state, Done)
        self.assertEqual(state.value, 10)


class TestIfExpressions(unittest.TestCase):
    """Test if expression evaluation."""

    def test_if_true_branch(self) -> None:
        """Test that if takes the then branch when condition is truthy."""
        # (if 1 42 99)
        if_expr = IfExpr(IntLiteral(1), IntLiteral(42), IntLiteral(99))
        func = FunctionDef("main", [], if_expr)
        program = Program([func])
        state: State | None = create_initial_state(program)

        # Run until done
        while state is not None and not isinstance(state, Done):
            state = step(state)

        self.assertIsInstance(state, Done)
        assert isinstance(state, Done)
        self.assertEqual(state.value, 42)

    def test_if_false_branch(self) -> None:
        """Test that if takes the else branch when condition is falsy."""
        # (if 0 42 99)
        if_expr = IfExpr(IntLiteral(0), IntLiteral(42), IntLiteral(99))
        func = FunctionDef("main", [], if_expr)
        program = Program([func])
        state: State | None = create_initial_state(program)

        # Run until done
        while state is not None and not isinstance(state, Done):
            state = step(state)

        self.assertIsInstance(state, Done)
        assert isinstance(state, Done)
        self.assertEqual(state.value, 99)

    def test_if_empty_string_is_falsy(self) -> None:
        """Test that empty string is falsy."""
        # (if "" 42 99)
        if_expr = IfExpr(StringLiteral(""), IntLiteral(42), IntLiteral(99))
        func = FunctionDef("main", [], if_expr)
        program = Program([func])
        state: State | None = create_initial_state(program)

        # Run until done
        while state is not None and not isinstance(state, Done):
            state = step(state)

        self.assertIsInstance(state, Done)
        assert isinstance(state, Done)
        self.assertEqual(state.value, 99)

    def test_if_nonempty_string_is_truthy(self) -> None:
        """Test that non-empty string is truthy."""
        # (if "hello" 42 99)
        if_expr = IfExpr(StringLiteral("hello"), IntLiteral(42), IntLiteral(99))
        func = FunctionDef("main", [], if_expr)
        program = Program([func])
        state: State | None = create_initial_state(program)

        # Run until done
        while state is not None and not isinstance(state, Done):
            state = step(state)

        self.assertIsInstance(state, Done)
        assert isinstance(state, Done)
        self.assertEqual(state.value, 42)


class TestLetExpressions(unittest.TestCase):
    """Test let expression evaluation."""

    def test_single_binding(self) -> None:
        """Test let with a single binding."""
        # (let ((x 10)) x)
        let_expr = LetExpr([LetBinding("x", IntLiteral(10))], Variable("x"))
        func = FunctionDef("main", [], let_expr)
        program = Program([func])
        state: State | None = create_initial_state(program)

        # Run until done
        while state is not None and not isinstance(state, Done):
            state = step(state)

        self.assertIsInstance(state, Done)
        assert isinstance(state, Done)
        self.assertEqual(state.value, 10)

    def test_multiple_bindings(self) -> None:
        """Test let with multiple bindings."""
        # (let ((x 10) (y 20)) y)
        let_expr = LetExpr(
            [LetBinding("x", IntLiteral(10)), LetBinding("y", IntLiteral(20))],
            Variable("y"),
        )
        func = FunctionDef("main", [], let_expr)
        program = Program([func])
        state: State | None = create_initial_state(program)

        # Run until done
        while state is not None and not isinstance(state, Done):
            state = step(state)

        self.assertIsInstance(state, Done)
        assert isinstance(state, Done)
        self.assertEqual(state.value, 20)

    def test_let_sequential_bindings(self) -> None:
        """Test that let bindings are sequential (later bindings can't see earlier ones)."""
        # This is based on the implementation where bindings are evaluated left-to-right
        # but each in the original environment
        # (let ((x 10) (y x)) y) would fail because x is not in scope for y's value
        # Instead test: (let ((x 10)) (let ((y x)) y))
        inner_let = LetExpr([LetBinding("y", Variable("x"))], Variable("y"))
        outer_let = LetExpr([LetBinding("x", IntLiteral(10))], inner_let)
        func = FunctionDef("main", [], outer_let)
        program = Program([func])
        state: State | None = create_initial_state(program)

        # Run until done
        while state is not None and not isinstance(state, Done):
            state = step(state)

        self.assertIsInstance(state, Done)
        assert isinstance(state, Done)
        self.assertEqual(state.value, 10)


class TestFunctionCalls(unittest.TestCase):
    """Test function call evaluation."""

    def test_nullary_function_call(self) -> None:
        """Test calling a function with no arguments."""
        # (defun foo () 42)
        # (defun main () (foo))
        foo_func = FunctionDef("foo", [], IntLiteral(42))
        main_func = FunctionDef("main", [], FunctionCall("foo", []))
        program = Program([foo_func, main_func])
        state: State | None = create_initial_state(program)

        # Run until done
        while state is not None and not isinstance(state, Done):
            state = step(state)

        self.assertIsInstance(state, Done)
        assert isinstance(state, Done)
        self.assertEqual(state.value, 42)

    def test_unary_function_call(self) -> None:
        """Test calling a function with one argument."""
        # (defun identity (x) x)
        # (defun main () (identity 42))
        identity_func = FunctionDef("identity", ["x"], Variable("x"))
        main_func = FunctionDef("main", [], FunctionCall("identity", [IntLiteral(42)]))
        program = Program([identity_func, main_func])
        state: State | None = create_initial_state(program)

        # Run until done
        while state is not None and not isinstance(state, Done):
            state = step(state)

        self.assertIsInstance(state, Done)
        assert isinstance(state, Done)
        self.assertEqual(state.value, 42)

    def test_binary_function_call(self) -> None:
        """Test calling a function with two arguments."""
        # (defun second (x y) y)
        # (defun main () (second 10 20))
        second_func = FunctionDef("second", ["x", "y"], Variable("y"))
        main_func = FunctionDef(
            "main", [], FunctionCall("second", [IntLiteral(10), IntLiteral(20)])
        )
        program = Program([second_func, main_func])
        state: State | None = create_initial_state(program)

        # Run until done
        while state is not None and not isinstance(state, Done):
            state = step(state)

        self.assertIsInstance(state, Done)
        assert isinstance(state, Done)
        self.assertEqual(state.value, 20)


class TestSystemCalls(unittest.TestCase):
    """Test system call primitives."""

    def test_write_primitive(self) -> None:
        """Test that write creates a WriteCall interop state."""
        # (write "hello")
        write_expr = WriteExpr(StringLiteral("hello"))
        func = FunctionDef("main", [], write_expr)
        program = Program([func])
        state: State | None = create_initial_state(program)

        # Step until we hit the interop state
        while isinstance(state, Computing):
            next_state = step(state)
            if next_state is None:
                break
            state = next_state

        # Should be an Interop state with WriteCall
        self.assertIsInstance(state, Interop)
        assert isinstance(state, Interop)
        self.assertIsInstance(state.syscall, WriteCall)
        assert isinstance(state.syscall, WriteCall)
        self.assertEqual(state.syscall.text, "hello")

        # Continue with syscall handler
        def handler(sc: SysCall) -> str:
            return ""

        state = step_with_syscall(state, handler)
        assert state is not None

        # Should complete with empty string
        while state is not None and not isinstance(state, Done):
            state = step(state)

        self.assertIsInstance(state, Done)
        assert isinstance(state, Done)
        self.assertEqual(state.value, "")

    def test_read_primitive(self) -> None:
        """Test that read creates a ReadCall interop state."""
        # (read)
        read_expr = ReadExpr()
        func = FunctionDef("main", [], read_expr)
        program = Program([func])
        state: State | None = create_initial_state(program)
        assert state is not None

        # Step once should create interop state
        state = step(state)
        assert state is not None

        # Should be an Interop state with ReadCall
        self.assertIsInstance(state, Interop)
        assert isinstance(state, Interop)
        self.assertIsInstance(state.syscall, ReadCall)

        # Continue with syscall handler that returns "user input"
        def handler(sc: SysCall) -> str:
            return "user input"

        state = step_with_syscall(state, handler)
        assert state is not None

        # Should complete with the input value
        while state is not None and not isinstance(state, Done):
            state = step(state)

        self.assertIsInstance(state, Done)
        assert isinstance(state, Done)
        self.assertEqual(state.value, "user input")

    def test_tell_primitive(self) -> None:
        """Test that tell creates a TellCall interop state."""
        # (tell "message")
        tell_expr = TellExpr(StringLiteral("message"))
        func = FunctionDef("main", [], tell_expr)
        program = Program([func])
        state: State | None = create_initial_state(program)

        # Step until we hit the interop state
        while isinstance(state, Computing):
            next_state = step(state)
            if next_state is None:
                break
            state = next_state

        assert state is not None

        # Should be an Interop state with TellCall
        self.assertIsInstance(state, Interop)
        assert isinstance(state, Interop)
        self.assertIsInstance(state.syscall, TellCall)
        assert isinstance(state.syscall, TellCall)
        self.assertEqual(state.syscall.text, "message")

    def test_ask_primitive(self) -> None:
        """Test that ask creates an AskCall interop state."""
        # (ask "question")
        ask_expr = AskExpr(StringLiteral("question"))
        func = FunctionDef("main", [], ask_expr)
        program = Program([func])
        state: State | None = create_initial_state(program)

        # Step until we hit the interop state
        while isinstance(state, Computing):
            next_state = step(state)
            if next_state is None:
                break
            state = next_state

        assert state is not None

        # Should be an Interop state with AskCall
        self.assertIsInstance(state, Interop)
        assert isinstance(state, Interop)
        self.assertIsInstance(state.syscall, AskCall)
        assert isinstance(state.syscall, AskCall)
        self.assertEqual(state.syscall.question, "question")

        # Continue with syscall handler that returns "answer"
        def handler(sc: SysCall) -> str:
            return "answer"

        state = step_with_syscall(state, handler)
        assert state is not None

        # Should complete with the answer
        while state is not None and not isinstance(state, Done):
            state = step(state)

        self.assertIsInstance(state, Done)
        assert isinstance(state, Done)
        self.assertEqual(state.value, "answer")


class TestTestAlisp(unittest.TestCase):
    """Test that test.alisp executes correctly."""

    def test_test_alisp_execution(self) -> None:
        """Test that test.alisp produces the expected output."""
        # Parse test.alisp
        with open("test.alisp", "r") as f:
            program_text = f.read()

        program = parse_program(program_text)
        state: State | None = create_initial_state(program)

        # Track write calls
        write_outputs: list[str] = []

        def syscall_handler(sc: SysCall) -> str:
            if isinstance(sc, WriteCall):
                write_outputs.append(sc.text)
                return ""
            return ""

        # Run until done
        while state is not None and not isinstance(state, Done):
            if isinstance(state, Interop):
                state = step_with_syscall(state, syscall_handler)
            else:
                state = step(state)

        # Should complete with empty string (write returns "")
        self.assertIsInstance(state, Done)
        assert isinstance(state, Done)
        self.assertEqual(state.value, "")

        # Should have written "Hello from AgentLisp!"
        self.assertEqual(write_outputs, ["Hello from AgentLisp!"])


class TestStepByStepEvaluation(unittest.TestCase):
    """Test the small-step evaluation semantics."""

    def test_evaluation_steps_are_deterministic(self) -> None:
        """Test that stepping through evaluation is deterministic."""
        # (let ((x 10)) x)
        let_expr = LetExpr([LetBinding("x", IntLiteral(10))], Variable("x"))
        func = FunctionDef("main", [], let_expr)
        program = Program([func])

        # Run twice and compare
        state1: State | None = create_initial_state(program)
        state2: State | None = create_initial_state(program)

        for _ in range(10):  # Step multiple times
            if state1 is None or isinstance(state1, Done):
                break
            if state2 is None or isinstance(state2, Done):
                break

            state1 = step(state1)
            state2 = step(state2)

            # States should be equal (same type and values)
            self.assertEqual(type(state1), type(state2))

    def test_done_state_returns_none_on_step(self) -> None:
        """Test that stepping a Done state returns None."""
        done_state: State = Done(42)
        next_state = step(done_state)
        self.assertIsNone(next_state)

    def test_interop_state_returns_none_without_handler(self) -> None:
        """Test that stepping an Interop state without handler returns None."""
        continuation: State = Done(42)
        interop_state: State = Interop(WriteCall("test"), continuation)
        next_state = step(interop_state)
        self.assertIsNone(next_state)


class TestErrorConditions(unittest.TestCase):
    """Test error handling in the evaluator."""

    def test_undefined_variable(self) -> None:
        """Test that referencing undefined variable raises error."""
        func = FunctionDef("main", [], Variable("undefined"))
        program = Program([func])
        state: State | None = create_initial_state(program)

        # Should raise RuntimeError when trying to look up undefined variable
        with self.assertRaises(RuntimeError) as context:
            while state is not None and not isinstance(state, Done):
                state = step(state)

        self.assertIn("Undefined variable", str(context.exception))

    def test_undefined_function(self) -> None:
        """Test that calling undefined function raises error."""
        func = FunctionDef("main", [], FunctionCall("undefined", []))
        program = Program([func])
        state: State | None = create_initial_state(program)

        # Should raise RuntimeError when trying to call undefined function
        with self.assertRaises(RuntimeError) as context:
            while state is not None and not isinstance(state, Done):
                state = step(state)

        self.assertIn("Undefined function", str(context.exception))

    def test_wrong_number_of_arguments(self) -> None:
        """Test that calling function with wrong number of args raises error."""
        # (defun foo (x) x)
        # (defun main () (foo))  -- missing argument
        foo_func = FunctionDef("foo", ["x"], Variable("x"))
        main_func = FunctionDef("main", [], FunctionCall("foo", []))
        program = Program([foo_func, main_func])
        state: State | None = create_initial_state(program)

        # Should raise RuntimeError for argument count mismatch
        with self.assertRaises(RuntimeError) as context:
            while state is not None and not isinstance(state, Done):
                state = step(state)

        self.assertIn("expects", str(context.exception))


if __name__ == "__main__":
    unittest.main()
