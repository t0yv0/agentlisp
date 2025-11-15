"""Parser for the Lisp-like language."""

from typing import Any
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


class ParseError(Exception):
    """Raised when parsing fails."""
    pass


class Tokenizer:
    """Tokenizes input text into s-expressions."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def peek(self) -> str | None:
        """Peek at current character without consuming it."""
        if self.pos >= len(self.text):
            return None
        return self.text[self.pos]

    def advance(self) -> str | None:
        """Consume and return current character."""
        if self.pos >= len(self.text):
            return None
        char = self.text[self.pos]
        self.pos += 1
        return char

    def skip_whitespace(self) -> None:
        """Skip whitespace and comments."""
        char = self.peek()
        while char is not None and char in " \t\n\r":
            self.advance()
            char = self.peek()

    def read_string(self) -> str:
        """Read a string literal."""
        self.advance()  # Skip opening quote
        chars: list[str] = ['"']  # Start with opening quote
        while True:
            char = self.peek()
            if char is None:
                raise ParseError("Unterminated string literal")
            if char == '"':
                self.advance()
                chars.append('"')  # Add closing quote
                break
            if char == "\\":
                self.advance()
                next_char = self.advance()
                # Store escape sequences as-is so parser can handle them
                chars.append("\\")
                if next_char is not None:
                    chars.append(next_char)
            else:
                chars.append(char)
                self.advance()
        return "".join(chars)

    def read_atom(self) -> str:
        """Read an atom (identifier or number)."""
        chars: list[str] = []
        while True:
            char = self.peek()
            if char is None or char in " \t\n\r()":
                break
            chars.append(char)
            self.advance()
        return "".join(chars)

    def tokenize(self) -> Any:
        """Tokenize into nested lists representing s-expressions."""
        self.skip_whitespace()
        char = self.peek()

        if char is None:
            return None

        if char == "(":
            self.advance()
            items: list[Any] = []
            while True:
                self.skip_whitespace()
                if self.peek() == ")":
                    self.advance()
                    break
                if self.peek() is None:
                    raise ParseError("Unterminated list")
                item = self.tokenize()
                if item is not None:
                    items.append(item)
            return items

        if char == ")":
            raise ParseError("Unexpected closing parenthesis")

        if char == '"':
            return self.read_string()

        return self.read_atom()


def parse_expr(sexp: Any) -> Expr:
    """Parse an s-expression into an Expr."""
    # String literal (already parsed by tokenizer)
    if isinstance(sexp, str) and sexp.startswith('"'):
        # Process escape sequences
        content = sexp[1:-1]  # Strip quotes
        content = content.replace("\\n", "\n")
        content = content.replace("\\t", "\t")
        content = content.replace("\\\\", "\\")
        content = content.replace('\\"', '"')
        return StringLiteral(content)

    # Integer literal or variable
    if isinstance(sexp, str):
        try:
            return IntLiteral(int(sexp))
        except ValueError:
            # Not a number, treat as a variable reference
            return Variable(sexp)

    # List form
    if isinstance(sexp, list):
        if len(sexp) == 0:
            raise ParseError("Empty list is not a valid expression")

        head = sexp[0]
        if not isinstance(head, str):
            raise ParseError("Function name must be an identifier")

        # Special form: if
        if head == "if":
            if len(sexp) != 4:
                raise ParseError("if requires 3 arguments: condition, then, else")
            condition = parse_expr(sexp[1])
            then_expr = parse_expr(sexp[2])
            else_expr = parse_expr(sexp[3])
            return IfExpr(condition, then_expr, else_expr)

        # Special form: let
        if head == "let":
            if len(sexp) != 3:
                raise ParseError("let requires 2 arguments: bindings and body")

            bindings_sexp = sexp[1]
            if not isinstance(bindings_sexp, list):
                raise ParseError("let bindings must be a list")

            bindings: list[LetBinding] = []
            for binding in bindings_sexp:
                if not isinstance(binding, list) or len(binding) != 2:
                    raise ParseError("Each let binding must be a list of (name value)")

                name = binding[0]
                if not isinstance(name, str):
                    raise ParseError("Binding name must be an identifier")

                value = parse_expr(binding[1])
                bindings.append(LetBinding(name, value))

            body = parse_expr(sexp[2])
            return LetExpr(bindings, body)

        # Primitive form: write
        if head == "write":
            if len(sexp) != 2:
                raise ParseError("write requires 1 argument: expression to write")
            return WriteExpr(parse_expr(sexp[1]))

        # Primitive form: read
        if head == "read":
            if len(sexp) != 1:
                raise ParseError("read takes no arguments")
            return ReadExpr()

        # Primitive form: tell
        if head == "tell":
            if len(sexp) != 2:
                raise ParseError("tell requires 1 argument: expression to tell")
            return TellExpr(parse_expr(sexp[1]))

        # Primitive form: ask
        if head == "ask":
            if len(sexp) != 2:
                raise ParseError("ask requires 1 argument: question expression")
            return AskExpr(parse_expr(sexp[1]))

        # Function call
        args = [parse_expr(arg) for arg in sexp[1:]]
        return FunctionCall(head, args)

    raise ParseError(f"Invalid expression: {sexp}")


def parse_function_def(sexp: Any) -> FunctionDef:
    """Parse a function definition."""
    if not isinstance(sexp, list):
        raise ParseError("Function definition must be a list")

    if len(sexp) != 4:
        raise ParseError("Function definition must have form: (defun name (params) body)")

    if sexp[0] != "defun":
        raise ParseError("Function definition must start with 'defun'")

    name = sexp[1]
    if not isinstance(name, str):
        raise ParseError("Function name must be an identifier")

    params_list = sexp[2]
    if not isinstance(params_list, list):
        raise ParseError("Function parameters must be a list")

    params: list[str] = []
    for param in params_list:
        if not isinstance(param, str):
            raise ParseError("Function parameter must be an identifier")
        params.append(param)

    body = parse_expr(sexp[3])

    return FunctionDef(name, params, body)


def parse_program(text: str) -> Program:
    """Parse a complete program from text."""
    tokenizer = Tokenizer(text)
    functions: list[FunctionDef] = []

    while True:
        sexp = tokenizer.tokenize()
        if sexp is None:
            break
        func_def = parse_function_def(sexp)
        functions.append(func_def)

    if len(functions) == 0:
        raise ParseError("Program must contain at least one function")

    return Program(functions)
