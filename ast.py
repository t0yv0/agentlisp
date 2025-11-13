"""Typed AST definitions for the Lisp interpreter."""

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class IntLiteral:
    """Integer literal expression."""
    value: int


@dataclass(frozen=True)
class StringLiteral:
    """String literal expression."""
    value: str


@dataclass(frozen=True)
class FunctionCall:
    """Function call expression."""
    func_name: str
    args: list["Expr"]


@dataclass(frozen=True)
class IfExpr:
    """If conditional expression."""
    condition: "Expr"
    then_expr: "Expr"
    else_expr: "Expr"


@dataclass(frozen=True)
class Variable:
    """Variable reference expression."""
    name: str


@dataclass(frozen=True)
class LetBinding:
    """A single variable binding in a let expression."""
    name: str
    value: "Expr"


@dataclass(frozen=True)
class LetExpr:
    """Let expression with local variable bindings."""
    bindings: list[LetBinding]
    body: "Expr"


# Key type representing any expression in the language
Expr = Union[IntLiteral, StringLiteral, FunctionCall, IfExpr, Variable, LetExpr]


@dataclass(frozen=True)
class FunctionDef:
    """Function definition with name, parameters, and body."""
    name: str
    params: list[str]
    body: Expr


@dataclass(frozen=True)
class Program:
    """Top-level program containing function definitions."""
    functions: list[FunctionDef]

    def get_main(self) -> FunctionDef | None:
        """Get the main function if it exists."""
        for func in self.functions:
            if func.name == "main":
                return func
        return None
