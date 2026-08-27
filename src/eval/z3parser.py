"""
FOL (Prover9-format) string → Z3 expression parser.

Provides:
    fol_to_z3(fol_str, timeout_ms=30000) → z3.BoolRef
    normalize_fol(fol_str) → str
    FOLParseError

Used by z3_equiv.py for logical equivalence checking.

Supports:
    Quantifiers:  ∀ ∃
    Connectives:  ∧ ∨ → ↔ ¬ ⊕
    Predicates:   P(x, y), Q(const), R()
    Constants:    Capitalized or quoted identifiers
    Variables:    single lowercase letters (x, y, z, ...) + multi-char lowercase for Willow
    Equality:     built-in Z3 equality (for Z3 LE checking)
"""

from __future__ import annotations

import re
import time
from typing import List, Optional, Tuple

import z3


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class FOLParseError(Exception):
    """Raised when a FOL string cannot be parsed into a Z3 expression."""
    pass


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_fol(fol_str: str) -> str:
    """Normalise a FOL string for comparison (whitespace + symbol canonicalisation)."""
    s = str(fol_str).strip()
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s)
    # normalise arrows
    s = s.replace('->', '→').replace('-->', '→').replace('implies', '→')
    s = s.replace('<->', '↔').replace('↔', '↔')
    # normalise xor
    s = s.replace('xor', '⊕')
    # remove trailing dots (common in Prover9 format)
    s = re.sub(r'\s*\.\s*$', '', s)
    # strip trailing semicolons
    s = re.sub(r'[;]\s*$', '', s)
    # remove stray commas before closing parens
    s = re.sub(r',\s*\)', ')', s)
    return s.strip()


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

# Token types
TOK_QUANT  = 'QUANT'    # ∀ or ∃
TOK_LPAREN = 'LPAREN'   # (
TOK_RPAREN = 'RPAREN'   # )
TOK_COMMA  = 'COMMA'    # ,
TOK_NOT    = 'NOT'      # ¬
TOK_AND    = 'AND'      # ∧
TOK_OR     = 'OR'       # ∨
TOK_IMPL   = 'IMPL'     # →
TOK_IFF    = 'IFF'      # ↔
TOK_XOR    = 'XOR'      # ⊕
TOK_EQ     = 'EQ'       # =
TOK_SYM    = 'SYM'      # predicate / constant / variable name
TOK_EOF    = 'EOF'

# Symbol regex: CamelCase predicates, lowercase vars, quoted strings
_SYM_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|'[^']*'")

_TOKEN_SPEC = [
    (TOK_QUANT,  r'[∀∃]'),
    (TOK_LPAREN, r'\('),
    (TOK_RPAREN, r'\)'),
    (TOK_COMMA,  r','),
    (TOK_NOT,    r'¬'),
    (TOK_AND,    r'∧'),
    (TOK_OR,     r'∨'),
    (TOK_IMPL,   r'→'),
    (TOK_IFF,    r'↔'),
    (TOK_XOR,    r'⊕'),
    (TOK_EQ,     r'='),
]


class Token:
    def __init__(self, kind: str, value: str, pos: int):
        self.kind = kind
        self.value = value
        self.pos = pos

    def __repr__(self) -> str:
        return f"Token({self.kind}, {self.value!r})"


def _lex(fol_str: str) -> List[Token]:
    """Tokenize a FOL string."""
    tokens: List[Token] = []
    s = normalize_fol(fol_str)
    i = 0
    while i < len(s):
        c = s[i]
        # skip whitespace
        if c.isspace():
            i += 1
            continue
        # check fixed tokens
        matched = False
        for kind, pat in _TOKEN_SPEC:
            m = re.match(pat, s[i:])
            if m:
                tokens.append(Token(kind, m.group(0), i))
                i += len(m.group(0))
                matched = True
                break
        if matched:
            continue
        # symbol (predicate / constant / variable)
        m = _SYM_RE.match(s[i:])
        if m:
            tokens.append(Token(TOK_SYM, m.group(0), i))
            i += len(m.group(0))
            continue
        raise FOLParseError(f"Unexpected character {c!r} at position {i} in: {s[:80]}")
    tokens.append(Token(TOK_EOF, '', len(s)))
    return tokens


# ---------------------------------------------------------------------------
# Recursive-descent parser → Z3
# ---------------------------------------------------------------------------

class _Parser:
    """Recursive-descent parser that builds a Z3 expression tree."""

    def __init__(self, tokens: List[Token], timeout_ms: int = 30000, ctx: Optional[z3.Context] = None):
        self.tokens = tokens
        self.pos = 0
        self.timeout_ms = timeout_ms
        self._deadline = time.perf_counter() + timeout_ms / 1000.0
        self._ctx = ctx
        self._sorts: dict = {}        # (name, arity) → Z3 Sort
        self._funcs: dict = {}        # (name, arity) → Z3 FuncDecl
        self._domain_sort = z3.DeclareSort('U', ctx=ctx) if ctx else z3.DeclareSort('U')

    def _check_timeout(self):
        if time.perf_counter() > self._deadline:
            raise FOLParseError("Z3 FOL parser timeout")

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, kind: str) -> Token:
        tok = self._advance()
        if tok.kind != kind:
            raise FOLParseError(
                f"Expected {kind}, got {tok.kind}({tok.value!r}) at pos {tok.pos}"
            )
        return tok

    def _get_func(self, name: str, arity: int) -> z3.FuncDecl:
        """Create Z3 predicate declarations within the parser's context."""
        key = (name, arity)
        if key not in self._funcs:
            domain_sorts = [self._domain_sort] * arity
            bs = z3.BoolSort(ctx=self._ctx) if self._ctx else z3.BoolSort()
            self._funcs[key] = z3.Function(name, *domain_sorts, bs)
        return self._funcs[key]

    # ---- Grammar methods ------------------------------------------------

    def parse(self) -> z3.BoolRef:
        """Entry point: S → formula EOF"""
        self._check_timeout()
        expr = self._parse_formula()
        if self._peek().kind != TOK_EOF:
            raise FOLParseError(
                f"Trailing tokens after formula: {self._peek()}"
            )
        return expr

    def _parse_formula(self) -> z3.BoolRef:
        """formula → quant_expr | implication"""
        self._check_timeout()
        tok = self._peek()
        if tok.kind == TOK_QUANT:
            return self._parse_quantified()
        return self._parse_implication()

    def _parse_quantified(self) -> z3.BoolRef:
        """quant_expr → QUANT var ... ( formula )"""
        quants: List[Tuple[str, str]] = []  # (quant_kind, var_name)
        while self._peek().kind == TOK_QUANT:
            qtok = self._advance()
            vtok = self._expect(TOK_SYM)
            quants.append((qtok.value, vtok.value))
        body = self._parse_formula()
        # wrap quantifiers inside-out
        for qkind, vname in reversed(quants):
            z3_var = z3.Const(vname, self._domain_sort)
            if qkind == '∀':
                body = z3.ForAll([z3_var], body)
            else:
                body = z3.Exists([z3_var], body)
        return body

    def _parse_implication(self) -> z3.BoolRef:
        """implication → disjunction (→ disjunction)*   right-associative"""
        left = self._parse_disjunction()
        while self._peek().kind == TOK_IMPL:
            self._advance()
            right = self._parse_implication()  # right-assoc
            left = z3.Implies(left, right)
        return left

    def _parse_disjunction(self) -> z3.BoolRef:
        """disjunction → conjunction ((∨ | ⊕) conjunction)*"""
        left = self._parse_conjunction()
        while self._peek().kind in (TOK_OR, TOK_XOR):
            op = self._advance().kind
            right = self._parse_conjunction()
            if op == TOK_OR:
                left = z3.Or(left, right)
            else:
                left = z3.Xor(left, right)
        return left

    def _parse_conjunction(self) -> z3.BoolRef:
        """conjunction → unary (∧ unary)*"""
        left = self._parse_unary()
        while self._peek().kind == TOK_AND:
            self._advance()
            right = self._parse_unary()
            left = z3.And(left, right)
        return left

    def _parse_unary(self) -> z3.BoolRef:
        """unary → ¬ formula | atomic"""
        if self._peek().kind == TOK_NOT:
            self._advance()
            return z3.Not(self._parse_formula())
        return self._parse_atomic()

    def _parse_atomic(self) -> z3.BoolRef:
        """atomic → SYM ( term_list ) | ( formula ) | SYM = SYM"""
        self._check_timeout()
        tok = self._peek()
        if tok.kind == TOK_LPAREN:
            self._advance()
            expr = self._parse_formula()
            self._expect(TOK_RPAREN)
            return expr
        if tok.kind == TOK_SYM:
            name = self._advance().value
            # Look ahead: '(' → predicate, '=' → equality, else → zero-ary predicate
            nxt = self._peek()
            if nxt.kind == TOK_LPAREN:
                self._advance()  # consume '('
                args = self._parse_term_list()
                self._expect(TOK_RPAREN)
                func = self._get_func(name, len(args))
                return func(*args)
            elif nxt.kind == TOK_EQ:
                # equality:  SYM = SYM
                self._advance()
                rhs_tok = self._expect(TOK_SYM)
                return z3.Const(name, self._domain_sort) == z3.Const(rhs_tok.value, self._domain_sort)
            else:
                # zero-ary predicate
                func = self._get_func(name, 0)
                return func()
        raise FOLParseError(
            f"Unexpected token {tok.kind}({tok.value!r}) in atomic formula at pos {tok.pos}"
        )

    def _parse_term_list(self) -> List[z3.ExprRef]:
        """term_list → term (, term)* | empty"""
        args: List[z3.ExprRef] = []
        if self._peek().kind == TOK_RPAREN:
            return args  # zero-ary
        args.append(self._parse_term())
        while self._peek().kind == TOK_COMMA:
            self._advance()
            args.append(self._parse_term())
        return args

    def _parse_term(self) -> z3.ExprRef:
        """term → SYM  (treated as a constant in a Z3 uninterpreted sort)"""
        tok = self._expect(TOK_SYM)
        return z3.Const(tok.value, self._domain_sort)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fol_to_z3(fol_str: str, timeout_ms: int = 30000, ctx: Optional[z3.Context] = None) -> z3.BoolRef:
    """Parse a FOL string (Prover9 format) into a Z3 BoolRef expression.

    Args:
        fol_str: FOL formula string.
        timeout_ms: Parse timeout in milliseconds.
        ctx: Optional Z3 context. If None, uses global context.
             Pass a context to encapsulate all Z3 memory — delete the context
             after use to free C++ memory.

    Returns:
        z3.BoolRef expression suitable for z3.Solver checks.

    Raises:
        FOLParseError: If the string cannot be parsed.
    """
    if not fol_str or not str(fol_str).strip():
        raise FOLParseError("Empty FOL string")
    s = normalize_fol(str(fol_str))
    tokens = _lex(s)
    parser = _Parser(tokens, timeout_ms, ctx=ctx)
    return parser.parse()
