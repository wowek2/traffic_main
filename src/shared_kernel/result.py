from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar, Union, cast, final

# T - Success type (ex. BoundingBox)
# E - Error type (ex. InvalidBoundingBoxError)
# U - Mapped success type
# F - Mapped error type
T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")
F = TypeVar("F")

class ResultError(Exception):
    """Base class for Result-related errors."""
    pass

class Result(ABC, Generic[T, E]):
    """Monad Result representing success (Ok) or failure (Err)."""

    __slots__ = ()

    @abstractmethod
    def is_ok(self) -> bool: ...

    @abstractmethod
    def is_err(self) -> bool: ...

    @abstractmethod
    def unwrap(self) -> T:
        """Return the success value or throws ResultError."""
        ...

    @abstractmethod
    def unwrap_err(self) -> E:
        """Return the error value or throws ResultError."""
        ...

    @abstractmethod
    def unwrap_or(self, default: T) -> T:
        """Return the success value or default if Err."""
        ...

    @abstractmethod
    def unwrap_or_else(self, fn: Callable[[E], T]) -> T:
        """Return the success value or compute default value from error."""
        ...


    @abstractmethod
    def expect(self, msg: str) -> T:
        """Return value or raise error with custom message."""
        ...

    @abstractmethod
    def map(self, fn: Callable[[T], U]) -> Result[U, E]:
        """T -> U. If Err, return the same Err (changed type)."""
        ...

    @abstractmethod
    def map_err(self, fn: Callable[[E], F]) -> Result[T, F]:
        """E -> F. If Ok, return the same Ok (changed type)."""
        ...

    @abstractmethod
    def flat_map(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Binds function returning Result (T -> Result[U, E])."""
        ...

    @abstractmethod
    def or_else(self, fn: Callable[[E], Result[T, F]]) -> Result[T, F]:
        """Handles function returning Result."""
        ...
    # Aliases
    def bind(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        return self.flat_map(fn)

    @abstractmethod
    def inspect(self, fn: Callable[[T], None]) -> Result[T, E]:
        """Run side-effect function for success value."""
        ...

    @abstractmethod
    def inspect_err(self, fn: Callable[[E], None]) -> Result[T, E]:
        """Run side-effect function for error value."""
        ...

@final
@dataclass(frozen=True, slots=True)
class Ok(Result[T, E]):
    """Represents a successful result."""
    value: T

    def is_ok(self) -> bool: return True
    def is_err(self) -> bool: return False

    def unwrap(self) -> T: return self.value
    def unwrap_err(self) -> E: raise ValueError(f"Called unwrap_err on Ok: {self.value}")

    def unwrap_or(self, default: T) -> T: return self.value
    def unwrap_or_else(self, fn: Callable[[E], T]) -> T: return self.value

    def expect(self, msg: str) -> T: return self.value

    def map(self, fn: Callable[[T], U]) -> Result[U, E]: return Ok(fn(self.value))

    def map_err(self, fn: Callable[[E], F]) -> Result[T, F]: return cast("Result[T, F]", self)

    def flat_map(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]: return fn(self.value)

    def or_else(self, fn: Callable[[E], Result[T, F]]) -> Result[T, F]: return cast("Result[T, F]", self)

    def inspect(self, fn: Callable[[T], None]) -> Result[T, E]:
        fn(self.value)
        return self

    def inspect_err(self, fn: Callable[[E], None]) -> Result[T, E]: return self

    def __repr__(self) -> str: return f"Ok({self.value})"

@final
@dataclass(frozen=True, slots=True)
class Err(Result[T, E]):
    """Represents a failed result."""
    error: E

    def is_ok(self) -> bool: return False
    def is_err(self) -> bool: return True

    def unwrap(self) -> T: raise ValueError(f"Called unwrap on Err value: {self.error}")
    def unwrap_err(self) -> E: return self.error

    def unwrap_or(self, default: T) -> T: return default
    def unwrap_or_else(self, fn: Callable[[E], T]) -> T: return fn(self.error)

    def expect(self, msg: str) -> T: raise ValueError(f"{msg}: {self.error}")

    def map(self, fn: Callable[[T], U]) -> Result[U, E]: return cast("Result[U, E]", self)
    def map_err(self, fn) -> Result[T, F]: return Err(fn(self.error))

    def flat_map(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]: return cast("Result[U, E]", self)

    def or_else(self, fn: Callable[[E], Result[T, F]]) -> Result[T, F]: return fn(self.error)

    def inspect(self, fn: Callable[[T], None]) -> Result[T, E]:
        fn(self.value)
        return self

    def inspect_err(self, fn: Callable[[E], None]) -> Result[T, E]:
        return self

    def __repr__(self) -> str: return f"Err({self.error})"

# Aliases for easier usage
ResultType = Union(Ok[T, E], Err[T, E])
