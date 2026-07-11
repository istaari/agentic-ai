```
██████╗ ██╗   ██╗████████╗██╗  ██╗ ██████╗ ███╗   ██╗
██╔══██╗╚██╗ ██╔╝╚══██╔══╝██║  ██║██╔═══██╗████╗  ██║
██████╔╝ ╚████╔╝    ██║   ███████║██║   ██║██╔██╗ ██║
██╔═══╝   ╚██╔╝     ██║   ██╔══██║██║   ██║██║╚██╗██║
██║        ██║      ██║   ██║  ██║╚██████╔╝██║ ╚████║
╚═╝        ╚═╝      ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
```
> **Covers Python 3.8 → 3.13** · Basic · Advanced · Latest · Compact

---

## Table of Contents

1. [Core Syntax & Types](#1-core-syntax--types)
2. [Strings](#2-strings)
3. [Control Flow](#3-control-flow)
4. [Functions](#4-functions)
5. [Data Structures](#5-data-structures)
6. [OOP](#6-oop)
7. [Modern Python 3.8–3.13](#7-modern-python-38313)
8. [Power Tools — stdlib & Patterns](#8-power-tools--stdlib--patterns)

---

## 1. Core Syntax & Types

#### Variables & Assignment

```python
x = 42                          # dynamic typing
a, b, c = 1, 2, 3               # tuple unpacking
a, *rest, z = [1, 2, 3, 4, 5]  # starred: rest = [2, 3, 4]
a, b = b, a                     # swap

x: int   = 10                   # type-annotated variable
PI: float = 3.14159
```

#### Built-in Types

| Type       | Example        | Notes                  |
|------------|----------------|------------------------|
| `int`      | `1_000_000`    | underscores for readability |
| `float`    | `3.14`         | IEEE 754 double        |
| `complex`  | `2 + 3j`       |                        |
| `bool`     | `True / False` | subclass of int        |
| `str`      | `"hello"`      | immutable, Unicode     |
| `bytes`    | `b"hello"`     | immutable byte sequence|
| `NoneType` | `None`         | singleton              |

#### Type Conversion

```python
int("42"), int(3.9)          # → 42, 3  (truncates)
float("3.14")                # → 3.14
str(100), repr([1, 2])       # → '100', '[1, 2]'
bool(0), bool(""), bool([])  # → False, False, False
list("abc")                  # → ['a', 'b', 'c']
```

#### Operators

```python
5 // 2   # → 2   (floor div)     5 % 2    # → 1   (modulo)
2 ** 10  # → 1024 (power)        -5 // 2  # → -3  (floors toward -∞)

x == y;  x is y;  x in [1, 2, 3]
x != y;  x is not y;  x not in lst

# Bitwise
0b1010 & 0b1100  # → 0b1000 (AND)
0b1010 | 0b0101  # → 0b1111 (OR)
1 << 3           # → 8
```

#### Walrus Operator `:=` `3.8`

> **Walrus `:=`:** Assigns and returns a value in the same expression — eliminating repeated calls in `while` loops and comprehension filters. The name and value are both available after the `:=`.

```python
# assign + test in one expression
while chunk := f.read(8192):
    process(chunk)

if (n := len(data)) > 10:
    print(f"too long: {n}")     # n already computed, no second len()

# reuse inside comprehension
results = [y for x in data if (y := transform(x)) is not None]
```

---

## 2. Strings

#### f-strings — Full Power

```python
name, val = "Alice", 3.14159

f"{name!r}"             # → "'Alice'"      repr conversion
f"{val:.2f}"            # → '3.14'         format spec
f"{name=}"              # → 'name=Alice'   debug mode (3.8)
f"{'hello':>10}"        # → '     hello'   alignment
f"{2 ** 10:,}"          # → '1,024'        thousands separator

# 3.12: quotes and nesting inside f-strings
f"{'yes' if True else 'no'}"   # no backslash restriction
f"{f'{name}'}"                  # nested f-strings
```

#### String Flavors

```python
r"\n is not newline"   # raw string  — backslashes literal
b"bytes \x00 data"     # bytes literal
"""
multi
line
"""                    # triple-quoted
```

#### Key Methods

```python
s = "  Hello, World!  "
s.strip()                  # → 'Hello, World!'
s.lower() / s.upper()      # → case variants
s.replace("World", "PY")   # → '  Hello, PY!  '
s.split(", ")              # → ['  Hello', 'World!  ']
", ".join(["a", "b", "c"]) # → 'a, b, c'
s.startswith("  H")        # → True
s.find("World")            # → 9  (-1 if not found)
s.count("l")               # → 3
"42".zfill(5)              # → '00042'
```

| Old style                  | Preferred              |
|----------------------------|------------------------|
| `"%s %d" % (name, n)`     | `f"{name} {n}"`        |
| `"{}".format(x)`           | `f"{x}"`               |
| `s.format_map(d)`          | `f"{d['key']}"` (3.12+)|

---

## 3. Control Flow

#### Conditionals

```python
label = "even" if n % 2 == 0 else "odd"   # ternary

if 0 < x < 100: ...                        # chained comparison

value = x or default    # use default if x is falsy
safe  = x and x.method()
```

#### Loops

```python
for i, v in enumerate(lst, start=1):   # 1-indexed
    print(i, v)

for a, b in zip(lst1, lst2):           # parallel iteration
    ...

for i in range(0, 10, 2):             # → 0, 2, 4, 6, 8

# else: runs only if loop completed without break
for x in items:
    if pred(x): break
else:
    print("not found")
```

#### `match` / `case` — Structural Pattern Matching `3.10`

> **`match`/`case`:** Goes beyond a switch statement — it destructures sequences, mappings, and class instances, binding sub-values as variables. Guards (`if`) add extra conditions after a pattern match.

```python
match command:
    case "quit":                          # literal match
        sys.exit()
    case {"action": action, "obj": obj}:  # mapping destructure
        do(action, obj)
    case [x, y]:                          # sequence destructure
        move(x, y)
    case Point(x=x, y=y) if x > 0:       # class + guard
        print(f"right half: {x},{y}")
    case _:                               # wildcard (required last)
        raise ValueError(command)
```

#### Exception Handling

```python
try:
    result = 1 / x
except ZeroDivisionError as e:
    print(e)
except (TypeError, ValueError):
    raise
else:               # runs only if no exception was raised
    save(result)
finally:            # always runs — cleanup here
    cleanup()

raise RuntimeError("msg") from original_exc   # chained exceptions
```

#### Exception Groups + `except*` `3.11`

> **Exception groups:** `TaskGroup` and `asyncio.gather` can raise multiple simultaneous exceptions — `except*` filters by type and handles each matching sub-exception, letting other types propagate.

```python
try:
    async with asyncio.TaskGroup() as tg:
        tg.create_task(task1())
        tg.create_task(task2())
except* ValueError as eg:        # catches all ValueErrors in the group
    for e in eg.exceptions:
        print(e)
except* TypeError:                # other types handled separately
    ...
```

---

## 4. Functions

#### Signature Anatomy

```python
#           positional-only ↓     keyword-only ↓
def func(pos1, pos2, /, normal, *, kw_only, **kwargs): ...

def greet(name: str, *, loud: bool = False) -> str:
    return name.upper() if loud else name
```

#### `*args` / `**kwargs`

```python
def variadic(*args, **kwargs):
    print(args)    # → tuple
    print(kwargs)  # → dict

variadic(1, 2, x=3)   # (1, 2)  {'x': 3}

lst = [1, 2, 3]; d = {"x": 1}
func(*lst, **d)        # unpack at call site
```

#### Lambda & Higher-Order

```python
square = lambda x: x ** 2
sorted(people, key=lambda p: p.age)
list(filter(lambda x: x > 0, nums))
list(map(str, [1, 2, 3]))             # → ['1', '2', '3']
```

#### Closures & `nonlocal`

> **`nonlocal`:** Without it, assigning to an outer variable inside a nested function creates a new local — `nonlocal` tells Python to rebind the enclosing scope's variable instead of shadowing it.

```python
def counter(start=0):
    count = start
    def inc():
        nonlocal count   # rebind outer 'count', not create a new local
        count += 1
        return count
    return inc

c = counter()
c()  # → 1
c()  # → 2
```

#### Decorators

> **Decorators:** A decorator replaces a function with a wrapper; `@functools.wraps` copies `__name__`, `__doc__`, and `__wrapped__` onto the wrapper so introspection and tooling still see the original function.

```python
import functools

def timer(func):
    @functools.wraps(func)              # preserves __name__, __doc__
    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {time.perf_counter() - t:.4f}s")
        return result
    return wrapper

@timer
def slow(): time.sleep(0.1)

# Parameterized decorator — factory returns the actual decorator
def repeat(n):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(n): func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def hello(): print("hi")
```

#### Generators

> **Generators:** `yield` suspends execution and returns a value lazily — no list is built until iterated. `yield from` delegates to a sub-iterable, forwarding values and `send()`/`throw()` calls transparently.

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

gen = fibonacci()
[next(gen) for _ in range(8)]  # → [0, 1, 1, 2, 3, 5, 8, 13]

# yield from — delegates to sub-iterable (not just syntactic sugar)
def chain(*iterables):
    for it in iterables:
        yield from it              # forwards send/throw/close too

# Generator expression — lazy, no list allocated
total = sum(x**2 for x in range(1_000_000))
```

---

## 5. Data Structures

#### List

```python
lst = [1, 2, 3, 4, 5]
lst[1:4]     # → [2, 3, 4]
lst[::2]     # → [1, 3, 5]    (step)
lst[::-1]    # → [5, 4, 3, 2, 1]  (reverse)

lst.append(6); lst.extend([7, 8]); lst.insert(0, 0)
lst.pop()    # → removes & returns last
lst.pop(0)   # → removes & returns index 0
lst.sort(key=abs, reverse=True)
sorted(lst)  # → new sorted list; lst unchanged
3 in lst     # → True
```

#### List Comprehension

```python
squares = [x**2 for x in range(10)]
evens   = [x for x in range(20) if x % 2 == 0]
flat    = [x for row in matrix for x in row]    # flatten 2D
pairs   = [(x, y) for x in "AB" for y in range(3)]
```

#### Dict

```python
d = {"a": 1, "b": 2}
d["c"] = 3
d.get("z", 0)              # → 0  (default, no KeyError)
d.setdefault("x", []).append(1)
d.items(); d.keys(); d.values()

{**d, "extra": 99}         # merge via unpacking
d1 | d2                    # → new merged dict, d2 wins  (3.9)
d1 |= d2                   # in-place merge  (3.9)

inv     = {v: k for k, v in d.items()}  # invert
squares = {n: n**2 for n in range(5)}
```

#### Set

```python
s = {1, 2, 3}
s.add(4); s.discard(99)    # discard: no error if missing

a | b   # → union           a & b  # → intersection
a - b   # → difference      a ^ b  # → symmetric difference
a <= b  # → subset          a < b  # → proper subset

no_dups = {x.lower() for x in words}   # set comprehension
```

#### Tuple & NamedTuple

```python
point = (3, 4)
x, y = point    # unpack

from collections import namedtuple
Point = namedtuple("Point", ["x", "y"])
p = Point(3, 4); p.x   # → 3

from typing import NamedTuple
class Vec(NamedTuple):
    x: float
    y: float
    z: float = 0.0
```

---

## 6. OOP

#### Class Basics

```python
class Animal:
    kingdom = "Animalia"   # class variable — shared across instances

    def __init__(self, name: str, age: int):
        self.name = name   # instance variable
        self.age  = age

    def __repr__(self) -> str: return f"Animal({self.name!r}, {self.age})"
    def __str__(self)  -> str: return self.name
    def __eq__(self, other) -> bool:
        return isinstance(other, Animal) and self.name == other.name

    def speak(self) -> str: return "..."
```

#### Inheritance & MRO

> **MRO (Method Resolution Order):** Python uses C3 linearization to determine which class's method wins in multiple inheritance — always check `ClassName.__mro__` when mixing in behaviors to avoid surprises.

```python
class Dog(Animal):
    def __init__(self, name: str, age: int, breed: str):
        super().__init__(name, age)
        self.breed = breed

    def speak(self) -> str: return "Woof!"

class Cat(Animal):
    def speak(self) -> str: return "Meow!"

class Hybrid(Dog, Cat): ...
Hybrid.__mro__   # → (Hybrid, Dog, Cat, Animal, object)
# speak() resolves to Dog.speak() — leftmost parent wins
```

#### `@property`

> **`@property`:** Converts a method into an attribute-access call, hiding the implementation detail (`_radius`). The setter enables validation on assignment without callers ever calling a setter method explicitly.

```python
class Circle:
    def __init__(self, radius: float):
        self._radius = radius

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, value: float):
        if value < 0: raise ValueError("radius must be non-negative")
        self._radius = value

    @property
    def area(self) -> float:
        return 3.14159 * self._radius ** 2

    @classmethod
    def unit(cls) -> "Circle": return cls(1.0)   # factory

    @staticmethod
    def is_valid(r: float) -> bool: return r >= 0
```

#### Key Dunder Methods

| Method                      | Triggered by              |
|-----------------------------|---------------------------|
| `__init__`                  | `obj = Class()`           |
| `__repr__`                  | `repr(obj)`, REPL         |
| `__str__`                   | `str(obj)`, `print()`     |
| `__len__`                   | `len(obj)`                |
| `__getitem__`               | `obj[key]`                |
| `__setitem__`               | `obj[key] = val`          |
| `__contains__`              | `x in obj`                |
| `__iter__` / `__next__`     | `for x in obj`            |
| `__enter__` / `__exit__`    | `with obj:`               |
| `__call__`                  | `obj()`                   |
| `__add__` / `__radd__`      | `obj + other`             |
| `__hash__`                  | `hash(obj)`, dict key     |

#### Dataclasses `3.7`

> **Dataclasses:** Auto-generate `__init__`, `__repr__`, and `__eq__`; use `field(default_factory=list)` for mutable defaults (never use `[]` directly) and `frozen=True` to make instances hashable and immutable.

```python
from dataclasses import dataclass, field

@dataclass(order=True, frozen=True)
class Point:
    x: float
    y: float
    z: float = 0.0
    tags: list = field(default_factory=list)   # ✅ new list per instance

    def __post_init__(self):    # runs after generated __init__
        object.__setattr__(self, "magnitude", (self.x**2 + self.y**2) ** 0.5)

p = Point(3, 4)
p.x        # → 3   (frozen = immutable after construction)
p.magnitude  # → 5.0
```

#### Protocols `3.8`

> **Protocols:** Define an interface structurally — any class with the right methods satisfies it, with no explicit `class Foo(Protocol)` inheritance needed. Add `@runtime_checkable` to enable `isinstance()` checks.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    def draw(self)  -> None:  ...
    def area(self)  -> float: ...

class Circle:           # no inheritance — structural match
    def draw(self):  print("○")
    def area(self):  return 3.14

def render(shape: Drawable): shape.draw()

render(Circle())                    # ✅ structural match
isinstance(Circle(), Drawable)      # → True  (runtime_checkable)
```

---

## 7. Modern Python 3.8–3.13

#### `3.8` — Walrus & Positional-only

```python
# Walrus — see §1 for full coverage
while chunk := f.read(8192): process(chunk)
data = [y := f(x), y**2, y**3]    # reuse computed value inline

# Positional-only parameters (/)
def pow(base, exp, /, mod=None):   # base & exp cannot be passed as keywords
    return builtins.pow(base, exp, mod)
```

#### `3.9` — Built-in Generics & Dict Operators

```python
# Built-in generic types — no typing import needed
def process(items: list[int]) -> dict[str, int]: ...

# Dict merge/update
merged  = config | overrides    # → new dict, overrides wins
config |= overrides             # in-place update

"python".removeprefix("py")     # → 'thon'
"hello!".removesuffix("!")      # → 'hello'
```

#### `3.10` — Union `|` & Pattern Matching

```python
# Union with | — replaces Optional[X] and Union[X, Y]
def greet(name: str | None = None) -> str:
    return name or "stranger"

match status:
    case 200:               print("OK")
    case 404:               print("Not Found")
    case 500 | 502 | 503:   print("Server Error")   # OR pattern
    case int(n) if n > 0:   print(f"Success: {n}")
    case _:                 print("Unknown")
```

#### `3.11` — Exception Notes & `Self`

```python
# Exception notes — attach context without chaining
try: ...
except ValueError as e:
    e.add_note("hint: check input range")
    raise

# Self type — correct return type for fluent/builder methods
from typing import Self

class Builder:
    def set_name(self, name: str) -> Self:   # returns the exact subclass, not 'Builder'
        self.name = name
        return self

# tomllib — stdlib TOML parser
import tomllib
with open("pyproject.toml", "rb") as f:
    cfg = tomllib.load(f)
```

#### `3.12` — `type` Statement & PEP 695 Generics

> **PEP 695 syntax:** Replaces `TypeVar("T")` boilerplate with `[T]` in function/class definitions and the `type` keyword for aliases — cleaner and checked at definition time, not runtime.

```python
type Vector    = list[float]
type Matrix[T] = list[list[T]]   # generic alias

# Generic function — T is inferred, no TypeVar import
def first[T](lst: list[T]) -> T:
    return lst[0]

class Stack[T]:
    def __init__(self)         -> None: self._items: list[T] = []
    def push(self, item: T)    -> None: self._items.append(item)
    def pop(self)              -> T:    return self._items.pop()

# @override — compile error if parent lacks the method
from typing import override
class Child(Parent):
    @override
    def method(self) -> None: ...   # ❌ error if Parent.method doesn't exist

# f-strings: backslashes and nesting now allowed
print(f"{'\\n'.join(words)}")
```

#### `3.13` — Free-threaded Mode & `copy.replace`

> **Free-threaded Python:** An experimental build (`python3.13t`) removes the GIL, enabling true parallel execution across threads for CPU-bound code — API is identical, just the lock is gone.

```python
import sys
sys._is_gil_enabled()   # → False in free-threaded build (python3.13t)

# copy.replace() — non-destructive update for frozen dataclasses
from copy import replace
@dataclass(frozen=True)
class Config: host: str; port: int

cfg  = Config("localhost", 3000)
cfg2 = replace(cfg, port=8080)   # → Config(host='localhost', port=8080)
```

---

## 8. Power Tools — stdlib & Patterns

#### Comprehensions — All Four

```python
squares  = [x**2 for x in range(10)]                   # list
inv_map  = {v: k for k, v in d.items()}                # dict
unique   = {x.lower() for x in words}                  # set
lazy_sum = sum(x**2 for x in range(10**6))             # generator — no list built
```

#### Context Managers

> **`@contextmanager`:** The `yield` splits setup (before) from teardown (after) — the `with` block body runs at the `yield` point. Exceptions inside the block are re-raised at `yield` unless caught in the generator.

```python
with open("file.txt", "r", encoding="utf-8") as f:
    data = f.read()

# Multiple (parenthesized form — 3.10)
with (open("in.txt") as src, open("out.txt", "w") as dst):
    dst.write(src.read())

from contextlib import contextmanager

@contextmanager
def timer(label: str):
    t = time.perf_counter()
    yield                      # ← with-block body runs here
    print(f"{label}: {time.perf_counter() - t:.3f}s")

with timer("load"):
    load_data()
```

#### Async / Await

```python
import asyncio

async def fetch(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()

async def main():
    # gather — concurrent, but exceptions cancel others
    results = await asyncio.gather(
        fetch("https://api.one.com"),
        fetch("https://api.two.com"),
    )

    # TaskGroup (3.11) — structured concurrency, propagates all exceptions
    async with asyncio.TaskGroup() as tg:
        t1 = tg.create_task(fetch("url1"))
        t2 = tg.create_task(fetch("url2"))
    print(t1.result(), t2.result())

asyncio.run(main())
```

#### Type Hints — Advanced

> **`ParamSpec`:** Captures the full parameter signature of a callable so a decorator's wrapper can preserve the original function's argument types exactly — `TypeVar` alone can't express this.

```python
from typing import TypeVar, Callable, ParamSpec, TypedDict, Annotated, Literal, Generic

T = TypeVar("T")
P = ParamSpec("P")    # captures (args, kwargs) of a callable

# TypedDict — typed dict shape (no runtime enforcement)
class Config(TypedDict):
    host:  str
    port:  int
    debug: bool

# Generic class
class Pair(Generic[T]):
    def __init__(self, a: T, b: T): self.a, self.b = a, b
    def first(self) -> T: return self.a

# Callable with ParamSpec — wrapper preserves exact signature
def logged(func: Callable[P, T]) -> Callable[P, T]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        print(f"calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

# Annotated — attach metadata for validators / docs
Age  = Annotated[int, "must be >= 0"]

# Literal — restrict to specific values
Mode = Literal["r", "w", "rb", "wb"]
```

#### `itertools`

> **`groupby` gotcha:** Input **must be sorted** by the key first — `groupby` only groups consecutive elements, so unsorted input produces multiple groups for the same key.

```python
from itertools import chain, islice, groupby, product, combinations, permutations, accumulate, batched

list(chain([1, 2], [3, 4], [5]))    # → [1, 2, 3, 4, 5]
list(islice(fib_gen, 10))           # → first 10 Fibonacci numbers
list(product("AB", repeat=2))       # → [AA, AB, BA, BB]

# batched (3.12) — chunk an iterable
list(batched(range(10), 3))         # → [(0,1,2), (3,4,5), (6,7,8), (9,)]

# groupby — ✅ sort first or you'll get duplicate groups
for key, group in groupby(sorted(data, key=lambda x: x.dept), key=lambda x: x.dept):
    print(key, list(group))

list(accumulate([1, 2, 3, 4], operator.mul))   # → [1, 2, 6, 24]  (running product)
```

#### `functools`

> **`@cache` vs `@lru_cache`:** `@cache` (3.9) is unbounded — it never evicts entries, so memory grows forever. Use `@lru_cache(maxsize=N)` when the input space is large or unbounded.

```python
from functools import lru_cache, cache, partial, reduce

@cache                         # ✅ unbounded — fine for small input spaces (3.9)
def fib(n: int) -> int:
    return n if n < 2 else fib(n-1) + fib(n-2)

@lru_cache(maxsize=256)        # ✅ bounded LRU — evicts oldest when full
def expensive(x): ...

double = partial(pow, exp=2)   # fix a parameter
reduce(lambda a, b: a + b, [1, 2, 3, 4])   # → 10
```

#### `pathlib`

```python
from pathlib import Path

p = Path("data") / "raw" / "file.csv"
p.exists(); p.is_file(); p.suffix          # → '.csv'
p.read_text(encoding="utf-8")
p.write_text("hello"); p.write_bytes(b"")
p.parent; p.name; p.stem                   # dir / filename / no-ext
list(Path(".").glob("**/*.py"))            # recursive glob
p.rename(p.with_suffix(".tsv"))
p.mkdir(parents=True, exist_ok=True)
```

#### `collections`

```python
from collections import Counter, defaultdict, deque, ChainMap

c = Counter("aabbccc")       # → Counter({'c': 3, 'a': 2, 'b': 2})
c.most_common(2)             # → [('c', 3), ('a', 2)]
c1 + c2                      # combine counts

graph = defaultdict(list)
graph["a"].append("b")       # no KeyError on missing key

q = deque([1, 2, 3], maxlen=5)
q.appendleft(0); q.popleft() # O(1) at both ends
q.rotate(2)                  # rotate right by 2

env = ChainMap(local_cfg, default_cfg)
env["key"]                   # searches local_cfg first, then default_cfg
```

#### Concurrency Quick Reference

```python
# Thread pool — I/O bound (network, disk)
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(fetch, urls))

# Process pool — CPU bound (computation, no GIL contention)
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor() as ex:
    futures = [ex.submit(crunch, data) for data in chunks]
    results = [f.result() for f in futures]

# asyncio — I/O bound, single-threaded cooperative concurrency
asyncio.run(main())                      # entry point
asyncio.gather(*coros)                   # run coroutines concurrently
asyncio.wait_for(coro, timeout=5.0)     # timeout wrapper
```

---

> **Common Gotchas**

```python
# 1. Mutable default argument — shared across all calls
def bad(lst=[]):   lst.append(1); return lst   # ❌ shared state!
def good(lst=None): lst = lst or []; ...       # ✅ new list each call

# 2. is vs == — identity vs equality
a = [1, 2]; b = [1, 2]
a == b   # → True   (equal values)
a is b   # → False  (different objects)
x is None   # ✅ always use 'is' for None, not ==

# 3. Late binding closures — 'i' is looked up at call time, not capture time
fns = [lambda: i for i in range(3)]
fns[0]()   # → 2  ❌ not 0 — all share the same 'i'
fns = [lambda i=i: i for i in range(3)]   # ✅ fix: default arg freezes value

# 4. Float precision
0.1 + 0.2 == 0.3     # → False ❌
import math
math.isclose(0.1 + 0.2, 0.3)   # → True ✅

# 5. Modifying a collection while iterating — iterate a copy
for k in list(d.keys()): ...     # ✅ copy keys first
```
