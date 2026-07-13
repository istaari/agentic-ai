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
7. [Comprehensions & Generators](#7-comprehensions--generators)
8. [Collections (stdlib)](#8-collections-stdlib)
9. [Iterators & `itertools`](#9-iterators--itertools)
10. [`functools` & Caching](#10-functools--caching)
11. [Context Managers](#11-context-managers)
12. [Concurrency — Threads & Processes](#12-concurrency--threads--processes)
13. [Async / Await](#13-async--await)
14. [Type Hints](#14-type-hints)
15. [Modern Python 3.8–3.13](#15-modern-python-38313)
16. [Modules, Packages & Imports](#16-modules-packages--imports)
17. [File I/O — JSON & CSV](#17-file-io--json--csv)
18. [Numbers & Math](#18-numbers--math)
19. [Dates & Times](#19-dates--times)
20. [Regular Expressions](#20-regular-expressions)
21. [Enums](#21-enums)
22. [Testing & Tooling](#22-testing--tooling)
23. [Mental Models & Gotchas](#23-mental-models--gotchas)

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

## 7. Comprehensions & Generators

#### Comprehensions — All Four Forms

> A comprehension builds a collection in one expression: `output for item in iterable if condition`. More readable and faster than an explicit loop with `.append()`. The bracket type picks the result: `[]` list, `{}` dict/set, `()` generator.

```python
squares = [x**2 for x in range(10)]                 # list
inv_map = {v: k for k, v in d.items()}              # dict
unique  = {x.lower() for x in words}                # set
flat    = [x for row in matrix for x in row]        # nested → flatten 2D
pairs   = [(x, y) for x in "AB" for y in range(3)]  # cartesian product
```

#### Generator Expressions — Lazy

> Swap `[]` for `()` and nothing is materialized: values are produced one at a time, on demand. Ideal for large or infinite streams and for feeding aggregators (`sum`, `any`, `max`) without building an intermediate list.

```python
total = sum(x**2 for x in range(1_000_000))   # no million-element list built
any(n < 0 for n in data)                       # short-circuits on first True
```

#### Generator Functions — `yield`

> **`yield`:** Suspends the function, emits a value, and resumes on the next `next()` — state (locals) is preserved between calls. Lets you express infinite or streaming sequences with normal control flow. `yield from` delegates to a sub-iterable, forwarding values and `send()`/`throw()`.

```python
def fibonacci():
    a, b = 0, 1
    while True:              # "infinite" — safe because it's lazy
        yield a
        a, b = b, a + b

gen = fibonacci()
[next(gen) for _ in range(8)]   # → [0, 1, 1, 2, 3, 5, 8, 13]

def chain(*iterables):
    for it in iterables:
        yield from it        # delegate; forwards send/throw/close
```

---

## 8. Collections (stdlib)

> The `collections` module provides container types that replace hand-written boilerplate for counting, grouping, queuing, and layered lookups.

```python
from collections import Counter, defaultdict, deque, ChainMap
```

#### `Counter` — Tallying

> A `dict` subclass mapping each item to its count. Missing keys return `0` instead of raising. Supports arithmetic (`+`, `-`, `&`, `|`) between counters.

```python
c = Counter("aabbccc")       # → Counter({'c': 3, 'a': 2, 'b': 2})
c.most_common(2)             # → [('c', 3), ('a', 2)]   (highest first)
c["z"]                       # → 0   (no KeyError)
c.update("a/z")              # increment counts from another iterable
c1 + c2                      # add counts;  c1 - c2 keeps only positive
```

#### `defaultdict` — Auto-Initialized Values

> Takes a zero-arg factory (`int`, `list`, `set`) called the first time a key is missing. Replaces the `if k not in d: d[k] = ...` pattern. `int` → 0 for counters, `list` → `[]` for grouping.

```python
groups = defaultdict(list)
for name, dept in employees:
    groups[dept].append(name)    # key auto-created as [] on first access

counts = defaultdict(int)
for word in text.split():
    counts[word] += 1            # starts from 0
```

#### `deque` — Double-Ended Queue

> A list is O(n) to insert/pop at the front; a `deque` is O(1) at **both** ends — use it for queues, stacks, and sliding windows. `maxlen` makes a bounded ring buffer that discards from the opposite end when full.

```python
q = deque([1, 2, 3], maxlen=5)
q.append(4);     q.pop()        # O(1) right (stack)
q.appendleft(0); q.popleft()    # O(1) left  (queue / BFS)
q.rotate(2)                     # rotate right by 2
```

#### `ChainMap` — Layered Lookup

> A single view over several dicts searched in order, without merging/copying them. Perfect for config precedence (CLI > env > defaults). Writes affect only the first map.

```python
config = ChainMap(cli_args, env_vars, defaults)
config["timeout"]            # first map that has the key wins
config["debug"] = True       # written to cli_args (the first map)
```

#### `namedtuple` — see §5

Lightweight immutable record with named fields; covered under Data Structures.

---

## 9. Iterators & `itertools`

> The iterator protocol (`__iter__` / `__next__`) underlies every `for` loop. `itertools` composes iterators lazily — memory-flat pipelines over huge or infinite streams.

```python
from itertools import (chain, islice, groupby, product,
                       combinations, permutations, accumulate, batched)

list(chain([1, 2], [3, 4], [5]))     # → [1, 2, 3, 4, 5]   (concatenate)
list(islice(fibonacci(), 10))        # → first 10 from an infinite generator
list(product("AB", repeat=2))        # → [('A','A'),('A','B'),('B','A'),('B','B')]
list(combinations([1, 2, 3], 2))     # → [(1,2), (1,3), (2,3)]
list(accumulate([1, 2, 3, 4]))       # → [1, 3, 6, 10]  (running sum)
list(accumulate([1, 2, 3, 4], operator.mul))   # → [1, 2, 6, 24]

# batched (3.12) — fixed-size chunks
list(batched(range(10), 3))          # → [(0,1,2), (3,4,5), (6,7,8), (9,)]
```

> **`groupby` gotcha:** It groups only *consecutive* equal keys, so you MUST sort by the same key first or identical keys split into multiple groups.

```python
data.sort(key=lambda x: x.dept)
for dept, members in groupby(data, key=lambda x: x.dept):
    print(dept, list(members))
```

---

## 10. `functools` & Caching

> Higher-order helpers: memoization, partial application, and reduction.

```python
from functools import lru_cache, cache, partial, reduce, wraps
```

#### Memoization — `@cache` vs `@lru_cache`

> Both cache results keyed by arguments (which must be hashable). `@cache` (3.9) is **unbounded** — never evicts, so memory grows without limit. `@lru_cache(maxsize=N)` evicts the least-recently-used entry when full — use it whenever the input space is large or unbounded.

```python
@cache                          # unbounded — fine for small/finite input space
def fib(n: int) -> int:
    return n if n < 2 else fib(n - 1) + fib(n - 2)

@lru_cache(maxsize=256)         # bounded LRU
def query(user_id): ...
query.cache_info()              # → CacheInfo(hits=…, misses=…, maxsize=256, currsize=…)
query.cache_clear()             # reset
```

#### `partial` & `reduce`

```python
from functools import partial, reduce
import operator

cube = partial(pow, exp=3)          # freeze an argument → cube(2) == 8
reduce(operator.add, [1, 2, 3, 4])  # → 10   (fold left; prefer sum() when adding)
reduce(operator.mul, range(1, 6))   # → 120  (5!)
```

---

## 11. Context Managers

> A context manager guarantees setup/teardown around a block via `with` — files close, locks release, transactions commit/rollback even on exception. Any object with `__enter__`/`__exit__` qualifies.

```python
with open("file.txt", encoding="utf-8") as f:   # auto-closes
    data = f.read()

# Multiple managers (parenthesized form — 3.10)
with (open("in.txt") as src, open("out.txt", "w") as dst):
    dst.write(src.read())
```

#### `@contextmanager` — Build One From a Generator

> The single `yield` splits setup (before) from teardown (after); the `with` body runs at the `yield`. Wrap the `yield` in `try/finally` if teardown must run even on error.

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(label: str):
    start = time.perf_counter()
    try:
        yield                       # ← with-block body runs here
    finally:
        print(f"{label}: {time.perf_counter() - start:.3f}s")

with timer("load"):
    load_data()
```

---

## 12. Concurrency — Threads & Processes

> **Choose by workload, not habit.** The GIL (§23) lets only one thread run Python bytecode at a time, so **threads never speed up computation** — they help only while tasks *wait* on I/O. Rule: **I/O-bound → threads (or async, §13); CPU-bound → processes.** `concurrent.futures` gives both the same `Executor` API, so switching is a one-line change.

#### `ThreadPoolExecutor` — I/O-Bound

> Threads share memory (cheap to pass data, but mind data races) and the GIL is released during blocking I/O, so many network/disk calls overlap.

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(fetch, urls))   # ordered results, like built-in map()
```

#### `ProcessPoolExecutor` — CPU-Bound

> Separate processes = separate interpreters = separate GILs = true multi-core parallelism. Cost: arguments/results are **pickled** across process boundaries, and startup has overhead — worthwhile only when the work per task is substantial.

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

with ProcessPoolExecutor() as ex:
    futures = [ex.submit(crunch, chunk) for chunk in chunks]
    for f in as_completed(futures):       # yields in completion order
        print(f.result())                 # re-raises any worker exception here
```

#### `Future` Essentials

```python
f = ex.submit(fn, *args)   # returns immediately with a Future
f.result(timeout=5)        # blocks until done; re-raises exceptions
f.done()                   # → bool, non-blocking status check
as_completed(futures)      # iterate results as each finishes
```

| Approach              | Parallel?          | Best for             | Main caveat                 |
|-----------------------|--------------------|----------------------|-----------------------------|
| `ThreadPoolExecutor`  | no (GIL)           | I/O-bound, few tasks | shared state → data races   |
| `ProcessPoolExecutor` | yes (real cores)   | CPU-bound            | pickling + startup overhead |
| `asyncio` (§13)       | no (single thread) | I/O-bound, many tasks| requires async libraries    |

---

## 13. Async / Await

> **Core idea:** `async def` defines a *coroutine* that can pause. `await` is where it pauses, yielding control to the event loop so other coroutines run during the wait. It is **single-threaded cooperative** concurrency — nothing runs in true parallel; tasks take turns while others are blocked on I/O. Calling a coroutine does nothing until it is awaited or scheduled.

```python
import asyncio

async def fetch(url: str) -> str:
    async with aiohttp.ClientSession() as session:   # async context manager
        async with session.get(url) as resp:
            return await resp.text()                 # pauses; loop runs others
```

#### Concurrent vs Sequential

> `await a(); await b()` runs them one after another. To overlap, **schedule first, await later** — via `gather` or `create_task`. The concurrency comes from scheduling multiple coroutines before awaiting, not from `await` itself.

```python
async def main():
    # ❌ sequential — total ≈ t1 + t2
    r1 = await fetch("u1"); r2 = await fetch("u2")

    # ✅ gather — concurrent; total ≈ max(t1, t2). One failure cancels the rest.
    r1, r2 = await asyncio.gather(fetch("u1"), fetch("u2"))

    # ✅ TaskGroup (3.11) — structured concurrency (preferred)
    async with asyncio.TaskGroup() as tg:            # awaits all on block exit
        t1 = tg.create_task(fetch("u1"))             # starts running now
        t2 = tg.create_task(fetch("u2"))
    print(t1.result(), t2.result())   # errors surface as ExceptionGroup (§3)

asyncio.run(main())                    # entry point: create loop, run to done
```

| Primitive                    | Use for                                             |
|------------------------------|-----------------------------------------------------|
| `asyncio.run(coro)`          | top-level entry point — one per program             |
| `asyncio.gather(*coros)`     | run many concurrently, results in order             |
| `asyncio.TaskGroup()` `3.11` | structured concurrency; auto-awaits + propagates all|
| `create_task(coro)`          | schedule a coroutine to start now                   |
| `await asyncio.sleep(s)`     | non-blocking sleep (yields to loop)                 |
| `asyncio.wait_for(c, t)`     | await with timeout → `TimeoutError`                 |
| `await asyncio.to_thread(fn)`| offload blocking/CPU work to a thread               |

> ⚠️ **Never block the loop.** `time.sleep`, sync `requests`, or heavy CPU inside a coroutine freezes *all* tasks. Use async equivalents or `asyncio.to_thread`.

---

## 14. Type Hints

> Hints are optional and not enforced at runtime — `mypy`/`ruff`/IDEs check them statically. They document intent and catch bugs before execution.

```python
def greet(name: str, times: int = 1) -> str: ...
x: list[int] = []
y: dict[str, float] = {}
z: str | None = None            # 3.10 union — replaces Optional[str]
```

#### `TypedDict`, `Literal`, `Annotated`

```python
from typing import TypedDict, Literal, Annotated

class Config(TypedDict):        # typed shape for a dict (no runtime enforcement)
    host: str
    port: int
    debug: bool

Mode = Literal["r", "w", "rb", "wb"]        # restrict to specific values
Age  = Annotated[int, "must be >= 0"]        # attach metadata for validators/docs
```

#### Generics — `TypeVar`, `Generic`, `ParamSpec`

> **`TypeVar`** links input and output types ("returns the same type it received"). **`ParamSpec`** captures a callable's *entire* signature so a decorator's wrapper preserves argument types exactly — something `TypeVar` alone cannot express. (3.12 has cleaner `[T]` syntax — see §15.)

```python
from typing import TypeVar, Generic, Callable, ParamSpec

T = TypeVar("T")
P = ParamSpec("P")

class Pair(Generic[T]):
    def __init__(self, a: T, b: T): self.a, self.b = a, b
    def first(self) -> T: return self.a

def logged(func: Callable[P, T]) -> Callable[P, T]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:   # exact signature kept
        print(f"calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper
```

---

## 15. Modern Python 3.8–3.13

#### `3.8` — Walrus & Positional-only

```python
while chunk := f.read(8192):       # assign + test in one expression (§1)
    process(chunk)

def pow(base, exp, /, mod=None):   # '/' → base & exp are positional-only
    return builtins.pow(base, exp, mod)
```

#### `3.9` — Built-in Generics & Dict Operators

```python
def process(items: list[int]) -> dict[str, int]: ...   # no typing import needed

merged = config | overrides        # → new dict, overrides wins
config |= overrides                # in-place update

"python".removeprefix("py")        # → 'thon'
"hello!".removesuffix("!")         # → 'hello'
```

#### `3.10` — Union `|` & Pattern Matching

```python
def greet(name: str | None = None) -> str:   # replaces Optional/Union
    return name or "stranger"

match status:
    case 200:               print("OK")
    case 500 | 502 | 503:   print("Server Error")   # OR pattern
    case int(n) if n > 0:   print(f"Success: {n}")  # capture + guard
    case _:                 print("Unknown")
```

#### `3.11` — Exception Notes, `Self`, `tomllib`

```python
try: ...
except ValueError as e:
    e.add_note("hint: check input range")   # attach context without chaining
    raise

from typing import Self
class Builder:
    def set_name(self, name: str) -> Self:   # correct type for fluent chaining
        self.name = name
        return self

import tomllib                                # stdlib TOML reader
with open("pyproject.toml", "rb") as f:
    cfg = tomllib.load(f)
```

#### `3.12` — `type` Statement & PEP 695 Generics

> **PEP 695:** Replaces `TypeVar("T")` boilerplate with inline `[T]` on functions/classes and the `type` keyword for aliases — cleaner and checked at definition time.

```python
type Vector    = list[float]
type Matrix[T] = list[list[T]]     # generic alias

def first[T](lst: list[T]) -> T:   # inline generic — no TypeVar import
    return lst[0]

class Stack[T]:
    def __init__(self)      -> None: self._items: list[T] = []
    def push(self, x: T)    -> None: self._items.append(x)
    def pop(self)           -> T:    return self._items.pop()

from typing import override
class Child(Parent):
    @override                        # error if Parent has no such method
    def method(self) -> None: ...

print(f"{'\n'.join(words)}")         # backslashes/nesting now allowed in f-strings
```

#### `3.13` — Free-threaded Mode & `copy.replace`

> **Free-threaded build (`python3.13t`):** experimentally removes the GIL, enabling true multi-threaded parallelism for CPU-bound code. Same API, no lock.

```python
import sys
sys._is_gil_enabled()              # → False in the free-threaded build

from copy import replace           # non-destructive update for frozen dataclasses
@dataclass(frozen=True)
class Config: host: str; port: int
cfg2 = replace(Config("localhost", 3000), port=8080)
```

---

## 16. Modules, Packages & Imports

#### Import Forms

```python
import math                        # whole module → math.sqrt(9)
import numpy as np                 # aliased
from math import sqrt, pi          # specific names → sqrt(9)
from math import sqrt as s         # aliased name
from math import *                 # ❌ avoid — pollutes the namespace
```

#### Packages & Relative Imports

> A **package** is a directory of modules (`__init__.py` makes it explicit and runs on first import). Relative imports (`.`) resolve against the current package and work only *inside* an imported package — never in a script run directly.

```python
from . import sibling              # module in the same package
from .sub import thing             # module in a sub-package
from ..pkg import other            # parent package
```

#### The `__main__` Guard

> **`if __name__ == "__main__"`:** A file run directly has `__name__ == "__main__"`; when imported it's the module name. The guard lets one file serve as both a runnable script and an importable library without executing its main logic on import.

```python
def main(): ...

if __name__ == "__main__":         # runs only when executed directly
    main()

# Introspection
import math
math.__name__      # → 'math'
dir(math)          # → names defined in the module
```

---

## 17. File I/O — JSON & CSV

#### `open()` — Modes & Encoding

> Pass `encoding="utf-8"` explicitly — the platform default differs (Windows ≠ Linux) and causes subtle bugs. A `with` block closes the file even on exception.

```python
with open("file.txt", "r", encoding="utf-8") as f:
    text = f.read()                # whole file → str

for line in open("big.txt", encoding="utf-8"):   # lazy, one line at a time
    process(line.rstrip("\n"))

with open("out.txt", "w", encoding="utf-8") as f:   # 'w' truncates, 'a' appends
    f.write("hello\n")
```

| Mode      | Meaning                          |
|-----------|----------------------------------|
| `r`       | read (default), error if missing |
| `w`       | write, **truncates** existing    |
| `a`       | append to end                    |
| `x`       | create, error if exists          |
| `rb`/`wb` | binary (bytes, no encoding)      |

#### `pathlib` — Modern Paths

```python
from pathlib import Path

p = Path("data") / "raw" / "file.csv"      # '/' joins, OS-independent
p.exists(); p.is_file(); p.suffix          # → '.csv'
p.stem; p.name; p.parent                   # 'file' / 'file.csv' / .../raw
p.read_text(encoding="utf-8")
p.write_text("hello")
list(Path(".").glob("**/*.py"))            # recursive glob
p.mkdir(parents=True, exist_ok=True)
```

#### JSON

```python
import json

obj = json.loads('{"a": 1, "b": [2, 3]}')      # str → object
s   = json.dumps(obj, indent=2, sort_keys=True) # object → str

with open("config.json", encoding="utf-8") as f:
    cfg = json.load(f)                          # file → object
with open("out.json", "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2, default=str)    # default= serializes odd types
```

#### CSV

> Open with `newline=""` for the `csv` module — it manages line endings itself; omitting this adds blank rows on Windows.

```python
import csv

with open("data.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):        # each row → dict keyed by header
        print(row["name"], row["age"])

with open("out.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["name", "age"])          # header
    w.writerows([["Alice", 30], ["Bob", 25]])
```

---

## 18. Numbers & Math

```python
import math
math.sqrt(16); math.floor(3.7); math.ceil(3.2)   # → 4.0, 3, 4
math.pi; math.e; math.inf; math.nan
math.gcd(12, 18); math.factorial(5)
math.isclose(0.1 + 0.2, 0.3)                      # → True (float-safe compare)

import random
random.random()                # → float in [0.0, 1.0)
random.randint(1, 6)           # → int in [1, 6] inclusive
random.choice(["a", "b"])      # → one random element
random.sample(range(100), 5)   # → 5 unique elements
random.shuffle(lst)            # in-place
```

> **`Decimal`:** Floats can't represent `0.1` exactly (binary fractions), so use `Decimal` for money and anywhere exactness matters. Construct from a **string** — `Decimal(0.1)` inherits the float's error.

```python
from decimal import Decimal
Decimal("0.1") + Decimal("0.2")   # → Decimal('0.3')   ✅ exact
Decimal(0.1)                       # → 0.1000000000000000055...  ❌ float leak
```

---

## 19. Dates & Times

> Prefer **timezone-aware** datetimes (`tz=timezone.utc`) over naive ones — naive values silently assume local time and break across DST and servers. Store/transmit in UTC; convert only for display.

```python
from datetime import datetime, date, timedelta, timezone

datetime.now(timezone.utc)                    # aware "now"
date.today()                                  # → 2026-07-12
datetime.fromisoformat("2026-07-12T10:30:00") # parse ISO 8601

# Format ↔ parse
datetime.now().strftime("%Y-%m-%d %H:%M")     # datetime → str
datetime.strptime("2026-07-12", "%Y-%m-%d")   # str → datetime

# Arithmetic
tomorrow = date.today() + timedelta(days=1)
(datetime(2026, 12, 25) - datetime.now()).days   # → days remaining
```

---

## 20. Regular Expressions

> `re.match` anchors at the **start**; `re.search` scans **anywhere**. Compile patterns you reuse. Use raw strings (`r"..."`) so `\d`, `\w` reach the regex engine unmangled.

```python
import re

re.search(r"\d+", "abc123").group()   # → '123'   (anywhere)
re.match(r"\d+", "abc123")             # → None    (must match at start)
re.findall(r"\d+", "a1 b22 c333")      # → ['1', '22', '333']
re.sub(r"\s+", "_", "a  b   c")        # → 'a_b_c'
re.split(r"[,;]", "a,b;c")             # → ['a', 'b', 'c']

# Capture groups (positional + named), and compile for reuse
m = re.search(r"(\w+)@(\w+)\.com", "bob@acme.com")
m.group(1), m.group(2)                 # → ('bob', 'acme')

pat = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})")
pat.search("2026-07").group("year")    # → '2026'
```

| Token   | Matches                    | Token   | Matches                  |
|---------|----------------------------|---------|--------------------------|
| `\d`    | digit                      | `\w`    | word char `[A-Za-z0-9_]` |
| `\s`    | whitespace                 | `.`     | any char (not `\n`)      |
| `+`     | 1 or more                  | `*`     | 0 or more                |
| `?`     | 0 or 1                     | `{n,m}` | n to m repetitions       |
| `^` `$` | start / end of string      | `\|`    | alternation (OR)         |

---

## 21. Enums

> **`Enum`:** Named constants that are singletons compared by identity — safer and clearer than bare strings/ints. `auto()` numbers members for you. Subclass `IntEnum` when members must also behave as plain ints (comparable, sortable, JSON-serializable).

```python
from enum import Enum, auto, IntEnum

class Color(Enum):
    RED   = auto()     # → 1
    GREEN = auto()     # → 2
    BLUE  = auto()

Color.RED            # → <Color.RED: 1>
Color.RED.name       # → 'RED'
Color.RED.value      # → 1
Color(1)             # → <Color.RED: 1>   (lookup by value)
list(Color)          # → [Color.RED, Color.GREEN, Color.BLUE]

class Status(IntEnum):
    PENDING = 1
    DONE    = 2
Status.DONE > Status.PENDING   # → True   (behaves like int)
```

---

## 22. Testing & Tooling

#### `pytest`

> Auto-discovers `test_*` functions and rewrites plain `assert` to show rich failure diffs — no special assertion methods needed.

```python
def add(a, b): return a + b

def test_add():
    assert add(2, 3) == 5

import pytest

def test_raises():
    with pytest.raises(ValueError):
        int("not a number")

@pytest.mark.parametrize("a,b,expected", [(1, 2, 3), (0, 0, 0), (-1, 1, 0)])
def test_add_cases(a, b, expected):
    assert add(a, b) == expected
```

```bash
pytest                          # run all tests
pytest -v                       # one line per test
pytest -k "add"                 # only tests matching "add"
pytest test_math.py::test_add   # a single test
```

#### Environments & Tooling

```bash
python -m venv .venv            # create isolated environment
source .venv/bin/activate       # activate (Unix); .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip freeze > requirements.txt   # snapshot exact versions
```

| Tool             | Purpose                                      |
|------------------|----------------------------------------------|
| `uv`             | fast all-in-one installer / venv / resolver  |
| `ruff`           | lightning-fast linter + formatter            |
| `mypy`           | static type checker (validates hints)        |
| `black`          | opinionated code formatter                   |
| `pyproject.toml` | modern project + tool config (PEP 621)       |

---

## 23. Mental Models & Gotchas

#### Names, Objects & Mutability

> **The model behind most surprises:** a variable is a *name bound to an object*, not a box holding a value. `b = a` binds a second name to the *same* object; mutating via one is visible via the other — but only for **mutable** objects (`list`, `dict`, `set`, most instances). Immutable ones (`int`, `str`, `tuple`, `frozenset`) can't change in place, so this never bites there.

```python
a = [1, 2, 3]
b = a               # same list, two names
b.append(4)
a                   # → [1, 2, 3, 4]  ❗ a changed too

def add_one(lst):   # args pass by binding — same object
    lst.append(1)   # mutates the caller's list
```

#### Shallow vs Deep Copy

> A **shallow** copy duplicates the outer container but shares nested objects; a **deep** copy duplicates recursively. Reach for `deepcopy` only with nested mutable structure — it's slower.

```python
import copy
a = [[1, 2], [3, 4]]
b = a.copy()              # or list(a), a[:] — SHALLOW: inner lists shared
b[0].append(99)
a                          # → [[1, 2, 99], [3, 4]]  ❗
c = copy.deepcopy(a)       # fully independent
```

#### Truthiness

> Every object is truthy or falsy in a boolean context — powering `if items:` and `x or default`. Empty containers and zeroes are falsy; everything else is truthy.

```python
# Falsy: 0  0.0  ''  []  {}  set()  ()  None  False        Truthy: everything else
if items:                        # "if non-empty"
    process(items)

port = config or 8080            # ❌ if config == 0, you wrongly get 8080
port = config if config is not None else 8080   # ✅ explicit
```

#### The GIL & Concurrency Choice

> **The GIL (Global Interpreter Lock)** lets only one thread run Python bytecode at a time in CPython. Threads give **no speedup for CPU-bound work** — use *processes* (§12). Threads/async help **I/O-bound** work because the GIL is released during waits. (3.13's free-threaded build lifts this — §15.)

#### Common Gotchas

```python
# 1. Mutable default argument — shared across ALL calls
def bad(lst=[]):    lst.append(1); return lst   # ❌ persists between calls
def good(lst=None): lst = lst or []; ...         # ✅ fresh list each call

# 2. is vs == — identity vs equality
[1, 2] == [1, 2]    # → True   (equal values)
[1, 2] is [1, 2]    # → False  (different objects)
x is None           # ✅ always use 'is' for None

# 3. Late-binding closures — 'i' looked up at call time
fns = [lambda: i for i in range(3)]
fns[0]()            # → 2  ❌ (all share one 'i')
fns = [lambda i=i: i for i in range(3)]   # ✅ default arg freezes value

# 4. Float precision
0.1 + 0.2 == 0.3            # → False ❌
math.isclose(0.1 + 0.2, 0.3)  # → True ✅   (or use Decimal, §18)

# 5. Modifying a collection while iterating
for k in list(d.keys()):    # ✅ iterate a copy of the keys
    ...
```
