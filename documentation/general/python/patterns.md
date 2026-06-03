---
mandatory_for:
  skills: [fix-issue, feat-issue]
  rules: [backend]
---

# Python Patterns

Reusable patterns for modern Python applications. Framework-agnostic where possible.

---

## Async-First

All I/O operations are async. No sync alternatives maintained.

Repository methods, service methods, and API endpoints are all `async def`. Sync code only exists at the very edges (CLI entry points, test utilities).

```python
# Repository — always async
async def get_by_id(self, uuid: str) -> Entity | None:
    return await self.db.query_single("...", {"uuid": uuid})

# Service — always async
async def process(self, uuid: str) -> ProcessResult:
    entity = require_entity(await self.repo.get_by_id(uuid), "Entity", uuid)
    return await self._run_pipeline(entity)

# API endpoint — always async
@router.get("/{uuid}")
async def get_entity(uuid: str) -> EntityResponse:
    return await entity_service.get_by_id(uuid)
```

### Streaming with AsyncIterator

```python
async def stream_chunks(self, doc_id: str) -> AsyncIterator[Chunk]:
    async for record in self.db.stream("...", {"id": doc_id}):
        yield Chunk.from_record(record)
```

### Resource lifecycle with @asynccontextmanager

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    db = await Database.connect(settings.db_url)
    try:
        yield {"db": db}
    finally:
        await db.close()
```

### Offloading CPU-bound work

```python
import asyncio

async def hash_file(path: Path) -> str:
    return await asyncio.to_thread(_compute_sha256, path)

def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()
```

### Concurrent independent operations

```python
title, summary, keywords = await asyncio.gather(
    extract_title(content),
    generate_summary(content),
    extract_keywords(content),
)
```

---

## Error Handling

Minimal exception types. No deep custom hierarchies unless genuinely needed.

| Exception | Meaning | HTTP mapping |
|-----------|---------|-------------|
| `ValueError` | Client error — bad input, missing entity | 400 |
| `RuntimeError` | Server error — unexpected failure | 500 |

### Global exception handler

```python
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc):
    logger.error("Unexpected error", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
```

### Exception chaining

Always preserve the original cause:

```python
try:
    data = parse_document(raw)
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid document format: {path}") from e
```

### require_entity helper

```python
def require_entity(value: T | None, name: str, identifier: str) -> T:
    if value is None:
        raise ValueError(f"{name} not found: {identifier}")
    return value

# Usage — one-liner guard
entity = require_entity(await repo.get_by_id(uuid), "Entity", uuid)
```

### Decorator for endpoint error wrapping

```python
from functools import wraps

def handle_service_errors(fn):
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except ValueError:
            raise  # let global handler convert to 400
        except Exception as e:
            raise RuntimeError(f"Unexpected error in {fn.__name__}") from e
    return wrapper
```

---

## Guard Clauses / Early Returns

Validate at the top, fail fast. Flatten control flow with early returns.

```python
# Bad — deeply nested
async def process(self, uuid: str, content: str) -> Result:
    entity = await self.repo.get(uuid)
    if entity:
        if entity.status == "active":
            if content:
                return await self._do_work(entity, content)
            else:
                raise ValueError("Content is empty")
        else:
            raise ValueError("Entity is not active")
    else:
        raise ValueError("Entity not found")

# Good — flat with early returns
async def process(self, uuid: str, content: str) -> Result:
    entity = require_entity(await self.repo.get(uuid), "Entity", uuid)
    if entity.status != "active":
        raise ValueError(f"Entity is not active: {uuid}")
    if not content:
        raise ValueError("Content is empty")
    return await self._do_work(entity, content)
```

Short-circuit returns for optional work:

```python
async def enrich(self, doc: Document) -> Document:
    if doc.is_enriched:
        return doc
    metadata = await self.extractor.extract(doc.content)
    return doc.with_metadata(metadata)
```

---

## Composition Over Inheritance

Services compose sub-services by concern. No deep class hierarchies.

```python
# Good — composition (has-a)
class DocumentService:
    def __init__(self, repo: DocumentRepository, parser: ParserService, indexer: IndexService):
        self._repo = repo
        self._parser = parser
        self._indexer = indexer

    async def ingest(self, file: UploadFile) -> Document:
        parsed = await self._parser.parse(file)
        doc = await self._repo.create(parsed)
        await self._indexer.index(doc)
        return doc
```

### ABC interface + concrete implementation (max two levels)

```python
from abc import ABC, abstractmethod

class DocumentRepository(ABC):
    @abstractmethod
    async def get_by_id(self, uuid: str) -> Document | None: ...

    @abstractmethod
    async def create(self, data: DocumentCreate) -> Document: ...

class GraphDocumentRepository(DocumentRepository):
    async def get_by_id(self, uuid: str) -> Document | None:
        return await self._db.query_single("...", {"uuid": uuid})

    async def create(self, data: DocumentCreate) -> Document:
        return await self._db.query_single("...", data.model_dump())
```

### Mixins for shared query fragments

```python
class PaginationMixin:
    def paginate_query(self, query: str, skip: int, limit: int) -> str:
        return f"{query} SKIP {skip} LIMIT {limit}"

class GraphDocumentRepository(DocumentRepository, PaginationMixin):
    async def list(self, skip: int = 0, limit: int = 50) -> list[Document]:
        query = self.paginate_query(f"MATCH (d:{NodeType.DOCUMENT}) RETURN d", skip, limit)
        return await self._db.query(query)
```

---

## Design Patterns

### Factory — @classmethod for object creation

```python
class Pipeline:
    def __init__(self, steps: list[Step], config: PipelineConfig):
        self._steps = steps
        self._config = config

    @classmethod
    def from_config(cls, config: dict) -> "Pipeline":
        steps = [Step.from_dict(s) for s in config["steps"]]
        return cls(steps=steps, config=PipelineConfig(**config))

    @classmethod
    def default(cls) -> "Pipeline":
        return cls.from_config(load_yaml("default_pipeline.yaml"))
```

### Strategy — interchangeable algorithms

```python
class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]: ...

class SentenceChunker(ChunkingStrategy):
    def chunk(self, text: str) -> list[str]:
        return split_sentences(text)

class TokenChunker(ChunkingStrategy):
    def chunk(self, text: str) -> list[str]:
        return split_tokens(text, max_tokens=512)

# Selected at runtime
chunker = strategies[config.chunking_method]
chunks = chunker.chunk(document.content)
```

### Builder — incremental construction

```python
class QueryBuilder:
    def __init__(self):
        self._clauses: list[str] = []
        self._params: dict = {}

    def match(self, pattern: str) -> "QueryBuilder":
        self._clauses.append(f"MATCH {pattern}")
        return self

    def where(self, condition: str, **params) -> "QueryBuilder":
        self._clauses.append(f"WHERE {condition}")
        self._params.update(params)
        return self

    def build(self) -> tuple[str, dict]:
        return " ".join(self._clauses), self._params
```

---

## Modern Python Syntax

### match/case for complex branching

```python
def classify_status(code: int) -> str:
    match code:
        case 200 | 201:
            return "success"
        case 400:
            return "client_error"
        case 404:
            return "not_found"
        case code if 500 <= code < 600:
            return "server_error"
        case _:
            return "unknown"
```

### Walrus operator for assignment in conditionals

```python
if (doc := await repo.get_by_id(uuid)):
    return doc.to_response()

if (match := pattern.search(text)):
    return match.group(1)
```

### List comprehensions and generator expressions

```python
# Transform
active = [u.name for u in users if u.is_active]

# Aggregate
success_count = sum(1 for r in results if r.success)
total_size = sum(f.size for f in files)

# Concurrent async
results = await asyncio.gather(*[process(item) for item in items])
```

---

## No Hardcoded Values

Thresholds, timeouts, limits, model names, feature flags, retry counts, page sizes, file size limits — all belong in configuration, never in code. Code defines the mechanism. Configuration defines the policy.

```python
# Bad — hardcoded
async def search(self, query: str) -> list[Result]:
    results = await self._repo.search(query, limit=50, timeout=30)
    if len(results) > 200:
        results = results[:200]
    return results

# Good — from config
async def search(self, query: str) -> list[Result]:
    results = await self._repo.search(
        query,
        limit=config.search.default_limit,
        timeout=config.search.timeout,
    )
    if len(results) > config.search.max_results:
        results = results[:config.search.max_results]
    return results
```

The test: if a product decision changes (bigger page size, longer timeout, different model), can you change it without touching code? If yes, it is decoupled. If no, the business rule is hardcoded.

Default to extracting. Any literal that is not structurally trivial — `0`, `1`, `""`, an index — is a named constant or config value until proven otherwise. "It's not really a policy" is the rationalization that leaves magic numbers scattered through the code; if you have to argue the case, extract it.

A module-level constant is not the same as config. Hoisting `MAX_RESULTS = 200` to the top of the same file removes the magic number, but the value still lives in code. If it is a policy a non-developer might tune — limits, timeouts, model names, thresholds, feature flags — it belongs in the project's configuration (see project conventions for where config lives), not just a local constant. Local constants are for values that are structural to that module and will only ever change in a code review.

This extends to any value that a non-developer might want to change: email subjects, default labels, retry delays, batch sizes, notification thresholds. If it is a policy decision, it belongs in config.

---

## Function Design

Keep functions short (5-50 lines). One responsibility per function.

### Before writing a new function, check what exists

Do not create a new helper if one already exists. Before writing a utility:

1. Check `utils/` or `helpers/` in the current module
2. Check `domain/constants/` for existing constants
3. Check sibling services for similar logic
4. Grep the codebase for the operation you need

If similar logic exists in another module, extract it into a shared location rather than duplicating it. If a private helper (`_do_something()`) is useful beyond its current class, promote it to a module-level function or a shared utility.

### Where helpers belong

| Scope | Location | Example |
|---|---|---|
| Used within one class | Private method (`_validate_input`) | Validation, formatting for one service |
| Used within one module | Module-level function (`_build_query`) | Query builders, record mappers |
| Used across modules in one layer | Shared `utils/` or `helpers.py` in that layer | Date formatting, string sanitization |
| Used across layers | `domain/` or dedicated utility package | UUID generation, generic type converters |

Do not scatter helpers across random files. A function that lives in `services/document/document_service.py` but is also called from `services/page/page_service.py` should be extracted into a shared location, not copied.

### Extract private helpers

```python
class ImportService:
    async def import_file(self, file: UploadFile) -> Document:
        content = await self._read_and_validate(file)
        doc = await self._repo.create(DocumentCreate(content=content, filename=file.filename))
        await self._publish_event("document.imported", doc.uuid)
        return doc

    async def _read_and_validate(self, file: UploadFile) -> str:
        content = (await file.read()).decode("utf-8")
        if not content.strip():
            raise ValueError(f"Empty file: {file.filename}")
        return content

    async def _publish_event(self, event_type: str, entity_id: str) -> None:
        await self._events.publish(Event(type=event_type, entity_id=entity_id))
```

### Properties for computed attributes

```python
class Document:
    def __init__(self, title: str, chunks: list[Chunk]):
        self.title = title
        self.chunks = chunks

    @property
    def word_count(self) -> int:
        return sum(len(c.text.split()) for c in self.chunks)

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0
```

### @staticmethod for stateless utilities

```python
class FileService:
    @staticmethod
    def sanitize_filename(name: str) -> str:
        return re.sub(r"[^\w\-.]", "_", name).lower()
```

---

## Date/Time

UTC-only. ISO 8601 as exchange format.

```python
from datetime import UTC, datetime, timedelta

# Always UTC
now = datetime.now(UTC)

# ISO 8601 for serialization
timestamp = now.isoformat()  # "2026-04-11T14:30:00+00:00"

# Safe parsing — return None on invalid input
def parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None

# Relative time helper
def utc_cutoff(*, minutes: int = 0, days: int = 0) -> datetime:
    return datetime.now(UTC) - timedelta(minutes=minutes, days=days)

# Usage
recent = await repo.find_since(utc_cutoff(days=7))
```
