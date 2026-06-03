---
mandatory_for:
  skills: [fix-issue, feat-issue]
  rules: [backend]
---

# Python Backend Architecture

Guidelines for structuring Python backend applications using layered architecture, dependency inversion, and clean module organization.

## Layered Architecture

Four layers, each with a single responsibility:

```
API  →  Application  →  Infrastructure  →  Domain
```

**Domain** -- the core. Entities, value objects, repository interfaces (ABCs), constants, and enums. This layer has zero external dependencies. It defines *what* the system is, not how it works.

**Application** -- business logic orchestration. Services coordinate domain objects and repository interfaces to fulfill use cases. No database calls, no HTTP concerns, no framework imports.

**Infrastructure** -- concrete implementations. Database repositories, API clients, file storage adapters, message publishers, connectors. This is the only layer that touches external systems.

**API** -- HTTP endpoints, request/response schemas, authentication middleware. Thin translation layer between HTTP and application services. No business logic.

### Import Direction

Strictly downward. Each layer may only import from layers below it:

```
API → Application → Domain
API → Infrastructure → Domain
Application → Domain
```

Never upward. Infrastructure never imports from API. Domain never imports from anything above it. Application never imports from API or Infrastructure (it uses domain interfaces that infrastructure implements).

## Interface-First Design

Every repository and external connector is defined as an abstract class in the domain layer. The infrastructure layer provides the concrete implementation.

```python
# domain/repositories/user_repository.py
from abc import ABC, abstractmethod
from domain.models.entities.user import User

class IUserRepository(ABC):
    @abstractmethod
    def get_by_uuid(self, uuid: str) -> User | None:
        pass

    @abstractmethod
    def create(self, user: User) -> User:
        pass

    @abstractmethod
    def update(self, user: User) -> User:
        pass

    @abstractmethod
    def delete(self, uuid: str) -> None:
        pass

    @abstractmethod
    def get_all(self, project_uuid: str) -> list[User]:
        pass
```

Rules:

- Abstract method bodies use `pass`, not `...` or `raise NotImplementedError`.
- Naming convention: `I<Entity>Repository` for repository interfaces, `I<Service>Client` for external API clients.
- Services depend on interfaces, never on concrete implementations. This is Dependency Inversion -- the D in SOLID.

## Dependency Injection

### Constructor Injection

Services receive their dependencies through `__init__`. This makes dependencies explicit and testable.

```python
# application/services/user_service.py
from domain.repositories.user_repository import IUserRepository
from domain.repositories.event_publisher import IEventPublisher

class UserService:
    def __init__(
        self,
        user_repo: IUserRepository,
        event_publisher: IEventPublisher,
    ) -> None:
        self._user_repo = user_repo
        self._event_publisher = event_publisher

    def get_user(self, uuid: str) -> User | None:
        return self._user_repo.get_by_uuid(uuid)

    def create_user(self, user: User) -> User:
        created = self._user_repo.create(user)
        self._event_publisher.publish("user.created", created)
        return created
```

### Framework-Level Injection

Web framework dependency injection (e.g., FastAPI's `Depends()`) handles HTTP-level concerns: authentication, request context, database sessions. Keep this at the API layer only.

```python
# api/endpoints/user_endpoints.py
from fastapi import APIRouter, Depends

router = APIRouter()

@router.get("/users/{uuid}")
def get_user(uuid: str, service: UserService = Depends(get_user_service)):
    return service.get_user(uuid)
```

### Singleton and Proxy Patterns

Expensive resources (database connections, event publishers) use the singleton pattern -- instantiated once at startup, shared across the application.

When a shared resource has lifecycle methods (e.g., `close()`), wrap it in a proxy that prevents individual consumers from prematurely cleaning up the shared instance.

### Circular Dependency Resolution

Use lazy imports inside method bodies when two modules would otherwise create a circular import at module load time. This is the one acceptable case for non-top-level imports.

```python
def process_document(self, doc_uuid: str) -> None:
    from application.services.enrichment_service import EnrichmentService
    enrichment = EnrichmentService(self._enrichment_repo)
    enrichment.enrich(doc_uuid)
```

## Module Organization

Group by family first, then nest as a family grows. Organize related concepts into sub-packages rather than flat directories -- a flat `models/` directory with 40 files is harder to navigate than a tree grouped by domain. Add depth when a group grows large enough that unrelated files crowd it, not preemptively. (This is Pillar 7, Organize by Family, applied to Python packages.)

```
project/
    domain/
        models/
            entities/
                auth/
                    __init__.py
                    user.py
                    session.py
                document/
                    __init__.py
                    document.py
                    metadata.py
            value_objects/
                __init__.py
                file_path.py
                content_type.py
        repositories/
            __init__.py
            user_repository.py
            document_repository.py
        constants/
            __init__.py
            processing.py
    application/
        services/
            user/
                __init__.py
                user_service.py
                user_validation.py
            document/
                __init__.py
                document_service.py
                processing_service.py
    infrastructure/
        graph/
            repositories/
                __init__.py
                graph_user_repository.py
                graph_document_repository.py
            connectors/
                __init__.py
                database_connector.py
        storage/
            __init__.py
            file_storage.py
    api/
        endpoints/
            __init__.py
            user_endpoints.py
            document_endpoints.py
        schemas/
            __init__.py
            user_schemas.py
```

Rules:

- Every package has `__init__.py` with explicit `__all__` exports.
- No star imports (`from module import *`) in production code.
- Group related concepts into sub-packages: `services/document/`, `services/user/`.

## Standard CRUD Interface

Repository interfaces follow a consistent method signature pattern:

| Method | Signature | Returns |
|---|---|---|
| `create` | `create(entity: T) -> T` | Created entity |
| `get_by_uuid` | `get_by_uuid(uuid: str) -> T \| None` | Entity or None |
| `get_by_<field>` | `get_by_email(email: str) -> T \| None` | Entity or None |
| `get_all` | `get_all(scope: str) -> list[T]` | List of entities |
| `update` | `update(entity: T) -> T` | Updated entity |
| `delete` | `delete(uuid: str) -> None` | Nothing |
| `count` | `count(scope: str) -> int` | Count |

The interface lives in the domain layer. The implementation lives in infrastructure. The service instantiates the concrete repository in its constructor (or receives it via injection).

## No String Literals in Queries

Every entity name, table name, collection name, node label, or relationship type that appears in a database query must come from a constant or enum — never a raw string literal. This applies regardless of database technology (SQL, Cypher, MongoDB, Elasticsearch).

Define constants in the domain layer (e.g., `domain/constants/`). Repository implementations import and use them:

```python
# domain/constants/schema.py
from enum import StrEnum

class NodeType(StrEnum):
    USER = "User"
    DOCUMENT = "Document"
    TAG = "Tag"

class RelType(StrEnum):
    HAS_TAG = "HAS_TAG"
    BELONGS_TO = "BELONGS_TO"

# For SQL:
class TableName(StrEnum):
    USERS = "users"
    DOCUMENTS = "documents"
```

```python
# Good — constants
from domain.constants.schema import NodeType

result = connector.execute(
    f"MATCH (u:{NodeType.USER} {{uuid: $uuid}}) RETURN u",
    uuid=uuid,
)

# Bad — string literals
result = connector.execute(
    "MATCH (u:User {uuid: $uuid}) RETURN u",  # "User" is a magic string
    uuid=uuid,
)
```

Why this matters:
- Renaming an entity updates one constant, not every query that mentions it.
- Typos in constants fail at import time. Typos in strings fail at query time (or silently return empty results).
- A grep for `NodeType.USER` finds every query that touches users. A grep for `"User"` finds everything — logs, comments, unrelated strings.

This is an instance of Pillar 6 (Decouple Everything) applied to database queries — the query does not hardcode knowledge about the schema; it references the schema through a constant.

### Reserved Keyword Collisions

Some database engines (notably LadybugDB) fail when property names collide with query language reserved keywords. For example, `a.order` fails because `ORDER` is a Cypher reserved keyword. Either escape with backticks (`` a.`order` ``) or, better, avoid the collision entirely by naming the property `sort_order` or `display_order`. This applies to any property matching a reserved keyword (`ORDER`, `MATCH`, `RETURN`, `WHERE`, `SET`, `DELETE`, etc.).

## Repository Implementation

```python
# infrastructure/graph/repositories/graph_user_repository.py
from domain.constants.schema import NodeType
from domain.repositories.user_repository import IUserRepository

class GraphUserRepository(IUserRepository):
    def __init__(self, connector: DatabaseConnector) -> None:
        self._connector = connector

    def get_by_uuid(self, uuid: str) -> User | None:
        result = self._connector.execute(
            f"MATCH (u:{NodeType.USER} {{uuid: $uuid}}) RETURN u",
            uuid=uuid,
        )
        if not result:
            return None
        return User.from_record(result[0])

    def create(self, user: User) -> User:
        self._connector.execute(
            f"CREATE (u:{NodeType.USER} {{uuid: $uuid, name: $name, email: $email}})",
            uuid=user.uuid, name=user.name, email=user.email,
        )
        return user
```
