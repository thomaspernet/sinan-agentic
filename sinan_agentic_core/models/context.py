"""Generic Context model for agents - stores database connector and query results.

This is a generic version that can be extended for specific use cases.
Replace 'database_connector' with your specific database implementation.
"""

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Replace with your database connector type
    pass


@dataclass
class AgentContext:
    """Generic context passed to all agents.

    This context is populated by the orchestrator as agents execute:
    - database_connector: Your database connection (Neo4j, PostgreSQL, etc.)
    - schema: Formatted schema string for agent instructions
    - query_results: Results from database queries
    - filters: User pre-selected filters/parameters
    - discovered_data: Data discovered during workflow execution

    The context owns its ``schema_data``, ``query_results``, ``filters``, and
    ``discovered_data`` collections in every direction: what seeds them is
    copied on construction, what :meth:`add_query_result` and
    :meth:`add_discovered_item` collect into them is copied on the way in, and
    what :meth:`get_discovered_items` hands back is copied on the way out. No
    caller keeps a handle on what a run reads, and no consumer of a reading can
    write back into it. ``database_connector`` is the one caller-supplied value
    left shared — it is a live handle the run uses, not data the context
    accumulates into.

    The accessors are the read boundary; the fields themselves stay live. A
    holder of the context reads ``query_results`` or writes ``discovered_data``
    directly and sees exactly what the run sees, the same split
    :class:`~sinan_agentic_core.session.agent_session.AgentSession` draws
    between its ``history`` attribute and ``get_history()``.

    Extend this class for your specific use case by adding domain-specific fields.
    A subclass that defines its own ``__post_init__`` must call
    ``super().__post_init__()``, or its instances go back to aliasing the seeds.
    That call copies the four collections declared here and nothing else — the
    base class cannot reach a field it does not declare — so a subclass that
    adds a collection of its own must copy that one itself.

    Example:
        @dataclass
        class MyAppContext(AgentContext):
            user_id: str = ""
            workspace_id: str = ""
            custom_metadata: dict[str, Any] = field(default_factory=dict)

            def __post_init__(self) -> None:
                super().__post_init__()
                self.custom_metadata = copy.deepcopy(self.custom_metadata)
    """

    database_connector: Any  # Replace with your specific database connector type
    schema: str = ""  # Formatted schema string for agent instructions
    schema_data: dict[str, Any] | None = None  # Raw schema data
    query_results: list[dict[str, Any]] = field(default_factory=list)
    filters: dict[str, Any] | None = None  # User pre-selected filters
    discovered_data: dict[str, Any] = field(default_factory=dict)  # Data discovered during workflow

    def __post_init__(self) -> None:
        """Take ownership of the seeded collections.

        ``default_factory`` only covers the omitted-argument case. The context
        is built from caller keyword data — ``setup_context(**context_data)``
        forwards straight into ``AgentContext(**context_data)`` — so a caller
        that seeds a collection keeps it aliased into the object a whole run
        mutates in place: :meth:`add_query_result` extends ``query_results``
        and :meth:`add_discovered_item` appends into ``discovered_data[key]``.
        Copying here means a run cannot write back into the caller's own
        containers, a caller editing them mid-run cannot change what agents
        read, and two contexts seeded from one collection do not share it.

        The copies are deep. Each of the four nests further mutable values — a
        result row, a schema block, a list of filter values, the list
        :meth:`add_discovered_item` appends into — and a shallow copy would
        detach only the outer container, leaving every one of those still
        aliased. Live objects belong in ``database_connector``, which is not
        copied; these four carry data.
        """
        self.schema_data = copy.deepcopy(self.schema_data)
        self.query_results = copy.deepcopy(self.query_results)
        self.filters = copy.deepcopy(self.filters)
        self.discovered_data = copy.deepcopy(self.discovered_data)

    @property
    def has_data(self) -> bool:
        """Check if any query results have been collected."""
        return len(self.query_results) > 0

    def add_query_result(self, result: dict[str, Any]) -> None:
        """Collect the rows of a database query result into the context.

        The rows are copied on the way in, for the reason the constructor
        copies the seeded list: the context accumulates them for the rest of
        the run, so a caller that keeps its payload and edits a row would
        rewrite what every later reader sees. The copy is deep — a row is a
        dict that nests further mutable values, and a shallow copy would
        detach only the list of rows.

        Args:
            result: Dictionary containing query results under a ``data`` key
        """
        if not isinstance(result, dict) or "data" not in result:
            return

        data = result["data"]
        if not isinstance(data, list):
            return

        self.query_results.extend(copy.deepcopy(data))

    def clear_results(self) -> None:
        """Clear all accumulated results."""
        self.query_results = []
        self.discovered_data = {}

    def add_discovered_item(self, key: str, value: Any) -> None:
        """Collect a discovery under *key*, as a snapshot of *value*.

        The value is copied on the way in, so what the run discovered stays
        what it discovered even if the caller edits its object afterwards, and
        adding one object twice records two independent entries. ``value`` is
        declared ``Any`` — the case a container-level copy cannot reach — so
        the copy goes all the way down, which makes deep-copyability the
        contract for a discovery and this the wrong place for a live handle.
        Those belong in ``database_connector``.

        Args:
            key: Category/type of discovered data
            value: The discovered data, taken as a snapshot
        """
        stored = copy.deepcopy(value)

        if key not in self.discovered_data:
            self.discovered_data[key] = []

        if isinstance(self.discovered_data[key], list):
            self.discovered_data[key].append(stored)
        else:
            self.discovered_data[key] = stored

    def get_discovered_items(self, key: str) -> Any:
        """Get a snapshot of the discoveries collected under *key*.

        The value is copied on the way out: without it a consumer appending to
        the returned list would write straight into the context, and two
        readers of one key would share an object. The copy is deep for the
        same reason :meth:`add_discovered_item` copies deeply — the value is
        declared ``Any`` and nests whatever the run put there. A key that was
        never collected copies to ``None``, so the miss needs no branch.

        Args:
            key: Category/type of discovered data

        Returns:
            A detached copy of the discovered data, or None if not found
        """
        return copy.deepcopy(self.discovered_data.get(key))
