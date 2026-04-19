"""Generic Context model for agents - stores database connector and query results.

This is a generic version that can be extended for specific use cases.
Replace 'database_connector' with your specific database implementation.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Dict, Any, List

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
    
    Extend this class for your specific use case by adding domain-specific fields.
    
    Example:
        @dataclass
        class MyAppContext(AgentContext):
            user_id: str = ""
            workspace_id: str = ""
            custom_metadata: Dict[str, Any] = field(default_factory=dict)
    """
    
    database_connector: Any  # Replace with your specific database connector type
    schema: str = ""  # Formatted schema string for agent instructions
    schema_data: Optional[Dict[str, Any]] = None  # Raw schema data
    query_results: list[dict] = field(default_factory=list)
    filters: Optional[Dict[str, Any]] = None  # User pre-selected filters
    discovered_data: Dict[str, Any] = field(default_factory=dict)  # Data discovered during workflow
    
    @property
    def has_data(self) -> bool:
        """Check if any query results have been collected."""
        return len(self.query_results) > 0
    
    def add_query_result(self, result: dict) -> None:
        """Add a database query result to the context.
        
        Args:
            result: Dictionary containing query results
        """
        if isinstance(result, dict) and 'data' in result:
            data = result['data']
            if isinstance(data, list):
                self.query_results.extend(data)
    
    def clear_results(self) -> None:
        """Clear all accumulated results."""
        self.query_results = []
        self.discovered_data = {}
    
    def add_discovered_item(self, key: str, value: Any) -> None:
        """Add discovered data during workflow execution.
        
        Args:
            key: Category/type of discovered data
            value: The discovered data
        """
        if key not in self.discovered_data:
            self.discovered_data[key] = []
        
        if isinstance(self.discovered_data[key], list):
            self.discovered_data[key].append(value)
        else:
            self.discovered_data[key] = value
    
    def get_discovered_items(self, key: str) -> Any:
        """Get discovered data by key.
        
        Args:
            key: Category/type of discovered data
            
        Returns:
            Discovered data or None if not found
        """
        return self.discovered_data.get(key)
