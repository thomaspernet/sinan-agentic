"""Generic Agent Session - Custom session implementation for OpenAI Agent SDK.

This is a generic conversation history manager that works with the OpenAI Agents SDK.
"""

from typing import List, Optional, Dict, Any
from agents.memory.session import SessionABC
from agents.items import TResponseInputItem
import json


class ConversationHistory:
    """Generic conversation history container.
    
    Stores messages in a simple list format compatible with OpenAI SDK.
    Extend this for your specific storage needs (database persistence, etc.).
    """
    
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
    
    def to_list_dict(self) -> List[Dict[str, Any]]:
        """Convert to list of message dictionaries."""
        return self.messages.copy()
    
    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Add a message to history.
        
        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            **kwargs: Additional fields (name, tool_call_id, etc.)
        """
        msg = {"role": role, "content": content}
        msg.update(kwargs)
        self.messages.append(msg)
    
    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()


class AgentSession(SessionABC):
    """Generic session implementation for OpenAI Agents SDK.
    
    Manages conversation history for multi-agent workflows with automatic
    summarization when history becomes too long.
    
    Usage:
        session = AgentSession(session_id="unique-id")
        
        # Add messages manually
        await session.add_items([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ])
        
        # Get history
        history = await session.get_items()
    """
    
    def __init__(
        self, 
        session_id: str, 
        initial_history: Optional[ConversationHistory] = None,
        max_items: int = 50
    ):
        """Initialize session.
        
        Args:
            session_id: Unique identifier for this session
            initial_history: Optional existing conversation history
            max_items: Maximum messages before triggering summarization
        """
        self.session_id = session_id
        self.history = initial_history or ConversationHistory()
        self.max_items = max_items
        self.metadata: Dict[str, Any] = {}  # Store arbitrary session metadata
    
    async def get_items(self, limit: int | None = None) -> List[TResponseInputItem]:
        """Retrieve conversation history as list of dicts.
        
        Args:
            limit: Optional limit on number of items to return (from end)
            
        Returns:
            List of message dicts compatible with Agent SDK
        """
        items = self.history.to_list_dict()
        if limit:
            return items[-limit:]
        return items
    
    async def add_items(self, items: List[TResponseInputItem]) -> None:
        """Add new messages to conversation history.
        
        Args:
            items: List of message dicts to add
        """
        for item in items:
            # Skip empty messages
            content = item.get("content", "")
            if not content or (isinstance(content, str) and not content.strip()):
                continue
            
            # Clean up SDK's structured output format
            # SDK returns: [{'text': '{"response":...}', 'type': 'output_text', ...}]
            # We want: just the JSON string
            if isinstance(content, list) and len(content) > 0:
                if isinstance(content[0], dict) and 'text' in content[0]:
                    content = content[0]['text']
            
            # For assistant messages with structured output, extract response
            if item.get("role") == "assistant" and isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "response" in parsed:
                        content = json.dumps(parsed["response"])
                except (json.JSONDecodeError, ValueError):
                    pass
            
            # Create message dict
            msg = {
                "role": item.get("role"),
                "content": str(content)
            }
            
            # Add optional fields
            if "name" in item:
                msg["name"] = item["name"]
            if "tool_call_id" in item:
                msg["tool_call_id"] = item["tool_call_id"]
            
            self.history.messages.append(msg)
    
    async def pop_item(self) -> TResponseInputItem | None:
        """Remove and return the most recent message.
        
        Returns:
            Last message as dict, or None if empty
        """
        if self.history.messages:
            return self.history.messages.pop()
        return None
    
    async def clear_session(self) -> None:
        """Clear all messages from history."""
        self.history.messages.clear()
    
    def get_history(self) -> ConversationHistory:
        """Get full ConversationHistory object.
        
        Returns:
            ConversationHistory for debugging/persistence
        """
        return self.history
    
    def needs_summarization(self) -> bool:
        """Check if session has exceeded max items threshold.
        
        Returns:
            True if session should be summarized
        """
        return len(self.history.messages) > self.max_items
    
    def get_message_count(self) -> int:
        """Get current number of messages in session.
        
        Returns:
            Number of messages
        """
        return len(self.history.messages)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Store metadata in session.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve session metadata.
        
        Args:
            key: Metadata key
            default: Default value if key not found
            
        Returns:
            Metadata value or default
        """
        return self.metadata.get(key, default)
