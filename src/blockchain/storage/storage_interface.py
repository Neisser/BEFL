from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class StorageInterface(ABC):
    """Abstract base class for storage implementations."""
    
    @abstractmethod
    async def store(self, data: Any, metadata: Optional[Dict] = None) -> str:
        """
        Store data and return its unique identifier.
        
        Args:
            data: The data to store
            metadata: Optional metadata about the data
            
        Returns:
            str: Unique identifier for the stored data
        """
        pass
    
    @abstractmethod
    async def retrieve(self, identifier: str) -> Any:
        """
        Retrieve data using its identifier.
        
        Args:
            identifier: Unique identifier of the data
            
        Returns:
            The retrieved data
        """
        pass
    
    @abstractmethod
    async def delete(self, identifier: str) -> bool:
        """
        Delete data using its identifier.
        
        Args:
            identifier: Unique identifier of the data
            
        Returns:
            bool: True if deletion was successful
        """
        pass
    
    @abstractmethod
    async def exists(self, identifier: str) -> bool:
        """
        Check if data exists for the given identifier.
        
        Args:
            identifier: Unique identifier to check
            
        Returns:
            bool: True if data exists
        """
        pass 