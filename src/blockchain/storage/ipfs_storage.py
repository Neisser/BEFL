import json
import asyncio
from typing import Any, Dict, Optional
import aiohttp
from .storage_interface import StorageInterface

class IPFSStorage(StorageInterface):
    """IPFS implementation of the storage interface."""
    
    def __init__(self, ipfs_api_url: str = "http://localhost:5001"):
        """
        Initialize IPFS storage.
        
        Args:
            ipfs_api_url: URL of the IPFS API server
        """
        self.api_url = ipfs_api_url
        self.session = None
        
    async def __aenter__(self):
        """Create aiohttp session when entering context."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session when exiting context."""
        if self.session:
            await self.session.close()
            
    async def _make_request(self, endpoint: str, method: str = "POST", **kwargs) -> Dict:
        """
        Make a request to the IPFS API.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            **kwargs: Additional request parameters
            
        Returns:
            Dict: Response from IPFS API
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        url = f"{self.api_url}/api/v0/{endpoint}"
        async with self.session.request(method, url, **kwargs) as response:
            response.raise_for_status()
            return await response.json()
            
    async def store(self, data: Any, metadata: Optional[Dict] = None) -> str:
        """
        Store data in IPFS.
        
        Args:
            data: The data to store
            metadata: Optional metadata about the data
            
        Returns:
            str: IPFS hash of the stored data
        """
        # Convert data to bytes if it's not already
        if isinstance(data, (dict, list)):
            data_bytes = json.dumps(data).encode()
        elif isinstance(data, str):
            data_bytes = data.encode()
        else:
            data_bytes = data
            
        # Prepare form data
        form = aiohttp.FormData()
        form.add_field('file', data_bytes)
        
        if metadata:
            form.add_field('metadata', json.dumps(metadata))
            
        # Add to IPFS
        response = await self._make_request('add', data=form)
        return response['Hash']
        
    async def retrieve(self, identifier: str) -> Any:
        """
        Retrieve data from IPFS.
        
        Args:
            identifier: IPFS hash of the data
            
        Returns:
            The retrieved data
        """
        response = await self._make_request(f'cat?arg={identifier}')
        return response
        
    async def delete(self, identifier: str) -> bool:
        """
        Delete data from IPFS (note: IPFS is immutable, this just removes local pin).
        
        Args:
            identifier: IPFS hash of the data
            
        Returns:
            bool: True if unpinning was successful
        """
        try:
            await self._make_request(f'pin/rm?arg={identifier}')
            return True
        except Exception:
            return False
            
    async def exists(self, identifier: str) -> bool:
        """
        Check if data exists in IPFS.
        
        Args:
            identifier: IPFS hash to check
            
        Returns:
            bool: True if data exists
        """
        try:
            await self._make_request(f'object/stat?arg={identifier}')
            return True
        except Exception:
            return False 