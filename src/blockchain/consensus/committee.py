from typing import List, Dict, Optional
from dataclasses import dataclass
from .vrf import VRF

@dataclass
class CommitteeMember:
    """Represents a member of the consensus committee."""
    node_id: str
    public_key: bytes
    stake: int
    is_active: bool = True

class CommitteeManager:
    """Manages the consensus committee selection and operations."""
    
    def __init__(self, vrf: VRF, min_stake: int = 1000):
        """
        Initialize committee manager.
        
        Args:
            vrf: VRF instance for random selection
            min_stake: Minimum stake required to be a committee member
        """
        self.vrf = vrf
        self.min_stake = min_stake
        self.members: Dict[str, CommitteeMember] = {}
        self.current_committee: List[str] = []
        
    def add_member(self, node_id: str, public_key: bytes, stake: int) -> bool:
        """
        Add a new potential committee member.
        
        Args:
            node_id: Unique identifier of the node
            public_key: Public key of the node
            stake: Stake amount of the node
            
        Returns:
            bool: True if member was added successfully
        """
        if stake < self.min_stake:
            return False
            
        self.members[node_id] = CommitteeMember(
            node_id=node_id,
            public_key=public_key,
            stake=stake
        )
        return True
        
    def update_stake(self, node_id: str, new_stake: int) -> bool:
        """
        Update a member's stake.
        
        Args:
            node_id: ID of the member to update
            new_stake: New stake amount
            
        Returns:
            bool: True if update was successful
        """
        if node_id not in self.members:
            return False
            
        if new_stake < self.min_stake:
            self.members[node_id].is_active = False
        else:
            self.members[node_id].stake = new_stake
            self.members[node_id].is_active = True
            
        return True
        
    def select_committee(self, committee_size: int, seed: bytes) -> List[str]:
        """
        Select a new committee using VRF.
        
        Args:
            committee_size: Number of members to select
            seed: Random seed for selection
            
        Returns:
            List[str]: List of selected node IDs
        """
        # Get active members with sufficient stake
        eligible_members = [
            m for m in self.members.values()
            if m.is_active and m.stake >= self.min_stake
        ]
        
        if not eligible_members:
            return []
            
        # Calculate total stake
        total_stake = sum(m.stake for m in eligible_members)
        
        # Select committee members
        selected = []
        while len(selected) < committee_size and eligible_members:
            # Get random value from VRF
            random_value = self.vrf.get_random_value(seed + b''.join(selected))
            
            # Select member based on stake
            threshold = (random_value % total_stake) + 1
            current_sum = 0
            
            for member in eligible_members:
                current_sum += member.stake
                if current_sum >= threshold:
                    selected.append(member.node_id)
                    eligible_members.remove(member)
                    total_stake -= member.stake
                    break
                    
        self.current_committee = selected
        return selected
        
    def is_committee_member(self, node_id: str) -> bool:
        """
        Check if a node is in the current committee.
        
        Args:
            node_id: ID of the node to check
            
        Returns:
            bool: True if node is in committee
        """
        return node_id in self.current_committee
        
    def get_committee_size(self) -> int:
        """
        Get current committee size.
        
        Returns:
            int: Number of members in committee
        """
        return len(self.current_committee)
        
    def get_member(self, node_id: str) -> Optional[CommitteeMember]:
        """
        Get committee member by ID.
        
        Args:
            node_id: ID of the member
            
        Returns:
            Optional[CommitteeMember]: Member if found, None otherwise
        """
        return self.members.get(node_id) 