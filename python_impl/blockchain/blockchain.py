import json
import time
from typing import List, Dict, Any, Optional
from .block import Block

class Blockchain:
    def __init__(self, difficulty: int = 4):
        self.chain: List[Block] = [self.create_genesis_block()]
        self.difficulty = difficulty
        self.pending_transactions: List[Dict[str, Any]] = []
        self.mining_reward = 10.0

    def create_genesis_block(self) -> Block:
        return Block(0, [], time.time(), "0")

    def get_latest_block(self) -> Block:
        return self.chain[-1]

    def mine_pending_transactions(self, miner_address: str) -> None:
        # Create new block with all pending transactions
        block = Block(
            len(self.chain),
            self.pending_transactions,
            time.time(),
            self.get_latest_block().hash
        )

        # Mine the block
        block.mine_block(self.difficulty)

        # Add the block to the chain
        self.chain.append(block)

        # Reset pending transactions and add mining reward
        self.pending_transactions = [
            {"from": "network", "to": miner_address, "amount": self.mining_reward}
        ]

    def add_transaction(self, transaction: Dict[str, Any]) -> None:
        # Verify transaction has required fields
        if not all(k in transaction for k in ["from", "to", "amount"]):
            raise ValueError("Transaction must include 'from', 'to', and 'amount' fields")

        # Verify transaction amount is positive
        if transaction["amount"] <= 0:
            raise ValueError("Transaction amount must be positive")

        self.pending_transactions.append(transaction)

    def get_balance(self, address: str) -> float:
        balance = 0.0

        for block in self.chain:
            for transaction in block.transactions:
                if transaction["from"] == address:
                    balance -= transaction["amount"]
                if transaction["to"] == address:
                    balance += transaction["amount"]

        return balance

    def is_chain_valid(self) -> bool:
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]

            # Verify current block's hash
            if current_block.hash != current_block.calculate_hash():
                return False

            # Verify chain linkage
            if current_block.previous_hash != previous_block.hash:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain": [block.to_dict() for block in self.chain],
            "difficulty": self.difficulty,
            "pending_transactions": self.pending_transactions,
            "mining_reward": self.mining_reward
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Blockchain':
        blockchain = cls(difficulty=data["difficulty"])
        blockchain.chain = [Block.from_dict(block_data) for block_data in data["chain"]]
        blockchain.pending_transactions = data["pending_transactions"]
        blockchain.mining_reward = data["mining_reward"]
        return blockchain

    def save_to_file(self, filename: str) -> None:
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_from_file(cls, filename: str) -> 'Blockchain':
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data) 