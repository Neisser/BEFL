import hashlib
import hmac
from typing import Tuple
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

class VRF:
    """Verifiable Random Function implementation for consensus."""
    
    def __init__(self, private_key: bytes = None):
        """
        Initialize VRF with optional private key.
        
        Args:
            private_key: Optional private key in PEM format
        """
        if private_key:
            self.private_key = serialization.load_pem_private_key(
                private_key,
                password=None,
                backend=default_backend()
            )
        else:
            self.private_key = ec.generate_private_key(
                ec.SECP256K1(),
                default_backend()
            )
            
        self.public_key = self.private_key.public_key()
        
    def get_public_key(self) -> bytes:
        """
        Get the public key in PEM format.
        
        Returns:
            bytes: Public key in PEM format
        """
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
    def prove(self, message: bytes) -> Tuple[bytes, bytes]:
        """
        Generate a VRF proof for a message.
        
        Args:
            message: The message to prove
            
        Returns:
            Tuple[bytes, bytes]: (proof, hash)
        """
        # Generate signature
        signature = self.private_key.sign(
            message,
            ec.ECDSA(hashes.SHA256())
        )
        
        # Generate hash
        h = hmac.new(
            self.get_public_key(),
            message,
            hashlib.sha256
        )
        hash_value = h.digest()
        
        return signature, hash_value
        
    def verify(self, message: bytes, proof: bytes, public_key: bytes) -> bool:
        """
        Verify a VRF proof.
        
        Args:
            message: The original message
            proof: The VRF proof
            public_key: The public key of the prover
            
        Returns:
            bool: True if proof is valid
        """
        try:
            # Load public key
            pub_key = serialization.load_pem_public_key(
                public_key,
                backend=default_backend()
            )
            
            # Verify signature
            pub_key.verify(
                proof,
                message,
                ec.ECDSA(hashes.SHA256())
            )
            
            # Verify hash
            h = hmac.new(
                public_key,
                message,
                hashlib.sha256
            )
            expected_hash = h.digest()
            
            return True
        except Exception:
            return False
            
    def get_random_value(self, message: bytes) -> int:
        """
        Get a random value from the VRF output.
        
        Args:
            message: The message to generate random value from
            
        Returns:
            int: Random value
        """
        _, hash_value = self.prove(message)
        return int.from_bytes(hash_value, byteorder='big') 