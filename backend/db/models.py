"""
SQLAlchemy ORM models for user accounts.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Boolean
from backend.db.session import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Upstox OAuth tokens (stored per-user so we can make server-side API calls)
    upstox_access_token = Column(String, nullable=True)
    upstox_token_expiry = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<User {self.email}>"
