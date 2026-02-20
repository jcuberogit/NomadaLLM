"""
Usage Tracker for NomadaLLM SDK

Tracks API call usage locally to enforce daily limits.
Data is stored in a local SQLite database.
"""

import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from nomadallm.licensing.validator import LicenseTier


@dataclass
class UsageStats:
    """Usage statistics for the current period."""
    calls_today: int
    calls_remaining: int
    daily_limit: int
    reset_time: datetime
    tier: LicenseTier
    
    @property
    def is_limit_reached(self) -> bool:
        """Check if daily limit has been reached."""
        if self.daily_limit == -1:  # Unlimited
            return False
        return self.calls_today >= self.daily_limit
    
    @property
    def usage_percentage(self) -> float:
        """Get usage as percentage of daily limit."""
        if self.daily_limit == -1:
            return 0.0
        return (self.calls_today / self.daily_limit) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "calls_today": self.calls_today,
            "calls_remaining": self.calls_remaining,
            "daily_limit": self.daily_limit,
            "reset_time": self.reset_time.isoformat(),
            "tier": self.tier.value,
            "is_limit_reached": self.is_limit_reached,
            "usage_percentage": round(self.usage_percentage, 2),
        }


class UsageTracker:
    """
    Tracks SDK usage locally.
    
    Usage data is stored in a local SQLite database.
    Daily limits reset at midnight UTC.
    
    Usage:
        tracker = UsageTracker(tier=LicenseTier.FREE)
        
        # Check if can make a call
        if tracker.can_call():
            # Make the call
            tracker.record_call()
        else:
            print("Daily limit reached")
    """
    
    def __init__(
        self,
        tier: LicenseTier = LicenseTier.FREE,
        db_path: Optional[str] = None
    ):
        """
        Initialize the usage tracker.
        
        Args:
            tier: The license tier (determines daily limit).
            db_path: Path to SQLite database. Defaults to ~/.nomadallm/usage.db
        """
        self.tier = tier
        self.daily_limit = tier.daily_limit
        
        # Set up database path
        if db_path is None:
            home = Path.home()
            nomadallm_dir = home / ".nomadallm"
            nomadallm_dir.mkdir(exist_ok=True)
            db_path = str(nomadallm_dir / "usage.db")
        
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    calls INTEGER DEFAULT 0,
                    tier TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_date ON usage(date)
            """)
            conn.commit()
    
    def _get_today(self) -> str:
        """Get today's date in UTC as string."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    def _get_reset_time(self) -> datetime:
        """Get the next reset time (midnight UTC)."""
        now = datetime.now(timezone.utc)
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if tomorrow <= now:
            from datetime import timedelta
            tomorrow += timedelta(days=1)
        return tomorrow
    
    def _get_or_create_today_record(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Get or create today's usage record."""
        today = self._get_today()
        now = datetime.now(timezone.utc).isoformat()
        
        cursor = conn.execute(
            "SELECT id, calls FROM usage WHERE date = ?",
            (today,)
        )
        row = cursor.fetchone()
        
        if row:
            return {"id": row[0], "calls": row[1]}
        
        # Create new record for today
        conn.execute(
            """
            INSERT INTO usage (date, calls, tier, created_at, updated_at)
            VALUES (?, 0, ?, ?, ?)
            """,
            (today, self.tier.value, now, now)
        )
        conn.commit()
        
        return {"id": conn.execute("SELECT last_insert_rowid()").fetchone()[0], "calls": 0}
    
    def get_usage(self) -> UsageStats:
        """
        Get current usage statistics.
        
        Returns:
            UsageStats with current usage info.
        """
        with sqlite3.connect(self.db_path) as conn:
            record = self._get_or_create_today_record(conn)
            calls_today = record["calls"]
        
        calls_remaining = self.daily_limit - calls_today if self.daily_limit != -1 else -1
        
        return UsageStats(
            calls_today=calls_today,
            calls_remaining=max(0, calls_remaining) if calls_remaining != -1 else -1,
            daily_limit=self.daily_limit,
            reset_time=self._get_reset_time(),
            tier=self.tier,
        )
    
    def can_call(self) -> bool:
        """
        Check if a call can be made within the daily limit.
        
        Returns:
            True if call is allowed, False if limit reached.
        """
        if self.daily_limit == -1:  # Unlimited
            return True
        
        usage = self.get_usage()
        return not usage.is_limit_reached
    
    def record_call(self) -> UsageStats:
        """
        Record a single API call.
        
        Returns:
            Updated UsageStats after recording the call.
            
        Raises:
            RuntimeError: If daily limit has been reached.
        """
        if not self.can_call():
            usage = self.get_usage()
            raise RuntimeError(
                f"Daily limit reached ({usage.daily_limit} calls). "
                f"Upgrade at https://nomadallm.nomadahealth.com/pricing"
            )
        
        with sqlite3.connect(self.db_path) as conn:
            record = self._get_or_create_today_record(conn)
            now = datetime.now(timezone.utc).isoformat()
            
            conn.execute(
                "UPDATE usage SET calls = calls + 1, updated_at = ? WHERE id = ?",
                (now, record["id"])
            )
            conn.commit()
        
        return self.get_usage()
    
    def record_calls(self, count: int) -> UsageStats:
        """
        Record multiple API calls at once.
        
        Args:
            count: Number of calls to record.
            
        Returns:
            Updated UsageStats after recording.
        """
        for _ in range(count):
            self.record_call()
        return self.get_usage()
    
    def reset_usage(self) -> None:
        """
        Reset today's usage count.
        
        This is mainly for testing purposes.
        """
        with sqlite3.connect(self.db_path) as conn:
            today = self._get_today()
            now = datetime.now(timezone.utc).isoformat()
            
            conn.execute(
                "UPDATE usage SET calls = 0, updated_at = ? WHERE date = ?",
                (now, today)
            )
            conn.commit()
    
    def update_tier(self, tier: LicenseTier) -> None:
        """
        Update the license tier.
        
        Args:
            tier: New license tier.
        """
        self.tier = tier
        self.daily_limit = tier.daily_limit
    
    def get_history(self, days: int = 30) -> list:
        """
        Get usage history for the past N days.
        
        Args:
            days: Number of days to retrieve.
            
        Returns:
            List of daily usage records.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT date, calls, tier FROM usage
                ORDER BY date DESC
                LIMIT ?
                """,
                (days,)
            )
            
            return [
                {"date": row[0], "calls": row[1], "tier": row[2]}
                for row in cursor.fetchall()
            ]
