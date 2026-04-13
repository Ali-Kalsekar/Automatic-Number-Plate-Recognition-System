from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.image_processing import ensure_directory


class DatabaseManager:
    """Persist recognized plates in SQLite and optionally CSV."""

    def __init__(
        self,
        sqlite_path: str | Path,
        csv_path: str | Path | None = None,
        save_to_database: bool = True,
        save_to_csv: bool = True,
    ) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.csv_path = Path(csv_path) if csv_path else None
        self.save_to_database = save_to_database
        self.save_to_csv = save_to_csv
        self._seen_plates: set[str] = set()

        ensure_directory(self.sqlite_path.parent)
        if self.csv_path is not None:
            ensure_directory(self.csv_path.parent)

        self.connection: sqlite3.Connection | None = None
        if self.save_to_database:
            self.connection = sqlite3.connect(self.sqlite_path)
            self.connection.execute("PRAGMA journal_mode=WAL")
            self._create_table()
            self._load_seen_plates_from_database()

        self._load_seen_plates_from_csv()

    def _create_table(self) -> None:
        """Create the plates table if needed."""

        if self.connection is None:
            return

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL UNIQUE,
                timestamp TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def plate_exists(self, plate_number: str) -> bool:
        """Check whether a plate already exists in the database."""

        plate_number = plate_number.strip().upper()
        if plate_number in self._seen_plates:
            return True

        if self.connection is None:
            return False

        cursor = self.connection.execute(
            "SELECT 1 FROM plates WHERE plate_number = ? LIMIT 1",
            (plate_number,),
        )
        exists = cursor.fetchone() is not None
        if exists:
            self._seen_plates.add(plate_number)
        return exists

    def _load_seen_plates_from_database(self) -> None:
        """Warm the in-memory duplicate cache from SQLite."""

        if self.connection is None:
            return

        cursor = self.connection.execute("SELECT plate_number FROM plates")
        self._seen_plates.update(row[0].strip().upper() for row in cursor.fetchall() if row and row[0])

    def _load_seen_plates_from_csv(self) -> None:
        """Warm the in-memory duplicate cache from the CSV log if it exists."""

        if self.csv_path is None or not self.csv_path.exists():
            return

        try:
            frame = pd.read_csv(self.csv_path)
        except Exception:
            return

        if "plate_number" not in frame.columns:
            return

        self._seen_plates.update(
            str(value).strip().upper()
            for value in frame["plate_number"].dropna().tolist()
            if str(value).strip()
        )

    def save_plate(self, plate_number: str, timestamp: Optional[str] = None) -> bool:
        """Insert a new plate if it has not already been recorded."""

        plate_number = plate_number.strip().upper()
        if not plate_number:
            return False

        if plate_number in self._seen_plates or self.plate_exists(plate_number):
            return False

        timestamp = timestamp or datetime.now().isoformat(timespec="seconds")
        inserted = False

        if self.save_to_database and self.connection is not None:
            try:
                self.connection.execute(
                    "INSERT INTO plates (plate_number, timestamp) VALUES (?, ?)",
                    (plate_number, timestamp),
                )
                self.connection.commit()
                inserted = True
            except sqlite3.IntegrityError:
                inserted = False
                self._seen_plates.add(plate_number)

        if self.save_to_csv and self.csv_path is not None:
            frame = pd.DataFrame(
                [{"plate_number": plate_number, "timestamp": timestamp}]
            )
            write_header = not self.csv_path.exists()
            frame.to_csv(self.csv_path, mode="a", header=write_header, index=False)
            inserted = True

        self._seen_plates.add(plate_number)

        return inserted

    def close(self) -> None:
        """Close the database connection."""

        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self) -> "DatabaseManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
