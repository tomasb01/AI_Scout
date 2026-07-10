"""Finding workflow states — open │ accepted_risk │ resolved (Sprint 2).

A local, human-editable JSON store keyed by stable finding IDs
(Sprint 0.1), so a security team's decisions persist across scans:
a finding marked ``accepted_risk`` stays accepted next month — visible
in the report with a badge, no longer screaming in the critical counts.

Self-hosted discipline: the store is a plain file next to the scans
(default ``.aiscout/findings.json``), no server, no phone-home. It
also tracks ``first_seen`` per finding ("this key has been here for
two scans now"), stamped automatically on every scan that uses the
store.

``resolved`` is never set by hand here — it is an automatic
observation of the diff (the finding is no longer detected).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from aiscout.models import AIAsset, now_utc

DEFAULT_STATE_PATH = ".aiscout/findings.json"

STATUS_OPEN = "open"
STATUS_ACCEPTED = "accepted_risk"
STATUS_RESOLVED = "resolved"

_MANUAL_STATUSES = {STATUS_OPEN, STATUS_ACCEPTED}


@dataclass
class FindingsState:
    """Persistent per-finding workflow state."""

    path: Path
    entries: dict[str, dict] = field(default_factory=dict)

    # ── Persistence ─────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path = DEFAULT_STATE_PATH) -> "FindingsState":
        p = Path(path)
        if not p.exists():
            return cls(path=p)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(path=p, entries=data.get("findings", {}))

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "aiscout-findings-state/1",
            "findings": {
                fid: self.entries[fid] for fid in sorted(self.entries)
            },
        }
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return self.path

    # ── Workflow transitions (manual) ───────────────────────────────────

    def accept(self, finding_id: str, note: str = "") -> dict:
        """Mark a finding as accepted risk (with an audit note)."""
        entry = self.entries.setdefault(finding_id, {})
        entry["status"] = STATUS_ACCEPTED
        entry["note"] = note
        entry["updated_at"] = now_utc().isoformat()
        return entry

    def reopen(self, finding_id: str) -> dict:
        """Return an accepted finding to the open state."""
        entry = self.entries.setdefault(finding_id, {})
        entry["status"] = STATUS_OPEN
        entry["updated_at"] = now_utc().isoformat()
        return entry

    def status_of(self, finding_id: str) -> str:
        return self.entries.get(finding_id, {}).get("status", STATUS_OPEN)

    # ── Scan integration ────────────────────────────────────────────────

    def apply_to_assets(self, assets: list[AIAsset]) -> dict:
        """Stamp statuses + first_seen onto scan findings; record new ones.

        Returns counts: how many findings are open / accepted in this
        scan. Must run BEFORE enrichment so risk derivation can respect
        accepted findings.
        """
        now = now_utc().isoformat()
        counts = {STATUS_OPEN: 0, STATUS_ACCEPTED: 0}
        for asset in assets:
            for f in asset.raw_findings:
                if not f.id:
                    continue
                entry = self.entries.setdefault(f.id, {})
                entry.setdefault("first_seen", now)
                status = entry.get("status", STATUS_OPEN)
                if status not in _MANUAL_STATUSES:
                    status = STATUS_OPEN
                f.status = status
                f.first_seen = entry["first_seen"]
                counts[status] += 1
        return counts
