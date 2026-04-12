"""JSON export — machine-readable scan results."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

from aiscout.engine.enrichment import AssetInsight
from aiscout.models import AIAsset, ScanResult


class JSONExporter:
    """Export scan results as structured JSON for analysis or integration."""

    def __init__(
        self,
        scan_results: list[ScanResult],
        output_path: str = "aiscout_report.json",
        insights: dict[str, AssetInsight] | None = None,
    ):
        self.scan_results = scan_results
        self.output_path = output_path
        self.insights = insights or {}

    def generate(self) -> Path:
        """Render scan results as JSON and write to disk."""
        data = self._build_data()
        out = Path(self.output_path)
        out.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
        return out

    def _build_data(self) -> dict:
        """Build the full JSON structure."""
        all_assets: list[AIAsset] = []
        all_errors: list[str] = []
        repos: list[dict] = []
        total_files = 0

        for result in self.scan_results:
            all_assets.extend(result.assets)
            all_errors.extend(result.errors)
            total_files += result.metadata.get("files_scanned", 0)
            repos.append({
                "name": result.metadata.get("repository", "unknown"),
                "branch": result.metadata.get("branch", "main"),
                "url": result.metadata.get("repo_url", ""),
                "files_scanned": result.metadata.get("files_scanned", 0),
                "assets_found": len(result.assets),
                "errors": len(result.errors),
                "started_at": result.started_at.isoformat() if result.started_at else None,
                "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            })

        # Risk counts
        critical = sum(1 for a in all_assets if a.risk_score >= 0.7)
        warning = sum(1 for a in all_assets if 0.4 <= a.risk_score < 0.7)
        ok = sum(1 for a in all_assets if a.risk_score < 0.4)

        # Build solutions
        solutions = [self._asset_to_dict(asset) for asset in all_assets]

        # Group by category
        categories: dict[str, int] = {}
        for asset in all_assets:
            insight = self.insights.get(asset.id)
            cat = insight.category if insight and insight.category else "Other"
            categories[cat] = categories.get(cat, 0) + 1

        # Tech stack distribution
        tech_counts: dict[str, int] = {}
        for asset in all_assets:
            insight = self.insights.get(asset.id)
            if insight:
                for tech in insight.tech_stack:
                    tech_counts[tech] = tech_counts.get(tech, 0) + 1

        # Author coverage
        author_counts: dict[str, int] = {}
        for asset in all_assets:
            if asset.owner and asset.owner != "unknown":
                for author in asset.owner.split(", "):
                    author = author.strip()
                    if author:
                        author_counts[author] = author_counts.get(author, 0) + 1

        # Overlap detection
        from collections import defaultdict
        by_purpose: dict[str, list[AIAsset]] = defaultdict(list)
        for asset in all_assets:
            insight = self.insights.get(asset.id)
            key = insight.solution_name if insight else asset.name
            by_purpose[key].append(asset)
        overlaps = [
            {
                "solution_name": name,
                "count": len(group),
                "asset_ids": [a.id for a in group],
                "authors": sorted({a.owner for a in group if a.owner != "unknown"}),
                "repositories": sorted({a.repository for a in group}),
            }
            for name, group in by_purpose.items() if len(group) > 1
        ]
        overlaps.sort(key=lambda x: -x["count"])

        return {
            "scout_version": _scout_version(),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "summary": {
                "total_solutions": len(all_assets),
                "critical": critical,
                "warning": warning,
                "ok": ok,
                "files_scanned": total_files,
                "repositories_scanned": len(repos),
                "unique_categories": len(categories),
                "unique_technologies": len(tech_counts),
                "unique_contributors": len(author_counts),
                "overlap_areas": len(overlaps),
                "duplicate_solutions": sum(o["count"] for o in overlaps),
            },
            "repositories": repos,
            "categories": dict(sorted(categories.items(), key=lambda x: -x[1])),
            "tech_stack": dict(sorted(tech_counts.items(), key=lambda x: -x[1])),
            "authors": dict(sorted(author_counts.items(), key=lambda x: -x[1])),
            "overlaps": overlaps,
            "solutions": solutions,
            "errors": all_errors,
        }

    def _asset_to_dict(self, asset: AIAsset) -> dict:
        """Convert AIAsset + insight to a flat dict."""
        insight = self.insights.get(asset.id)

        result = {
            "id": asset.id,
            "name": insight.solution_name if insight and insight.solution_name else asset.name,
            "raw_name": asset.name,
            "repository": asset.repository,
            "files": asset.file_path.split(", ") if asset.file_path else [],
            "file_count": len(asset.file_path.split(", ")) if asset.file_path else 0,
            "owner": asset.owner if asset.owner != "unknown" else None,
            "users": asset.users,
            "risk_score": round(asset.risk_score, 3),
            "risk_level": _risk_level(asset.risk_score),
            "task_types": [t.value for t in getattr(asset, "task_types", [])],
            "tags": list(getattr(asset, "tags", [])),
            "dependencies": asset.dependencies,
        }

        if asset.provider:
            result["provider"] = {
                "name": asset.provider.name,
                "display_name": insight.provider_profile.display_name if insight and insight.provider_profile else asset.provider.name,
            }

        if insight:
            result["category"] = insight.category
            result["summary"] = insight.summary
            result["tech_stack"] = insight.tech_stack
            result["data_involved"] = insight.data_involved
            result["risk_reasons"] = [
                {"severity": r.severity, "title": r.title, "detail": r.detail}
                for r in insight.risk_reasons
            ]
            result["recommendations"] = insight.recommendations

            if insight.provider_profile and insight.provider_profile.name != "unknown":
                p = insight.provider_profile
                result["provider_details"] = {
                    "vendor": p.vendor,
                    "category": p.category,
                    "data_residency": p.data_residency,
                    "training_policy": p.training_policy,
                    "certifications": p.certifications,
                }

        # Findings (without raw API key contents — they're already redacted)
        result["findings"] = [
            {
                "type": f.type.value,
                "file_path": f.file_path,
                "line_number": f.line_number,
                "content": f.redacted_content if f.redacted_content else f.content,
                "provider": f.provider,
            }
            for f in asset.raw_findings
        ]
        result["finding_count"] = len(asset.raw_findings)

        # LLM classification (if available)
        if asset.data_classification:
            dc = asset.data_classification
            result["llm_classification"] = {
                "categories": [c.value for c in dc.categories],
                "confidence": dc.confidence.value if hasattr(dc.confidence, "value") else str(dc.confidence),
                "details": dc.details,
                "risk_score": dc.risk_score,
                "recommendations": dc.recommendations,
            }

        return result


# ── Helpers ────────────────────────────────────────────────────────────────


def _risk_level(score: float) -> str:
    if score >= 0.7:
        return "critical"
    if score >= 0.4:
        return "warning"
    return "ok"


def _scout_version() -> str:
    try:
        from aiscout import __version__
        return __version__
    except ImportError:
        return "unknown"


def _json_default(obj):
    """Fallback JSON serializer for non-standard types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "value"):  # Enum
        return obj.value
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):  # Pydantic
        return obj.model_dump()
    return str(obj)
