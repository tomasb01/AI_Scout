"""HTML Report Generator for AI Scout."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from jinja2 import Environment, PackageLoader

from aiscout import __version__
from aiscout.engine.enrichment import AssetInsight, enrich_assets
from aiscout.knowledge.providers import get_provider
from aiscout.models import AIAsset, FindingType, ScanResult


class ReportGenerator:
    """Generates a self-contained HTML report from scan results."""

    def __init__(
        self,
        scan_results: list[ScanResult],
        output_path: str = "aiscout_report.html",
        insights: dict[str, AssetInsight] | None = None,
    ):
        self.scan_results = scan_results
        self.output_path = output_path
        self.insights = insights
        self._env = Environment(
            loader=PackageLoader("aiscout", "report/templates"),
            autoescape=True,
        )
        self._env.filters["risk_class"] = self._get_risk_class

    def generate(self) -> Path:
        """Render the report and write it to disk."""
        context = self._build_context()
        template = self._env.get_template("report.html.j2")
        html = template.render(**context)

        out = Path(self.output_path)
        out.write_text(html, encoding="utf-8")
        return out

    def _build_context(self) -> dict:
        """Build the template context from scan results."""
        all_assets: list[AIAsset] = []
        all_errors: list[str] = []
        repos: list[str] = []
        total_files_scanned = 0

        for result in self.scan_results:
            all_assets.extend(result.assets)
            all_errors.extend(result.errors)
            repo = result.metadata.get("repository", "unknown")
            if repo not in repos:
                repos.append(repo)
            total_files_scanned += result.metadata.get("files_scanned", 0)

        # Enrich assets with insights (if not already provided)
        if self.insights is None:
            self.insights = enrich_assets(all_assets)

        # Sort by risk score descending (enrichment may have updated scores)
        all_assets.sort(key=lambda a: a.risk_score, reverse=True)

        # Risk counts
        critical = sum(1 for a in all_assets if a.risk_score >= 0.7)
        warning = sum(1 for a in all_assets if 0.4 <= a.risk_score < 0.7)
        ok = sum(1 for a in all_assets if a.risk_score < 0.4)

        # Cross-repo overlap detection
        provider_repos: dict[str, set[str]] = defaultdict(set)
        for asset in all_assets:
            if asset.provider:
                provider_repos[asset.provider.name].add(asset.repository)
        cross_repo_overlaps = {
            provider: sorted(repo_set)
            for provider, repo_set in provider_repos.items()
            if len(repo_set) > 1
        }

        # Scan date from first result
        scan_date = ""
        if self.scan_results:
            scan_date = self.scan_results[0].started_at.strftime("%Y-%m-%d %H:%M UTC")

        has_llm_data = any(a.data_classification for a in all_assets)

        # Data egress map: provider -> {display_name, regions, category, repos}
        data_egress = {}
        for asset in all_assets:
            if asset.provider:
                pname = asset.provider.name
                if pname not in data_egress:
                    profile = get_provider(pname)
                    data_egress[pname] = {
                        "display_name": profile.display_name,
                        "regions": profile.data_residency,
                        "category": profile.category,
                        "repos": set(),
                        "asset_count": 0,
                    }
                data_egress[pname]["repos"].add(asset.repository)
                data_egress[pname]["asset_count"] += 1
        # Sort: external APIs first, then by asset count
        category_order = {"llm_api": 0, "embedding_db": 1, "framework": 2, "local_runtime": 3}
        data_egress_sorted = dict(sorted(
            data_egress.items(),
            key=lambda x: (category_order.get(x[1]["category"], 9), -x[1]["asset_count"]),
        ))

        # Unique authors across all assets
        all_authors = sorted({
            author
            for asset in all_assets
            for author in asset.users
        })

        # Scan results per repo (for header)
        repo_details = []
        for result in self.scan_results:
            repo_details.append({
                "name": result.metadata.get("repository", "unknown"),
                "branch": result.metadata.get("branch", "main"),
                "files_scanned": result.metadata.get("files_scanned", 0),
                "assets_found": len(result.assets),
                "errors": len(result.errors),
            })

        return {
            "version": __version__,
            "scan_date": scan_date,
            "repos": repos,
            "total_assets": len(all_assets),
            "total_files_scanned": total_files_scanned,
            "critical_count": critical,
            "warning_count": warning,
            "ok_count": ok,
            "assets": all_assets,
            "insights": self.insights,
            "cross_repo_overlaps": cross_repo_overlaps,
            "errors": all_errors,
            "has_llm_data": has_llm_data,
            "FindingType": FindingType,
            "data_egress": data_egress_sorted,
            "all_authors": all_authors,
            "repo_details": repo_details,
            "categories": self._group_by_category(all_assets),
        }

    def _group_by_category(self, assets: list[AIAsset]) -> dict[str, list[AIAsset]]:
        """Group assets by category for dashboard display."""
        from collections import defaultdict
        cats: dict[str, list[AIAsset]] = defaultdict(list)
        for asset in assets:
            insight = self.insights.get(asset.id) if self.insights else None
            cat = insight.category if insight and insight.category else "Other AI Solutions"
            cats[cat].append(asset)
        # Sort categories by count descending
        return dict(sorted(cats.items(), key=lambda x: -len(x[1])))

    @staticmethod
    def _get_risk_class(score: float) -> str:
        if score >= 0.7:
            return "critical"
        if score >= 0.4:
            return "warning"
        return "ok"
