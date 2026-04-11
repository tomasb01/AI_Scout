"""HTML Report Generator for AI Scout."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from jinja2 import Environment, PackageLoader

from aiscout import __version__
from aiscout.engine.enrichment import AssetInsight, _deduplicate_tech_stack, enrich_assets
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
        self._env.filters["md_bold"] = self._md_bold

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
        repo_urls = {}  # repo_name -> url
        for result in self.scan_results:
            repo_name = result.metadata.get("repository", "unknown")
            repo_url = result.metadata.get("repo_url", "")
            if repo_url:
                repo_urls[repo_name] = repo_url
            repo_details.append({
                "name": repo_name,
                "branch": result.metadata.get("branch", "main"),
                "files_scanned": result.metadata.get("files_scanned", 0),
                "assets_found": len(result.assets),
                "errors": len(result.errors),
                "url": repo_url,
            })

        # Analytics
        overlaps = self._detect_overlaps(all_assets)
        tech_radar = self._build_tech_radar(all_assets)
        data_sensitivity = self._build_data_sensitivity(all_assets)
        author_coverage = self._build_author_coverage(all_assets)
        categories = self._group_by_category(all_assets)
        exec_summary = self._build_executive_summary(
            all_assets, overlaps, tech_radar, data_sensitivity,
            author_coverage, data_egress_sorted, critical, warning,
        )

        # New stats
        overlap_count = sum(o["count"] for o in overlaps)
        unique_count = len(all_assets) - overlap_count
        at_risk = critical + warning
        people_count = len([a for a in author_coverage if a["count"] > 0])

        return {
            "version": __version__,
            "scan_date": scan_date,
            "repos": repos,
            "total_assets": len(all_assets),
            "unique_count": max(0, unique_count),
            "overlap_count": overlap_count,
            "overlap_areas": len(overlaps),
            "people_count": people_count,
            "at_risk_count": at_risk,
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
            "categories": categories,
            "overlaps": overlaps,
            "tech_radar": tech_radar,
            "data_sensitivity": data_sensitivity,
            "author_coverage": author_coverage,
            "exec_summary": exec_summary,
            "graph_data": self._build_graph_data(all_assets),
            "repo_urls": repo_urls,
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

    def _detect_overlaps(self, assets: list[AIAsset]) -> list[dict]:
        """Find solutions that do the same thing (functional overlap)."""
        # Group by category + similar purpose
        by_purpose: dict[str, list[AIAsset]] = defaultdict(list)
        for asset in assets:
            insight = self.insights.get(asset.id) if self.insights else None
            if not insight:
                continue
            # Create a fingerprint: category + key function names + data types
            cat = insight.category
            funcs = set()
            for ctx in asset.code_contexts:
                for f in ctx.functions:
                    name = f.get("name", "")
                    if name and not name.startswith("_") and name not in (
                        "main", "__init__", "setup", "run", "config",
                    ):
                        funcs.add(name)
            # Use solution_name as grouping key (duplicates = overlap)
            key = insight.solution_name
            by_purpose[key].append(asset)

        overlaps = []
        for purpose, group in sorted(by_purpose.items(), key=lambda x: -len(x[1])):
            if len(group) < 2:
                continue
            authors = sorted({a.owner for a in group if a.owner != "unknown"})
            repos = sorted({a.repository for a in group})
            # Tech stack across duplicates
            techs = set()
            for a in group:
                i = self.insights.get(a.id)
                if i:
                    techs.update(i.tech_stack)

            overlaps.append({
                "purpose": purpose,
                "count": len(group),
                "authors": authors,
                "repos": repos,
                "tech_stacks": sorted(_deduplicate_tech_stack(techs)),
                "assets": group,
            })

        return overlaps

    def _build_tech_radar(self, assets: list[AIAsset]) -> list[dict]:
        """Count solutions per technology for radar chart."""
        tech_counts: dict[str, int] = defaultdict(int)
        for asset in assets:
            insight = self.insights.get(asset.id) if self.insights else None
            if insight:
                for tech in insight.tech_stack:
                    tech_counts[tech] += 1
        # Sort by count
        return [
            {"name": tech, "count": count}
            for tech, count in sorted(tech_counts.items(), key=lambda x: -x[1])
        ]

    def _build_data_sensitivity(self, assets: list[AIAsset]) -> list[dict]:
        """Build data type × count matrix for sensitivity heatmap."""
        data_counts: dict[str, dict] = defaultdict(lambda: {"count": 0, "warning": 0, "critical": 0})
        for asset in assets:
            insight = self.insights.get(asset.id) if self.insights else None
            if not insight:
                continue
            risk = self._get_risk_class(asset.risk_score)
            for d in insight.data_involved:
                data_counts[d]["count"] += 1
                if risk in ("warning", "critical"):
                    data_counts[d][risk] += 1
        return [
            {"type": dtype, **counts}
            for dtype, counts in sorted(data_counts.items(), key=lambda x: -x[1]["count"])
        ]

    def _build_author_coverage(self, assets: list[AIAsset]) -> list[dict]:
        """Analyze who built what — single-point-of-failure detection."""
        by_author: dict[str, list[AIAsset]] = defaultdict(list)
        for asset in assets:
            if asset.owner and asset.owner != "unknown":
                # Split comma-separated authors
                for author in asset.owner.split(", "):
                    author = author.strip()
                    if author:
                        by_author[author].append(asset)
        total = len(assets)
        result = []
        for author, author_assets in sorted(by_author.items(), key=lambda x: -len(x[1])):
            cats = set()
            for a in author_assets:
                i = self.insights.get(a.id)
                if i:
                    cats.add(i.category)
            pct = round(100 * len(author_assets) / total) if total else 0
            result.append({
                "name": author,
                "count": len(author_assets),
                "percentage": pct,
                "categories": sorted(cats),
                "is_spof": pct >= 30,  # 30%+ = single point of failure risk
            })
        return result

    def _build_executive_summary(
        self, assets, overlaps, tech_radar, data_sensitivity,
        author_coverage, data_egress, critical_count, warning_count,
    ) -> list[str]:
        """Generate executive summary bullet points."""
        points = []
        total = len(assets)

        # Total count
        points.append(f"Found **{total} AI solutions** across scanned repositories.")

        # Overlap
        if overlaps:
            overlap_solutions = sum(o["count"] for o in overlaps)
            points.append(
                f"**{overlap_solutions} solutions** functionally overlap in "
                f"**{len(overlaps)} areas** — consolidation opportunity."
            )

        # Risk
        if critical_count:
            points.append(
                f"**{critical_count} solutions** with critical risk "
                f"(hardcoded API keys, sensitive data) — requires immediate attention."
            )

        # Data egress
        us_providers = [
            eg["display_name"] for eg in data_egress.values()
            if any("US" in r for r in eg.get("regions", []))
            and eg.get("category") == "llm_api"
        ]
        if us_providers:
            us_assets = sum(
                eg["asset_count"] for eg in data_egress.values()
                if any("US" in r for r in eg.get("regions", []))
                and eg.get("category") == "llm_api"
            )
            points.append(
                f"**{us_assets} solutions** send data to US ({', '.join(us_providers)}) "
                f"— verify that all legal requirements are in place (DPA, data residency)."
            )

        # Author SPOF
        spof_authors = [a for a in author_coverage if a["is_spof"]]
        if spof_authors:
            names = ", ".join(a["name"] for a in spof_authors)
            pct = spof_authors[0]["percentage"]
            points.append(
                f"**{len(spof_authors)} developer{'s' if len(spof_authors) > 1 else ''}** "
                f"created over {pct}% of all solutions ({names}) "
                f"— single-point-of-failure risk."
            )

        # Tech concentration
        if tech_radar and tech_radar[0]["count"] > total * 0.3:
            top = tech_radar[0]
            points.append(
                f"Highest dependency on **{top['name']}** — "
                f"used by {top['count']} of {total} solutions ({round(100*top['count']/total)}%)."
            )

        # Data sensitivity
        sensitive_types = [d for d in data_sensitivity if d["type"] in (
            "Personal data / PII", "Financial data", "Credentials / Secrets",
        )]
        if sensitive_types:
            for st in sensitive_types:
                points.append(
                    f"**{st['count']} solutions** process **{st['type']}** data "
                    f"— requires elevated compliance attention."
                )

        return points

    def _build_graph_data(self, assets: list[AIAsset]) -> dict:
        """Build node/edge data for the 3 graph views."""
        import json

        cat_colors = {
            "AI Agents": "#6366f1", "Chatbot & Conversation": "#3b82f6",
            "RAG & Search": "#22c55e", "Fine-tuning & Training": "#f59e0b",
            "Workflows & Pipelines": "#a855f7", "Web Automation": "#ef4444",
            "Model & Inference": "#14b8a6", "MCP & Integration": "#f97316",
            "Other AI Solutions": "#8b8fa3",
        }

        # ── Solutions graph ──
        sol_nodes = []
        sol_edges = []
        for asset in assets:
            insight = self.insights.get(asset.id) if self.insights else None
            cat = insight.category if insight else "Other"
            name = insight.solution_name if insight else asset.name
            sol_nodes.append({
                "id": asset.id,
                "label": name[:30],
                "group": cat,
                "color": cat_colors.get(cat, "#8b8fa3"),
                "size": max(6, min(18, len(asset.raw_findings) * 2)),
                "risk": self._get_risk_class(asset.risk_score),
            })

        # Edges: connect solutions with same solution_name (overlap)
        seen_names: dict[str, list[str]] = defaultdict(list)
        for asset in assets:
            insight = self.insights.get(asset.id) if self.insights else None
            name = insight.solution_name if insight else asset.name
            seen_names[name].append(asset.id)
        for name, ids in seen_names.items():
            if len(ids) > 1:
                for i in range(len(ids) - 1):
                    sol_edges.append({"from": ids[i], "to": ids[i + 1], "type": "overlap"})

        # ── Tech graph ──
        tech_nodes = []
        tech_edges = []
        tech_ids = {}
        for asset in assets:
            insight = self.insights.get(asset.id) if self.insights else None
            if not insight:
                continue
            for tech in insight.tech_stack:
                if tech not in tech_ids:
                    tid = f"tech_{len(tech_ids)}"
                    tech_ids[tech] = tid
                    tech_nodes.append({
                        "id": tid, "label": tech, "group": "tech",
                        "color": "#6366f1", "size": 12, "is_tech": True,
                    })
                tech_edges.append({"from": asset.id, "to": tech_ids[tech]})

        # Add solution nodes (smaller) to tech graph
        tech_sol_nodes = [
            {**n, "size": max(3, n["size"] // 2), "is_tech": False}
            for n in sol_nodes
        ]

        # ── People graph ──
        people_nodes = []
        people_edges = []
        author_ids = {}
        author_solutions: dict[str, list[str]] = defaultdict(list)

        for asset in assets:
            if asset.owner and asset.owner != "unknown":
                for author in asset.owner.split(", "):
                    author = author.strip()
                    if not author:
                        continue
                    if author not in author_ids:
                        aid = f"author_{len(author_ids)}"
                        author_ids[author] = aid
                    author_solutions[author].append(asset.id)

        for author, aid in author_ids.items():
            count = len(author_solutions[author])
            people_nodes.append({
                "id": aid, "label": f"{author} ({count} solutions)", "group": "author",
                "color": "#f59e0b", "size": max(14, min(28, count // 2)),
                "is_author": True,
            })

        # Add solution nodes to people graph
        people_sol_nodes = [
            {**n, "size": max(3, n["size"] // 2), "is_author": False}
            for n in sol_nodes
        ]

        for author, asset_ids in author_solutions.items():
            aid = author_ids[author]
            for sid in asset_ids:
                people_edges.append({"from": aid, "to": sid})

        return {
            "solutions": json.dumps({"nodes": sol_nodes, "edges": sol_edges}),
            "tech": json.dumps({"nodes": tech_sol_nodes + tech_nodes, "edges": tech_edges}),
            "people": json.dumps({"nodes": people_sol_nodes + people_nodes, "edges": people_edges}),
        }

    @staticmethod
    def _md_bold(text: str) -> str:
        """Convert **bold** markdown to <strong> tags."""
        import re
        return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

    @staticmethod
    def _get_risk_class(score: float) -> str:
        if score >= 0.7:
            return "critical"
        if score >= 0.4:
            return "warning"
        return "ok"
