"""Configuration reader for HTML report generation."""

from pathlib import Path
import json


class ReportConfigReader:
    """
    Reads and provides access to HTML report configuration from a JSON file.
    """

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> dict:
        """Load the JSON configuration file."""
        with open(self.config_path, "r") as f:
            return json.load(f)

    def get_title(self) -> str:
        """Return the report title."""
        return self._config.get("title", "")

    def get_executive_summary(self) -> str:
        """Return the executive summary text."""
        return self._config.get("executive_summary", "")

    def get_html_header(self) -> str:
        """Return the HTML header template."""
        return self._config.get("html_header", "")

    def get_html_footer(self) -> str:
        """Return the HTML footer template."""
        return self._config.get("html_footer", "")

    def get_all_sections(self) -> list[dict]:
        """Return all section configurations."""
        return self._config.get("sections", [])

    def get_section_by_id(self, section_id: str) -> dict | None:
        """Return a specific section by its ID."""
        for section in self.get_all_sections():
            if section.get("id") == section_id:
                return section
        return None

    def get_section_title(self, section_id: str) -> str:
        """Return the title of a section by ID."""
        section = self.get_section_by_id(section_id)
        return section.get("title", "") if section else ""

    def get_section_charts(self, section_id: str) -> list[dict]:
        """Return the charts configuration for a section by ID."""
        section = self.get_section_by_id(section_id)
        return section.get("charts", []) if section else []

    def get_section_ids(self) -> list[str]:
        """Return a list of all section IDs."""
        return [section.get("id", "") for section in self.get_all_sections()]

