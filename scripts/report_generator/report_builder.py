"""HTML Report Builder for generating reports from JSON configuration."""

from pathlib import Path
import base64

from .config_reader import ReportConfigReader
from .templates import (
    CHART_TEMPLATE,
    SECTION_TEMPLATE,
    BODY_TEMPLATE,
    DOCUMENT_TEMPLATE,
)


class HTMLReportBuilder:
    """
    Base HTML report builder that generates reports from JSON configuration.
    Uses ReportConfigReader to populate sections and chart configurations.

    Structure:
        - build_* methods: Handle structure traversal and data preparation
        - render_* methods: Handle HTML layout using Jinja2 templates

    Subclasses should override:
        - build_chart(): Define how to load and render chart data
        - render_chart(): Define chart template (optional)
        - render_body(): Add custom body sections (optional)
    """

    def __init__(self, output_path: Path, config_path: Path):
        """
        Initialize the report builder.

        Args:
            output_path: Path where the HTML report will be saved
            config_path: Path to the JSON configuration file
        """
        self.output_path = output_path
        self.config = ReportConfigReader(config_path)

    # -------------------------------------------------------------------------
    # Data/Utility Methods
    # -------------------------------------------------------------------------

    def get_image_base64(self, image_path: Path) -> str | None:
        """Read an image file and return its base64-encoded string."""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"Error getting image data from {image_path}: {e}")
            return None

    def get_report_title(self) -> str:
        """Return the main title for the report."""
        return self.config.get_title()

    def get_executive_summary(self) -> str:
        """Return the executive summary HTML content."""
        return self.config.get_executive_summary()

    def get_sections(self) -> list[dict]:
        """Return all section configurations from config."""
        return self.config.get_all_sections()

    # -------------------------------------------------------------------------
    # Render Methods (Jinja2 Templates) - Override for custom layouts
    # -------------------------------------------------------------------------

    def render_chart(self, **kwargs) -> str:
        """Render HTML layout for a single chart using Jinja2."""
        return CHART_TEMPLATE.render(**kwargs)

    def render_section(self, title: str, charts_html: str) -> str:
        """Render HTML layout for a section using Jinja2."""
        return SECTION_TEMPLATE.render(
            title=title,
            charts_html=charts_html,
        )

    def render_body(self, sections_html: str) -> str:
        """Render HTML layout for the body using Jinja2."""
        return BODY_TEMPLATE.render(
            title=self.get_report_title(),
            summary=self.get_executive_summary(),
            sections_html=sections_html,
        )

    def render_document(self, body: str) -> str:
        """Render the complete HTML document layout using Jinja2."""
        return DOCUMENT_TEMPLATE.render(
            header=self.config.get_html_header(),
            body=body,
            footer=self.config.get_html_footer(),
        )

    # -------------------------------------------------------------------------
    # Build Methods (Structure Traversal) - Override for custom data handling
    # -------------------------------------------------------------------------

    def build_chart(self, chart_config: dict, section_title: str) -> str:
        """
        Build a chart from config. Override in subclass to define image loading.

        Args:
            chart_config: Chart configuration dict with file, name, alt, description
            section_title: Title of the parent section (for context)

        Returns:
            Rendered chart HTML string
        """
        raise NotImplementedError("Subclass must implement build_chart")

    def build_section(self, section: dict) -> str:
        """Build a section by iterating through its charts."""
        charts_html = ""
        charts = section.get("charts", [])
        for chart in charts:
            charts_html += self.build_chart(chart, section["title"])
        return self.render_section(section["title"], charts_html)

    def build_body(self) -> str:
        """Traverse all sections and build the body HTML."""
        sections_html = ""
        for section in self.get_sections():
            sections_html += self.build_section(section)
        return self.render_body(sections_html)

    def build(self) -> str:
        """Build the complete HTML document."""
        return self.render_document(self.build_body())

    def save(self) -> None:
        """Build and save the HTML report to the output path."""
        final_html = self.build()
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(final_html)
        print(f"Final HTML file created at: {self.output_path}")


class SingleReportBuilder(HTMLReportBuilder):
    """
    Report builder for single-source reports with one plot directory.
    Only overrides build_chart to define single-image loading.
    """

    def __init__(self, plot_dir: Path, output_path: Path, config_path: Path):
        """
        Initialize the single report builder.

        Args:
            plot_dir: Directory containing plot images
            output_path: Path where the HTML report will be saved
            config_path: Path to the JSON configuration file
        """
        super().__init__(output_path, config_path)
        self.plot_dir = plot_dir

    def build_chart(self, chart_config: dict, section_title: str) -> str:
        """Build a single chart by loading one image from plot_dir."""
        image_path = self.plot_dir / chart_config["file"]
        image_data = self.get_image_base64(image_path)
        if image_data is None:
            return ""
        return self.render_chart(
            name=chart_config["name"],
            image_data=image_data,
            alt=chart_config["alt"],
            description=chart_config["description"],
        )
