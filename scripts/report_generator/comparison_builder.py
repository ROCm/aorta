"""Comparison Report Builder for generating side-by-side comparison reports."""

from pathlib import Path

from .report_builder import HTMLReportBuilder
from .comparison_templates import (
    COMPARISON_CHART_TEMPLATE,
    SWEEP_INFO_TEMPLATE,
    DATA_FILES_TEMPLATE,
    COMPARISON_BODY_TEMPLATE,
)


class ComparisonReportBuilder(HTMLReportBuilder):
    """
    HTML report builder for comparing two experiment sweeps.
    Extends HTMLReportBuilder with side-by-side comparison capabilities.

    Overrides:
        - build_chart: Loads images from two sweep directories
        - render_chart: Uses comparison template for side-by-side display
        - render_body: Adds sweep info and data files sections
    """

    def __init__(
        self,
        sweep1_path: Path,
        sweep2_path: Path,
        output_path: Path,
        config_path: Path,
        label1: str | None = None,
        label2: str | None = None,
    ):
        """
        Initialize the comparison report builder.

        Args:
            sweep1_path: Path to first sweep directory
            sweep2_path: Path to second sweep directory
            output_path: Path where the HTML report will be saved
            config_path: Path to the JSON configuration file
            label1: Label for first sweep (default: directory name)
            label2: Label for second sweep (default: directory name)
        """
        super().__init__(output_path, config_path)
        self.sweep1_path = sweep1_path
        self.sweep2_path = sweep2_path

        # Use directory names as labels if not provided
        self.label1 = label1 if label1 else sweep1_path.name
        self.label2 = label2 if label2 else sweep2_path.name

    # -------------------------------------------------------------------------
    # Data/Utility Methods
    # -------------------------------------------------------------------------

    def get_plots_dir(self, sweep_path: Path) -> Path:
        """Get the plots directory for a sweep."""
        return sweep_path / "tracelens_analysis" / "plots"

    # -------------------------------------------------------------------------
    # Render Methods (Jinja2 Templates) - Comparison-specific
    # -------------------------------------------------------------------------

    def render_chart(self, **kwargs) -> str:
        """Render HTML layout for a side-by-side chart comparison."""
        return COMPARISON_CHART_TEMPLATE.render(
            label1=self.label1,
            label2=self.label2,
            **kwargs,
        )

    def render_sweep_info(self) -> str:
        """Render HTML layout for sweep information section."""
        return SWEEP_INFO_TEMPLATE.render(
            label1=self.label1,
            label2=self.label2,
        )

    def render_data_files(self) -> str:
        """Render HTML layout for data files section."""
        return DATA_FILES_TEMPLATE.render(
            label1=self.label1,
            label2=self.label2,
            sweep1_path=self.sweep1_path,
            sweep2_path=self.sweep2_path,
        )

    def render_body(self, sections_html: str) -> str:
        """Render HTML layout for the body with comparison-specific elements."""
        return COMPARISON_BODY_TEMPLATE.render(
            title=self.get_report_title(),
            summary=self.get_executive_summary(),
            sweep_info=self.render_sweep_info(),
            sections_html=sections_html,
            data_files=self.render_data_files(),
        )

    # -------------------------------------------------------------------------
    # Build Methods (Structure Traversal) - Only override build_chart
    # -------------------------------------------------------------------------

    def build_chart(self, chart_config: dict, section_title: str) -> str:
        """Build a comparison chart with images from both sweeps."""
        plots_dir1 = self.get_plots_dir(self.sweep1_path)
        plots_dir2 = self.get_plots_dir(self.sweep2_path)

        image_path1 = plots_dir1 / chart_config["file"]
        image_path2 = plots_dir2 / chart_config["file"]

        print(f"  Processing: {chart_config['name']}")
        image_data1 = self.get_image_base64(image_path1)
        image_data2 = self.get_image_base64(image_path2)

        if image_data1:
            print(f"    Sweep 1 ({self.label1}): [OK]")
        else:
            print(f"    Sweep 1 ({self.label1}): [MISSING] {image_path1}")

        if image_data2:
            print(f"    Sweep 2 ({self.label2}): [OK]")
        else:
            print(f"    Sweep 2 ({self.label2}): [MISSING] {image_path2}")

        return self.render_chart(
            title=section_title,
            alt=chart_config["alt"],
            image_data1=image_data1 or "",
            image_data2=image_data2 or "",
        )

    # def build_body(self) -> str:
    #    """Build the HTML body (adds logging message)."""
    #    print("Converting images to base64...")
    #    return super().build_body()

    # def save(self) -> None:
    #    """Build and save the HTML report with file size info."""
    #    final_html = self.build()
    #   with open(self.output_path, "w", encoding="utf-8") as f:
    #        f.write(final_html)
    #    print(f"\n[OK] HTML report created: {self.output_path}")
    #    print(f"     File size: {self.output_path.stat().st_size / 1024 / 1024:.2f} MB")
