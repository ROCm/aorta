"""Jinja2 templates for HTML report generation."""

from jinja2 import Template


# =============================================================================
# Single Report Templates
# =============================================================================

CHART_TEMPLATE = Template(
    """
    <h4>{{ name }}</h4>
    <img src="data:image/png;base64,{{ image_data }}" alt="{{ alt }}" class="chart-image">
    <p>{{ description }}</p>
"""
)

SECTION_TEMPLATE = Template(
    """
    <h3>{{ title }}</h3>
    {{ charts_html }}
"""
)

BODY_TEMPLATE = Template(
    """
<body>

<h1>{{ title }}</h1>

<hr>

<h2>Executive Summary</h2>

<p>{{ summary }}</p>

{{ sections_html }}

</body>
"""
)

DOCUMENT_TEMPLATE = Template("""{{ header }}{{ body }}{{ footer }}""")


# =============================================================================
# Comparison Report Templates (Side-by-Side)
# =============================================================================

COMPARISON_CHART_TEMPLATE = Template(
    """
<table class="comparison-table">
<tr>
<th>{{ label1 }}</th>
<th>{{ label2 }}</th>
</tr>
<tr>
<td>
{% if image_data1 %}
<img src="data:image/png;base64,{{ image_data1 }}" alt="{{ alt }} - {{ label1 }}">
{% else %}
<p><em>Image not available</em></p>
{% endif %}
</td>
<td>
{% if image_data2 %}
<img src="data:image/png;base64,{{ image_data2 }}" alt="{{ alt }} - {{ label2 }}">
{% else %}
<p><em>Image not available</em></p>
{% endif %}
</td>
</tr>
</table>
"""
)

COMPARISON_SECTION_TEMPLATE = Template(
    """
<h2>{{ title }}</h2>
{{ charts_html }}
<hr>
"""
)

COMPARISON_INFO_TABLE_TEMPLATE = Template(
    """
<h2>Sweep Information</h2>
<table>
<tr>
<th>Sweep</th>
<th>Path</th>
</tr>
<tr>
<td><strong>Sweep 1</strong></td>
<td>{{ label1 }} ({{ path1 }})</td>
</tr>
<tr>
<td><strong>Sweep 2</strong></td>
<td>{{ label2 }} ({{ path2 }})</td>
</tr>
</table>
<hr>
"""
)

COMPARISON_BODY_TEMPLATE = Template(
    """
<h1>{{ title }}</h1>

<div class="info-box">
<p><strong>{{ summary }}</strong></p>
</div>

<hr>

{{ info_table }}

{{ sections_html }}

<div class="data-section">
<h2>Data Files Information</h2>

<h3>Sweep 1: {{ label1 }}</h3>
<ul>
<li>Path: {{ path1 }}</li>
<li>GEMM Variance CSV</li>
<li>TraceLens Reports</li>
<li>Plots</li>
</ul>

<h3>Sweep 2: {{ label2 }}</h3>
<ul>
<li>Path: {{ path2 }}</li>
<li>GEMM Variance CSV</li>
<li>TraceLens Reports</li>
<li>Plots</li>
</ul>
</div>
"""
)

