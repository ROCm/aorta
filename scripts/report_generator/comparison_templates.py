"""Jinja2 templates for comparison HTML report generation."""

from jinja2 import Template


COMPARISON_CHART_TEMPLATE = Template(
    """
<h2>{{ title }}</h2>

<table class="comparison-table">
<tr>
<th>{{ label1 }}</th>
<th>{{ label2 }}</th>
</tr>
<tr>
<td>
<img src="data:image/png;base64,{{ image_data1 }}" alt="{{ alt }} - {{ label1 }}">
</td>
<td>
<img src="data:image/png;base64,{{ image_data2 }}" alt="{{ alt }} - {{ label2 }}">
</td>
</tr>
</table>

<hr>
"""
)

SWEEP_INFO_TEMPLATE = Template(
    """
<h2>Sweep Information</h2>

<table>
<tr>
<th>Sweep</th>
<th>Path</th>
</tr>
<tr>
<td><strong>Sweep 1</strong></td>
<td>{{ label1 }}</td>
</tr>
<tr>
<td><strong>Sweep 2</strong></td>
<td>{{ label2 }}</td>
</tr>
</table>

<hr>
"""
)

DATA_FILES_TEMPLATE = Template(
    """
<div class="data-section">
<h2>Data Files Information</h2>

<h3>Sweep 1: {{ label1 }}</h3>
<ul>
<li>Path: {{ sweep1_path }}</li>
<li>GEMM Variance CSV</li>
<li>TraceLens Reports</li>
<li>Plots</li>
</ul>

<h3>Sweep 2: {{ label2 }}</h3>
<ul>
<li>Path: {{ sweep2_path }}</li>
<li>GEMM Variance CSV</li>
<li>TraceLens Reports</li>
<li>Plots</li>
</ul>

</div>
"""
)

COMPARISON_BODY_TEMPLATE = Template(
    """
<h1>{{ title }}</h1>

<div class="info-box">
<p><strong>{{ summary }}</strong></p>
</div>

<hr>

{{ sweep_info }}

{{ sections_html }}

{{ data_files }}
"""
)

COMPARISON_DOCUMENT_TEMPLATE = Template("""{{ header }}{{ body }}{{ footer }}""")
