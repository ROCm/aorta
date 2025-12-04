"""HTML template for GEMM sweep comparison report."""

def get_comparison_template(label1, label2, sweep1_path, sweep2_path, image_data):
    """
    Generate HTML content for sweep comparison report.

    Args:
        label1: Label for first sweep
        label2: Label for second sweep
        sweep1_path: Path to first sweep directory
        sweep2_path: Path to second sweep directory
        image_data: Dictionary of base64-encoded images

    Returns:
        HTML content as string
    """
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>GEMM Kernel Variance - Sweep Comparison</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            border-bottom: 3px solid #333;
            padding-bottom: 10px;
            color: #2c3e50;
        }}
        h2 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
            margin-top: 40px;
            color: #2c3e50;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 10px auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
            background-color: white;
        }}
        .comparison-table {{ width: 100%; }}
        .comparison-table td {{ width: 50%; vertical-align: top; }}
        .comparison-table th {{
            background-color: #34495e;
        }}
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }}
        .info-box {{
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .data-section {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 4px;
            margin: 20px 0;
        }}
        .data-section h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        @media print {{
            body {{ margin: 0; background-color: white; }}
            .container {{ box-shadow: none; }}
            h2 {{ page-break-before: always; }}
            h2:first-of-type {{ page-break-before: auto; }}
        }}
    </style>
</head>
<body>
<div class="container">

<h1>GEMM Kernel Variance - Sweep Comparison</h1>

<div class="info-box">
<p><strong>Visual comparison of GEMM kernel performance variance between two training sweeps.</strong></p>
<p>This report compares kernel variance across different thread counts, channel configurations, and ranks.</p>
</div>

<hr>

<h2>Sweep Information</h2>

<table>
<tr>
<th>Sweep</th>
<th>Path</th>
</tr>
<tr>
<td><strong>Sweep 1</strong></td>
<td>{label1}</td>
</tr>
<tr>
<td><strong>Sweep 2</strong></td>
<td>{label2}</td>
</tr>
</table>

<hr>

<h2>Variance by Thread Count</h2>

<table class="comparison-table">
<tr>
<th>{label1}</th>
<th>{label2}</th>
</tr>
<tr>
<td>
<img src="{image_data.get('threads_sweep1', '')}" alt="Threads Sweep 1">
</td>
<td>
<img src="{image_data.get('threads_sweep2', '')}" alt="Threads Sweep 2">
</td>
</tr>
</table>

<hr>

<h2>Variance by Channel Count</h2>

<table class="comparison-table">
<tr>
<th>{label1}</th>
<th>{label2}</th>
</tr>
<tr>
<td>
<img src="{image_data.get('channels_sweep1', '')}" alt="Channels Sweep 1">
</td>
<td>
<img src="{image_data.get('channels_sweep2', '')}" alt="Channels Sweep 2">
</td>
</tr>
</table>

<hr>

<h2>Variance by Rank</h2>

<table class="comparison-table">
<tr>
<th>{label1}</th>
<th>{label2}</th>
</tr>
<tr>
<td>
<img src="{image_data.get('ranks_sweep1', '')}" alt="Ranks Sweep 1">
</td>
<td>
<img src="{image_data.get('ranks_sweep2', '')}" alt="Ranks Sweep 2">
</td>
</tr>
</table>

<hr>

<h2>Variance Distribution (Violin Plots)</h2>

<table class="comparison-table">
<tr>
<th>{label1}</th>
<th>{label2}</th>
</tr>
<tr>
<td>
<img src="{image_data.get('violin_sweep1', '')}" alt="Violin Sweep 1">
</td>
<td>
<img src="{image_data.get('violin_sweep2', '')}" alt="Violin Sweep 2">
</td>
</tr>
</table>

<hr>

<h2>Thread-Channel Interaction</h2>

<table class="comparison-table">
<tr>
<th>{label1}</th>
<th>{label2}</th>
</tr>
<tr>
<td>
<img src="{image_data.get('interaction_sweep1', '')}" alt="Interaction Sweep 1">
</td>
<td>
<img src="{image_data.get('interaction_sweep2', '')}" alt="Interaction Sweep 2">
</td>
</tr>
</table>

<hr>

<div class="data-section">
<h2>Data Files Information</h2>

<h3>Sweep 1: {label1}</h3>
<ul>
<li>Path: {sweep1_path}</li>
<li>GEMM Variance CSV</li>
<li>TraceLens Reports</li>
<li>Plots</li>
</ul>

<h3>Sweep 2: {label2}</h3>
<ul>
<li>Path: {sweep2_path}</li>
<li>GEMM Variance CSV</li>
<li>TraceLens Reports</li>
<li>Plots</li>
</ul>

</div>

</div>
</body>
</html>
"""
