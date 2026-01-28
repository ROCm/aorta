"""HTML template for single GEMM variance analysis report."""


def get_gemm_report_template(label, sweep_path, image_data, csv_path=None):
    """
    Generate HTML content for single GEMM variance analysis report.

    Args:
        label: Label for this analysis
        sweep_path: Path to sweep directory
        image_data: Dictionary of base64-encoded images
        csv_path: Optional path to the CSV data file

    Returns:
        HTML content as string
    """
    csv_info = f"<p><strong>Data:</strong> {csv_path}</p>" if csv_path else ""
    
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>GEMM Kernel Variance Analysis - {label}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            background-color: #1a1a2e;
            color: #eee;
        }}
        .container {{
            background-color: #16213e;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        h1 {{
            border-bottom: 3px solid #e94560;
            padding-bottom: 15px;
            color: #e94560;
            font-size: 2.2em;
        }}
        h2 {{
            border-bottom: 2px solid #0f3460;
            padding-bottom: 8px;
            margin-top: 40px;
            color: #e94560;
            font-size: 1.5em;
        }}
        h3 {{
            color: #94b3fd;
            margin-top: 25px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 15px auto;
            border: 2px solid #0f3460;
            border-radius: 8px;
            padding: 10px;
            background-color: white;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 25px;
            margin: 20px 0;
        }}
        .plot-card {{
            background-color: #0f3460;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }}
        .plot-card h3 {{
            margin-top: 0;
            text-align: center;
            color: #94b3fd;
        }}
        hr {{
            border: none;
            border-top: 2px solid #0f3460;
            margin: 40px 0;
        }}
        .info-box {{
            background-color: #0f3460;
            border-left: 4px solid #e94560;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }}
        .info-box strong {{
            color: #e94560;
        }}
        .summary-section {{
            background-color: #0f3460;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .missing-plot {{
            background-color: #1a1a2e;
            border: 2px dashed #e94560;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            color: #94b3fd;
        }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #0f3460;
            padding-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 GEMM Kernel Variance Analysis</h1>
        
        <div class="info-box">
            <p><strong>Analysis:</strong> {label}</p>
            <p><strong>Source:</strong> {sweep_path}</p>
            {csv_info}
        </div>

        <h2>📊 Variance Distribution by Configuration</h2>
        
        <div class="plot-grid">
            <div class="plot-card">
                <h3>By Thread Count</h3>
                {_image_or_missing(image_data.get('threads', ''), 'Thread variance plot')}
            </div>
            <div class="plot-card">
                <h3>By Channel Count</h3>
                {_image_or_missing(image_data.get('channels', ''), 'Channel variance plot')}
            </div>
        </div>
        
        <div class="plot-grid">
            <div class="plot-card">
                <h3>By Rank</h3>
                {_image_or_missing(image_data.get('ranks', ''), 'Rank variance plot')}
            </div>
            <div class="plot-card">
                <h3>Combined Violin Plot</h3>
                {_image_or_missing(image_data.get('violin', ''), 'Violin plot')}
            </div>
        </div>

        <hr>

        <h2>🔗 Thread-Channel Interaction</h2>
        <p>This plot shows how variance changes across different thread and channel configurations, helping identify optimal settings.</p>
        
        <div class="summary-section">
            {_image_or_missing(image_data.get('interaction', ''), 'Interaction plot')}
        </div>

        <hr>

        <h2>📖 How to Interpret</h2>
        <div class="info-box">
            <p><strong>Time Variance:</strong> The difference between maximum and minimum kernel execution times (μs).</p>
            <p><strong>Higher variance</strong> indicates less consistent performance, which may cause synchronization issues in distributed training.</p>
            <p><strong>Box plots</strong> show the distribution: boxes represent 25th-75th percentile, whiskers show the range, diamonds show outliers.</p>
            <p><strong>Violin plots</strong> show the full distribution shape.</p>
        </div>

        <div class="footer">
            <p>Generated by aorta-report | GEMM Variance Analysis Pipeline</p>
        </div>
    </div>
</body>
</html>"""


def _image_or_missing(image_data: str, alt_text: str) -> str:
    """Return image tag or missing placeholder."""
    if image_data:
        return f'<img src="{image_data}" alt="{alt_text}">'
    else:
        return f'<div class="missing-plot">Plot not available: {alt_text}</div>'
