"""
Simple tests for merge_gpu_trace_ranks.py script.
"""

import json
import pytest
import tempfile
from pathlib import Path
import sys

# Add scripts/utils directory to path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts' / 'utils'))

from merge_gpu_trace_ranks import merge_gpu_traces, categorize_process_type


def test_process_categorization():
    """Test that process types are correctly identified."""
    # GPU process
    assert categorize_process_type({'kernel'}, set()) == 'gpu_stream'

    # CPU process
    assert categorize_process_type({'cuda_runtime'}, set()) == 'cpu_main'

    # Unknown
    assert categorize_process_type({'unknown'}, {'unknown'}) == 'other'


@pytest.fixture
def temp_trace_dir():
    """Create temporary directory with mock trace files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_dir = Path(tmpdir)

        # Simple mock trace data
        mock_data = {
            "traceEvents": [
                {"pid": 100, "tid": 1, "ts": 1000, "dur": 100,
                 "ph": "X", "name": "kernel", "cat": "kernel", "args": {}},
                {"pid": 200, "tid": 2, "ts": 2000, "dur": 50,
                 "ph": "X", "name": "cudaMalloc", "cat": "cuda_runtime", "args": {}}
            ]
        }

        # Create rank directories
        for rank in range(2):
            rank_dir = trace_dir / f'rank{rank}'
            rank_dir.mkdir()
            with open(rank_dir / 'trace_step19.json', 'w') as f:
                json.dump(mock_data, f)

        yield trace_dir


def test_basic_merge_works(temp_trace_dir):
    """Test that script can merge traces without crashing."""
    output_file = temp_trace_dir / 'merged.json'

    result = merge_gpu_traces(
        trace_dir=str(temp_trace_dir),
        output_file=str(output_file),
        num_ranks=2
    )

    # Should succeed and create output file
    assert result == 0
    assert output_file.exists()

    # Output should be valid JSON with events
    with open(output_file) as f:
        data = json.load(f)
    assert 'traceEvents' in data
    assert len(data['traceEvents']) > 0


def test_missing_ranks_handled(temp_trace_dir):
    """Test that missing rank directories don't crash the script."""
    import shutil
    shutil.rmtree(temp_trace_dir / 'rank1')

    output_file = temp_trace_dir / 'merged.json'
    result = merge_gpu_traces(
        trace_dir=str(temp_trace_dir),
        output_file=str(output_file),
        num_ranks=2
    )

    # Should still succeed
    assert result == 0
    assert output_file.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
