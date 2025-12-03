#!/usr/bin/env python3
"""
Merge GPU profiling traces from multiple distributed training ranks into a single file.

WHAT IT DOES:
    Combines trace files from rank0/, rank1/, rank2/, etc. into one JSON file
    that can be viewed at https://ui.perfetto.dev

DIRECTORY STRUCTURE:
    Your trace directory should contain:
        trace_dir/
        ├── rank0/trace_step19.json
        ├── rank1/trace_step19.json
        ├── rank2/trace_step19.json
        └── ...

BASIC USAGE:
    python merge_gpu_trace_ranks.py <trace_directory>

    Example:
        python merge_gpu_trace_ranks.py experiments/my_run/torch_profiler

OPTIONS:
    -o FILE             Output file (default: merged_gpu_trace_enhanced.json)
    -n N                Number of ranks (default: 8)
    --trace-name NAME   Trace filename in each rank folder (default: trace_step19.json)

EXAMPLES:
    # Basic - merge 8 ranks
    python merge_gpu_trace_ranks.py experiments/my_run/torch_profiler

    # Custom output and trace name
    python merge_gpu_trace_ranks.py experiments/my_run/torch_profiler \
        -o merged.json \
        --trace-name customer_trace.json

    # Different number of ranks
    python merge_gpu_trace_ranks.py experiments/my_run/torch_profiler -n 16

OUTPUT:
    Creates a single JSON file with all ranks organized by:
    - Rank 0 GPU: PID 0,    Rank 0 CPU: PID 500
    - Rank 1 GPU: PID 1000, Rank 1 CPU: PID 1500
    - Rank 2 GPU: PID 2000, Rank 2 CPU: PID 2500
    etc.

    View the output at: https://ui.perfetto.dev
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def categorize_process_type(categories, event_names):
    """Determine if a process is GPU, CPU, or metadata based on its events."""

    # GPU stream processes have kernel and gpu_ events
    gpu_indicators = {'kernel', 'gpu_memcpy', 'gpu_memset', 'gpu_user_annotation'}
    if categories & gpu_indicators:
        return 'gpu_stream'

    # CPU processes have cpu_op, cuda_runtime, etc.
    cpu_indicators = {'cpu_op', 'cuda_runtime', 'user_annotation', 'fwdbwd'}
    if categories & cpu_indicators:
        return 'cpu_main'

    # Metadata processes only have labels
    if all('process_' in name or name in ['Record Window End', 'Iteration Start: PyTorch Profiler']
           for name in event_names):
        return 'metadata'

    return 'other'


def merge_gpu_traces(trace_dir, output_file, num_ranks=8, trace_name='trace_step19.json'):
    """
    Merge traces with better process organization:
    - Rank N GPU streams: PID = N * 1000
    - Rank N CPU process: PID = N * 1000 + 500
    """

    merged = {"traceEvents": []}

    # Categories to keep for GPU visualization
    gpu_categories = {
        'kernel', 'gpu_memcpy', 'gpu_memset', 'cuda_runtime',
        'cuda_driver', 'gpu_user_annotation', 'Kernel', 'ac2g'
    }

    # Also keep some CPU events for context
    cpu_categories = {
        'user_annotation',  # User markers
        'fwdbwd',          # Forward/backward pass markers
        'cpu_instant_event' # Important instant events
    }

    keep_categories = gpu_categories | cpu_categories

    print(f"Merging traces from {trace_dir}")

    for rank in range(num_ranks):
        trace_file = Path(trace_dir) / f'rank{rank}' / trace_name
        if not trace_file.exists():
            print(f"  Skipping rank {rank} - file not found")
            continue

        print(f"  Processing rank {rank}...")

        try:
            with open(trace_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Error reading {trace_file}: {e}")
            continue

        # First pass: identify process types
        process_info = defaultdict(lambda: {'categories': set(), 'event_names': set()})
        for event in data.get('traceEvents', []):
            if 'pid' in event:
                pid = event['pid']
                if 'cat' in event:
                    process_info[pid]['categories'].add(event['cat'])
                if 'name' in event:
                    process_info[pid]['event_names'].add(event['name'])

        # Map old PIDs to new PIDs based on process type
        # Also need to map TIDs to avoid collisions when multiple old PIDs map to same new PID
        pid_mapping = {}
        tid_mapping = {}  # Maps (old_pid, old_tid) -> new_tid

        # PID assignment scheme for organizing processes in trace viewer:
        # - GPU streams: rank * 1000 (e.g., Rank 0 = 0, Rank 1 = 1000, Rank 2 = 2000)
        #   The 1000 spacing allows each rank to have distinct PID ranges
        # - CPU/CUDA runtime: rank * 1000 + 500 (e.g., Rank 0 = 500, Rank 1 = 1500)
        #   The +500 offset separates CPU processes from GPU within each rank
        #   This offset is used during event processing to distinguish process types via modulo check
        gpu_stream_pid = rank * 1000
        cpu_main_pid = rank * 1000 + 500
        next_gpu_tid = {}  # Counter for GPU stream TIDs per new PID
        next_cpu_tid = {}  # Counter for CPU TIDs per new PID

        for old_pid, info in process_info.items():
            proc_type = categorize_process_type(info['categories'], info['event_names'])
            if proc_type == 'gpu_stream':
                pid_mapping[old_pid] = gpu_stream_pid
                tid_mapping[old_pid] = {}  # Will map old TIDs to new ones
            elif proc_type == 'cpu_main':
                pid_mapping[old_pid] = cpu_main_pid
                tid_mapping[old_pid] = {}
            # Skip metadata processes

        # Second pass: add events with new PIDs and remapped TIDs
        gpu_events = 0
        cpu_events = 0

        for event in data.get('traceEvents', []):
            # Skip if no category or not in keep list
            if 'cat' not in event or event['cat'] not in keep_categories:
                continue

            # Skip if PID not in mapping (metadata processes)
            if 'pid' in event and event['pid'] not in pid_mapping:
                continue

            # Update PID and TID if present
            if 'pid' in event:
                old_pid = event['pid']
                old_tid = event.get('tid', 0)
                new_pid = pid_mapping[old_pid]

                # Assign new TID if we haven't seen this (old_pid, old_tid) combo
                if old_tid not in tid_mapping[old_pid]:
                    # Check if GPU or CPU based on PID assignment scheme:
                    # GPU PIDs (0, 1000, 2000...) have modulo 0
                    # CPU PIDs (500, 1500, 2500...) have modulo 500
                    if new_pid % 1000 == 0:  # GPU process
                        if new_pid not in next_gpu_tid:
                            next_gpu_tid[new_pid] = 0
                        tid_mapping[old_pid][old_tid] = next_gpu_tid[new_pid]
                        next_gpu_tid[new_pid] += 1
                    else:  # CPU process
                        if new_pid not in next_cpu_tid:
                            next_cpu_tid[new_pid] = 0
                        tid_mapping[old_pid][old_tid] = next_cpu_tid[new_pid]
                        next_cpu_tid[new_pid] += 1

                event['pid'] = new_pid
                event['tid'] = tid_mapping[old_pid][old_tid]

                if new_pid % 1000 == 0:
                    gpu_events += 1
                else:
                    cpu_events += 1

            # Add rank info
            if 'args' not in event:
                event['args'] = {}
            event['args']['rank'] = rank

            # Prefix name with rank
            if 'name' in event:
                event['name'] = f"[R{rank}] {event['name']}"

            merged['traceEvents'].append(event)

        print(f"    Added {gpu_events} GPU events, {cpu_events} CPU context events")

    # Add process metadata for better visualization
    print("\n  Adding process metadata...")
    for rank in range(num_ranks):
        # GPU stream process
        merged['traceEvents'].extend([
            {
                "pid": rank * 1000,
                "tid": 0,
                "ts": 0,
                "ph": "M",
                "cat": "__metadata",
                "name": "process_name",
                "args": {"name": f"Rank {rank} - GPU Stream"}
            },
            {
                "pid": rank * 1000,
                "tid": 0,
                "ts": 0,
                "ph": "M",
                "cat": "__metadata",
                "name": "process_sort_index",
                "args": {"sort_index": rank * 2}
            }
        ])

        # CPU process
        merged['traceEvents'].extend([
            {
                "pid": rank * 1000 + 500,
                "tid": 0,
                "ts": 0,
                "ph": "M",
                "cat": "__metadata",
                "name": "process_name",
                "args": {"name": f"Rank {rank} - CPU/CUDA Runtime"}
            },
            {
                "pid": rank * 1000 + 500,
                "tid": 0,
                "ts": 0,
                "ph": "M",
                "cat": "__metadata",
                "name": "process_sort_index",
                "args": {"sort_index": rank * 2 + 1}
            }
        ])

    # Write merged trace
    print(f"\nWriting merged trace to {output_file}...")
    try:
        with open(output_file, 'w') as f:
            json.dump(merged, f)
    except IOError as e:
        print(f"Error writing output file: {e}")
        return 1

    print(f"\n Successfully merged traces")
    print(f" Total events: {len(merged['traceEvents'])}")
    print(f"\nView in Perfetto: https://ui.perfetto.dev")
    print("\nProcess organization:")
    print("  - PIDs N000: Rank N GPU streams (kernels, memcpy)")
    print("  - PIDs N500: Rank N CPU/CUDA runtime context")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced GPU trace merger with better process organization'
    )

    parser.add_argument(
        'trace_dir',
        type=str,
        help='Directory containing rank subdirectories with trace files'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='merged_gpu_trace_enhanced.json',
        help='Output merged trace file'
    )
    parser.add_argument(
        '-n', '--num-ranks',
        type=int,
        default=8,
        help='Number of ranks to process (default: 8)'
    )
    parser.add_argument(
        '--trace-name',
        type=str,
        default='trace_step19.json',
        help='Name of trace file in each rank directory'
    )

    args = parser.parse_args()

    return merge_gpu_traces(
        args.trace_dir,
        args.output,
        args.num_ranks,
        args.trace_name
    )


if __name__ == '__main__':
    exit(main())
