#!/usr/bin/env python3
"""
Generate synthetic test data for GEMM analysis regression tests.
Creates realistic PyTorch profiler traces based on actual MI350X traces
for two channel configurations, two threads, and 7 ranks.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List
import random
import argparse
from datetime import datetime
import pandas as pd

# Set random seed for reproducibility
random.seed(42)


class SyntheticDataGenerator:
    """Generate synthetic test data for regression testing."""

    def __init__(
        self,
        output_dir: Path,
        thread_configs: List[int] | None = None,
        channel_configs: List[int] | None = None,
        num_ranks: int = 8,
        num_batches: int = 2,
    ):
        self.output_dir = Path(output_dir)
        self.sweep_dir = self.output_dir / "test_sweep"  # Fixed name without timestamp

        # Fixed configurations for testing (can be overridden)
        self.thread_configs = thread_configs or [256, 512]
        # include 28 by default so synthetic sweeps cover test expectations
        self.channel_configs = channel_configs or [28, 56]
        self.num_ranks = num_ranks
        self.num_batches = num_batches  # Generate batches (ProfilerSteps) per trace

        # Realistic GEMM kernel names from actual AMD MI350X traces
        self.gemm_kernels = [
            # Most common GEMM kernels from actual traces
            "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB2_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_15_MO40_NTn1_NTA0_NTB0_NTC6_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO1_SRVW0_SSO1_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1",
            "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT8_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW8_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA8_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1",
            # Additional GEMM variants
            "Cijk_Alik_Bljk_SB_MT128x128x32_MI32x32x1x2",
            "Cijk_Alik_Bljk_SB_MT256x256x64_MI64x64x2x2",
            "Cijk_Alik_Bljk_SB_MT64x64x16_MI16x16x1x1",
        ]

        # Other kernel types seen in real traces
        self.other_kernels = [
            "void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)",
            "void at::native::unrolled_elementwise_kernel<at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast)",
            "ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)",
        ]

        # NCCL operations from real traces
        self.nccl_ops = [
            "nccl:all_reduce",
            "nccl:broadcast",
            "nccl:all_gather",
            "nccl:reduce_scatter"
        ]

    def create_pytorch_trace(self, rank: int, thread: int, channel: int) -> Dict:
        """Create a realistic PyTorch profiler trace matching actual MI350X format.

        Generates trace events for multiple batches (ProfilerSteps).
        Each ProfilerStep represents one training batch/iteration.
        """

        # Device properties for MI350X GPUs (exact format from real traces)
        device_properties = []
        for gpu_id in range(8):
            device_properties.append({
                "id": gpu_id,
                "name": "AMD Instinct MI350X",
                "totalGlobalMem": 309220868096,
                "computeMajor": 9,
                "computeMinor": 5,
                "maxThreadsPerBlock": 1024,
                "maxThreadsPerMultiprocessor": 2048,
                "regsPerBlock": 65536,
                "warpSize": 64,
                "sharedMemPerBlock": 163840,
                "numSms": 256,
                "maxSharedMemoryPerMultiProcessor": 163840
            })

        # Metadata matching real traces
        metadata = {
            "schemaVersion": 1,
            "deviceProperties": device_properties,
            "roctracer_version": 4.1,
            "with_flops": 1,
            "distributedInfo": {
                "backend": "nccl",
                "rank": rank,
                "world_size": 8,
                "pg_count": 1,
                "pg_config": [{
                    "pg_name": "0",
                    "pg_desc": "default_pg",
                    "backend_config": "cuda:nccl",
                    "pg_size": 8,
                    "ranks": list(range(8))
                }],
                "nccl_version": "2.26.6"
            },
            "record_shapes": 1,
            "hip_runtime_version": 70051831,
            "profile_memory": 1,
            "hip_driver_version": 70051831,
            "with_stack": 1,
            "trace_id": f"F063B8B18C2C424DB348840DD69E22BB",
            "displayTimeUnit": "ms",
            "baseTimeNanoseconds": 1759300074000000000
        }

        # Generate trace events for multiple batches
        events = []
        base_ts = 5917770837674.380  # Base timestamp in microseconds
        pid = 258  # CPU Process ID
        gpu_pid = 2  # GPU Process ID
        tid_main = thread  # Main thread ID
        tid_nccl = 683  # NCCL thread ID (from real traces)
        external_id_counter = 1
        correlation_counter = 3900

        # Generate events for each batch (2 batches as requested)
        for batch in range(self.num_batches):
            batch_start_ts = base_ts + batch * 600000000  # ~600ms per batch (realistic timing)
            current_ts = batch_start_ts

            # ProfilerStep event (marks each batch/iteration)
            step_num = batch + 2  # Start from ProfilerStep#2 like real traces
            step_duration = 536019.180 + random.uniform(-10000, 10000)  # Slight variation per batch

            events.append({
                "ph": "X",
                "cat": "user_annotation",
                "name": f"ProfilerStep#{step_num}",
                "pid": pid,
                "tid": tid_main,
                "ts": current_ts,
                "dur": step_duration,
                "args": {
                    "External id": external_id_counter,
                    "Record function id": 0,
                    "Ev Idx": batch * 1000
                }
            })
            external_id_counter += 1

            # DataLoader event (loading next batch of data)
            events.append({
                "ph": "X",
                "cat": "user_annotation",
                "name": "enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__",
                "pid": pid,
                "tid": tid_main,
                "ts": current_ts + 50,
                "dur": 141.768,
                "args": {
                    "External id": external_id_counter,
                    "Record function id": 0,
                    "Ev Idx": batch * 1000 + 1
                }
            })
            external_id_counter += 1

            # Add CPU operations that TraceLens expects
            cpu_op_names = ["aten::to", "aten::_to_copy", "aten::mul", "aten::add", "aten::matmul"]
            for cpu_idx in range(5):
                events.append({
                    "ph": "X",
                    "cat": "cpu_op",
                    "name": cpu_op_names[cpu_idx % len(cpu_op_names)],
                    "pid": pid,
                    "tid": tid_main,
                    "ts": current_ts + 200 + cpu_idx * 100,
                    "dur": 100 + random.uniform(-20, 20),
                    "args": {
                        "External id": external_id_counter,
                        "Sequence number": 2351 + cpu_idx,
                        "Fwd thread id": 0,
                        "Record function id": 0,
                        "Concrete Inputs": ["", "6", "0", "", "", "True", "False", ""],
                        "Input type": ["float", "Scalar", "Scalar", "", "", "Scalar", "Scalar", ""],
                        "Input Strides": [[1441792, 8192, 256, 1], [], [], [], [], [], [], []],
                        "Input Dims": [[512, 176, 32, 256], [], [], [], [], [], [], []],
                        "Ev Idx": batch * 1000 + 2 + cpu_idx
                    }
                })
                external_id_counter += 1

            current_ts += 1000

            # Generate NCCL broadcast at start of batch (parameter sync)
            events.append({
                "ph": "X",
                "cat": "user_annotation",
                "name": "nccl:broadcast",
                "pid": pid,
                "tid": tid_main,
                "ts": current_ts,
                "dur": 5000 + random.uniform(-500, 500),
                "args": {
                    "External id": external_id_counter,
                    "Record function id": 0,
                    "Ev Idx": batch * 1000 + 10
                }
            })
            external_id_counter += 1
            current_ts += 6000

            # Generate mix of kernels for this batch
            num_kernels_per_batch = 30 + random.randint(-5, 5)

            for kernel_idx in range(num_kernels_per_batch):
                # Mix GEMM kernels with other kernels (70% GEMM, 30% other)
                if random.random() < 0.7:
                    # GEMM kernel
                    kernel_name_idx = (rank + kernel_idx + batch) % len(self.gemm_kernels)
                    kernel_name = self.gemm_kernels[kernel_name_idx]

                    # Realistic timing based on kernel type and configuration
                    if "MT256x240" in kernel_name:
                        base_dur = 45000 + random.uniform(-5000, 5000)
                        grid_size = [385024, 1, 1]
                    elif "MT256x256" in kernel_name:
                        base_dur = 50000 + random.uniform(-5000, 5000)
                        grid_size = [65536, 1, 1]
                    else:
                        base_dur = 30000 + random.uniform(-3000, 3000)
                        grid_size = [32768, 1, 1]

                    # Configuration-based variance
                    dur_variance = base_dur * (1 + (thread-256)/2000 + (channel-28)/200 + rank * 0.005)
                else:
                    # Other kernel (elementwise, etc.)
                    kernel_name = random.choice(self.other_kernels)
                    dur_variance = 5000 + random.uniform(-1000, 1000)
                    grid_size = [16384, 1, 1]

                # Create a CPU launcher op that envelops the runtime + kernel so TraceLens can assign parents
                cpu_launcher_ts = current_ts - 40
                cpu_launcher_dur = dur_variance + 80
                launcher_name = cpu_op_names[kernel_idx % len(cpu_op_names)]
                kernel_detail_entry = {
                    "name": kernel_name,
                    "dur": dur_variance,
                    "stream": 0,
                    "grid": grid_size,
                    "block": [256, 1, 1],
                }
                events.append({
                    "ph": "X",
                    "cat": "cpu_op",
                    "name": launcher_name,
                    "pid": pid,
                    "tid": tid_main,
                    "ts": cpu_launcher_ts,
                    "dur": cpu_launcher_dur,
                    "args": {
                        "External id": external_id_counter,
                        "Sequence number": 3000 + kernel_idx,
                        "Fwd thread id": 0,
                        "Record function id": 0,
                        "Concrete Inputs": ["", "6", "0", "", "", "True", "False", ""],
                        "Input type": ["float", "Scalar", "Scalar", "", "", "Scalar", "Scalar", ""],
                        "Input Strides": [[1441792, 8192, 256, 1], [], [], [], [], [], [], []],
                        "Input Dims": [[512, 176, 32, 256], [], [], [], [], [], [], []],
                        "Ev Idx": batch * 1000 + 600 + kernel_idx
                    },
                    "kernel_details": [kernel_detail_entry],
                    "kernel_names": [kernel_name],
                })
                external_id_counter += 1

                # Create runtime launch and ac2g bridge to link kernels for TraceLens
                corr_id = correlation_counter
                runtime_ts = current_ts - 20  # slightly before kernel start
                runtime_dur = 10

                # cuda runtime launch event
                events.append({
                    "ph": "X",
                    "cat": "cuda_runtime",
                    "name": "hipLaunchKernel",
                    "pid": pid,
                    "tid": tid_main,
                    "ts": runtime_ts,
                    "dur": runtime_dur,
                    "args": {
                        "External id": external_id_counter,
                        "correlation": corr_id,
                        "kernel": kernel_name,
                        "grid": grid_size,
                        "block": [256, 1, 1],
                    }
                })
                external_id_counter += 1

                # ac2g start/end events to allow TraceLens to map runtime->GPU
                events.append({
                    "ph": "s",
                    "id": corr_id,
                    "pid": pid,
                    "tid": tid_main,
                    "ts": runtime_ts,
                    "cat": "ac2g",
                    "name": "ac2g"
                })
                events.append({
                    "ph": "f",
                    "id": corr_id,
                    "pid": gpu_pid,
                    "tid": 0,
                    "ts": current_ts,
                    "cat": "ac2g",
                    "name": "ac2g",
                    "bp": "e"
                })

                # GPU kernel event
                events.append({
                    "ph": "X",
                    "cat": "kernel",
                    "name": kernel_name,
                    "pid": gpu_pid,
                    "tid": 0,  # GPU stream 0
                    "ts": current_ts,
                    "dur": dur_variance,
                    "args": {
                        "External id": external_id_counter,
                        "device": rank % 8,  # Device ID based on rank
                        "stream": 0,  # Stream ID
                        "correlation": corr_id,
                        "kind": "Dispatch Kernel",  # Required by TraceLens
                        "grid": grid_size,
                        "block": [256, 1, 1],
                        "kernel": kernel_name  # Keep kernel name in args too
                    }
                })

                external_id_counter += 1
                correlation_counter += 1
                current_ts += dur_variance + random.uniform(100, 500)

                # Add periodic NCCL all_reduce operations (every 5 kernels for gradient sync)
                if kernel_idx % 5 == 4:
                    events.append({
                        "ph": "X",
                        "cat": "user_annotation",
                        "name": "nccl:all_reduce",
                        "pid": pid,
                        "tid": tid_nccl,  # NCCL operations on different thread
                        "ts": current_ts,
                        "dur": 15000 + random.uniform(-2000, 2000),
                        "args": {
                            "External id": external_id_counter,
                            "Record function id": 0,
                            "Ev Idx": batch * 1000 + 100 + kernel_idx
                        }
                    })
                    external_id_counter += 1

                    # Corresponding NCCL runtime launch and kernel
                    nccl_corr_id = correlation_counter
                    runtime_ts = current_ts + 50
                    runtime_dur = 8

                    # CPU launcher for NCCL kernel (gives parent for short-kernel stats)
                    events.append({
                        "ph": "X",
                        "cat": "cpu_op",
                        "name": "nccl::launch",
                        "pid": pid,
                        "tid": tid_nccl,
                        "ts": runtime_ts - 20,
                        "dur": 16000,
                        "args": {
                            "External id": external_id_counter,
                            "Sequence number": 5000 + kernel_idx,
                            "Fwd thread id": 0,
                            "Record function id": 0,
                            "Concrete Inputs": ["nccl_all_reduce"],
                            "Input type": ["string"],
                            "Input Strides": [[]],
                            "Input Dims": [[]],
                            "Ev Idx": batch * 1000 + 700 + kernel_idx
                        },
                        "kernel_details": [{
                            "name": "ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)",
                            "dur": 14000,
                            "stream": 4,
                            "grid": [1, 1, 1],
                            "block": [1024, 1, 1],
                        }],
                        "kernel_names": ["ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)"],
                    })
                    external_id_counter += 1

                    events.append({
                        "ph": "X",
                        "cat": "cuda_runtime",
                        "name": "hipLaunchKernel",
                        "pid": pid,
                        "tid": tid_nccl,
                        "ts": runtime_ts,
                        "dur": runtime_dur,
                        "args": {
                            "External id": external_id_counter,
                            "correlation": nccl_corr_id,
                            "kernel": "ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)",
                            "grid": [1, 1, 1],
                            "block": [1024, 1, 1],
                        }
                    })
                    external_id_counter += 1

                    events.append({
                        "ph": "s",
                        "id": nccl_corr_id,
                        "pid": pid,
                        "tid": tid_nccl,
                        "ts": runtime_ts,
                        "cat": "ac2g",
                        "name": "ac2g"
                    })

                    events.append({
                        "ph": "X",
                        "cat": "kernel",
                        "name": "ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)",
                        "pid": gpu_pid,
                        "tid": 4,  # Different GPU stream for NCCL
                        "ts": current_ts + 100,
                        "dur": 14000 + random.uniform(-2000, 2000),
                        "args": {
                            "External id": external_id_counter,
                            "device": rank % 8,  # Device ID based on rank
                            "stream": 4,  # NCCL stream
                            "correlation": nccl_corr_id,
                            "kind": "Dispatch Kernel",  # Required by TraceLens
                            "grid": [1, 1, 1],
                            "block": [1024, 1, 1],
                            # Metadata required by TraceLens NcclAnalyser
                            "Collective name": "allreduce",
                            "In msg nelems": 1048576,
                            "Out msg nelems": 1048576,
                            "Group size": 8,
                            "Process Group Name": "0",
                            "Process Group Ranks": list(range(8)),
                            "dtype": "Float",
                            "Process Group Description": "default_pg",
                            "Process Group Ranks": list(range(8)),
                        }
                    })
                    external_id_counter += 1
                    events.append({
                        "ph": "f",
                        "id": nccl_corr_id,
                        "pid": gpu_pid,
                        "tid": 4,
                        "ts": current_ts + 100,
                        "cat": "ac2g",
                        "name": "ac2g",
                        "bp": "e"
                    })

                    correlation_counter += 1
                    current_ts += 16000

            # Add GPU annotation events (match the ProfilerStep on GPU side)
            for gpu_tid in [0, 5, 6]:  # Multiple GPU threads like real traces
                events.append({
                    "ph": "X",
                    "cat": "gpu_user_annotation",
                    "name": f"ProfilerStep#{step_num}",
                    "pid": gpu_pid,
                    "tid": gpu_tid,
                    "ts": batch_start_ts,
                    "dur": step_duration,
                    "args": {
                        "External id": external_id_counter,
                        "Record function id": 0,
                        "Ev Idx": batch * 1000 + 500 + gpu_tid
                    }
                })
                external_id_counter += 1

        # Sort events by timestamp (important for trace validity)
        events.sort(key=lambda x: x["ts"])

        # Combine metadata and events
        trace_data = metadata.copy()
        trace_data["traceEvents"] = events

        return trace_data

    def generate_all_data(self) -> None:
        """Generate all synthetic test data matching real trace structure."""
        print(f"Generating realistic synthetic test data in: {self.sweep_dir}")
        print(f"  Based on actual MI350X PyTorch profiler traces")
        print(f"  Generating {self.num_batches} batches (ProfilerSteps) per trace")

        for thread in self.thread_configs:
            for channel in self.channel_configs:
                config_name = f"{thread}thread/nccl_{channel}channels"
                print(f"\n  Generating traces for {config_name}...")

                # Create PyTorch traces for each rank
                for rank in range(self.num_ranks):
                    trace_dir = self.sweep_dir / f"{config_name}/torch_profiler/rank{rank}"
                    trace_dir.mkdir(parents=True, exist_ok=True)

                    # Generate realistic trace data
                    trace_data = self.create_pytorch_trace(rank, thread, channel)

                    # Write to customer_trace_step10.json (matching real traces)
                    trace_file = trace_dir / "customer_trace_step10.json"

                    with open(trace_file, 'w') as f:
                        json.dump(trace_data, f, indent=2)

                    # Verify event counts
                    num_events = len(trace_data.get("traceEvents", []))
                    print(f"    Rank {rank}: Generated {num_events} events")

        # Create a metadata file for the sweep
        metadata = {
            "thread_configs": self.thread_configs,
            "channel_configs": self.channel_configs,
            "num_ranks": self.num_ranks,
            "num_batches": self.num_batches,
            "generated_at": datetime.now().isoformat(),
            "based_on": "AMD MI350X actual traces",
            "trace_format": "PyTorch Profiler JSON",
            "profiler_steps": [f"ProfilerStep#{i+2}" for i in range(self.num_batches)]
        }

        with open(self.sweep_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nSynthetic test data generated successfully!")
        print(f"  Thread configs: {self.thread_configs}")
        print(f"  Channel configs: {self.channel_configs}")
        print(f"  Number of ranks: {self.num_ranks}")
        print(f"  Batches per trace: {self.num_batches}")
        print(f"  ProfilerSteps: {metadata['profiler_steps']}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test data for GEMM analysis regression tests")
    parser.add_argument("--output-dir", type=str, default="testdata",
                        help="Output directory for synthetic data (default: testdata)")
    parser.add_argument("--threads", type=str, default="256,512",
                        help="Comma separated thread configs (default: 256,512)")
    parser.add_argument("--channels", type=str, default="24,28,56",
                        help="Comma separated channel configs (default: 24,28,56)")
    parser.add_argument("--ranks", type=int, default=8,
                        help="Number of ranks to generate (default: 8)")
    parser.add_argument("--batches", type=int, default=2,
                        help="Profiler steps per trace (default: 2)")
    args = parser.parse_args()

    output_dir = args.output_dir
    thread_configs = [int(x) for x in args.threads.split(",") if x]
    channel_configs = [int(x) for x in args.channels.split(",") if x]

    generator = SyntheticDataGenerator(
        output_dir=output_dir,
        thread_configs=thread_configs,
        channel_configs=channel_configs,
        num_ranks=args.ranks,
        num_batches=args.batches,
    )
    generator.generate_all_data()


if __name__ == "__main__":
    main()
