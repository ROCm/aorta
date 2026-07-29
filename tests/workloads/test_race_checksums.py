"""CPU-only tests for FSDP correctness detectors.

These are CPU-only: no GPU, no torch.distributed. The method under test only
reads `self.layer_checksums`, `self.rank`, and appends to
`self.corruption_details` -- it never touches CUDA. So we bypass __init__ with
object.__new__ and set just those three attributes.

Contract (read from src/aorta/race/modes/fsdp.py):
    _verify_layer_checksums(iteration) -> bool
      - uses the modal value for each checksum key as the reference
      - returns True if every layer matches that consensus on all four keys
        (comm_input, comm_output, compute_input, compute_output)
      - returns True when fewer than two populated layers are available
      - returns False on any mismatch, logging LAYER_CHECKSUM_MISMATCH (<key>)
        and appending a {"type": "layer_checksum_mismatch_<key>", ...} record
      - None entries among later layers are skipped
"""

import torch

from aorta.race.modes.fsdp import (
    _REDUCE_SCATTER_ORACLE_DTYPE,
    FSDPModeReproducer,
)


def _make_reproducer(layer_checksums):
    """Build an FSDPModeReproducer with only the attrs the method reads.

    object.__new__ skips __init__, so no CUDA buffers are allocated.
    """
    r = object.__new__(FSDPModeReproducer)
    r.layer_checksums = layer_checksums
    r.rank = 0
    r.corruption_details = []
    # Observability counters normally set in __init__ (skipped by object.__new__).
    r.layers_verified = 0
    r.layer_checksum_mismatches = 0
    return r


def _checksums(comm_in=10, comm_out=20, compute_in=30, compute_out=40):
    return {
        "comm_input": comm_in,
        "comm_output": comm_out,
        "compute_input": compute_in,
        "compute_output": compute_out,
    }


def test_clean_layers_pass():
    """Identical checksum dicts across all layers -> pass, no corruption recorded."""
    r = _make_reproducer([_checksums(), _checksums(), _checksums(), _checksums()])
    assert r._verify_layer_checksums(iteration=0) is True
    assert r.corruption_details == []
    # Observability: a clean run must still PROVE the detector ran.
    assert r.layers_verified == 3  # layers 1..3 compared against layer 0
    assert r.layer_checksum_mismatches == 0


def test_compute_corruption_detected():
    """One layer with a divergent compute_output is flagged and localized to COMPUTE.

    comm_* still match, so the recorded mismatch type must be the compute key and
    the offending layer index must be exposed in corruption_details.
    """
    bad_layer = 2
    layers = [_checksums() for _ in range(4)]
    layers[bad_layer] = _checksums(compute_out=999)  # only compute_output differs

    r = _make_reproducer(layers)
    assert r._verify_layer_checksums(iteration=5) is False

    assert len(r.corruption_details) == 1
    detail = r.corruption_details[0]
    assert detail["type"] == "layer_checksum_mismatch_compute_output"
    assert detail["layer_cmp"] == bad_layer
    assert detail["layer_ref"] == 0
    # localized to compute, NOT comm
    assert "comm" not in detail["type"]
    assert r.layers_verified == 3
    assert r.layer_checksum_mismatches == 1


def test_comm_corruption_detected():
    """One layer with a divergent comm_output is flagged and localized to COMM/NIC."""
    bad_layer = 1
    layers = [_checksums() for _ in range(3)]
    layers[bad_layer] = _checksums(comm_out=777)  # only comm_output differs

    r = _make_reproducer(layers)
    assert r._verify_layer_checksums(iteration=9) is False

    assert len(r.corruption_details) == 1
    detail = r.corruption_details[0]
    assert detail["type"] == "layer_checksum_mismatch_comm_output"
    assert detail["layer_cmp"] == bad_layer
    assert detail["cmp_checksum"] == 777
    # localized to comm, NOT compute
    assert "compute" not in detail["type"]


def test_corrupted_layer_zero_is_localized_by_modal_reference():
    """A bad layer 0 must not mark every agreeing later layer as corrupted."""
    layers = [_checksums(comm_out=777)] + [_checksums() for _ in range(3)]

    r = _make_reproducer(layers)
    assert r._verify_layer_checksums(iteration=0) is False

    assert r.layer_checksum_mismatches == 1
    assert len(r.corruption_details) == 1
    detail = r.corruption_details[0]
    assert detail["layer_cmp"] == 0
    assert detail["layer_ref"] == 1
    assert detail["ref_checksum"] == 20
    assert detail["cmp_checksum"] == 777


def test_tied_layer_fingerprints_are_reported_as_ambiguous():
    r = _make_reproducer(
        [
            _checksums(compute_out=40),
            _checksums(compute_out=999),
        ]
    )

    assert r._verify_layer_checksums(iteration=6) is False
    assert r.layer_checksum_mismatches == 0
    assert r.corruption_details == [
        {
            "type": "layer_checksum_ambiguous_compute_output",
            "rank": 0,
            "iteration": 6,
            "checksum_groups": {40: 1, 999: 1},
        }
    ]


def test_mismatch_counter_is_per_layer_not_per_key():
    """A single corrupted layer with MULTIPLE bad keys counts ONCE, not 4x.

    layer_checksum_mismatches is a per-layer counter; a layer that diverges on
    several checksum keys must not inflate it (was previously +1 per key).
    """
    bad_layer = 1
    layers = [_checksums() for _ in range(3)]
    # All four keys differ on the one bad layer.
    layers[bad_layer] = _checksums(comm_in=1, comm_out=2, compute_in=3, compute_out=4)

    r = _make_reproducer(layers)
    assert r._verify_layer_checksums(iteration=0) is False
    # corruption_details still records each key (full localization detail)...
    assert len(r.corruption_details) == 4
    # ...but the per-layer metric counts the layer ONCE.
    assert r.layer_checksum_mismatches == 1


def test_single_layer_or_empty():
    """1 layer or empty -> nothing to compare against -> pass, no false positive."""
    # Single layer: loop over range(1, 1) never runs.
    single = _make_reproducer([_checksums()])
    assert single._verify_layer_checksums(iteration=0) is True
    assert single.corruption_details == []

    # Reference is None (e.g. compute disabled for layer 0) -> early True return.
    none_ref = _make_reproducer([None, _checksums()])
    assert none_ref._verify_layer_checksums(iteration=0) is True
    assert none_ref.corruption_details == []


def test_reduce_scatter_oracle_dtype_is_fp32():
    assert _REDUCE_SCATTER_ORACLE_DTYPE == torch.float32


def test_reduce_scatter_exact_oracle_passes_and_reports_real_mismatch():
    r = object.__new__(FSDPModeReproducer)
    r.rank = 0
    r.world_size = 24
    r.reduce_scatter_oracle_dtype = "float32"
    r.corruption_details = []
    r.grad_shard = torch.full((8,), 300.0, dtype=torch.float32)

    assert r._verify_reduce_scatter(iteration=3) is True

    r.grad_shard[5] = 298.0
    assert r._verify_reduce_scatter(iteration=4) is False
    detail = r.corruption_details[-1]
    assert detail["iteration"] == 4
    assert detail["first_mismatch_index"] == 5
    assert detail["expected"] == 300.0
    assert detail["actual"] == 298.0
    assert detail["mismatch_count"] == 1
    assert detail["max_abs_error"] == 2.0


def test_all_gather_exact_check_localizes_layer_source_and_index():
    r = object.__new__(FSDPModeReproducer)
    r.rank = 1
    r.world_size = 3
    r.shard_size = 4
    r.corruption_details = []
    r.full_param = torch.cat(
        [
            torch.full((4,), float(rank + 1), dtype=torch.bfloat16)
            for rank in range(3)
        ]
    )

    assert r._verify_all_gather(iteration=2, layer_idx=7, phase="forward") is True

    r.full_param[6] = 9.0
    assert r._verify_all_gather(iteration=2, layer_idx=7, phase="forward") is False
    detail = r.corruption_details[-1]
    assert detail["src_rank"] == 1
    assert detail["layer"] == 7
    assert detail["phase"] == "forward"
    assert detail["first_mismatch_index"] == 2
    assert detail["actual"] == 9.0
    assert detail["expected"] == 2.0


def test_rank_fill_pattern_is_nonzero_for_rank_zero():
    r = object.__new__(FSDPModeReproducer)
    r.rank = 0
    r.param_shards = [torch.empty(4), torch.empty(4)]
    r.full_grad = torch.empty(4, dtype=torch.float32)

    r._fill_patterns()

    assert all(torch.equal(shard, torch.ones_like(shard)) for shard in r.param_shards)
    assert torch.equal(r.full_grad, torch.ones_like(r.full_grad))
