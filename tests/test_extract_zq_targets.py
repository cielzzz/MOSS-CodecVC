from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ver3_1/extract_zq_targets.py"
SPEC = importlib.util.spec_from_file_location("extract_zq_targets_v3_1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)


class _FakeQuantizer:
    def __init__(self, latent_dim: int = 3, num_quantizers: int = 2, codebook_size: int = 1024) -> None:
        self.latent_dim = latent_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.calls = 0

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        # codes: (NQ, B, T); retain a deterministic dependence on every codebook.
        base = codes.to(torch.float32).sum(dim=0).unsqueeze(1)
        offsets = torch.arange(self.latent_dim, dtype=torch.float32, device=codes.device).view(1, -1, 1)
        return base + offsets


class _FakeCodec:
    def __init__(self, encoded: dict[str, torch.Tensor] | None = None, latent_dim: int = 3) -> None:
        self.device = torch.device("cpu")
        self.model = SimpleNamespace(quantizer=_FakeQuantizer(latent_dim))
        self.encoded = encoded or {}
        self.encode_calls: list[tuple[str, int | None]] = []

    def encode_path(self, audio_path: str, *, n_vq: int | None = None) -> dict[str, object]:
        self.encode_calls.append((audio_path, n_vq))
        codes = self.encoded[audio_path]
        if n_vq is not None:
            codes = codes[:, :n_vq]
        return {"codes": codes.clone(), "num_frames": codes.shape[0], "n_vq": codes.shape[1]}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _config(path: Path) -> Path:
    codec_path = path.parent / f"{path.stem}_codec"
    codec_path.mkdir(parents=True, exist_ok=True)
    (codec_path / "config.json").write_text(
        json.dumps(
            {
                "sampling_rate": 24000,
                "downsample_rate": 1920,
                "quantizer_kwargs": {
                    "codebook_size": 1024,
                    "num_quantizers": 2,
                    "output_dim": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    path.write_text(
        json.dumps(
            {
                "moss": {
                    "codec_path": str(codec_path),
                    "root": "/fake/moss",
                    "default_n_vq": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_byte_range_sharding_covers_each_line_exactly_once(tmp_path: Path) -> None:
    manifest = tmp_path / "large.jsonl"
    rows = [
        {"sample_id": f"row-{index}", "payload": "x" * (17 + index * 101)}
        for index in range(19)
    ]
    _write_jsonl(manifest, rows)

    seen: list[tuple[int, bytes]] = []
    for shard_id in range(7):
        owned = list(extractor.iter_jsonl_byte_range(manifest, shard_id=shard_id, num_shards=7))
        start, end = extractor.byte_range(manifest.stat().st_size, shard_id, 7)
        assert all(start <= offset < end for offset, _ in owned)
        seen.extend(owned)

    assert [json.loads(line)["sample_id"] for _, line in sorted(seen)] == [
        row["sample_id"] for row in rows
    ]
    assert len({offset for offset, _ in seen}) == len(rows)


def test_stable_collision_safe_filename_and_ids() -> None:
    digest_a = extractor.row_sha256(b'{"sample_id":"a/b"}')
    digest_b = extractor.row_sha256(b'{"sample_id":"a:b"}')
    name_a = extractor.safe_filename("no_text", "a/b", digest_a)
    name_b = extractor.safe_filename("no_text", "a:b", digest_b)

    assert name_a == extractor.safe_filename("no_text", "a/b", digest_a)
    assert name_a != name_b
    assert name_a.endswith(".npy")
    assert "/" not in name_a
    assert extractor.logical_record_id("no_text", "same") != extractor.logical_record_id("text", "same")
    assert extractor.resolve_utterance_id({}, "a" * 64) == "row-" + "a" * 24


def test_atomic_npy_round_trip_and_shape_validation(tmp_path: Path) -> None:
    path = tmp_path / "nested/zq.npy"
    value = np.arange(15, dtype=np.float16).reshape(3, 5)
    size = extractor.atomic_save_npy(path, value)
    assert size == path.stat().st_size
    loaded = extractor.validate_npy(
        path,
        expected_dim=3,
        expected_dtype="float16",
        expected_frames=5,
    )
    np.testing.assert_array_equal(loaded, value)
    assert not list(path.parent.glob("*.tmp-*"))

    with pytest.raises(ValueError, match="frame count"):
        extractor.validate_npy(path, expected_dim=3, expected_dtype="float16", expected_frames=4)
    with pytest.raises(ValueError, match="dtype"):
        extractor.validate_npy(path, expected_dim=3, expected_dtype="float32", expected_frames=5)


def test_manifest_encode_and_verify_codes_sources() -> None:
    codes = torch.tensor([[1, 2], [3, 4], [5, 6]])
    audio = "/fake/target.wav"
    row = {
        "audio_codes": codes.tolist(),
        "moss_codecvc_meta": {"target_audio": audio, "target_codec_frames": 3},
    }
    codec = _FakeCodec({audio: codes})

    manifest, target = extractor.acquire_codes(
        row, codes_source="manifest", codec=None, n_vq=2, codebook_size=1024
    )
    torch.testing.assert_close(manifest, codes)
    assert target == audio
    encoded, target = extractor.acquire_codes(
        row, codes_source="encode", codec=codec, n_vq=2, codebook_size=1024
    )
    torch.testing.assert_close(encoded, codes)
    assert target == audio
    verified, target = extractor.acquire_codes(
        row, codes_source="verify", codec=codec, n_vq=2, codebook_size=1024
    )
    torch.testing.assert_close(verified, codes)
    assert target == audio

    bad_codec = _FakeCodec({audio: codes.clone()})
    bad_codec.encoded[audio][1, 1] = 99
    with pytest.raises(ValueError, match="tokens differ"):
        extractor.acquire_codes(
            row, codes_source="verify", codec=bad_codec, n_vq=2, codebook_size=1024
        )


def test_decode_codes_batch_crops_each_item_to_valid_length() -> None:
    quantizer = _FakeQuantizer(latent_dim=3)
    first = torch.tensor([[1, 10], [2, 20]])
    second = torch.tensor([[3, 30], [4, 40], [5, 50], [6, 60]])
    decoded = extractor.decode_codes_batch(
        quantizer,
        [first, second],
        device=torch.device("cpu"),
        expected_dim=3,
    )
    assert [tuple(value.shape) for value in decoded] == [(3, 2), (3, 4)]
    torch.testing.assert_close(decoded[0][0], torch.tensor([11.0, 22.0]))
    torch.testing.assert_close(decoded[1][2], torch.tensor([35.0, 46.0, 57.0, 68.0]))


def test_extract_restart_and_finalize_with_fake_codec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {
            "sample_id": f"utt/{index}",
            "audio_codes": [[index + 1, 10], [index + 2, 20], [index + 3, 30]][: 1 + index % 3],
            "moss_codecvc_meta": {
                "target_audio": f"/fake/{index}.wav",
                "target_codec_frames": 1 + index % 3,
            },
        }
        for index in range(6)
    ]
    manifest = tmp_path / "train.jsonl"
    _write_jsonl(manifest, rows)
    config = _config(tmp_path / "config.json")
    output = tmp_path / "zq"
    fake = _FakeCodec(latent_dim=3)
    monkeypatch.setattr(extractor, "_codec_from_args", lambda _args: fake)

    common = [
        "extract",
        "--input",
        f"no_text={manifest}",
        "--output-root",
        str(output),
        "--config",
        str(config),
        "--expected-dim",
        "3",
        "--num-shards",
        "2",
        "--batch-size",
        "3",
        "--log-every",
        "0",
    ]
    assert extractor.main([*common, "--shard-id", "0"]) == 0
    assert extractor.main([*common, "--shard-id", "1"]) == 0

    records_0 = list(extractor._iter_jsonl_records(extractor.shard_prefix(output, 0, 2).with_suffix(".records.jsonl")))
    assert records_0
    corrupt = Path(records_0[0]["output_path"])
    extractor.atomic_save_npy(corrupt, np.zeros((2, 9), dtype=np.float32))
    calls_before_restart = fake.model.quantizer.calls

    # Existing valid arrays are reused, while the malformed one is atomically rebuilt.
    assert extractor.main([*common, "--shard-id", "0"]) == 0
    assert fake.model.quantizer.calls > calls_before_restart
    restored = np.load(corrupt, allow_pickle=False)
    assert restored.shape[0] == 3
    restarted = json.loads(
        extractor.shard_prefix(output, 0, 2).with_suffix(".COMPLETED.json").read_text(encoding="utf-8")
    )
    assert restarted["reused"] >= 1
    assert restarted["written"] >= 1

    assert extractor.main(["finalize", "--output-root", str(output), "--num-shards", "2"]) == 0
    completed = json.loads((output / "COMPLETED.json").read_text(encoding="utf-8"))
    assert completed["total_utterances"] == len(rows)
    assert completed["total_frames"] == sum(len(row["audio_codes"]) for row in rows)
    assert completed["latent_dim"] == 3
    assert completed["dtype"] == "float32"
    assert completed["errors"] == 0
    assert completed["latent_stats"]["value_count"] == completed["total_frames"] * 3
    assert len(list(extractor._iter_jsonl_records(output / "manifest.jsonl"))) == len(rows)


def test_finalize_rejects_duplicate_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "duplicate.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "sample_id": "duplicate",
                "audio_codes": [[1, 2]],
                "moss_codecvc_meta": {"target_codec_frames": 1},
            },
            {
                "sample_id": "duplicate",
                "audio_codes": [[3, 4], [5, 6]],
                "moss_codecvc_meta": {"target_codec_frames": 2},
            },
        ],
    )
    config = _config(tmp_path / "config.json")
    output = tmp_path / "duplicate-zq"
    monkeypatch.setattr(extractor, "_codec_from_args", lambda _args: _FakeCodec(latent_dim=3))
    assert (
        extractor.main(
            [
                "extract",
                "--input",
                f"no_text={manifest}",
                "--output-root",
                str(output),
                "--config",
                str(config),
                "--expected-dim",
                "3",
                "--log-every",
                "0",
            ]
        )
        == 0
    )
    with pytest.raises(ValueError, match="duplicate utterance ID"):
        extractor.main(["finalize", "--output-root", str(output), "--num-shards", "1"])


def test_strict_failure_publishes_error_stats_but_no_completion(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.jsonl"
    _write_jsonl(
        manifest,
        [{"sample_id": "missing-codes", "moss_codecvc_meta": {"target_codec_frames": 1}}],
    )
    config = _config(tmp_path / "config.json")
    output = tmp_path / "bad-zq"

    with pytest.raises(KeyError, match="audio_codes"):
        extractor.main(
            [
                "extract",
                "--input",
                f"no_text={manifest}",
                "--output-root",
                str(output),
                "--config",
                str(config),
                "--expected-dim",
                "3",
                "--log-every",
                "0",
            ]
        )

    prefix = extractor.shard_prefix(output, 0, 1)
    errors = list(extractor._iter_jsonl_records(prefix.with_suffix(".errors.jsonl")))
    stats = json.loads(prefix.with_suffix(".stats.json").read_text(encoding="utf-8"))
    assert len(errors) == 1
    assert stats["status"] == "failed"
    assert stats["errors"] == 1
    assert not prefix.with_suffix(".COMPLETED.json").exists()


def test_byte_range_handles_crlf_empty_no_final_newline_and_many_shards(tmp_path: Path) -> None:
    manifest = tmp_path / "edge.jsonl"
    huge = {"sample_id": "huge", "payload": "z" * 200_000}
    rows = [
        {"sample_id": "first"},
        huge,
        {"sample_id": "last"},
    ]
    manifest.write_bytes(
        b"\r\n"
        + json.dumps(rows[0]).encode("utf-8")
        + b"\r\n\r\n"
        + json.dumps(rows[1]).encode("utf-8")
        + b"\r\n"
        + json.dumps(rows[2]).encode("utf-8")
    )

    seen: list[tuple[int, bytes]] = []
    for shard_id in range(257):
        seen.extend(extractor.iter_jsonl_byte_range(manifest, shard_id=shard_id, num_shards=257))
    decoded = [json.loads(line)["sample_id"] for _, line in sorted(seen)]
    assert decoded == ["first", "huge", "last"]
    assert len({offset for offset, _ in seen}) == 3

    tiny = tmp_path / "tiny.jsonl"
    tiny.write_bytes(b'{"sample_id":"only"}')
    tiny_seen: list[tuple[int, bytes]] = []
    for shard_id in range(64):
        tiny_seen.extend(extractor.iter_jsonl_byte_range(tiny, shard_id=shard_id, num_shards=64))
    assert len(tiny_seen) == 1
    assert json.loads(tiny_seen[0][1])["sample_id"] == "only"


def test_codes_contract_rejects_nq_vocab_and_frame_mismatches() -> None:
    good = {
        "sample_id": "good",
        "audio_codes": [[1, 2], [3, 4]],
        "moss_codecvc_meta": {"target_audio": "/nested.wav", "target_codec_frames": 2},
    }
    codes, target = extractor.acquire_codes(
        good,
        codes_source="manifest",
        codec=None,
        n_vq=2,
        codebook_size=8,
    )
    assert tuple(codes.shape) == (2, 2)
    assert target == "/nested.wav"

    top_level = {
        "sample_id": "top",
        "target_audio": "/top.wav",
        "target_codec_frames": 1,
        "audio_codes": [[1, 2]],
    }
    _, target = extractor.acquire_codes(
        top_level,
        codes_source="manifest",
        codec=None,
        n_vq=2,
        codebook_size=8,
    )
    assert target == "/top.wav"

    wrong_nq = {**good, "audio_codes": [[1]]}
    with pytest.raises(ValueError, match="exactly 2 quantizers"):
        extractor.acquire_codes(
            wrong_nq, codes_source="manifest", codec=None, n_vq=2, codebook_size=8
        )
    bad_token = {**good, "audio_codes": [[1, 8], [3, 4]]}
    with pytest.raises(ValueError, match="outside codebook size"):
        extractor.acquire_codes(
            bad_token, codes_source="manifest", codec=None, n_vq=2, codebook_size=8
        )
    wrong_frames = {
        **good,
        "moss_codecvc_meta": {"target_audio": "/nested.wav", "target_codec_frames": 3},
    }
    with pytest.raises(ValueError, match="target_codec_frames mismatch"):
        extractor.acquire_codes(
            wrong_frames, codes_source="manifest", codec=None, n_vq=2, codebook_size=8
        )


def test_output_contract_blocks_incompatible_reuse_and_shard_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "train.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "sample_id": "one",
                "audio_codes": [[1, 2], [3, 4]],
                "moss_codecvc_meta": {"target_codec_frames": 2},
            }
        ],
    )
    config = _config(tmp_path / "config.json")
    output = tmp_path / "zq"
    fake = _FakeCodec(latent_dim=3)
    monkeypatch.setattr(extractor, "_codec_from_args", lambda _args: fake)
    common = [
        "extract",
        "--input",
        f"no_text={manifest}",
        "--output-root",
        str(output),
        "--config",
        str(config),
        "--expected-dim",
        "3",
        "--log-every",
        "0",
    ]
    assert extractor.main(common) == 0
    contract = json.loads((output / "CONTRACT.json").read_text(encoding="utf-8"))
    assert contract["codes_source"] == "manifest"
    assert contract["n_vq"] == 2
    assert contract["codec_provenance"]["fingerprint"]

    calls_before = fake.model.quantizer.calls
    assert extractor.main(common) == 0
    assert fake.model.quantizer.calls == calls_before
    completion = json.loads(
        extractor.shard_prefix(output, 0, 1).with_suffix(".COMPLETED.json").read_text(encoding="utf-8")
    )
    assert completion["reused"] == 1

    with pytest.raises(ValueError, match="contract mismatch"):
        extractor.main([*common, "--output-dtype", "float16"])

    lock = extractor.acquire_shard_lock(extractor.shard_prefix(output, 0, 1).with_suffix(".lock"))
    try:
        with pytest.raises(RuntimeError, match="already owns this shard"):
            extractor.main(common)
    finally:
        extractor.release_shard_lock(lock)


def test_max_rows_is_per_input_and_finalize_requires_explicit_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests = []
    for split in ("no_text", "text"):
        path = tmp_path / f"{split}.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "sample_id": f"{split}-{index}",
                    "audio_codes": [[index + 1, index + 2]],
                    "moss_codecvc_meta": {"target_codec_frames": 1},
                }
                for index in range(3)
            ],
        )
        manifests.append((split, path))
    config = _config(tmp_path / "config.json")
    output = tmp_path / "partial"
    monkeypatch.setattr(extractor, "_codec_from_args", lambda _args: _FakeCodec(latent_dim=3))
    args = [
        "extract",
        "--input",
        f"no_text={manifests[0][1]}",
        "--input",
        f"text={manifests[1][1]}",
        "--output-root",
        str(output),
        "--config",
        str(config),
        "--expected-dim",
        "3",
        "--max-rows",
        "1",
        "--log-every",
        "0",
    ]
    assert extractor.main(args) == 0
    completion = json.loads(
        extractor.shard_prefix(output, 0, 1).with_suffix(".COMPLETED.json").read_text(encoding="utf-8")
    )
    assert completion["processed_rows"] == 2
    assert [value["processed_rows"] for value in completion["input_progress"]] == [1, 1]
    assert completion["partial"] is True

    with pytest.raises(ValueError, match="partial"):
        extractor.main(["finalize", "--output-root", str(output), "--num-shards", "1"])
    with pytest.raises(ValueError, match="utterance gate"):
        extractor.main(
            [
                "finalize",
                "--output-root",
                str(output),
                "--num-shards",
                "1",
                "--allow-partial",
                "--expected-total-utterances",
                "3",
            ]
        )
    assert (
        extractor.main(
            [
                "finalize",
                "--output-root",
                str(output),
                "--num-shards",
                "1",
                "--allow-partial",
                "--expected-total-utterances",
                "2",
                "--expected-total-frames",
                "2",
            ]
        )
        == 0
    )


def test_no_strict_save_failure_accounts_for_every_processed_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "train.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "sample_id": f"row-{index}",
                "audio_codes": [[index + 1, index + 2]],
                "moss_codecvc_meta": {"target_codec_frames": 1},
            }
            for index in range(3)
        ],
    )
    config = _config(tmp_path / "config.json")
    output = tmp_path / "nonstrict"
    monkeypatch.setattr(extractor, "_codec_from_args", lambda _args: _FakeCodec(latent_dim=3))
    real_save = extractor.atomic_save_npy
    calls = 0

    def fail_first(path: str | Path, array: np.ndarray) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected save failure")
        return real_save(path, array)

    monkeypatch.setattr(extractor, "atomic_save_npy", fail_first)
    assert (
        extractor.main(
            [
                "extract",
                "--input",
                f"no_text={manifest}",
                "--output-root",
                str(output),
                "--config",
                str(config),
                "--expected-dim",
                "3",
                "--batch-size",
                "3",
                "--no-strict",
                "--log-every",
                "0",
            ]
        )
        == 0
    )
    completion = json.loads(
        extractor.shard_prefix(output, 0, 1).with_suffix(".COMPLETED.json").read_text(encoding="utf-8")
    )
    assert completion["processed_rows"] == 3
    assert completion["records"] == 2
    assert completion["errors"] == 1
    assert completion["records"] + completion["errors"] == completion["processed_rows"]


def test_finalize_rejects_same_size_npy_content_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "train.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "sample_id": "hash-me",
                "audio_codes": [[1, 2], [3, 4]],
                "moss_codecvc_meta": {"target_codec_frames": 2},
            }
        ],
    )
    config = _config(tmp_path / "config.json")
    output = tmp_path / "hash-zq"
    monkeypatch.setattr(extractor, "_codec_from_args", lambda _args: _FakeCodec(latent_dim=3))
    assert (
        extractor.main(
            [
                "extract",
                "--input",
                f"no_text={manifest}",
                "--output-root",
                str(output),
                "--config",
                str(config),
                "--expected-dim",
                "3",
                "--log-every",
                "0",
            ]
        )
        == 0
    )
    record = next(
        extractor._iter_jsonl_records(
            extractor.shard_prefix(output, 0, 1).with_suffix(".records.jsonl")
        )
    )
    latent_path = Path(record["output_path"])
    before_size = latent_path.stat().st_size
    value = np.load(latent_path, allow_pickle=False).copy()
    value[0, 0] += 1.0
    extractor.atomic_save_npy(latent_path, value)
    assert latent_path.stat().st_size == before_size
    with pytest.raises(ValueError, match="sha256 mismatch"):
        extractor.main(["finalize", "--output-root", str(output), "--num-shards", "1"])


def test_real_v1_first_rows_match_nested_target_and_code_contract() -> None:
    root = extractor.ROOT / "trainset/ver2_9_prepared_speaker_split_wavlm_sv_seq_20260709"
    manifests = [root / "no_text.train.jsonl", root / "text.train.jsonl"]
    if not all(path.is_file() for path in manifests):
        pytest.skip("real v1 manifests are not present in this checkout")
    for manifest in manifests:
        with manifest.open("rb") as handle:
            row = json.loads(handle.readline())
        codes = extractor.manifest_codes(row, expected_n_vq=32, codebook_size=1024)
        assert int(codes.shape[0]) == extractor.target_codec_frames_from_row(row)
        target_audio = extractor.target_audio_from_row(row, required=True)
        assert target_audio == row["moss_codecvc_meta"]["target_audio"]
