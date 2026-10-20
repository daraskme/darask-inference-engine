"""Match exploratory sample hashes against a completed, verified source file."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("verification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    proof = json.loads(args.verification.read_text())
    if proof.get("verified") is not True or proof.get("sha256") != proof.get("expected_sha256"):
        parser.error("Expected a successful complete-file verification record")
    model = Path(proof["file"])
    before = model.stat()
    if (before.st_size, before.st_mtime_ns) != (proof["size"], proof["mtime_ns"]):
        parser.error("Model changed since complete-file verification")
    experiment = json.loads(args.experiment.read_text())
    seen, rows = set(), []
    with model.open("rb") as file:
        for record in experiment["records"]:
            kind = record["kind"]
            itemsize = {0: 4, 1: 2, 30: 2}[kind]
            key = (record["byte_offset"], record["source_sha256"], kind)
            if key in seen:
                continue
            seen.add(key)
            offset = record["byte_offset"]
            if not 0 <= offset <= before.st_size - 128*itemsize:
                raise ValueError("Sample offset outside source")
            file.seek(offset)
            raw = file.read(128*itemsize)
            if hashlib.sha256(raw).hexdigest() != record["source_sha256"]:
                raise ValueError(f"Sample hash mismatch: {record['tensor']} at {offset}")
            if kind == 30:
                values = [struct.unpack("<f", struct.pack("<I", x << 16))[0]
                          for (x,) in struct.iter_unpack("<H", raw)]
            else:
                values = [x for (x,) in struct.iter_unpack("<f" if kind == 0 else "<e", raw)]
            rows.append({"tensor": record["tensor"], "block_index": record["block_index"],
                         "byte_offset": offset, "source_sha256": record["source_sha256"],
                         "nonzero_values": sum(x != 0 for x in values),
                         "max_abs": max(abs(x) for x in values)})
    after = model.stat()
    if not rows:
        raise ValueError("Experiment contains no sample records")
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Model changed during sample audit")
    report = {"experiment": str(args.experiment),
              "experiment_sha256": hashlib.sha256(args.experiment.read_bytes()).hexdigest(),
              "verification": str(args.verification), "model_sha256": proof["sha256"],
              "unique_samples": len(rows), "all_source_hashes_match": True,
              "zero_blocks": sum(r["nonzero_values"] == 0 for r in rows), "records": rows}
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        file.write(serialized)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
