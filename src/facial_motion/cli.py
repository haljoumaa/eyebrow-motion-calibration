"""Commands for Face Landmarker acquisition, frame review, extraction and evaluation."""

import argparse
import csv
import json

from .acquire import ROOT, acquire_recordings, download, inventory, read_manifest


def main():
    parser = argparse.ArgumentParser(description="Face Landmarker eyebrow measurement and pitch correction")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inventory", help="Refresh public OSF metadata; preserve review decisions")
    commands.add_parser("status", help="Show local acquisition and review counts")
    commands.add_parser("model", help="Acquire the official, versioned model asset")
    commands.add_parser("audit-raw", help="Make raw condition/reference/state review sheets for acquired pitch clips")
    commands.add_parser("audit-overlap", help="Screen neutral clips for re-encoded temporal overlap; raw review remains necessary")
    commands.add_parser("audit-landmarks", help="Render the recorded reference/mid/max-tilt QA frames")
    rederive = commands.add_parser("rederive", help="Apply verified references to cached landmarks")
    rederive.add_argument("recording_ids", nargs="+")
    acquire = commands.add_parser("acquire", help="Download named recordings; verify source hashes")
    acquire.add_argument("recording_ids", nargs="+")
    review = commands.add_parser("review", help="Generate raw or landmark contact sheets locally")
    review.add_argument("recording_id")
    review.add_argument("--landmarks", action="store_true")
    extract = commands.add_parser("extract", help="Native-rate extraction; candidate or reviewed reference")
    extract.add_argument("recording_ids", nargs="+")
    extract.add_argument("--overlay", action="store_true")
    extract.add_argument("--reference", type=float, nargs=2, metavar=("START_S", "END_S"))
    plots = commands.add_parser("pilot-figures", help="Plot cached raw pilot measurements")
    plots.add_argument("recording_ids", nargs="+")
    evaluation = commands.add_parser("evaluate", help="Evaluate pitch correction with nested validation across camera distances")
    evaluation.add_argument("--output", help="Fresh output directory; existing results are protected")
    experiment = commands.add_parser("experiment-figures", help="Plot the saved pilot comparison; refit selected models for training curves")
    experiment.add_argument("--recording-ids", nargs=3, required=True, metavar="LOCAL_ID",
                            help="Original locally selected trajectory examples")
    demo = commands.add_parser("demo", help="Create a local preview from user-supplied footage; raw view by default")
    demo.add_argument("--video", required=True)
    demo.add_argument("--output", required=True)
    demo.add_argument("--reference", type=float, nargs=2, required=True, metavar=("START_S", "END_S"))
    demo.add_argument("--method", choices=["raw", "gam", "mlp"], default="raw")
    demo.add_argument("--calibration-fold", choices=["close", "middle", "far"], default="close")
    args = parser.parse_args()
    if args.command == "inventory":
        inventory()
    elif args.command == "status":
        rows = read_manifest()
        print(json.dumps({"source_entries": len(rows), "downloaded": sum(r["downloaded"] == "true" for r in rows),
                         "condition_verified": sum(r["condition_verified"] == "true" for r in rows),
                         "included": sum(r["include"] == "true" for r in rows)}, indent=2))
    elif args.command == "acquire":
        acquire_recordings(args.recording_ids)
    elif args.command == "model":
        from .extract import load_config
        config = load_config()
        metadata = json.loads((ROOT / "models/model_metadata.json").read_text())
        checksum = download(config["model"]["url"], ROOT / config["model"]["path"], metadata["sha256"])
        print("Model SHA-256:", checksum)
    elif args.command == "audit-raw":
        from .audit import review_all
        review_all()
    elif args.command == "audit-overlap":
        from .audit import screen_neutral_overlap
        screen_neutral_overlap()
    elif args.command == "audit-landmarks":
        from .audit import review_geometry
        review_geometry()
    elif args.command == "rederive":
        from .extract import rederive_recording
        for rid in args.recording_ids:
            table, metadata = rederive_recording(rid)
            print(rid, metadata["reference_valid_frames"], "reference frames")
    elif args.command == "extract":
        from .extract import extract_recording
        with (ROOT / "data/references.csv").open(newline="") as f:
            references = {r["recording_id"]: r for r in csv.DictReader(f)}
        for rid in args.recording_ids:
            reference = args.reference
            if reference is None and rid in references and references[rid]["verified"].lower() == "true":
                reference = (float(references[rid]["start_s"]), float(references[rid]["end_s"]))
            extract_recording(rid, reference=reference, overlay=args.overlay)
    elif args.command == "review":
        from .review import contact_sheet
        if args.landmarks:
            from .extract import load_config
            contact_sheet(args.recording_id, landmarks_path=ROOT / f"data/cache/{args.recording_id}_landmarks.npz", config=load_config())
        else:
            contact_sheet(args.recording_id)
    elif args.command == "pilot-figures":
        from .figures import pilot_figure
        pilot_figure(args.recording_ids)
    elif args.command == "evaluate":
        from .evaluate import run_evaluation
        run_evaluation(output=args.output)
    elif args.command == "experiment-figures":
        from .figures import experiment_figures
        experiment_figures(examples=args.recording_ids)
    elif args.command == "demo":
        from .demo import video_preview
        video_preview(args.video, args.output, args.reference, args.method, args.calibration_fold)
