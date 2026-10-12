"""Publish explicit custom-run domain bindings without changing release packs."""

import argparse
import hashlib
import json
from pathlib import Path

from pydantic import RootModel, TypeAdapter

from tau2.data_model.language import DomainBinding, PackExtension, TaskMetadata
from tau2.multilingual.factory.localization_artifact import RegionalMapArtifact
from tau2.multilingual.factory.localization_publish import build_regional_bundle
from tau2.multilingual.localization import LocalizationSource
from tau2.multilingual.runtime import language_root, load_pack
from tau2.multilingual.scenario_dispatch import bundle_model
from tau2.utils.artifacts import ReviewApproval, content_digest, write_artifact


def compile_approved_regional(
    *,
    map_path: Path,
    source_path: Path,
    approval_path: Path,
    metadata_path: Path,
    output: Path,
    data_root: Path,
) -> None:
    """Compile an exact approved map for an opt-in extension, with no model calls."""
    artifact = RegionalMapArtifact.model_validate_json(map_path.read_bytes())
    source = LocalizationSource.model_validate_json(source_path.read_bytes())
    approval = ReviewApproval.model_validate_json(approval_path.read_bytes())
    approval.require(artifact)
    metadata = TypeAdapter(dict[str, TaskMetadata]).validate_json(
        metadata_path.read_bytes()
    )
    bundle = build_regional_bundle(artifact, source, metadata, data_root)
    write_artifact(bundle, output)
    print(
        json.dumps(
            {
                "stage": "compile-approved-regional",
                "output": str(output),
                "sha256": content_digest(bundle),
            }
        )
    )


def publish_pack_extension(
    *,
    name: str,
    language: str,
    revision: str,
    bundle_paths: list[Path],
    output: Path,
) -> PackExtension:
    """Publish already approved regional bundles for opt-in custom runs."""
    pack = load_pack(language)
    pack_bytes = (language_root() / language / "pack.yaml").read_bytes()
    bundles = {}
    for path in bundle_paths:
        raw = path.read_bytes()
        domain = json.loads(raw)["domain"]
        if domain in bundles or domain in pack.domains:
            raise ValueError(f"duplicate or released domain: {domain}")
        bundle = bundle_model(domain).model_validate_json(raw)
        if bundle.settings.language != language:
            raise ValueError(f"bundle language differs from {language}: {domain}")
        bundles[domain] = bundle
    if not bundles:
        raise ValueError("pack extension needs at least one regional bundle")
    bindings = {
        domain: DomainBinding(
            revision=bundle.revision,
            task_metadata_file=f"tasks/{domain}.metadata.json",
            regional_bundle_file=f"regional/{domain}.json",
        )
        for domain, bundle in sorted(bundles.items())
    }
    extension = PackExtension(
        name=name,
        language=language,
        base_pack_sha256=hashlib.sha256(pack_bytes).hexdigest(),
        revision=revision,
        domains=bindings,
    )
    for domain, bundle in sorted(bundles.items()):
        binding = bindings[domain]
        write_artifact(
            RootModel[dict[str, TaskMetadata]](root=bundle.metadata),
            output / binding.task_metadata_file,
        )
        write_artifact(bundle, output / binding.regional_bundle_file)
    # Write the manifest last so a partially published extension is not loadable.
    write_artifact(extension, output / "extension.json")
    return extension


def _run_publish(args: argparse.Namespace) -> None:
    extension = publish_pack_extension(
        name=args.name,
        language=args.language,
        revision=args.revision,
        bundle_paths=args.bundle,
        output=args.output,
    )
    print(
        f"Published {extension.name}/{extension.language}: "
        f"{', '.join(sorted(extension.domains))}"
    )


def add_pack_extension_args(commands: argparse._SubParsersAction) -> None:
    """Register opt-in language-pack extension publication."""
    compile_parser = commands.add_parser(
        "compile-approved-regional",
        help="Compile a digest-approved regional map for a separate-run extension",
    )
    for flag in ("map", "source", "approval", "metadata", "output"):
        compile_parser.add_argument("--" + flag, type=Path, required=True)
    compile_parser.add_argument("--data-root", type=Path)

    def run_compile(args: argparse.Namespace) -> None:
        from tau2.utils import DATA_DIR

        compile_approved_regional(
            map_path=args.map,
            source_path=args.source,
            approval_path=args.approval,
            metadata_path=args.metadata,
            output=args.output,
            data_root=args.data_root or DATA_DIR,
        )

    compile_parser.set_defaults(func=run_compile)
    publish = commands.add_parser(
        "publish-pack-extension",
        help="Bind approved regional bundles for separate runs only",
    )
    publish.add_argument("--name", required=True)
    publish.add_argument("--language", required=True)
    publish.add_argument("--revision", required=True)
    publish.add_argument("--bundle", type=Path, action="append", required=True)
    publish.add_argument("--output", type=Path, required=True)
    publish.set_defaults(func=_run_publish)
