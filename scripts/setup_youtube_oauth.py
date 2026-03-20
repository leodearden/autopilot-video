"""One-time OAuth2 setup for YouTube Data API access.

Run this script interactively to authorize autopilot-video to upload
videos to your YouTube channel. It opens a browser for consent, then
saves the refresh token to the specified output path.

Usage:
    python scripts/setup_youtube_oauth.py \
        --client-secrets /path/to/client_secrets.json \
        --output ~/.config/autopilot/youtube_oauth.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

_YOUTUBE_UPLOAD_SCOPE = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the OAuth setup script."""
    parser = argparse.ArgumentParser(
        description="Set up YouTube OAuth2 credentials for autopilot-video.",
    )
    parser.add_argument(
        "--client-secrets",
        required=True,
        type=Path,
        help="Path to the OAuth2 client secrets JSON file from Google Cloud Console.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("~/.config/autopilot/youtube_oauth.json"),
        help=(
            "Output path for the saved OAuth2 credentials "
            "(default: ~/.config/autopilot/youtube_oauth.json)."
        ),
    )
    return parser


def run_oauth_flow(
    client_secrets_path: Path,
    output_path: Path,
) -> None:
    """Run the installed-app OAuth2 flow and save credentials.

    Args:
        client_secrets_path: Path to client_secrets.json from Google Cloud.
        output_path: Where to write the serialized credentials JSON.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(
        str(client_secrets_path),
        scopes=_YOUTUBE_UPLOAD_SCOPE,
    )
    creds = flow.run_local_server(port=0)

    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(creds.to_json())
    print(f"Credentials saved to {output_path}")  # noqa: T201


def main() -> None:
    """Entry point for the OAuth setup script."""
    parser = build_parser()
    args = parser.parse_args()
    run_oauth_flow(
        client_secrets_path=args.client_secrets,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
