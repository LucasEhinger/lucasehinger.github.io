#!/usr/bin/env python3
"""One-time Strava OAuth setup.

Run this once to authorize the site's Strava app and store a refresh token.
After that, strava_sync.py runs unattended.

Setup:
  1. Go to https://www.strava.com/settings/api and create an app.
     - "Authorization Callback Domain" must be exactly: localhost
  2. Put the Client ID / Client Secret in local/strava_client.json:
       {"client_id": "12345", "client_secret": "abc..."}
     (or set STRAVA_CLIENT_ID / STRAVA_CLIENT_SECRET in the environment)
  3. python3 scripts/strava_auth.py

Tokens land in local/strava_tokens.json, which is gitignored.
"""

import http.server
import json
import os
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
LOCAL = REPO / "local"
CLIENT_FILE = LOCAL / "strava_client.json"
TOKEN_FILE = LOCAL / "strava_tokens.json"

PORT = 8721
REDIRECT_URI = f"http://localhost:{PORT}/exchange_token"
# activity:read_all also covers activities you've marked private.
SCOPE = "activity:read_all"


def load_client():
    client_id = os.environ.get("STRAVA_CLIENT_ID")
    client_secret = os.environ.get("STRAVA_CLIENT_SECRET")
    if client_id and client_secret:
        return client_id, client_secret

    if not CLIENT_FILE.exists():
        sys.exit(
            f"No credentials found.\n"
            f"Create {CLIENT_FILE} containing:\n"
            f'  {{"client_id": "12345", "client_secret": "abc..."}}\n'
            f"Get those from https://www.strava.com/settings/api"
        )
    data = json.loads(CLIENT_FILE.read_text())
    return str(data["client_id"]), data["client_secret"]


class CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Catches Strava's redirect so you never have to copy/paste a code."""

    code = None
    error = None

    def do_GET(self):
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)

        if "code" in params:
            CallbackHandler.code = params["code"][0]
            granted = params.get("scope", [""])[0]
            if "activity:read_all" not in granted:
                CallbackHandler.error = (
                    f"Missing activity:read_all scope (got: {granted}). "
                    "Re-run and check the box for private activities."
                )
            body = b"<h2>Authorized.</h2><p>You can close this tab.</p>"
        else:
            CallbackHandler.error = params.get("error", ["unknown error"])[0]
            body = b"<h2>Authorization failed.</h2><p>Check the terminal.</p>"

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # keep the terminal clean


def main():
    client_id, client_secret = load_client()

    authorize_url = "https://www.strava.com/oauth/authorize?" + urllib.parse.urlencode(
        {
            "client_id": client_id,
            "redirect_uri": REDIRECT_URI,
            "response_type": "code",
            "approval_prompt": "force",
            "scope": SCOPE,
        }
    )

    server = http.server.HTTPServer(("localhost", PORT), CallbackHandler)
    server.timeout = 300

    print("Opening Strava authorization in your browser.")
    print("If it doesn't open, paste this URL manually:\n")
    print(f"  {authorize_url}\n")
    threading.Timer(1.0, webbrowser.open, args=[authorize_url]).start()

    print("Waiting for the redirect (5 min timeout)...")
    timed_out = []
    server.handle_timeout = lambda: timed_out.append(True)

    # Serve until the callback lands; ignore stray requests like /favicon.ico.
    while CallbackHandler.code is None and CallbackHandler.error is None:
        server.handle_request()
        if timed_out:
            server.server_close()
            sys.exit("Timed out waiting for authorization. Re-run to try again.")
    server.server_close()

    if CallbackHandler.error:
        sys.exit(f"Authorization problem: {CallbackHandler.error}")
    if not CallbackHandler.code:
        sys.exit("No authorization code received. Try again.")

    resp = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": CallbackHandler.code,
            "grant_type": "authorization_code",
        },
        timeout=30,
    )
    resp.raise_for_status()
    tokens = resp.json()

    LOCAL.mkdir(exist_ok=True)
    TOKEN_FILE.write_text(
        json.dumps(
            {
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": tokens["refresh_token"],
                "access_token": tokens["access_token"],
                "expires_at": tokens["expires_at"],
            },
            indent=2,
        )
        + "\n"
    )
    TOKEN_FILE.chmod(0o600)

    athlete = tokens.get("athlete", {})
    name = " ".join(filter(None, [athlete.get("firstname"), athlete.get("lastname")]))
    print(f"\nAuthorized{' as ' + name if name else ''}.")
    print(f"Tokens written to {TOKEN_FILE} (gitignored).")
    print("Next: python3 scripts/strava_sync.py")


if __name__ == "__main__":
    main()
