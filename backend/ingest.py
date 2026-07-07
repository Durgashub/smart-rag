"""
ingest.py — thin CLI wrapper.

Called by server.py upload/delete handlers via:
    subprocess.run([sys.executable, "ingest.py", "--session", session_id])

All real logic lives in ingestion/pipeline.py.
This file stays as a thin shim so the subprocess call from server.py
(and Railway's filesystem layout) doesn't need to change.
"""

import argparse
from ingestion.pipeline import ingest_session

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, help="Session UUID")
    args = parser.parse_args()
    ingest_session(args.session)
