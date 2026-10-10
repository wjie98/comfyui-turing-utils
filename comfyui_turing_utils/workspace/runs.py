"""Recoverable publication of material results across SQLite and project JSON.

The database records an intent before the selection is written. A retry (or
reopening the project) finishes that intent without incrementing a selection
twice and without overwriting a newer manual selection.
"""

import json

from . import workflow_files


def finish_run(project, run_id, asset=None, text=None):
    with workflow_files.LOCK:
        with project.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            run = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if run is None:
                raise ValueError("Unknown material execution")
            if run["status"] == "success":
                return bool(run["selected"])
            if run["status"] == "publishing":
                content = json.loads(run["payload"])
            else:
                content = {"text": text} if text is not None else {"asset": asset}
                if text is not None and not isinstance(text, str):
                    raise ValueError("Text must be an inline string")
                if text is None:
                    project.asset(asset)
                db.execute(
                    "UPDATE runs SET status='publishing',asset=?,payload=? WHERE id=?",
                    (content.get("asset"), json.dumps(content), run_id),
                )

        # The intent is durable before entering the second persistence boundary.
        current = project.selections().get(run["target"], {})
        revision = current.get("revision", 0)
        if revision == run["revision"]:
            project.select(run["target"], content, revision)
            selected = True
        else:
            selected = revision == run["revision"] + 1 and all(
                current.get(key) == value for key, value in content.items()
            )

        with project.connect() as db:
            db.execute(
                "UPDATE runs SET status='success',selected=?,payload=NULL WHERE id=?",
                (int(selected), run_id),
            )
        return selected


def recover_publications(project):
    with project.connect() as db:
        pending = db.execute("SELECT id FROM runs WHERE status='publishing'").fetchall()
    for row in pending:
        finish_run(project, row["id"])
