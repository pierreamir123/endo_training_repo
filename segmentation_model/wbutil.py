"""Tiny .env loader + W&B init shared by train.py / eval.py.

.env lives at the repo root and holds at least:
    WANDB_API_KEY=...
optionally: WANDB_PROJECT=endo-prad  WANDB_ENTITY=your-team
"""
import os

_ENV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))


def load_dotenv(path=_ENV):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def wandb_init(name, job_type, config, mode):
    """mode: 'online' | 'offline' | 'disabled'. Returns a wandb Run (no-op if disabled)."""
    load_dotenv()
    import wandb
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        print("wbutil: no WANDB_API_KEY in .env / env -> falling back to offline mode")
        mode = "offline"
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "prad-segmentation-model"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=name, job_type=job_type, config=config, mode=mode,
    )
