"""
License service — BioSpark-Light

Key format:  BSL2-{base64url(json_payload)}.{base64url(hmac_sha256)[:24]}
Payload:     {"email": str, "tier": "paid", "exp": int, "iat": int}

License file (USER_DATA_DIR/license.json):
{
  "runs_used": int,
  "key":       str | null,
  "status":    "trial" | "licensed" | "expired",
  "email":     str | null,
  "expiry":    "YYYY-MM-DD" | null
}
"""

import base64
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone

from backend.config import LICENSE_FILE, TRIAL_RUNS_LIMIT

# 32-byte HMAC signing secret. Keep private — anyone with this can forge keys.
# Rotate with: python -c "import secrets; print(secrets.token_hex(32))"
# When rotating, update tools/keygen.py to match.
_SECRET = bytes.fromhex(
    "404442ba5e8d3fe766e0c716c507d81a"
    "41ebd5970a7fcad553346a1f37046123"
)


def _b64u_enc(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64u_dec(s: str) -> bytes:
    pad = (-len(s)) % 4
    return base64.urlsafe_b64decode(s + "=" * pad)


def _sign(payload_b64: str) -> str:
    h = hmac.new(_SECRET, payload_b64.encode(), hashlib.sha256)
    return _b64u_enc(h.digest())[:24]


class LicenseService:
    def __init__(self) -> None:
        self._path = LICENSE_FILE

    # ------------------------------------------------------------------ I/O

    def load(self) -> dict:
        if self._path.exists():
            try:
                d = json.loads(self._path.read_text(encoding="utf-8"))
                d.setdefault("runs_used", 0)
                d.setdefault("key", None)
                d.setdefault("status", "trial")
                d.setdefault("email", None)
                d.setdefault("expiry", None)
                return d
            except Exception:
                pass
        return {"runs_used": 0, "key": None, "status": "trial",
                "email": None, "expiry": None}

    def save(self, data: dict) -> None:
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # -------------------------------------------------------- Key validation

    def validate_key(self, key: str) -> dict | None:
        """Return payload dict if key has valid signature and is not expired."""
        try:
            if not key.startswith("BSL2-"):
                return None
            rest = key[5:]
            dot = rest.rfind(".")
            if dot == -1:
                return None
            payload_b64, sig = rest[:dot], rest[dot + 1:]
            if not hmac.compare_digest(sig, _sign(payload_b64)):
                return None
            payload = json.loads(_b64u_dec(payload_b64))
            if payload.get("exp", 0) < int(time.time()):
                return None
            return payload
        except Exception:
            return None

    def _is_key_structurally_valid_but_expired(self, key: str) -> bool:
        """True when signature is correct but key is past its expiry."""
        try:
            rest = key[5:]
            dot = rest.rfind(".")
            if dot == -1:
                return False
            payload_b64, sig = rest[:dot], rest[dot + 1:]
            if not hmac.compare_digest(sig, _sign(payload_b64)):
                return False
            payload = json.loads(_b64u_dec(payload_b64))
            return payload.get("exp", 0) < int(time.time())
        except Exception:
            return False

    # ------------------------------------------------------- Public methods

    def activate(self, key: str) -> dict:
        """Validate and persist a license key.

        Returns {success, message, email, expiry}.
        """
        payload = self.validate_key(key)
        if payload is None:
            if key.startswith("BSL2-") and self._is_key_structurally_valid_but_expired(key):
                return {"success": False,
                        "message": "激活码已过期 / License key has expired."}
            return {"success": False,
                    "message": "无效激活码 / Invalid license key."}

        exp_ts = payload.get("exp", 0)
        expiry_str = datetime.fromtimestamp(exp_ts, tz=timezone.utc).strftime("%Y-%m-%d")
        email = payload.get("email", "")

        data = self.load()
        data.update({"key": key, "status": "licensed",
                     "email": email, "expiry": expiry_str})
        self.save(data)

        return {
            "success": True,
            "message": f"激活成功 · {email} · 有效期至 {expiry_str}",
            "email": email,
            "expiry": expiry_str,
        }

    def get_status(self) -> dict:
        """Return current license status dict for the API."""
        data = self.load()
        runs_used = data.get("runs_used", 0)

        if data.get("key"):
            if self.validate_key(data["key"]) is not None:
                data["status"] = "licensed"
            else:
                data["status"] = "expired"
                self.save(data)
        elif runs_used >= TRIAL_RUNS_LIMIT:
            data["status"] = "expired"
        else:
            data["status"] = "trial"

        return {
            "status": data["status"],
            "runs_used": runs_used,
            "runs_limit": TRIAL_RUNS_LIMIT,
            "email": data.get("email"),
            "expiry": data.get("expiry"),
        }

    def check_can_train(self) -> tuple[bool, str]:
        """Return (allowed, message). Call at the start of every training job."""
        s = self.get_status()
        if s["status"] == "licensed":
            return True, ""
        if s["status"] == "trial" and s["runs_used"] < TRIAL_RUNS_LIMIT:
            return True, ""
        remaining = TRIAL_RUNS_LIMIT - s["runs_used"]
        if remaining > 0:
            return True, ""
        return False, (
            f"试用次数已用完（共 {TRIAL_RUNS_LIMIT} 次）。"
            "请激活授权码以继续使用。"
        )

    def record_run(self) -> None:
        """Increment the used-run counter. Call after training starts successfully."""
        data = self.load()
        if data.get("status") == "licensed" and data.get("key"):
            # Licensed users are not counted down.
            return
        data["runs_used"] = data.get("runs_used", 0) + 1
        if data["runs_used"] >= TRIAL_RUNS_LIMIT and not data.get("key"):
            data["status"] = "expired"
        self.save(data)


# Module-level singleton used by routers and main.py
license_service = LicenseService()
