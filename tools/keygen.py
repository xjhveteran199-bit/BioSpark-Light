#!/usr/bin/env python3
"""
BioSpark-Light — License Key Generator
Developer-only tool. NOT bundled in dist/BioSpark-Light/.

Usage:
  python tools/keygen.py --email user@lab.edu --tier paid --days 365
  python tools/keygen.py --email tester@example.com --days 30

Output:
  BSL2-{payload}.{signature}
"""
import argparse
import base64
import hashlib
import hmac
import json
import time

# Must match _SECRET in backend/services/license.py exactly.
_SECRET = bytes.fromhex(
    "404442ba5e8d3fe766e0c716c507d81a"
    "41ebd5970a7fcad553346a1f37046123"
)


def _b64u_enc(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _sign(payload_b64: str) -> str:
    h = hmac.new(_SECRET, payload_b64.encode(), hashlib.sha256)
    return _b64u_enc(h.digest())[:24]


def generate_key(email: str, tier: str, days: int) -> str:
    now = int(time.time())
    payload = {
        "email": email,
        "tier": tier,
        "iat": now,
        "exp": now + days * 86400,
    }
    payload_b64 = _b64u_enc(json.dumps(payload, separators=(",", ":")).encode())
    return f"BSL2-{payload_b64}.{_sign(payload_b64)}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a BioSpark-Light license key."
    )
    parser.add_argument("--email", required=True, help="Recipient email address")
    parser.add_argument(
        "--tier", default="paid", choices=["paid", "trial"],
        help="License tier (default: paid)"
    )
    parser.add_argument(
        "--days", type=int, default=365,
        help="Validity period in days (default: 365)"
    )
    args = parser.parse_args()

    key = generate_key(args.email, args.tier, args.days)
    exp_date = time.strftime(
        "%Y-%m-%d",
        time.localtime(int(time.time()) + args.days * 86400),
    )

    print()
    print("  BioSpark-Light License Key")
    print("  " + "─" * 42)
    print(f"  Email  : {args.email}")
    print(f"  Tier   : {args.tier}")
    print(f"  Expires: {exp_date} ({args.days} days)")
    print()
    print(f"  Key: {key}")
    print()


if __name__ == "__main__":
    main()
