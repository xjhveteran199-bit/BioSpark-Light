#!/usr/bin/env python3
"""
BioSpark-Light — Issue a license key and email it to the recipient.
Developer-only tool. NOT bundled in dist/BioSpark-Light/.

First-time setup (one-off):
  Copy tools/smtp_config.example.json → tools/smtp_config.json
  Fill in your SMTP credentials. The file is gitignored.

Usage:
  python tools/issue_key.py --email user@lab.edu
  python tools/issue_key.py --email user@lab.edu --name "Zhang San" --days 365 --tier paid
  python tools/issue_key.py --email user@lab.edu --dry-run   # print without sending

What it does:
  1. Generates a signed BSL2-... license key (same logic as keygen.py)
  2. Sends a bilingual (Chinese/English) email with the key + instructions
  3. Prints a summary to stdout
"""
import argparse
import json
import smtplib
import sys
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

# Reuse key generation from keygen.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.keygen import generate_key  # noqa: E402

_TOOLS_DIR = Path(__file__).resolve().parent
_CONFIG_FILE = _TOOLS_DIR / "smtp_config.json"
_CONFIG_EXAMPLE = _TOOLS_DIR / "smtp_config.example.json"


# ------------------------------------------------------------------ config

def _load_config() -> dict:
    if not _CONFIG_FILE.exists():
        print(f"[error] SMTP config not found: {_CONFIG_FILE}")
        print(f"        Copy {_CONFIG_EXAMPLE.name} → smtp_config.json and fill in your credentials.")
        sys.exit(1)
    return json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ email

def _build_email(cfg: dict, to_email: str, name: str, key: str,
                 expiry: str, tier: str) -> MIMEMultipart:
    greeting = f"您好 {name}，" if name else "您好，"
    greeting_en = f"Hi {name}," if name else "Hi,"

    subject = "BioSpark-Light 激活码 / License Key"

    html = f"""\
<html><body style="font-family:sans-serif;color:#222;max-width:600px;margin:auto">
<h2 style="color:#2563eb">BioSpark-Light 激活码</h2>
<p>{greeting}感谢您的支持！以下是您的激活码：</p>
<p style="background:#f1f5f9;padding:12px 16px;border-radius:8px;
          font-family:monospace;font-size:15px;word-break:break-all">{key}</p>
<p>有效期至：<strong>{expiry}</strong>（{tier} 授权）</p>
<h3>如何激活</h3>
<ol>
  <li>启动 BioSpark-Light（运行 <code>python run.py</code>）</li>
  <li>点击顶部横幅中的 <strong>输入激活码</strong></li>
  <li>粘贴上方激活码，点击 <strong>激活</strong></li>
  <li>看到绿色"已授权"提示即完成</li>
</ol>
<hr style="border:none;border-top:1px solid #e2e8f0;margin:24px 0">
<h2 style="color:#2563eb">License Key (English)</h2>
<p>{greeting_en} Thank you for your support! Here is your license key:</p>
<p style="background:#f1f5f9;padding:12px 16px;border-radius:8px;
          font-family:monospace;font-size:15px;word-break:break-all">{key}</p>
<p>Valid until: <strong>{expiry}</strong> ({tier} license)</p>
<h3>How to activate</h3>
<ol>
  <li>Launch BioSpark-Light (<code>python run.py</code>)</li>
  <li>Click <strong>Enter Activation Key</strong> in the top banner</li>
  <li>Paste the key above and click <strong>Activate</strong></li>
  <li>You'll see a green "Licensed" status when done</li>
</ol>
<p style="color:#64748b;font-size:13px">
  Questions? Reply to this email.<br>
  BioSpark-Light · <a href="mailto:{cfg['sender_email']}">{cfg['sender_email']}</a>
</p>
</body></html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = cfg.get("sender_name", "BioSpark-Light") + f" <{cfg['sender_email']}>"
    msg["To"] = to_email
    msg.attach(MIMEText(html, "html", "utf-8"))
    return msg


def _send(cfg: dict, msg: MIMEMultipart) -> None:
    host = cfg["smtp_host"]
    port = int(cfg.get("smtp_port", 465))
    use_ssl = cfg.get("use_ssl", True)

    if use_ssl:
        with smtplib.SMTP_SSL(host, port) as s:
            s.login(cfg["sender_email"], cfg["smtp_password"])
            s.send_message(msg)
    else:
        with smtplib.SMTP(host, port) as s:
            s.ehlo()
            s.starttls()
            s.login(cfg["sender_email"], cfg["smtp_password"])
            s.send_message(msg)


# ------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a license key and email it to the recipient."
    )
    parser.add_argument("--email", required=True, help="Recipient email address")
    parser.add_argument("--name", default="", help="Recipient name (optional)")
    parser.add_argument("--tier", default="paid", choices=["paid", "trial"])
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print key and email body without sending")
    args = parser.parse_args()

    key = generate_key(args.email, args.tier, args.days)
    expiry = time.strftime(
        "%Y-%m-%d",
        time.localtime(int(time.time()) + args.days * 86400),
    )

    print()
    print("  BioSpark-Light — Issue License Key")
    print("  " + "─" * 44)
    print(f"  To    : {args.email}" + (f" ({args.name})" if args.name else ""))
    print(f"  Tier  : {args.tier}  |  Expires: {expiry} ({args.days} days)")
    print(f"  Key   : {key}")
    print()

    if args.dry_run:
        print("  [dry-run] Email NOT sent.")
        print()
        return

    cfg = _load_config()
    msg = _build_email(cfg, args.email, args.name, key, expiry, args.tier)
    _send(cfg, msg)
    print(f"  [ok] Email sent to {args.email}")
    print()


if __name__ == "__main__":
    main()
