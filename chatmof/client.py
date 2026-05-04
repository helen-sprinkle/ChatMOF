"""
ChatMOF command-line client.

Usage:
    # Interactive mode
    python -m chatmof.client

    # Single question
    python -m chatmof.client "What is the bandgap of ACOGEF_clean?"

    # Point at a non-default server
    CHATMOF_SERVICE_URL=http://other-host:8001 python -m chatmof.client
"""

import os
import sys
import json
import httpx

DEFAULT_URL = "http://localhost:8001"


def ask(question: str, base_url: str = DEFAULT_URL) -> dict:
    try:
        r = httpx.post(f"{base_url}/ask", json={"question": question}, timeout=120)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        print(f"[error] Cannot reach ChatMOF service at {base_url}")
        print("        Start it with: uvicorn chatmof.service:app --host 0.0.0.0 --port 8001")
        sys.exit(1)


def main():
    base_url = os.environ.get("CHATMOF_SERVICE_URL", DEFAULT_URL)

    # Check service is up
    try:
        health = httpx.get(f"{base_url}/health", timeout=5).json()
        print(f"Connected to ChatMOF  backend={health['moftransformer_backend']}  url={base_url}")
        print()
    except httpx.ConnectError:
        print(f"[error] ChatMOF service not running at {base_url}")
        sys.exit(1)

    # Single question from CLI arg
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = ask(question, base_url)
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")
        return

    # Interactive loop
    print("Type your question and press Enter. Ctrl-C or 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not question or question.lower() == "exit":
            break
        result = ask(question, base_url)
        print(f"ChatMOF: {result['answer']}\n")


if __name__ == "__main__":
    main()
