"""
Functional test runner. Executes all 15 required messages plus edge cases.
All results are logged to route_log.jsonl automatically.
Run with: python -m tests.run_tests
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifier import classify_intent
from src.router import route_and_respond
from src.logger import log_route

MESSAGES = [
    "how do i sort a list of objects in python?",
    "explain this sql query for me",
    "This paragraph sounds awkward, can you help me fix it?",
    "I'm preparing for a job interview, any tips?",
    "what's the average of these numbers: 12, 45, 23, 67, 34",
    "Help me make this better.",
    "I need to write a function that takes a user id and returns their profile, but also i need help with my resume.",
    "hey",
    "Can you write me a poem about clouds?",
    "Rewrite this sentence to be more professional.",
    "I'm not sure what to do with my career.",
    "what is a pivot table",
    "fxi thsi bug pls: for i in range(10) print(i)",
    "How do I structure a cover letter?",
    "My boss says my writing is too verbose.",
    "@code Fix this bug: def add(a,b) return a+b",
    "@data What chart type shows monthly revenue trends?",
    "what is 2 + 2",
    "asdfjkl;",
    "I have 10000 rows of customer data. Values seem bimodal. What should I do?",
]

def run():
    print("LLM Prompt Router — Test Suite")
    print("=" * 60)
    passed = failed = 0
    intent_counts: dict[str, int] = {}
    for i, msg in enumerate(MESSAGES, 1):
        print(f"[{i:02d}/{len(MESSAGES)}] {msg[:75]!r}")
        try:
            intent = classify_intent(msg)
            intent_counts[intent["intent"]] = intent_counts.get(intent["intent"], 0) + 1
            response = route_and_respond(msg, intent)
            log_route(msg, intent["intent"], intent["confidence"], response)
            print(f"       Intent: {intent['intent']}  Confidence: {intent['confidence']:.4f}")
            print(f"       Response: {response[:90].replace(chr(10),' ')}...")
            print(f"       PASS")
            passed += 1
        except Exception as exc:
            print(f"       FAIL — {exc}")
            failed += 1
        print("-" * 60)
    print(f"\n{passed} passed, {failed} failed out of {len(MESSAGES)}.")
    if intent_counts:
        print(f"Intent distribution: {intent_counts}")
    if os.path.exists("route_log.jsonl"):
        with open("route_log.jsonl", "r", encoding="utf-8") as log_file:
            lines = sum(1 for l in log_file if l.strip())
        print(f"route_log.jsonl: {lines} entries.")
    if failed:
        sys.exit(1)

if __name__ == "__main__":
    run()
