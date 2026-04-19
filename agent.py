"""Math agent that solves questions using tools in a ReAct loop."""

import json
import time
import os

from dotenv import load_dotenv
from pydantic_ai import Agent
from calculator import calculate

load_dotenv(override=True)

# Configure your model below. Examples:
# ...
MODEL = "google-gla:gemini-2.5-flash-lite"
print("API key ends with:", os.getenv("GOOGLE_API_KEY", "")[-4:])
print("MODEL:", MODEL)

agent = Agent(
    MODEL,
    system_prompt=(
        "You are a helpful assistant. Solve each question step by step. "
        "Use the calculator tool for arithmetic. "
        "Use the product_lookup tool when a question mentions products from the catalog. "
        "If a question cannot be answered with the information given, say so."
    ),
)


@agent.tool_plain
def calculator_tool(expression: str) -> str:
    """Evaluate a math expression and return the result.

    Examples: "847 * 293", "10000 * (1.07 ** 5)", "23 % 4"
    """
    return calculate(expression)


# TODO: Implement this tool by uncommenting the code below and replacing
# the ... with your implementation. The tool should:
#   1. Read products.json using json.load() (json is already imported above)
#   2. If the product_name is in the catalog, return its price as a string
#   3. If not found, return the list of available product names so the agent
#      can try again with the correct name
#
@agent.tool_plain
def product_lookup(product_name: str) -> str:
    """Look up the price of a product by name.
    Use this when a question asks about product prices from the catalog.
    """
    with open("products.json", "r", encoding="utf-8") as f:
        catalog = json.load(f)

    if product_name in catalog:
        return str(catalog[product_name])

    available = ", ".join(catalog.keys())
    return f"Product not found. Available products: {available}"


def load_questions(path: str = "math_questions.md") -> list[str]:
    """Load numbered questions from the markdown file."""
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and line[0].isdigit() and ". " in line[:4]:
                questions.append(line.split(". ", 1)[1])
    return questions


def main():
    questions = load_questions()
    for i, question in enumerate(questions, 1):
        print(f"## Question {i}")
        print(f"> {question}\n")

        result = agent.run_sync(question)

        print("### Trace")
        for message in result.all_messages():
            for part in message.parts:
                kind = part.part_kind
                if kind in ("user-prompt", "system-prompt"):
                    continue
                elif kind == "text":
                    print(f"- **Reason:** {part.content}")
                elif kind == "tool-call":
                    print(f"- **Act:** {part.tool_name}({part.args})")
                elif kind == "tool-return":
                    print(f"- **Result:** {part.content}")

        print(f"\n**Answer:** {result.output}\n")
        print("---\n")

        if i < len(questions):
            print("Waiting 13 seconds to avoid rate limits...\n")
            time.sleep(13)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
