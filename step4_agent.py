"""
STEP 4 - Agent: let the LLM choose which tool to use.

Concept: a tool is a normal Python function the LLM is allowed to call.
An agent is an LLM in a loop: it reads the question, picks a tool, reads
the tool's result, and then either picks another tool or answers.

The LLM decides using each tool's name and its description (the text in
triple quotes). Clear descriptions matter more than clever code.

Our three tools:
  search_products  - looks in the product catalogue (can filter by budget)
  search_policies  - looks in the store policies
  get_order_status - looks up an order in orders.csv

Run step2_build_index.py first.

Run:  python step4_agent.py
"""

import pandas as pd
from langchain.agents import create_agent
from langchain.tools import tool

from config import DATA_FOLDER, get_llm, get_vector_store

vector_store = get_vector_store()


@tool
def search_products(query: str, max_price: int = 0) -> str:
    """Search the VoltKart product catalogue. Use this for any question about
    products, features, prices or recommendations.

    query: what the customer wants, for example 'wireless earbuds for the gym'.
    max_price: the customer's budget in rupees. Use 0 if no budget was given.
    """
    # The filter keeps only products, and only those within budget.
    if max_price > 0:
        search_filter = {"$and": [{"source": "product"}, {"price": {"$lte": max_price}}]}
    else:
        search_filter = {"source": "product"}

    results = vector_store.similarity_search(query, k=4, filter=search_filter)
    if not results:
        return "No matching products found."
    return "\n".join(doc.page_content for doc in results)


@tool
def search_policies(query: str) -> str:
    """Search VoltKart's store policies. Use this for questions about returns,
    refunds, shipping, cancellation, warranty, payment or customer support.
    """
    results = vector_store.similarity_search(query, k=3, filter={"source": "policy"})
    if not results:
        return "No matching policy found."
    return "\n\n".join(doc.page_content for doc in results)


@tool
def get_order_status(order_id: str) -> str:
    """Look up the status of a customer's order. Use this when the customer
    gives an order ID such as ORD1002.
    """
    orders = pd.read_csv(DATA_FOLDER / "orders.csv")
    match = orders[orders["order_id"] == order_id.strip().upper()]
    if match.empty:
        return f"No order found with ID {order_id}."

    order = match.iloc[0]
    reply = f"Order {order.order_id}: {order.product_name}, status: {order.status}."
    if pd.notna(order.expected_delivery):
        reply += f" Expected delivery: {order.expected_delivery}."
    return reply


SYSTEM_PROMPT = """You are the shopping assistant for VoltKart, an online electronics store.

Rules:
- Always use a tool to find information. Never answer about products, policies or orders from your own knowledge.
- Only mention products and policies that a tool returned. Never invent a product, price or rule.
- If the tools return nothing useful, say you could not find it and suggest contacting customer support.
- If the customer asks about an order but gives no order ID, ask for the ID.
- Keep answers short and friendly. Write prices as Rs."""

TOOLS = [search_products, search_policies, get_order_status]


def build_agent():
    """Creates the agent. app.py also uses this function."""
    return create_agent(model=get_llm(), tools=TOOLS, system_prompt=SYSTEM_PROMPT)


def tools_used(messages):
    """Returns the names of the tools the agent called, in order."""
    names = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            names.append(call["name"])
    return names


if __name__ == "__main__":
    agent = build_agent()
    history = []   # the full conversation, so follow-up questions make sense

    print("Ask about products, policies or an order (try ORD1002). Type 'quit' to stop.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", ""):
            break

        history.append({"role": "user", "content": question})
        result = agent.invoke({"messages": history})

        # result["messages"] is the whole conversation including tool calls.
        new_messages = result["messages"][len(history):]
        history = result["messages"]

        print(f"\n[tools used: {', '.join(tools_used(new_messages)) or 'none'}]")
        print(f"Assistant: {history[-1].text}\n")
