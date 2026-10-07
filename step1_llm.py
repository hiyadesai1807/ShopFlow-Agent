"""
STEP 1 - Call an LLM from Python.

Concept: an LLM is a text model. You send it text, it sends text back.
Notice what goes wrong here: the LLM knows nothing about our store,
so it either refuses or makes something up. Steps 2 and 3 fix that.

Run:  python step1_llm.py
"""

from config import get_llm

llm = get_llm()

# A general question: the LLM can answer this from what it learned in training.
answer = llm.invoke("In one sentence, what is noise cancellation in headphones?")
print("General question:")
print(answer.text)

# A question about OUR store: the LLM has never seen our catalogue or policies.
answer = llm.invoke("What is the return policy of VoltKart for laptops?")
print("\nStore question (without our documents):")
print(answer.text)
