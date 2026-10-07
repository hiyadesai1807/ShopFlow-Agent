"""
STEP 3 - RAG: answer questions using the store's data.

Concept: RAG = Retrieval-Augmented Generation. Three moves:
  1. Retrieve : find the chunks closest to the question (vector database).
  2. Augment  : paste those chunks into the prompt.
  3. Generate : the LLM answers using only what was pasted.

Run step2_build_index.py first.

Run:  python step3_rag.py
"""

from langchain_core.prompts import ChatPromptTemplate

from config import get_llm, get_vector_store

vector_store = get_vector_store()
llm = get_llm()

prompt = ChatPromptTemplate.from_template(
    """You are the shopping assistant for VoltKart, an online electronics store.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I could not find that in the store information."
Keep the answer short. Write prices as Rs.

Context:
{context}

Question: {question}"""
)


def answer_question(question):
    # 1. Retrieve the 4 closest chunks.
    chunks = vector_store.similarity_search(question, k=4)

    # 2. Augment: join the chunks into one block of text and fill the prompt.
    context = "\n\n".join(chunk.page_content for chunk in chunks)
    filled_prompt = prompt.invoke({"context": context, "question": question})

    # 3. Generate the answer.
    answer = llm.invoke(filled_prompt)
    return answer.text, chunks


if __name__ == "__main__":
    print("Ask about products or store policies. Type 'quit' to stop.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", ""):
            break

        answer, chunks = answer_question(question)
        print(f"\nAssistant: {answer}\n")

        # Showing the sources lets you check the answer was not made up.
        print("Sources used:")
        for chunk in chunks:
            label = chunk.metadata.get("name", "Store policy")
            print(f" - [{label}] {chunk.page_content[:70]}...")
        print()
