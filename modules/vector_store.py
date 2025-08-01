# modules/retriever_chain.py (with exclusion-prioritized filtering)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
import re

async def get_answers_as_json(questions, vector_store):
    import os

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    prompt_template = """
You are an expert assistant for analyzing legal and insurance documents.
Your job is to answer each user question strictly based on the provided context.

RULES:
- Do NOT answer with just "Yes" or "No". Always explain WHY.
- If the document contains exclusions (e.g. "not covered", "excluded", "shall not be liable"), treat them as definitive denial of claim.
- Clearly state exclusions and quote phrases when possible.
- If something is covered with conditions (e.g. "only if hospitalised", "only due to accident/burns"), mention those exactly.
- For missing info, respond: "The provided document does not contain information about this topic."

Return answer in JSON like this:
{{
  "answers": ["answer 1", "answer 2", ...]
}}

Context:
---
{context}
---

Questions:
---
{questions}
---
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    parser = JsonOutputParser()

    answers = []
    for question in questions:
        all_docs_scores = vector_store.similarity_search_with_score(question, k=10)
        scored_docs = [(doc, score) for doc, score in all_docs_scores if score > 0.65]

        keywords = ["cover", "claim", "excluded", "allow", "treatment", "reimbursement"]
        if any(word in question.lower() for word in keywords):
            exclusion_docs = [doc for doc, _ in scored_docs if doc.metadata.get("tag") == "exclusion"]
            coverage_docs = [doc for doc, _ in scored_docs if doc.metadata.get("tag") == "coverage"]
            neutral_docs = [doc for doc, _ in scored_docs if doc.metadata.get("tag") == "neutral"]
            top_docs = exclusion_docs + coverage_docs + neutral_docs
        else:
            top_docs = [doc for doc, _ in scored_docs]

        if not top_docs:
            answers.append("The provided document does not contain information about this topic.")
            continue

        input_data = {
            "context": "\n\n".join(doc.page_content for doc in top_docs[:4]),
            "questions": question
        }

        rag_chain = prompt | llm | parser

        try:
            response = rag_chain.invoke(input_data)
            answer = response["answers"][0] if response["answers"] else ""
            answers.append(answer)
        except Exception as e:
            answers.append(f"Error answering question: {str(e)}")

    return {"answers": answers}
