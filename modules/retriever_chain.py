# modules/retriever_chain.py (with strong exclusion detection and coverage rules)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

async def get_answers_as_json(questions, vector_store):
    """
    Final RAG chain with prompt updates, filtering, explanation enforcement, and exclusion detection
    """
    import os

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    prompt_template = """
You are an expert assistant for analyzing legal and insurance documents.
Your job is to answer each user question strictly based on the provided context.

⚠️ RULES:
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
        docs_and_scores = vector_store.similarity_search_with_score(question, k=5)
        filtered_docs = [doc for doc, score in docs_and_scores if score > 0.75]

        if not filtered_docs:
            answers.append("The provided document does not contain information about this topic.")
            continue

        input_data = {
            "context": "\n\n".join(doc.page_content for doc in filtered_docs),
            "questions": question
        }

        result_chain = prompt | llm | parser
        try:
            response = result_chain.invoke(input_data)
            answer = response["answers"][0] if response["answers"] else ""
            answers.append(answer)
        except Exception as e:
            answers.append(f"Error answering question: {str(e)}")

    return {"answers": answers}
