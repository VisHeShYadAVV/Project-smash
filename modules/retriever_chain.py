# modules/retriever_chain.py

import os
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from modules.prompt import prompt_template  # ✅ External prompt file

async def get_answers_as_json(questions: List[str], vector_store) -> Dict[str, List[str]]:
    """
    Uses similarity search + GPT-4o with strict rules to generate JSON answers from a document.
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY", "")
    )

    prompt = ChatPromptTemplate.from_template(prompt_template)
    parser = JsonOutputParser()

    answers = []

    for question in questions:
        try:
            # Perform similarity search with scores
            docs_and_scores = vector_store.similarity_search_with_score(question, k=5)
            filtered_docs = [doc for doc, score in docs_and_scores if score > 0.75]

            if not filtered_docs:
                answers.append("The provided document does not contain information about this topic.")
                continue

            input_data = {
                "context": "\n\n".join(doc.page_content for doc in filtered_docs),
                "questions": question
            }

            chain = prompt | llm | parser
            result = chain.invoke(input_data)

            # Get first answer
            answer = result.get("answers", [""])[0]
            answers.append(answer)

        except Exception as e:
            error_message = f"Error answering question: {str(e)}"
            print(f"❌ {error_message}")
            answers.append(error_message)

    return {"answers": answers}
