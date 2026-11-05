prompt_gen = """
{documents}
These are regulatory documents on occupational safety.

Using these documents, answer the question that will be provided below.

In your answer, use quotations from the documents and cite the relevant sections/clauses.

Answer in the following format (example for the question: “Capital of France?”):

According to <source document1>Document 1</source>, the capital of France is Paris.

Documents:

Document 1

Question: {query}
"""
