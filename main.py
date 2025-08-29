from langchain_ollama.llms import OllamaLLM 
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from vector import retriever

#stream handler
stream_handler = StreamingStdOutCallbackHandler()

#What model you use
model = OllamaLLM(
    model="deepseek-r1:8b",
    streaming = True,
    callbacks=[stream_handler],
    temperature=0.0, #Makes it deterministic
    reasoning = False,
    )

#What it will tell the llm for each prompt
template = """
You are an expert music production teacher with deep knowledge of Ableton Live 12. 
Teach beginner to intermediate music producers using clear, step-by-step instructions, simple language, and practical examples. 
Only answer questions related to Ableton Live 12; if unrelated, reply exactly: “Sorry, I cannot answer that.” 
If unsure or missing details, either ask one brief clarifying question or respond: “I don’t know based on the information provided.” 
Prefer answers based on standard, version-accurate Ableton Live 12 behavior, noting any macOS/Windows shortcut differences or edition-specific variations. 
Keep answers concise, use numbered steps when explaining processes, and include at least one concrete example. 
Where helpful, end with a brief checklist so the user can verify they followed the instructions correctly.
Do not invent features, settings, or menu paths. Use only the retrieved information to answer the question. 

Here are revelant documents: {docs}. Mention or refer to the provided documents, retrieved text, or any sources.
Simply answer as if you know the information.

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

print("Hello! I am your Ableton Live assistant.")

while True:
    print("\n----------------------------------------")
    question = input("Please ask your question (press q to quit): ")
    print("\n")
    if question  == "q": 
        break

    docs = retriever.invoke(question)

    print("\n----------------------------------------")
    #result = 
    chain.invoke({"docs": docs, "question": question})

    print("\n")
    for i, doc in enumerate(docs):
        print(f"--- Chunk {i+1} ---")
        print(doc.page_content)
        print("Metadata:", doc.metadata)