import requests
import gradio as gr

API_URL = "http://localhost:8080/query"

def ask(q):
    r = requests.post(API_URL, json={"query": q}).json()
    return r["answer"], r["hits"]

iface = gr.Interface(
    fn=ask,
    inputs="text",
    outputs=["text", "json"],
    title="IPO RAG Agent",
    description="Ask questions about IPO dataset"
)
iface.launch()
