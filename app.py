import gradio as gr
from transformers import pipeline
from PIL import Image
import pytesseract
from docx import Document
import fitz  # PyMuPDF
from langdetect import detect

# Load models
simplifier_en = pipeline("text2text-generation", model="t5-small")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

# Extract text from various file formats
def extract_text(file_obj):
    filepath = file_obj.name.lower()

    if filepath.endswith(".pdf"):
        with open(file_obj.name, "rb") as f:
            doc = fitz.open(stream=f.read(), filetype="pdf")
            return "\n".join([page.get_text() for page in doc])

    elif filepath.endswith(".docx"):
        doc = Document(file_obj.name)
        return "\n".join([p.text for p in doc.paragraphs])

    elif filepath.endswith((".jpg", ".png", ".jpeg")):
        img = Image.open(file_obj.name)
        return pytesseract.image_to_string(img)

    elif filepath.endswith(".txt"):
        with open(file_obj.name, "r", encoding="utf-8") as f:
            return f.read()

    return ""

# Simplify with language detection
def smart_simplify(text):
    lang = detect(text)

    if lang in ["bn", "hi"]:
        translated = translator(text)[0]['translation_text']
    else:
        translated = text

    prompt = "simplify: " + translated
    result = simplifier_en(prompt, max_length=256, do_sample=False)
    return result[0]['generated_text']

# Stylize output with highlights and emojis
def stylize_text(text):
    rules = {
        "important": "🔴 <b>IMPORTANT</b>",
        "warning": "⚠️ <b>WARNING</b>",
        "do not": "⛔️ <b>DO NOT</b>",
        "step": "🧩 <b>STEP</b>",
        "note": "📝 <b>NOTE</b>",
        "tip": "💡 <b>TIP</b>",
        "make sure": "✅ <b>MAKE SURE</b>",
    }
    for keyword, replacement in rules.items():
        text = text.replace(keyword, f"<span class='highlight'>{replacement}</span>")
    return text

# Full pipeline
def process(file):
    if file is None:
        return "<p>Please upload a valid file.</p>"
    try:
        raw_text = extract_text(file)
        if not raw_text.strip():
            return "<p>No readable text found.</p>"
        chunks = [raw_text[i:i+400] for i in range(0, len(raw_text), 400)]
        simplified = ""
        for chunk in chunks:
            simplified += smart_simplify(chunk) + "\n"
        styled = stylize_text(simplified).replace('\n', '<br><br>')
        return f"""
        <div class='container'>
            <h2>🧠 AI Simplified Output</h2>
            <div class='output-box'>{styled}</div>
        </div>
        <style>
            .container {{
                font-family: 'Segoe UI', sans-serif;
                color: #000;
                padding: 20px;
            }}
            .output-box {{
                background-color: #fff2cc;
                border-left: 6px solid #fca311;
                padding: 15px;
                border-radius: 12px;
                font-size: 17px;
            }}
            .highlight {{
                color: #d62828;
                font-weight: bold;
            }}
        </style>
        """
    except Exception as e:
        return f"<p>Error: {str(e)}</p>"

# Gradio UI
app = gr.Interface(
    fn=process,
    inputs=gr.File(label="📤 Upload your file (PDF, DOCX, TXT, Image)"),
    outputs=gr.HTML(label="✨ Simplified Result"),
    title="📚 AI Text Simplifier with Language Detection",
    description="Upload a document in English, Hindi, or Bengali. The app will auto-detect the language, simplify the text, and highlight important info with icons."
)

app.launch()