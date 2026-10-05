import os, base64, mimetypes, uuid, requests
from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

AOAI_ENDPOINT=os.getenv("AZURE_OPENAI_ENDPOINT","").rstrip("/")
AOAI_KEY=os.getenv("AZURE_OPENAI_API_KEY","")
TEXT_MODEL=os.getenv("TEXT_MODEL_DEPLOYMENT","")
VISION_MODEL=os.getenv("VISION_MODEL_DEPLOYMENT") or TEXT_MODEL
IMAGE_MODEL=os.getenv("IMAGE_MODEL_DEPLOYMENT","")
SPEECH_ENDPOINT=os.getenv("SPEECH_ENDPOINT","").rstrip("/")
SPEECH_KEY=os.getenv("SPEECH_API_KEY","")
SPEECH_REGION=os.getenv("SPEECH_REGION","eastus")
CONTENT_ENDPOINT=os.getenv("CONTENT_ENDPOINT","").rstrip("/")
CONTENT_KEY=os.getenv("CONTENT_API_KEY","")
CONTENT_API_VERSION=os.getenv("CONTENT_API_VERSION","2025-11-01")

def client():
    if not AOAI_ENDPOINT or not AOAI_KEY:
        raise RuntimeError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env")
    endpoint = AOAI_ENDPOINT.removesuffix("/openai/v1").rstrip("/")
    return OpenAI(api_key=AOAI_KEY, base_url=f"{endpoint}/openai/v1/")

def image_client():
    if not CONTENT_ENDPOINT or not CONTENT_KEY:
        raise RuntimeError("Set CONTENT_ENDPOINT and CONTENT_API_KEY in .env")
    endpoint = CONTENT_ENDPOINT.removesuffix("/openai/v1").removesuffix("/openai").rstrip("/")
    return OpenAI(
        api_key=CONTENT_KEY,
        base_url=f"{endpoint}/openai/v1/",
        default_query={"api-version": "preview"},
    )

def text_response(prompt, system="You are a helpful AI assistant."):
    r=client().responses.create(model=TEXT_MODEL, instructions=system, input=prompt)
    return r.output_text

@app.get("/")
def home(): return render_template("home.html")

@app.get("/chat")
def chat(): return render_template("chat.html")

@app.post("/api/chat")
def api_chat():
    try:
        p=(request.json or {}).get("message","").strip()
        if not p: return jsonify(error="Enter a message."),400
        return jsonify(reply=text_response(p))
    except Exception as e: return jsonify(error=str(e)),500

@app.get("/text-analysis")
def text_analysis(): return render_template("text_analysis.html")

@app.post("/api/text-analysis")
def api_text_analysis():
    try:
        text=(request.json or {}).get("text","").strip()
        if not text: return jsonify(error="Enter text."),400
        prompt=f"""Analyze the following text and return exactly these sections:
1. Sentiment
2. Keywords
3. Entities (organization, person, location, date, product where applicable)
4. Summary

TEXT:
{text}"""
        return jsonify(result=text_response(prompt))
    except Exception as e: return jsonify(error=str(e)),500

@app.get("/vision")
def vision(): return render_template("vision.html")

@app.post("/api/vision")
def api_vision():
    try:
        f=request.files.get("image")
        prompt=request.form.get("prompt","Describe this image in detail.").strip()
        if not f: return jsonify(error="Upload an image."),400
        data=base64.b64encode(f.read()).decode()
        mime=f.mimetype or "image/jpeg"
        r=client().responses.create(
            model=VISION_MODEL,
            input=[{"role":"user","content":[
                {"type":"input_text","text":prompt},
                {"type":"input_image","image_url":f"data:{mime};base64,{data}"}
            ]}]
        )
        return jsonify(result=r.output_text)
    except Exception as e: return jsonify(error=str(e)),500

@app.get("/image-generation")
def image_generation(): return render_template("image_generation.html")

@app.post("/api/image-generation")
def api_image_generation():
    try:
        prompt=(request.json or {}).get("prompt","").strip()
        if not prompt: return jsonify(error="Enter a prompt."),400
        if not IMAGE_MODEL:
            raise RuntimeError("Set IMAGE_MODEL_DEPLOYMENT to your FLUX-1.1-pro deployment name.")
        r=image_client().images.generate(model=IMAGE_MODEL,prompt=prompt,n=1,size="1024x1024")
        item=r.data[0]
        if getattr(item,"b64_json",None):
            raw=base64.b64decode(item.b64_json)
            name=f"{uuid.uuid4().hex}.png"; path=os.path.join("uploads",name)
            open(path,"wb").write(raw)
            return jsonify(url=f"/uploads/{name}")
        return jsonify(url=getattr(item,"url",None))
    except Exception as e: return jsonify(error=str(e)),500

@app.get("/speech")
def speech(): return render_template("speech.html")

@app.post("/api/speech")
def api_speech():
    try:
        f=request.files.get("audio")
        if not f: return jsonify(error="Upload an audio file."),400
        if not SPEECH_KEY: raise RuntimeError("Set SPEECH_API_KEY.")
        # Azure Speech REST endpoint for short audio transcription.
        # The exact language can be changed in the UI.
        language=request.form.get("language","en-US")
        url=f"{SPEECH_ENDPOINT}/speechtotext/transcriptions:transcribe?api-version=2024-11-15"
        headers={"Ocp-Apim-Subscription-Key":SPEECH_KEY}
        files={"audio":(f.filename,f.stream,f.mimetype or "audio/wav")}
        data={"definition":'{"locales":["'+language+'"],"profanityFilterMode":"Masked"}'}
        r=requests.post(url,headers=headers,files=files,data=data,timeout=120)
        if not r.ok:
            # Fall back to the common Speech SDK route through a helpful error.
            return jsonify(error=f"Speech service returned {r.status_code}: {r.text}"),r.status_code
        return jsonify(result=r.json())
    except Exception as e: return jsonify(error=str(e)),500

@app.get("/content-understanding")
def content_understanding(): return render_template("content.html")

@app.post("/api/content-understanding")
def api_content():
    try:
        import time
        f=request.files.get("file")
        analyzer=request.form.get("analyzer","prebuilt-read").strip() or "prebuilt-read"
        if not f: return jsonify(error="Upload a file."),400
        if not CONTENT_ENDPOINT or not CONTENT_KEY:
            raise RuntimeError("Set CONTENT_ENDPOINT and CONTENT_API_KEY in .env")
        
        file_bytes = f.read()
        mime_type = f.mimetype or mimetypes.guess_type(f.filename or "")[0] or "application/pdf"
        
        url=f"{CONTENT_ENDPOINT}/contentunderstanding/analyzers/{analyzer}:analyzeBinary?api-version={CONTENT_API_VERSION}"
        headers={"Ocp-Apim-Subscription-Key":CONTENT_KEY, "Content-Type":mime_type}
        
        r=requests.post(url,headers=headers,data=file_bytes,timeout=120)
        if r.status_code not in (200, 201, 202):
            return jsonify(error=f"Content Understanding returned {r.status_code}: {r.text}"),r.status_code
        
        op_loc = r.headers.get("Operation-Location")
        if not op_loc:
            return jsonify(result=r.json())
        
        poll_headers = {"Ocp-Apim-Subscription-Key": CONTENT_KEY}
        for _ in range(30):
            time.sleep(2)
            pr = requests.get(op_loc, headers=poll_headers, timeout=30)
            if not pr.ok:
                return jsonify(error=f"Polling returned {pr.status_code}: {pr.text}"), pr.status_code
            pj = pr.json()
            status = pj.get("status")
            if status == "Succeeded":
                return jsonify(result=pj.get("result", pj))
            elif status in ("Failed", "canceled"):
                err = pj.get("error", {})
                msg = err.get("message") or "Analysis failed."
                inner = err.get("innererror", {}).get("message")
                if inner: msg += f" ({inner})"
                return jsonify(error=msg), 400
                
        return jsonify(error="Document analysis timed out after 60s."), 504
    except Exception as e: return jsonify(error=str(e)),500

@app.get("/agent")
def agent(): return render_template("agent.html")

@app.post("/api/agent")
def api_agent():
    try:
        p=(request.json or {}).get("message","").strip()
        if not p:return jsonify(error="Enter a message."),400
        # Uses the same deployed model with agent-like instructions if no persisted agent ID is supplied.
        system="""You are a Microsoft Foundry teaching agent. Explain technical topics clearly,
step by step, with a definition, example, and practical use case."""
        return jsonify(reply=text_response(p,system))
    except Exception as e:return jsonify(error=str(e)),500

@app.get("/uploads/<name>")
def uploads(name): return send_file(os.path.join("uploads",name))

@app.get("/status")
def status(): return render_template("status.html")

@app.get("/health")
def health():
    return jsonify(text_model=bool(TEXT_MODEL), vision_model=bool(VISION_MODEL),
                   image_model=bool(IMAGE_MODEL), speech=bool(SPEECH_KEY),
                   content_understanding=bool(CONTENT_KEY))

if __name__=="__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

