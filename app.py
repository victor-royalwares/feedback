from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import joblib, statistics, json, threading, time
from datetime import datetime
import openai, os

app = Flask(__name__)
CORS(app)

# Set your OpenAI API key in environment: export OPENAI_API_KEY="sk-..."
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Load emotion model
data = joblib.load("EmoModel.pkl")
if isinstance(data, dict):
    clf = data.get("clf")
    vectorizer = data.get("vectorizer")
    encoder = data.get("encoder")
else:
    clf = data
    vectorizer = None
    encoder = None

# Store all messages
messages = []

# --- Predict emotion ---
def predict_emotion(text):
    if vectorizer and encoder:
        vec = vectorizer.transform([text])
        pred = clf.predict(vec)[0]
        emotion = encoder.inverse_transform([pred])[0]
    else:
        emotion = "neutral"
    sentiment = "neutral"
    if emotion in ["joy", "surprise", "neutral"]:
        sentiment = "positive"
    elif emotion in ["anger","fear","sadness"]:
        sentiment = "negative"
    return {"emotion": emotion, "sentiment": sentiment}

# --- GPT-4 empathetic bot reply ---
def gpt4_bot_reply(user_text, emotion):
    prompt = f"""
You are an empathetic assistant. A user wrote: "{user_text}".
The predicted emotion is "{emotion}".
Reply in a consoling, human-like way that tries to address their concern before the admin comes online.
Keep it short, friendly, and empathetic.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role":"system","content":"You are an empathetic assistant."},
                {"role":"user","content":prompt}
            ],
            max_tokens=80,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except:
        return "Thanks for sharing! We're here for you."

# --- Bot fallback (60s delay) ---
def bot_fallback(msg_id):
    time.sleep(60)
    msg = messages[msg_id]
    if not msg.get("reply"):  # admin hasn't replied
        bot_text = gpt4_bot_reply(msg["text"], msg["ai"]["emotion"])
        msg["reply"] = bot_text
        msg["bot_replied"] = True

# --- Metrics calculation ---
def calculate_metrics():
    csat_scores = [m["csat"] for m in messages if m.get("csat") is not None]
    nps_scores = [m["nps"] for m in messages if m.get("nps") is not None]
    ces_scores = [m["ces"] for m in messages if m.get("ces") is not None]

    csat_avg = round(statistics.mean(csat_scores) if csat_scores else 0,2)
    if nps_scores:
        promoters = sum(1 for x in nps_scores if x>=9)
        detractors = sum(1 for x in nps_scores if x<=6)
        nps_avg = round(((promoters - detractors)/len(nps_scores))*100,2)
    else: nps_avg = 0
    ces_avg = round(statistics.mean(ces_scores) if ces_scores else 0,2)

    sentiment_count = {"positive":0,"neutral":0,"negative":0}
    for m in messages:
        sentiment_count[m["ai"]["sentiment"]] +=1

    return {"csat":csat_avg,"nps":nps_avg,"ces":ces_avg,"sentiment":sentiment_count}

# --- User sends message ---
@app.route("/send", methods=["POST"])
def send_message():
    data = request.json
    ai = predict_emotion(data["text"])
    msg = {
        "id": len(messages),
        "user_id": data.get("user_id",0),
        "text": data["text"],
        "csat": data.get("csat"),
        "nps": data.get("nps"),
        "ces": data.get("ces"),
        "ai": ai,
        "time": datetime.now().isoformat(),
        "reply": "",
        "bot_replied": False
    }
    messages.append(msg)
    threading.Thread(target=bot_fallback, args=(msg["id"],), daemon=True).start()
    return jsonify({"status":"ok"})

# --- Admin replies ---
@app.route("/admin_reply", methods=["POST"])
def admin_reply():
    data = request.json
    msg_id = data.get("msg_id")
    text = data.get("text", "").strip()
    if msg_id is None or not text:
        return jsonify({"status":"error","message":"msg_id or text missing"}),400
    if 0 <= msg_id < len(messages):
        messages[msg_id]["reply"] = text
        messages[msg_id]["bot_replied"] = False  # cancel bot
        return jsonify({"status":"ok"})
    return jsonify({"status":"error","message":"invalid msg_id"}),400

# --- SSE user stream ---
@app.route("/user_stream/<int:user_id>")
def user_stream(user_id):
    def event_stream():
        while True:
            updates = [m for m in messages if m["user_id"]==user_id and m.get("reply") and not m.get("pushed_to_user")]
            for u in updates:
                u["pushed_to_user"] = True
                yield f"data: {json.dumps(u)}\n\n"
            time.sleep(1)
    return Response(event_stream(), mimetype="text/event-stream")

# --- SSE admin stream ---
@app.route("/admin_stream")
def admin_stream():
    def event_stream():
        last_len = 0
        while True:
            new_msgs = [m for m in messages if not m.get("pushed_to_admin")]
            for m in new_msgs:
                m["pushed_to_admin"] = True
            if new_msgs:
                payload = {"messages": messages, "metrics": calculate_metrics()}
                yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(1)
    return Response(event_stream(), mimetype="text/event-stream")

if __name__=="__main__":
    app.run(debug=True)
