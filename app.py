from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import joblib, statistics, json, threading, time
from datetime import datetime
import random, openai, os

app = Flask(__name__)
CORS(app)

openai.api_key = os.environ.get("OPENAI_API_KEY")

messages = []

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

def predict_emotion(text):
    if vectorizer and encoder:
        vec = vectorizer.transform([text])
        pred = clf.predict(vec)[0]
        emotion = encoder.inverse_transform([pred])[0]
    else:
        emotion = "neutral"
    sentiment = "neutral"
    if emotion in ["joy","surprise","neutral"]:
        sentiment = "positive"
    elif emotion in ["anger","fear","sadness"]:
        sentiment = "negative"
    return {"emotion": emotion, "sentiment": sentiment}

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
            messages=[{"role":"system","content":"You are an empathetic assistant."},
                      {"role":"user","content":prompt}],
            max_tokens=80,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except:
        return "Thanks for sharing! We're here for you."

def bot_fallback(msg_id):
    time.sleep(60)
    msg = messages[msg_id]
    if not msg.get("reply"):
        bot_text = gpt4_bot_reply(msg["text"], msg["ai"]["emotion"])
        msg["reply"] = bot_text
        msg["bot_replied"] = True

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
        "bot_replied": False,
        "pushed_to_user": False,
        "pushed_to_admin": False
    }
    messages.append(msg)
    threading.Thread(target=bot_fallback, args=(msg["id"],), daemon=True).start()
    return jsonify({"status":"ok"})

@app.route("/admin_reply", methods=["POST"])
def admin_reply():
    data = request.json
    msg_id = data.get("msg_id")
    text = data.get("text")
    if msg_id is None or not (0 <= msg_id < len(messages)):
        return jsonify({"status":"error","message":"invalid msg_id"}),400
    messages[msg_id]["reply"] = text
    messages[msg_id]["bot_replied"] = False
    return jsonify({"status":"ok"})

# SSE user stream
@app.route("/user_stream/<int:user_id>")
def user_stream(user_id):
    def event_stream():
        while True:
            for m in messages:
                if m["user_id"]==user_id and (m.get("reply") or m.get("bot_replied")) and not m.get("pushed_to_user"):
                    m["pushed_to_user"] = True
                    yield f"data: {json.dumps(m)}\n\n"
            time.sleep(1)
    return Response(event_stream(), mimetype="text/event-stream")

# SSE admin stream (send **all existing + new messages**)
@app.route("/admin_stream")
def admin_stream():
    def event_stream():
        # Send existing messages first
        for m in messages:
            if not m.get("pushed_to_admin"):
                payload = {"msg": m, "metrics": calculate_metrics()}
                m["pushed_to_admin"] = True
                yield f"data: {json.dumps(payload)}\n\n"

        while True:
            for m in messages:
                if not m.get("pushed_to_admin"):
                    payload = {"msg": m, "metrics": calculate_metrics()}
                    m["pushed_to_admin"] = True
                    yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(1)
    return Response(event_stream(), mimetype="text/event-stream")

if __name__=="__main__":
    app.run(debug=True)
