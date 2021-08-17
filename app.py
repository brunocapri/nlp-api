from keyword_extraction import get_rake
from flask import Flask, request, jsonify
from model import get_pretrained_model_and_tokenizer
import torch
import torch.nn.functional as F

def pair_devices(inputs, device):
  return { i:tensor.to(device) for i, tensor in inputs.items() }

device = torch.device('cuda')
model, tokenizer = get_pretrained_model_and_tokenizer(device)

rake = get_rake()

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
  data = request.get_json()
  inputs = tokenizer(data['text'], return_tensors="pt")
  device_input = pair_devices(inputs, device)
  scores = model(**device_input)[:2][0]
  prob_pos = F.softmax(scores, dim=1)[:, 1]
  response = {
    "pos_probability": prob_pos.item(), 
  }
  return jsonify(response)

@app.route("/extract", methods=["POST"])
def extract():
  data = request.get_json()
  keywords = rake.run(data['text'], minCharacters = 1, maxWords = 3, minFrequency = 1);
  return jsonify(keywords);

app.run()