from transformers import BertForSequenceClassification, BertTokenizer

def get_pretrained_model_and_tokenizer(device):
  model = BertForSequenceClassification.from_pretrained("C:/Users/bruno/bsi/NLP/api/assets/bert/")
  tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
  model.to(device)
  return model, tokenizer

