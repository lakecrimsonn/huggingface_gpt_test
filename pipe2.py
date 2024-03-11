from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

res = classifier("I've been waiting for a huggingface course my whole life")

print(res)

seq = "Using a transformer network is simple"
res = tokenizer(seq)
print(res)

tokens = tokenizer.tokenize(seq)
print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

decoded_str = tokenizer.decode(ids)
print(decoded_str)