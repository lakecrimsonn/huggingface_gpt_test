from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res = classifier("I've been waiting for a huggingface course my whole life")

print(res)