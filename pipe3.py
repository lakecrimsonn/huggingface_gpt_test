from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

X_train = ["I've been waiting for a Huggingface course my whole life", "python is great!"]

res = classifier(X_train)
# print(res)

batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt") # 토큰나이저를 통해서 X_train 전처리
# padding은 입력 시퀀스가 최대 512토큰에 비해 부족한 토큰을 패딩으로 추가해줌
# truncation은 최대 512를 넘어가는 경우 입력데이터의 길이를 적절히 조정
# print(batch) # 토큰들과 어탠션 마스크

with torch.no_grad(): # 모델을 평가모드로 실행할 때, 그레디언트 계산을 비활성화해서 메모리 사용량 줄임, 계속 속도는 높임
    outputs = model(**batch)
    # print(outputs) # 로짓(최종 레이어에서 생성된 원시 출력값, 각 클래스에 대한 점수)
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)
    print(labels)