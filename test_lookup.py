"""检索效果测试：直接运行，不需要真实 API key"""
import os
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test")

from agent import lookup_db, _vectorizer, _matrix, _keys
from sklearn.metrics.pairwise import cosine_similarity

queries = [
    "2022世界杯冠军是谁",       # 应命中：世界杯冠军
    "去年世界杯谁夺冠了",       # 换个说法，应命中：世界杯冠军
    "金球奖得主是哪位球员",     # 应命中：金球奖
    "欧冠冠军是哪个队",         # 应命中：欧冠冠军
    "越位规则是什么",           # 库里没有 → 应返回"没找到"
    "明天天气怎么样",           # 无关 → 应返回"没找到"
]

for q in queries:
    sims = cosine_similarity(_vectorizer.transform([q]), _matrix)[0]
    scores = {k: round(float(s), 3) for k, s in zip(_keys, sims)}
    print(f"{q}\n  -> {lookup_db(q)[:30]}...\n  相似度: {scores}\n")

