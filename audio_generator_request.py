import requests

# 请求接口
response = requests.post(
    "http://localhost:8000/generate_speech",
    json={
        "text": "长亭外 古道边 芳草碧连天 问君此去几时还 夕阳山外山 天之涯 地之角",
        "format": "wav",
        "stream": True
    },
    stream=True  # 开启流式接收
)

# 保存为一个完整的文件
with open("speech.wav", "wb") as f:
    for chunk in response.iter_content(chunk_size=4096):
        print("正在保存音频文件...")
        if chunk:
            f.write(chunk)

print("保存完成：speech.wav")
