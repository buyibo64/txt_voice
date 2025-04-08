import sys
import asyncio
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, List
import torch

import torchaudio
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.append('third_party/Matcha-TTS')

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CosyVoice TTS API")
executor = ThreadPoolExecutor(max_workers=4)

class CosyVoiceLoader:
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.cosyvoice = None
            cls._instance.prompt_speech = None
            cls._instance.sample_rate = 24000  # 默认模型采样率
        return cls._instance

    def load_model(self):
        """惰性加载模型"""
        if self.cosyvoice is None:
            try:
                from cosyvoice.cli.cosyvoice import CosyVoice2
                from cosyvoice.utils.file_utils import load_wav
                
                self.cosyvoice = CosyVoice2(
                    'pretrained_models/CosyVoice2-0.5B',
                    load_jit=False,
                    load_trt=False,
                    fp16=False
                )
                self.prompt_speech = load_wav('./asset/zero_shot_prompt.wav', 16000)
                self.sample_rate = self.cosyvoice.sample_rate
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Model loading failed: {str(e)}")
                raise RuntimeError("Model initialization failed")

cosyvoice_loader = CosyVoiceLoader()

class TTSRequest(BaseModel):
    text: str  
    ref_text: str = "用四川话说这句话"
    stream: bool = True
    sample_rate: int = 24000
    format: str = "wav"

async def async_audio_generator(texts: List[str], ref_text: str) -> AsyncGenerator[bytes, None]:
    """异步音频生成器"""
    loop = asyncio.get_event_loop()
    output_queue = asyncio.Queue()

    def sync_generator():
        try:
            print(texts)
            

            # for chunk in cosyvoice_loader.cosyvoice.inference_zero_shot(
            #     (t for t in texts if t.strip()),
            #     ref_text,
            #     cosyvoice_loader.prompt_speech,
            #     stream=True
            # ):
            # for chunk in cosyvoice_loader.cosyvoice.inference_instruct2(
            #     (t for t in texts if t.strip()),
            #     ref_text,
            #     cosyvoice_loader.prompt_speech,
            #     stream=True
            # ):
            for i, chunk in enumerate(cosyvoice_loader.cosyvoice.inference_instruct2(texts[0], '用四川话说这句话', cosyvoice_loader.prompt_speech, stream=True)):
                buffer = io.BytesIO()
                torchaudio.save(
                    buffer,
                    chunk['tts_speech'],
                    cosyvoice_loader.sample_rate,
                    format="wav"
                )
                buffer.seek(0)
                loop.call_soon_threadsafe(
                    output_queue.put_nowait, buffer.getvalue()
                )
            loop.call_soon_threadsafe(output_queue.put_nowait, None)
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            loop.call_soon_threadsafe(
                output_queue.put_nowait, e
            )

    # 在后台线程运行同步生成器
    await loop.run_in_executor(executor, sync_generator)

    while True:
        chunk = await output_queue.get()
        if chunk is None:
            break
        if isinstance(chunk, Exception):
            raise chunk
        print("yielding chunk")
        yield chunk

def resample_audio(waveform: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """音频重采样"""
    if orig_rate == target_rate:
        return waveform
    
    # 明确使用torch模块
    waveform_tensor = torch.from_numpy(waveform).float()  # 这里需要torch
    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_rate,
        new_freq=target_rate
    )
    return resampler(waveform_tensor).numpy()

@app.post("/generate_speech")
async def generate_speech(request: TTSRequest):
    try:
        cosyvoice_loader.load_model()  # 确保模型已加载
        
        texts = [t.strip() for t in request.text.split('\n') if t.strip()]
        if not texts:
            raise HTTPException(status_code=400, detail="Empty text input")

        if request.stream:
            print("Streaming response")
            return StreamingResponse(
                async_audio_generator(texts, request.ref_text),
                media_type=f"audio/{request.format}",
                headers={"Content-Disposition": f"attachment; filename=speech.{request.format}"}
            )
        else:
            # 收集所有音频片段
            chunks = []
            async for chunk in async_audio_generator(texts, request.ref_text):
                chunks.append(chunk)

            # 合并并转换音频
            full_waveform = []
            for chunk in chunks:
                buffer = io.BytesIO(chunk)
                waveform, sr = torchaudio.load(buffer)
                full_waveform.append(waveform)

            merged = torch.cat(full_waveform, dim=1)
            
            # 重采样处理
            print("request.sample_rate", request.sample_rate)
            print("cosyvoice_loader.sample_rate", cosyvoice_loader.sample_rate)
            if request.sample_rate != cosyvoice_loader.sample_rate:
                merged = resample_audio(
                    merged.numpy(),
                    cosyvoice_loader.sample_rate,
                    request.sample_rate
                )
                sr = request.sample_rate
                merged_tensor = torch.from_numpy(merged)
            else:
                merged_tensor = merged 
                sr = cosyvoice_loader.sample_rate

            # 格式转换
            buffer = io.BytesIO()
            torchaudio.save(
                buffer,
                merged_tensor,
                sr,
                format=request.format
            )
            
            return Response(
                content=buffer.getvalue(),
                media_type=f"audio/{request.format}",
                headers={"Content-Disposition": f"attachment; filename=speech.{request.format}"}
            )

    except Exception as e:
        logger.error(f"API Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)