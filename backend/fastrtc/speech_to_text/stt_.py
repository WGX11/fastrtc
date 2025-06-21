from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol, Tuple

import click
import librosa
import numpy as np
from numpy.typing import NDArray

from ..utils import AudioChunk, audio_to_float32

import base64
import os


curr_dir = Path(__file__).parent


class STTModel(Protocol):
    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str: ...


class MoonshineSTT(STTModel):
    def __init__(
        self, model: Literal["moonshine/base", "moonshine/tiny"] = "moonshine/base"
    ):
        try:
            from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Install fastrtc[stt] for speech-to-text and stopword detection support."
            )

        self.model = MoonshineOnnxModel(model_name=model)
        self.tokenizer = load_tokenizer()

    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        sr, audio_np = audio  # type: ignore
        audio_np = audio_to_float32(audio_np)
        if sr != 16000:
            audio_np: NDArray[np.float32] = librosa.resample(
                audio_np, orig_sr=sr, target_sr=16000
            )
        if audio_np.ndim == 1:
            audio_np = audio_np.reshape(1, -1)
        tokens = self.model.generate(audio_np)
        return self.tokenizer.decode_batch(tokens)[0]
    
import os
import json
import base64
import asyncio
import gzip
import uuid
import hmac
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, List
from urllib.parse import urlparse

import click
import librosa
import numpy as np
from numpy.typing import NDArray
import websockets   
class VolcanoEngineWsSTT(STTModel):
    # 从火山引擎Demo中提取的常量
    _PROTOCOL_VERSION = 0b0001
    _FULL_CLIENT_REQUEST = 0b0001
    _AUDIO_ONLY_REQUEST = 0b0010
    _FULL_SERVER_RESPONSE = 0b1001
    _SERVER_ACK = 0b1011
    _SERVER_ERROR_RESPONSE = 0b1111
    _POS_SEQUENCE = 0b0001
    _NEG_WITH_SEQUENCE = 0b0011
    _JSON = 0b0001
    _GZIP = 0b0001
    _NO_COMPRESSION = 0b0000

    def __init__(self):
        self.ak = os.getenv("ACCESS_KEY", "")         
        self.app_id = os.getenv("APP_ID", "")
        self.sk = os.getenv("SECRET_KEY", "")
        self.ws_url = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"
        self.uid = f"fastrtc_user_{uuid.uuid4()}"

        if not all([self.ak, self.sk, self.app_id]):
            raise ValueError(
                "请设置环境变量：ACCESS_KEY, SECRET_KEY, 和 APP_ID"
            )
        print(click.style("INFO", fg="green") + ":\t  火山引擎 WebSocket STT 服务已初始化。")

    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        # 这是同步和异步世界的桥梁
        try:
            return asyncio.run(self._async_process_utterance(audio))
        except Exception as e:
            print(click.style("ERROR", fg="red") + f":\t  运行火山引擎异步任务时出错: {e}")
            return ""

    def _generate_header(self, mt, mfsf, ct=_GZIP):
        h=bytearray(4)
        h[0]=(self._PROTOCOL_VERSION<<4)|1
        h[1]=(mt<<4)|mfsf
        h[2]=(self._JSON<<4)|ct
        h[3]=0x00
        return h

    def _parse_response(self, res:bytes) -> dict:
        mt = res[1] >> 4
        compression_flag = res[2] & 0x0f
        
        is_last_package = bool((res[1] & 0b0010))
        has_sequence = bool((res[1] & 0b0001))
        
        payload_offset = 4
        if has_sequence:
            payload_offset = 8

        p = res[payload_offset:]
        # 【修改】初始化一个带 'utterances' 键的字典
        r = {'type': mt, 'text': '', 'utterances': [], 'is_last': is_last_package} 
        
        if mt == self._FULL_SERVER_RESPONSE and len(p) > 4:
            payload_size = int.from_bytes(p[:4], 'big')
            if payload_size == 0:
                return r

            payload_body = p[4 : 4 + payload_size]
            payload_msg = gzip.decompress(payload_body) if compression_flag == self._GZIP else payload_body
            
            try:
                msg = json.loads(payload_msg.decode('utf-8'))
                
                # 【修改】提取整个 'utterances' 列表
                if 'result' in msg and isinstance(msg['result'], dict) and 'utterances' in msg['result']:
                    r['utterances'] = msg['result']['utterances']

            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        return r

    async def _async_process_utterance(self, audio: tuple) -> str:
        sr,an=audio
        afk=audio_to_float32(an)
        if sr!=16000: afk=librosa.resample(afk,orig_sr=sr,target_sr=16000)
        ab=(afk*32767).astype(np.int16).tobytes(); connect_id=str(uuid.uuid4())
        
        # 【修改】不再用 parts 列表，而是用一个字典来存储最终句子
        final_utterances = {}
        
        try:
            connect_headers = {
                "X-Api-App-Key": self.app_id,
                "X-Api-Access-Key": self.ak,
                "X-Api-Resource-Id": "volc.bigasr.sauc.duration",
                "X-Api-Connect-Id": connect_id
            }

            async with websockets.connect(
                self.ws_url, 
                additional_headers=connect_headers, 
                max_size=None
            ) as ws:
                
                cp = {
                    "user": {"uid": self.uid},
                    "audio": {'format': 'pcm', "rate": 16000, "bits": 16, "channel": 1},
                    "request": {
                        "model_name": "bigmodel",
                        "enable_punc": True,
                        "result_type": "full",
                        # 【关键】请求服务器返回分句信息
                        "show_utterances": True, 
                    }
                }
                cb=gzip.compress(json.dumps(cp).encode('utf-8'))
                h=self._generate_header(self._FULL_CLIENT_REQUEST, 0b0000)
                h.extend((len(cb)).to_bytes(4,'big') + cb)
                await ws.send(h)
                
                await ws.recv()
                
                csz=1280;
                for i in range(0,len(ab),csz):
                    c=ab[i:i+csz]; il=(i+csz)>=len(ab)
                    f = 0b0010 if il else 0b0000
                    ap=gzip.compress(c)
                    h=self._generate_header(self._AUDIO_ONLY_REQUEST, f)
                    h.extend((len(ap)).to_bytes(4,'big') + ap)
                    await ws.send(h)
                    
                    rd=await ws.recv()
                    p=self._parse_response(rd)
                    
                    # 【修改】处理返回的句子列表
                    if p.get('utterances'):
                        for utterance in p['utterances']:
                            # 只关心被服务器确认为“最终”的句子
                            if utterance.get('definite'):
                                # 使用句子的 start_time 作为唯一ID来去重和排序
                                sentence_id = utterance.get('start_time')
                                sentence_text = utterance.get('text', '')
                                final_utterances[sentence_id] = sentence_text
                    
                    if p.get('is_last'):
                        break
                        
        except Exception as e: 
            import traceback
            print(click.style("ERROR", fg="red") + f":\t  火山引擎 WebSocket 通信时发生未知错误: {e}")
            traceback.print_exc()
        
        # 【修改】最终拼接结果
        if not final_utterances:
            return ""
        
        # 按照句子的开始时间排序，然后拼接成一整句话
        sorted_sentences = [text for _, text in sorted(final_utterances.items())]
        final_result = "".join(sorted_sentences)
        
        print(click.style("SUCCESS", fg="green") + f":\t  流式识别最终结果: {final_result}")
        return final_result

import time
import requests
# ===【全新的录音文件识别类】===
class VolcanoEngineFileSTT(STTModel):
    def __init__(self):
        # AK/SK/AppID 直接硬编码，方便调试
        self.ak = "B3H8hhmlxDMr-tN1xe3vH2P8Oabaf_Al"
        self.sk = "eZV0HLwsfSU3QAspdLUPZo4zSUWYBAeI"
        self.app_id = "5613321874"
        self.ak = os.getenv("APP_KEY", "")          # 第二个参数是默认值
        self.app_id = os.getenv("APP_ID", "")
        self.sk = os.getenv("APP_SECRET", "")
        # 检查环境变量是否设置
        if not self.ak or not self.app_id or not self.sk:
            raise ValueError("Missing required environment variables: APP_KEY and APP_ID and APP_SECRET")
        
        self.submit_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
        self.query_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"
        
        self.uid = f"fastrtc_user_{uuid.uuid4()}"
        print(click.style("INFO", fg="green") + ":\t  火山引擎录音文件 STT 服务已初始化。")

    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        sr, audio_np = audio
        
        # 1. 将音频数据转换为 Base64 编码的字符串
        # 首先确保是 16k, int16 格式
        audio_float_16k = audio_to_float32(audio_np)
        if sr != 16000:
            audio_float_16k = librosa.resample(audio_float_16k, orig_sr=sr, target_sr=16000)
        audio_int16 = (audio_float_16k * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        task_id = str(uuid.uuid4())

        # 2. 提交任务
        try:
            if not self._submit_task(task_id, audio_base64):
                print(click.style("ERROR", fg="red") + ":\t  提交识别任务失败。")
                return ""
        except Exception as e:
            print(click.style("ERROR", fg="red") + f":\t  提交任务时发生异常: {e}")
            return ""

        # 3. 轮询查询结果
        try:
            return self._query_result_loop(task_id)
        except Exception as e:
            print(click.style("ERROR", fg="red") + f":\t  查询结果时发生异常: {e}")
            return ""

    def _submit_task(self, task_id: str, audio_base64: str) -> bool:
        headers = {
            "X-Api-App-Key": self.app_id,
            "X-Api-Access-Key": self.ak,
            "X-Api-Resource-Id": "volc.bigasr.auc",
            "X-Api-Request-Id": task_id,
            "X-Api-Sequence": "-1",
            # 【关键修改】移除 Content-Type，让 requests 库不要自动添加
        }
        
        payload = {
            "user": {"uid": self.uid},
            "audio": {
                "format": "raw",
                "data": audio_base64,
                "rate": 16000,
                "bits": 16,
                "channel": 1,
            },
            "request": { "model_name": "bigmodel", "enable_punc": True, "enable_itn": True }
        }
        
        # 【关键修改】使用 data 参数，并手动进行 json.dumps
        response = requests.post(self.submit_url, headers=headers, data=json.dumps(payload))
        
        status_code = response.headers.get("X-Api-Status-Code")
        if status_code == "20000000":
            print(click.style("INFO", fg="green") + f":\t  任务提交成功, Task ID: {task_id}")
            return True
        else:
            message = response.headers.get("X-Api-Message", "未知错误")
            body_text = response.text
            print(click.style("ERROR", fg="red") + f":\t  任务提交失败, Code: {status_code}, Msg: {message}, Body: {body_text}")
            return False

    def _query_result_loop(self, task_id: str) -> str:
        headers = {
            "X-Api-App-Key": self.app_id,
            "X-Api-Access-Key": self.ak,
            "X-Api-Resource-Id": "volc.bigasr.auc",
            "X-Api-Request-Id": task_id,
            # 【关键修改】移除 Content-Type
        }
        
        max_wait_time = 10
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # 【关键修改】使用 data 参数，并手动进行 json.dumps
            response = requests.post(self.query_url, headers=headers, data=json.dumps({}))
            
            status_code = response.headers.get("X-Api-Status-Code")
            
            if status_code == "20000000":
                result_json = response.json()
                if result_json.get("result") and result_json["result"].get("text"):
                    final_text = result_json["result"]["text"]
                    print(click.style("SUCCESS", fg="green") + f":\t  识别成功: {final_text}")
                    return final_text
                else: return ""
            elif status_code in ["20000001", "20000002"]:
                time.sleep(0.5)
                continue
            else:
                message = response.headers.get("X-Api-Message", "查询时发生未知错误")
                print(click.style("ERROR", fg="red") + f":\t  查询失败, Code: {status_code}, Msg: {message}")
                return ""
        
        print(click.style("WARNING", fg="yellow") + ":\t  查询超时。")
        return ""

# --- 【修改后】的工厂函数 ---
def get_stt_model(model: str = "moonshine/base") -> STTModel:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if model.startswith("moonshine/"):
        # ... (这部分保持不变) ...
        return m
    elif model == "volcano_engine_ws":
        return VolcanoEngineWsSTT()
    # --- 【新增这个分支】 ---
    elif model == "volcano_engine_file":
        return VolcanoEngineFileSTT()
    # ----------------------
    else:
        raise ValueError(f"未知的 STT 模型: '{model}'. 可用: 'moonshine/base', 'moonshine/tiny', 'volcano_engine_ws', 'volcano_engine_file'")



# @lru_cache
# def get_stt_model(
#     model: Literal["moonshine/base", "moonshine/tiny"] = "moonshine/base",
# ) -> STTModel:
#     import os

#     os.environ["TOKENIZERS_PARALLELISM"] = "false"
#     m = MoonshineSTT(model)
#     from moonshine_onnx import load_audio

#     audio = load_audio(str(curr_dir / "test_file.wav"))
#     print(click.style("INFO", fg="green") + ":\t  Warming up STT model.")

#     m.stt((16000, audio))
#     print(click.style("INFO", fg="green") + ":\t  STT model warmed up.")
    # return m


def stt_for_chunks(
    stt_model: STTModel,
    audio: tuple[int, NDArray[np.int16 | np.float32]],
    chunks: list[AudioChunk],
) -> str:
    sr, audio_np = audio
    return " ".join(
        [
            stt_model.stt((sr, audio_np[chunk["start"] : chunk["end"]]))
            for chunk in chunks
        ]
    )
