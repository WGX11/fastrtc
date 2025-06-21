import asyncio
import os
import importlib.util
import re
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, Protocol, TypeVar

import numpy as np
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray

from fastrtc.utils import async_aggregate_bytes_to_16bit


class TTSOptions:
    pass


T = TypeVar("T", bound=TTSOptions, contravariant=True)


class TTSModel(Protocol[T]):
    def tts(
        self, text: str, options: T | None = None
    ) -> tuple[int, NDArray[np.float32] | NDArray[np.int16]]: ...

    def stream_tts(
        self, text: str, options: T | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32] | NDArray[np.int16]], None]: ...

    def stream_tts_sync(
        self, text: str, options: T | None = None
    ) -> Generator[tuple[int, NDArray[np.float32] | NDArray[np.int16]], None, None]: ...


@dataclass
class KokoroTTSOptions(TTSOptions):
    voice: str = "af_heart"
    speed: float = 1.0
    lang: str = "en-us"


@lru_cache
def get_tts_model(
    model: Literal["kokoro", "cartesia"] = "kokoro", **kwargs
) -> TTSModel:
    if model == "kokoro":
        m = KokoroTTSModel()
        m.tts("Hello, world!")
        return m
    elif model == "cartesia":
        m = CartesiaTTSModel(api_key=kwargs.get("cartesia_api_key", ""))
        return m
    elif model == "volcano_engine":
        return VolcanoEngineTTSModel()
    else:
        raise ValueError(f"Invalid model: {model}")



class KokoroFixedBatchSize:
    # Source: https://github.com/thewh1teagle/kokoro-onnx/issues/115#issuecomment-2676625392
    def _split_phonemes(self, phonemes: str) -> list[str]:
        MAX_PHONEME_LENGTH = 510
        max_length = MAX_PHONEME_LENGTH - 1
        batched_phonemes = []
        while len(phonemes) > max_length:
            # Find best split point within limit
            split_idx = max_length

            # Try to find the last period before max_length
            period_idx = phonemes.rfind(".", 0, max_length)
            if period_idx != -1:
                split_idx = period_idx + 1  # Include period

            else:
                # Try other punctuation
                match = re.search(
                    r"[!?;,]", phonemes[:max_length][::-1]
                )  # Search backwards
                if match:
                    split_idx = max_length - match.start()

                else:
                    # Try last space
                    space_idx = phonemes.rfind(" ", 0, max_length)
                    if space_idx != -1:
                        split_idx = space_idx

            # If no good split point is found, force split at max_length
            chunk = phonemes[:split_idx].strip()
            batched_phonemes.append(chunk)

            # Move to the next part
            phonemes = phonemes[split_idx:].strip()

        # Add remaining phonemes
        if phonemes:
            batched_phonemes.append(phonemes)
        return batched_phonemes

import base64
import websockets
import asyncio
import base64
import json
import re
import gzip
import uuid
from typing import AsyncGenerator, Generator, Literal, Protocol, TypeVar

import click  
import numpy as np
import requests
import websockets
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
class VolcanoEngineTTSOptions(TTSOptions):
    voice_type: str = "zh_male_sunwukong_mars_bigtts"

class VolcanoEngineTTSModel(TTSModel[VolcanoEngineTTSOptions]):
    def __init__(self):
        self.ak = os.getenv("ACCESS_KEY", "")         
        self.app_id = os.getenv("APP_ID", "")
        
        if not self.ak or not self.app_id:
            raise ValueError("Missing required environment variables: APP_KEY and APP_ID")
        
        self.http_url = "https://openspeech.bytedance.com/api/v1/tts"
        self.ws_url = "wss://openspeech.bytedance.com/api/v1/tts/ws_binary"
        
        print(click.style("INFO", fg="green") + ":\t  火山引擎 TTS 服务已初始化。")

    # 1. 实现 tts() 方法 (使用HTTP接口)
    def _create_tts_binary_header(self, payload_size: int) -> bytearray:
        # Protocol version: 1, Header size: 1 (4 bytes)
        byte0 = (0b0001 << 4) | 0b0001
        # Message type: 1 (full client request), Flags: 0
        byte1 = (0b0001 << 4) | 0b0000
        # Serialization: 1 (JSON), Compression: 1 (gzip)
        byte2 = (0b0001 << 4) | 0b0001
        # Reserved
        byte3 = 0x00
        
        header = bytearray([byte0, byte1, byte2, byte3])
        
        # 文档中没有明确提到 header 后要跟 payload_size，但这是标准做法，我们加上
        # 如果不工作，可以尝试去掉这4个字节
        # header.extend(payload_size.to_bytes(4, 'big'))
        
        return header
    
    def tts(
        self, text: str, options: VolcanoEngineTTSOptions | None = None
    ) -> tuple[int, NDArray[np.int16]]:
        options = options or VolcanoEngineTTSOptions()
        
        headers = {"Authorization": f"Bearer;{self.ak}"}
        
        payload = {
            "app": {"appid": self.app_id, "token": "access_token", "cluster": "volcano_tts"},
            "user": {"uid": f"fastrtc_user_{uuid.uuid4()}"},
            "audio": {
                "voice_type": options.voice_type,
                "encoding": "pcm", # 请求 pcm 原始数据
                "rate": 24000,     # 大模型音色通常是 24k
            },
            "request": {
                "reqid": str(uuid.uuid4()),
                "text": text,
                "operation": "query",
            }
        }
        
        try:
            response = requests.post(self.http_url, headers=headers, json=payload)
            response.raise_for_status() # 如果请求失败则抛出异常
            
            result_json = response.json()
            if result_json.get("code") == 3000 and "data" in result_json:
                audio_base64 = result_json["data"]
                audio_bytes = base64.b64decode(audio_base64)
                # 将原始字节转换为 int16 numpy 数组
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                return (24000, audio_np) # 返回采样率和音频数据
            else:
                message = result_json.get("message", "未知错误")
                print(click.style("ERROR", fg="red") + f":\t  TTS HTTP API 错误: {message}")
        except requests.exceptions.RequestException as e:
            print(click.style("ERROR", fg="red") + f":\t  TTS HTTP 请求失败: {e}")
        
        return (24000, np.array([], dtype=np.int16)) # 失败时返回空音频

    # 2. 实现 stream_tts() 方法 (使用WebSocket接口)
    async def stream_tts(
        self, text: str, options: VolcanoEngineTTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.int16]], None]:
        options = options or VolcanoEngineTTSOptions()
        headers = {"Authorization": f"Bearer;{self.ak}"}
        
        # 1. 构建 payload 字典。
        # 【关键修正】文档中 "token" 字段是必需的，但值可以是任意非空字符串。
        # 我们之前发送的是 "access_token"，现在改为更通用的 "token"
        payload_dict = {
            "app": {"appid": self.app_id, "token": "token", "cluster": "volcano_tts"},
            "user": {"uid": f"fastrtc_user_{uuid.uuid4()}"},
            "audio": {
                "voice_type": options.voice_type,
                "encoding": "pcm",
                "rate": 24000,
            },
            "request": {
                "reqid": str(uuid.uuid4()),
                "text": text,
                "operation": "submit",
            }
        }
        
        # 2. 将 payload 序列化，但不压缩
        payload_bytes = json.dumps(payload_dict).encode('utf-8')
        
        # 3. 构建二进制报头，明确设置“无压缩”
        header = bytearray()
        header.append((0b0001 << 4) | 0b0001)
        header.append((0b0001 << 4) | 0b0000)
        header.append((0b0001 << 4) | 0b0000) # JSON, NO_COMPRESSION
        header.append(0x00)
        
        payload_size_bytes = len(payload_bytes).to_bytes(4, 'big')
        
        # 最终的二进制消息 = header + payload_size + payload
        binary_message = header + payload_size_bytes + payload_bytes
        
        try:
            async with websockets.connect(self.ws_url, additional_headers=headers) as ws:
                # 4. 发送最终的二进制消息
                await ws.send(binary_message)
                
                # 5. 循环接收服务器返回的二进制数据
                while True:
                    response_bytes = await ws.recv()
                    
                    if len(response_bytes) < 4: continue

                    resp_msg_type = (response_bytes[1] >> 4)
                    
                    if resp_msg_type == 0b1011: # Audio-only response
                        resp_flags = response_bytes[1] & 0x0f
                        audio_offset = 4
                        if resp_flags > 0: audio_offset = 8
                        
                        audio_data = response_bytes[audio_offset:]
                        if audio_data:
                            audio_np = np.frombuffer(audio_data, dtype=np.int16)
                            if audio_np.size > 0: yield (24000, audio_np)

                        if resp_flags in [0b0010, 0b0011]: break
                    
                    elif resp_msg_type == 0b1111: # Error response
                        error_code = int.from_bytes(response_bytes[4:8], 'big')
                        error_msg_size = int.from_bytes(response_bytes[8:12], 'big')
                        error_msg = response_bytes[12:12+error_msg_size].decode('utf-8')
                        print(f"TTS WS Error - Code: {error_code}, Msg: {error_msg}")
                        break
                    
        except websockets.exceptions.ConnectionClosed as e:
            if e.code != 1000:
                print(f"TTS WebSocket 连接异常关闭: code={e.code}, reason={e.reason}")
        except Exception as e:
            import traceback
            print(click.style("ERROR", fg="red") + f":\t  TTS WebSocket 发生未知错误: {e}")
            traceback.print_exc()

    # 3. 实现 stream_tts_sync() 方法 (直接复用 fastrtc 的已有实现)
    def stream_tts_sync(
        self, text: str, options: VolcanoEngineTTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.float32]], None, None]:
        
        # --- 【最终优化】 ---
        # 1. 增加交叉区域长度至 10ms (240个采样点 @ 24kHz)
        crossfade_samples = 240
        
        # 2. 使用“等功率”的正弦/余弦曲线，使过渡更平滑自然
        #    np.linspace(0, np.pi / 2, ...) 会生成一个从0到π/2的序列
        fade_out = np.cos(np.linspace(0, np.pi / 2, crossfade_samples, dtype=np.float32))
        fade_in = np.sin(np.linspace(0, np.pi / 2, crossfade_samples, dtype=np.float32))
        # ------------------------

        buffer = np.array([], dtype=np.float32)
        previous_chunk_tail = np.zeros(crossfade_samples, dtype=np.float32)

        async_generator = self.stream_tts(text, options)
        loop = asyncio.new_event_loop()
        iterator = async_generator.__aiter__()
        
        is_finished = False
        while not is_finished:
            try:
                _samplerate, audio_chunk_int16 = loop.run_until_complete(iterator.__anext__())
                audio_chunk_float32 = audio_chunk_int16.astype(np.float32) / 32767.0
                
                # --- 交叉处理逻辑保持不变，但现在使用更优的参数 ---
                current_chunk_head = audio_chunk_float32[:crossfade_samples]
                if len(current_chunk_head) < crossfade_samples:
                    padding = np.zeros(crossfade_samples - len(current_chunk_head), dtype=np.float32)
                    current_chunk_head = np.concatenate([current_chunk_head, padding])

                crossfaded_region = (previous_chunk_tail * fade_out) + (current_chunk_head * fade_in)
                
                buffer = np.concatenate([buffer, crossfaded_region])
                buffer = np.concatenate([buffer, audio_chunk_float32[crossfade_samples:]])

                if len(audio_chunk_float32) >= crossfade_samples:
                    previous_chunk_tail = audio_chunk_float32[-crossfade_samples:]
                else:
                    tail_part = audio_chunk_float32
                    padding = np.zeros(crossfade_samples - len(tail_part), dtype=np.float32)
                    previous_chunk_tail = np.concatenate([tail_part, padding])

            except StopAsyncIteration:
                is_finished = True

            # --- 切分帧的逻辑保持不变 ---
            frame_size_in_samples = 480
            while len(buffer) >= frame_size_in_samples:
                frame_to_yield = buffer[:frame_size_in_samples]
                buffer = buffer[frame_size_in_samples:]
                yield (24000, frame_to_yield.astype(np.float32))
        
        if len(buffer) > 0:
            yield (24000, buffer.astype(np.float32))

class KokoroTTSModel(TTSModel):
    def __init__(self):
        from kokoro_onnx import Kokoro

        self.model = Kokoro(
            model_path=hf_hub_download("fastrtc/kokoro-onnx", "kokoro-v1.0.onnx"),
            voices_path=hf_hub_download("fastrtc/kokoro-onnx", "voices-v1.0.bin"),
        )

        self.model._split_phonemes = KokoroFixedBatchSize()._split_phonemes

    def tts(
        self, text: str, options: KokoroTTSOptions | None = None
    ) -> tuple[int, NDArray[np.float32]]:
        options = options or KokoroTTSOptions()
        a, b = self.model.create(
            text, voice=options.voice, speed=options.speed, lang=options.lang
        )
        return b, a

    async def stream_tts(
        self, text: str, options: KokoroTTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        options = options or KokoroTTSOptions()

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        for s_idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            chunk_idx = 0
            async for chunk in self.model.create_stream(
                sentence, voice=options.voice, speed=options.speed, lang=options.lang
            ):
                if s_idx != 0 and chunk_idx == 0:
                    yield chunk[1], np.zeros(chunk[1] // 7, dtype=np.float32)
                chunk_idx += 1
                yield chunk[1], chunk[0]

    def stream_tts_sync(
        self, text: str, options: KokoroTTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.float32]], None, None]:
        loop = asyncio.new_event_loop()

        # Use the new loop to run the async generator
        iterator = self.stream_tts(text, options).__aiter__()
        while True:
            try:
                yield loop.run_until_complete(iterator.__anext__())
            except StopAsyncIteration:
                break


@dataclass
class CartesiaTTSOptions(TTSOptions):
    voice: str = "71a7ad14-091c-4e8e-a314-022ece01c121"
    language: str = "en"
    emotion: list[str] = field(default_factory=list)
    cartesia_version: str = "2024-06-10"
    model: str = "sonic-2"
    sample_rate: int = 22_050


class CartesiaTTSModel(TTSModel):
    def __init__(self, api_key: str):
        if importlib.util.find_spec("cartesia") is None:
            raise RuntimeError(
                "cartesia is not installed. Please install it using 'pip install cartesia'."
            )
        from cartesia import AsyncCartesia

        self.client = AsyncCartesia(api_key=api_key)

    async def stream_tts(
        self, text: str, options: CartesiaTTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.int16]], None]:
        options = options or CartesiaTTSOptions()

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        for sentence in sentences:
            if not sentence.strip():
                continue
            async for output in async_aggregate_bytes_to_16bit(
                self.client.tts.bytes(
                    model_id="sonic-2",
                    transcript=sentence,
                    voice={"id": options.voice},  # type: ignore
                    language="en",
                    output_format={
                        "container": "raw",
                        "sample_rate": options.sample_rate,
                        "encoding": "pcm_s16le",
                    },
                )
            ):
                yield options.sample_rate, np.frombuffer(output, dtype=np.int16)

    def stream_tts_sync(
        self, text: str, options: CartesiaTTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.int16]], None, None]:
        loop = asyncio.new_event_loop()

        iterator = self.stream_tts(text, options).__aiter__()
        while True:
            try:
                yield loop.run_until_complete(iterator.__anext__())
            except StopAsyncIteration:
                break

    def tts(
        self, text: str, options: CartesiaTTSOptions | None = None
    ) -> tuple[int, NDArray[np.int16]]:
        loop = asyncio.new_event_loop()
        buffer = np.array([], dtype=np.int16)

        options = options or CartesiaTTSOptions()

        iterator = self.stream_tts(text, options).__aiter__()
        while True:
            try:
                _, chunk = loop.run_until_complete(iterator.__anext__())
                buffer = np.concatenate([buffer, chunk])
            except StopAsyncIteration:
                break
        return options.sample_rate, buffer
