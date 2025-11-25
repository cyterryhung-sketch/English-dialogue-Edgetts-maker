import edge_tts
import asyncio
import os
import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading # 导入 threading 模块
import wave # 用于静音文件生成 (如果需要合并的话，但目前是生成单独文件)
import contextlib # 用于文件操作
import numpy as np
import tempfile
import re  # 用于处理 pause 标记
# 使用 librosa 和 soundfile 来处理音频文件格式转换
try:
    import librosa
    import soundfile as sf
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False

# 尝试导入音频播放库
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# 尝试导入 playsound 库作为备选方案
try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    PLAYSOUND_AVAILABLE = False

# 可用的音色列表，按类别分类
available_voices = {
    "美式英语-男声": {
        "Andrew (美式)": "en-US-AndrewNeural",
        "Brian (美式)": "en-US-BrianNeural",
        "Christopher (美式)": "en-US-ChristopherNeural",
        "Roger (美式)": "en-US-RogerNeural",
        "Steffan (美式)": "en-US-SteffanNeural",
        "Guy (美式, 默认)": "en-US-GuyNeural",
    },
    "美式英语-女声": {
        "Ana (美式)": "en-US-AnaNeural",
        "Aria (美式)": "en-US-AriaNeural",
        "Ava (美式)": "en-US-AvaNeural",
        "Jenny (美式, 默认)": "en-US-JennyNeural",
        "Michelle (美式)": "en-US-MichelleNeural",
    },
    "英式英语-男声": {
        "Libby (英式)": "en-GB-LibbyNeural",
        "Ryan (英式)": "en-GB-RyanNeural",
    },
    "英式英语-女声": {
        "Sonia (英式)": "en-GB-SoniaNeural",
        "Maisie (英式)": "en-GB-MaisieNeural",
    },
    # 可以根据需要添加更多分类，例如童声
}

def play_audio_file(file_path):
    """
    播放音频文件，尝试多种播放方法
    :param file_path: 音频文件路径
    """
    # 方法1: 使用 pygame 播放（先尝试转换格式）
    if PYGAME_AVAILABLE:
        try:
            # 先尝试转换音频格式
            converted_file = None
            if AUDIO_PROCESSING_AVAILABLE:
                # 创建临时文件用于转换后的音频
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    converted_file = tmp_file.name
                
                # 使用 librosa 读取并重新保存为标准格式
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                sf.write(converted_file, audio_data, sample_rate)
                
                # 使用 pygame 播放转换后的文件
                pygame.mixer.init()
                pygame.mixer.music.load(converted_file)
                pygame.mixer.music.play()
                
                # 等待播放完成
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                
                pygame.mixer.quit()
                
                # 清理临时文件
                try:
                    os.unlink(converted_file)
                except:
                    pass
                    
                return True
            else:
                # 直接尝试播放
                pygame.mixer.init()
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                
                # 等待播放完成
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                
                pygame.mixer.quit()
                return True
                
        except Exception as e:
            print(f"使用 pygame 播放音频文件时出错: {e}")
            try:
                pygame.mixer.quit()
            except:
                pass
    
    # 方法2: 使用 playsound 播放
    if PLAYSOUND_AVAILABLE:
        try:
            playsound(file_path)
            return True
        except Exception as e:
            print(f"使用 playsound 播放音频文件时出错: {e}")
    
    # 方法3: 使用系统默认播放器
    try:
        import subprocess
        import platform
        
        system = platform.system()
        if system == "Windows":
            os.startfile(file_path)
        elif system == "Darwin":  # macOS
            subprocess.call(["afplay", file_path])
        else:  # Linux
            subprocess.call(["aplay", file_path])
        return True
    except Exception as e:
        print(f"使用系统播放器播放音频文件时出错: {e}")
    
    print("所有音频播放方法都失败了，请检查音频文件或安装音频播放库")
    return False

def create_silence(duration, sample_rate=22050):
    """
    创建指定时长的静音音频文件
    :param duration: 静音时长（秒）
    :param sample_rate: 采样率
    :return: 静音音频文件路径
    """
    # 创建静音numpy数组
    silence = np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    # 保存为临时文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        temp_filename = tmp_file.name
    
    # 使用soundfile保存（如果可用）否则使用其他方法
    if AUDIO_PROCESSING_AVAILABLE:
        sf.write(temp_filename, silence, sample_rate)
    else:
        # 如果没有soundfile，创建一个非常短的静音文件
        with wave.open(temp_filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)   # 16位
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b'\x00' * int(sample_rate * duration * 2))
    
    return temp_filename

def merge_audio_files_with_librosa(file_list, output_filename):
    """
    使用 librosa 合并多个音频文件为一个文件
    :param file_list: 要合并的音频文件列表
    :param output_filename: 输出文件名
    """
    if not file_list:
        return False
    
    try:
        # 初始化一个空的音频数组
        combined_audio = np.array([], dtype=np.float32)
        sample_rate = None
        
        # 依次读取并合并所有音频文件
        for filename in file_list:
            # 使用 librosa 读取音频文件
            audio_data, sr = librosa.load(filename, sr=None)
            
            # 保存采样率（使用第一个文件的采样率）
            if sample_rate is None:
                sample_rate = sr
            
            # 合并音频数据
            combined_audio = np.concatenate([combined_audio, audio_data])
        
        # 保存合并后的音频文件
        sf.write(output_filename, combined_audio, sample_rate)
        return True
    except Exception as e:
        print(f"使用 librosa 合并音频文件时出错: {e}")
        return False

def merge_wav_files(file_list, output_filename):
    """
    合并多个 WAV 文件为一个文件
    :param file_list: 要合并的 WAV 文件列表
    :param output_filename: 输出文件名
    """
    if not file_list:
        return False
    
    try:
        # 尝试使用 librosa 方法合并（更可靠）
        if AUDIO_PROCESSING_AVAILABLE:
            return merge_audio_files_with_librosa(file_list, output_filename)
        
        # 如果 librosa 不可用，使用 wave 模块
        # 读取第一个文件的参数
        with wave.open(file_list[0], 'rb') as first_wav:
            params = first_wav.getparams()
            frames = first_wav.readframes(first_wav.getnframes())
        
        # 创建输出文件
        with wave.open(output_filename, 'wb') as output_wav:
            output_wav.setparams(params)
            output_wav.writeframes(frames)
            
            # 添加后续文件的内容
            for filename in file_list[1:]:
                try:
                    with wave.open(filename, 'rb') as wav_file:
                        # 确保参数匹配
                        if wav_file.getparams()[:4] == params[:4]:  # 检查前4个参数是否一致
                            frames = wav_file.readframes(wav_file.getnframes())
                            output_wav.writeframes(frames)
                        else:
                            print(f"警告: {filename} 参数不匹配，跳过该文件")
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")
                    continue
        
        return True
    except Exception as e:
        print(f"合并文件时出错: {e}")
        return False

def process_text_with_pause(text):
    """
    处理文本中的[pause_X]标记，将其分离为文本段和停顿时长
    :param text: 包含pause标记的文本
    :return: 处理后的文本段列表，每个元素为(text_segment, pause_duration)
    """
    # 使用正则表达式查找[pause_X]标记
    pattern = r'\[pause_(\d+(?:\.\d+)?)\]'
    parts = re.split(pattern, text)
    
    result = []
    current_text = ""
    
    # 第一个元素总是文本
    if parts[0]:
        current_text = parts[0]
    
    # 处理剩余部分
    for i in range(1, len(parts)):
        if i % 2 == 1:  # 奇数索引是pause的时长
            pause_duration = float(parts[i])
            result.append((current_text, pause_duration))
            current_text = ""
        else:  # 偶数索引是文本
            current_text = parts[i]
    
    # 添加最后一段文本（如果没有以pause结尾）
    if current_text or len(result) == 0:
        result.append((current_text, 0.0))
    
    return result

def get_dialogue_from_gui(root, generate_callback):
    root.title("广东碧桂园学校小学部中文转英语对话音频生成器")
    root.geometry("800x700") # 调整窗口大小以适应新功能

    dialogue_data_storage = [] # 用于在 GUI 内部存储解析后的对话数据
    
    # 存储每个说话者的音色选择
    speaker_voice_vars = {
        'A': tk.StringVar(value="Guy (美式, 默认)"), # 默认值
        'B': tk.StringVar(value="Jenny (美式, 默认)"), # 默认值
        'C': tk.StringVar(value="Christopher (美式)"), # 默认值
        'D': tk.StringVar(value="Ana (美式)"), # 默认值
    }
    
    # 添加合并选项变量
    merge_option_var = tk.BooleanVar(value=False)  # 默认不合并
    merged_filename_var = tk.StringVar(value="merged_output.wav")  # 合并后的文件名

    # 扁平化可用的音色列表，用于下拉菜单
    all_voice_names = []
    voice_id_map = {}
    for category, voices_in_category in available_voices.items():
        for name, voice_id in voices_in_category.items():
            all_voice_names.append(name)
            voice_id_map[name] = voice_id
    all_voice_names.sort()

    # 状态显示变量和标签
    status_message = tk.StringVar(value="")
    filename_format_var = tk.StringVar(value="{index}_{speaker}.wav") # 更新：移除 text_preview

    def async_generate_wrapper(dialogue_list, status_callback, filename_format, root_instance, button):
        async def _run_generation():
            await generate_individual_audios(dialogue_list, status_callback, filename_format, root_instance)
            # 生成完成后，重新启用按钮，更新最终状态
            root_instance.after(0, lambda: button.config(state=tk.NORMAL))
            root_instance.after(0, lambda: status_callback("完成！所有音频文件已生成。"))

        # 适配 Python 3.13: 在新线程中创建并运行新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_run_generation())
        loop.close()

    def preview_audio():
        """试听功能 - 使用对话内容中所有说话者的文本和音色"""
        # 获取对话内容
        dialogue_text = text_area.get("1.0", tk.END).strip()
        lines = [L.strip() for L in dialogue_text.splitlines() if L.strip()]
        
        # 解析所有说话者及其文本
        preview_texts = []
        for line in lines:
            if ":" in line:
                speaker_part, text_part = line.split(":", 1)
                speaker = speaker_part.strip().upper()
                text = text_part.strip()
                if text:  # 确保文本不为空
                    # 使用对应说话者的音色设置
                    selected_voice_name = speaker_voice_vars[speaker].get()
                    voice_id = voice_id_map.get(selected_voice_name, "en-US-JennyNeural")
                    preview_texts.append((text, voice_id))
        
        # 如果在对话中没有找到说话者，则使用默认文本
        if not preview_texts:
            preview_texts.append(("Hello, this is a preview.", "en-US-JennyNeural"))
            
        def run_preview():
            try:
                temp_files = []
                # 为每个说话者创建临时音频文件
                for i, (text, voice_id) in enumerate(preview_texts):
                    # 处理文本中的pause标记
                    text_segments = process_text_with_pause(text)
                    
                    segment_files = []
                    for segment_text, pause_duration in text_segments:
                        if segment_text:  # 如果有文本内容
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                                temp_filename = tmp_file.name
                                segment_files.append(temp_filename)
                            
                            # 生成音频
                            async def generate_preview():
                                tts = edge_tts.Communicate(segment_text, voice=voice_id)
                                await tts.save(temp_filename)
                            
                            # 在新线程中运行异步函数
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(generate_preview())
                            loop.close()
                        
                        # 如果有停顿时长，添加静音文件
                        if pause_duration > 0:
                            silence_file = create_silence(pause_duration)
                            segment_files.append(silence_file)
                    
                    # 合并该说话者的多个片段
                    if len(segment_files) > 1:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                            merged_filename = tmp_file.name
                        
                        if merge_wav_files(segment_files, merged_filename):
                            temp_files.append(merged_filename)
                        else:
                            # 如果合并失败，只添加第一个文件
                            temp_files.append(segment_files[0])
                        
                        # 清理片段文件
                        for segment_file in segment_files:
                            try:
                                os.unlink(segment_file)
                            except:
                                pass
                    elif len(segment_files) == 1:
                        temp_files.append(segment_files[0])
                
                # 更新状态
                status_message.set("正在播放试听音频...")
                root.update()
                
                # 按顺序播放所有音频
                all_played = True
                for temp_filename in temp_files:
                    if not play_audio_file(temp_filename):
                        all_played = False
                
                if all_played:
                    status_message.set("试听完成")
                else:
                    status_message.set("试听完成（部分无法播放）")
                
                # 清理临时文件
                for temp_filename in temp_files:
                    try:
                        os.unlink(temp_filename)
                    except:
                        pass
                    
            except Exception as e:
                error_msg = f"试听失败: {e}"
                print(error_msg)
                status_message.set(error_msg)
                messagebox.showerror("错误", error_msg)
        
        # 在新线程中运行试听功能，避免阻塞GUI
        threading.Thread(target=run_preview, daemon=True).start()

    def on_generate_button_click():
        nonlocal dialogue_data_storage

        # 收集用户选择的音色
        selected_voices_map = {}
        for speaker, var in speaker_voice_vars.items():
            selected_voice_name = var.get()
            selected_voices_map[speaker] = voice_id_map.get(selected_voice_name, "en-US-JennyNeural") # 默认女声

        dialogue_text = text_area.get("1.0", tk.END).strip()
        if not dialogue_text:
            messagebox.showwarning("输入错误", "对话内容不能为空！")
            return

        lines = [L.strip() for L in dialogue_text.splitlines() if L.strip()]
        parsed_dialogue = []
        for line in lines:
            if ":" in line:
                speaker_part, text_part = line.split(":", 1)
                speaker = speaker_part.strip().upper()
                text = text_part.strip()
                if text: # 确保文本不为空
                    voice_id_for_speaker = selected_voices_map.get(speaker, "en-US-JennyNeural") # 未预设的角色使用默认女声
                    parsed_dialogue.append((speaker, text, voice_id_for_speaker))  # 添加音色ID
                else:
                    print(f"⚠️ 忽略空句子对话行: {line}")
            else:
                print(f"⚠️ 忽略格式不正确的对话行 (缺少冒号): {line}")
        
        if not parsed_dialogue:
            messagebox.showwarning("输入错误", "没有解析到有效的对话内容。请确保每行格式为 'A: 句子' 或 'B: 句子'。")
            return

        dialogue_data_storage = parsed_dialogue
        
        # 获取用户定义的文件命名格式
        custom_filename_format = filename_format_var.get()
        
        # 获取合并选项
        merge_option = merge_option_var.get()
        merged_filename = merged_filename_var.get()

        # 开始生成，禁用按钮，更新状态
        generate_button.config(state=tk.DISABLED)
        status_message.set("开始生成音频...")
        # 将自定义的文件命名格式和 root_instance 传递给生成函数
        generate_callback(dialogue_data_storage, status_message.set, custom_filename_format, generate_button, merge_option, merged_filename)

    # 试听功能区域
    preview_frame = tk.LabelFrame(root, text="试听功能")
    preview_frame.pack(padx=10, pady=5, fill="x")
    
    # 试听按钮
    preview_control_frame = tk.Frame(preview_frame)
    preview_control_frame.pack(fill="x", padx=5, pady=2)
    tk.Label(preview_control_frame, text="试听对话中所有说话者:").pack(side=tk.LEFT)
    tk.Button(preview_control_frame, text="试听", command=preview_audio, bg="lightblue").pack(side=tk.LEFT, padx=(5, 0))

    # 音色选择区域
    voice_selection_frame = tk.LabelFrame(root, text="选择说话者音色 (A, B, C, D)")
    voice_selection_frame.pack(padx=10, pady=5, fill="x")

    # 创建一个框架用于容纳两个行
    row_frame = tk.Frame(voice_selection_frame)
    row_frame.grid(row=0, column=0, padx=5, pady=2, sticky="w")

    # 第一行：A 和 B 的音色选择
    for i, speaker_char in enumerate(['A', 'B']):
        frame = tk.Frame(row_frame)
        frame.grid(row=0, column=i, padx=5, pady=2, sticky="w")
        
        tk.Label(frame, text=f"{speaker_char} 的音色:").pack(side=tk.LEFT)
        # 下拉菜单
        option_menu = tk.OptionMenu(frame, speaker_voice_vars[speaker_char], *all_voice_names)
        option_menu.pack(side=tk.LEFT, fill="x", expand=True)

    # 第二行：C 和 D 的音色选择
    row_frame2 = tk.Frame(voice_selection_frame)
    row_frame2.grid(row=1, column=0, padx=5, pady=2, sticky="w")

    for i, speaker_char in enumerate(['C', 'D']):
        frame = tk.Frame(row_frame2)
        frame.grid(row=0, column=i, padx=5, pady=2, sticky="w")
        
        tk.Label(frame, text=f"{speaker_char} 的音色:").pack(side=tk.LEFT)
        # 下拉菜单
        option_menu = tk.OptionMenu(frame, speaker_voice_vars[speaker_char], *all_voice_names)
        option_menu.pack(side=tk.LEFT, fill="x", expand=True)
    
    # 合并选项区域
    merge_frame = tk.LabelFrame(root, text="合并选项")
    merge_frame.pack(padx=10, pady=5, fill="x")
    
    tk.Checkbutton(merge_frame, text="合并所有音频为一个文件", variable=merge_option_var).pack(anchor="w", padx=5, pady=2)
    
    merged_filename_frame = tk.Frame(merge_frame)
    merged_filename_frame.pack(fill="x", padx=5, pady=2)
    tk.Label(merged_filename_frame, text="合并后的文件名:").pack(side=tk.LEFT)
    tk.Entry(merged_filename_frame, textvariable=merged_filename_var, width=30).pack(side=tk.LEFT, fill="x", expand=True)

    # 对话输入区域
    tk.Label(root, text="请粘贴对话内容（每行一个对话条目，例如 'A: Hello'）：").pack(pady=5)
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=15) 
    text_area.pack(padx=10, pady=5)
    text_area.insert("1.0", "A: 一段对话内容\nB: 另一段对话内容") # 更新默认示例对话

    # 文件命名格式输入框
    filename_format_frame = tk.LabelFrame(root, text="音频文件命名格式")
    filename_format_frame.pack(padx=10, pady=5, fill="x")
    tk.Entry(filename_format_frame, textvariable=filename_format_var, width=60).pack(padx=5, pady=2, fill="x", expand=True)
    tk.Label(filename_format_frame, text="可用占位符: {index}, {speaker}").pack(padx=5, pady=2, anchor="w") # 更新说明

    # 生成按钮
    generate_button = tk.Button(root, text="生成单独音频文件", command=on_generate_button_click, font=("微软雅黑", 12))
    generate_button.pack(pady=10)

    # 状态显示标签
    status_label = tk.Label(root, textvariable=status_message, fg="blue", font=("微软雅黑", 10))
    status_label.pack(pady=5)


async def generate_individual_audios(dialogue_list, status_callback=None, filename_format="{index}_{speaker}.wav", root_instance=None, merge_files=False, merged_filename="merged_output.wav"):
    generated_files = []

    for i, (speaker, text, voice_id) in enumerate(dialogue_list):
        current_status = f"正在生成第 {i + 1} / {len(dialogue_list)} 段音频 (说话者: {speaker})"
        if status_callback and root_instance:
            root_instance.after(0, lambda msg=current_status: status_callback(msg)) # 在主线程更新 GUI

        # 构建文件名
        # 移除了 text_preview 的生成
        formatted_filename = filename_format.format(
            index=i + 1,
            speaker=speaker,
            # text_preview=text_preview # 移除了 text_preview
        )
        filename = formatted_filename # 使用自定义格式的文件名

        print(f"✅ {current_status} 文件名: {filename} (文本: '{text[:30]}...', 音色: {voice_id})")
        try:
            # 处理文本中的pause标记
            text_segments = process_text_with_pause(text)
            
            # 如果有pause标记，需要生成多个音频片段并合并
            if len(text_segments) > 1 or (len(text_segments) == 1 and text_segments[0][1] > 0):
                # 生成多个片段
                segment_files = []
                for segment_text, pause_duration in text_segments:
                    if segment_text:  # 如果有文本内容
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                            temp_filename = tmp_file.name
                        
                        # 生成音频
                        tts = edge_tts.Communicate(segment_text, voice=voice_id)
                        await tts.save(temp_filename)
                        segment_files.append(temp_filename)
                    
                    # 如果有停顿时长，添加静音文件
                    if pause_duration > 0:
                        silence_file = create_silence(pause_duration)
                        segment_files.append(silence_file)
                
                # 合并所有片段
                if merge_wav_files(segment_files, filename):
                    print(f"✅ 音频文件生成成功: {filename}")
                    generated_files.append(filename)
                else:
                    print(f"❌ 合并音频文件失败: {filename}")
                
                # 清理临时片段文件
                for segment_file in segment_files:
                    try:
                        os.unlink(segment_file)
                    except:
                        pass
            else:
                # 没有pause标记，直接生成音频
                tts = edge_tts.Communicate(text, voice=voice_id)
                await tts.save(filename) # 修正缩进
                print(f"✅ 音频文件生成成功: {filename}")
                generated_files.append(filename)
        except Exception as e:
            error_msg = f"❌ 生成音频文件失败 {filename}: {e}"
            print(error_msg)
            if status_callback and root_instance:
                root_instance.after(0, lambda msg=error_msg: status_callback(msg)) # 在主线程更新 GUI
            messagebox.showerror("错误", error_msg)
            # 如果一个文件生成失败，我们仍然继续尝试其他文件，但会显示错误。

    # 如果选择了合并选项，则合并所有音频文件
    if merge_files and generated_files:
        merge_status = f"正在合并 {len(generated_files)} 个音频文件..."
        if status_callback and root_instance:
            root_instance.after(0, lambda msg=merge_status: status_callback(msg))
        print(merge_status)
        
        # 使用自定义函数合并音频文件
        success = merge_wav_files(generated_files, merged_filename)
        
        if success:
            merge_success_msg = f"✅ 音频合并成功: {merged_filename}"
            print(merge_success_msg)
            if status_callback and root_instance:
                root_instance.after(0, lambda msg=merge_success_msg: status_callback(msg))
        else:
            error_msg = "❌ 音频合并失败"
            print(error_msg)
            if status_callback and root_instance:
                root_instance.after(0, lambda msg=error_msg: status_callback(msg))
            messagebox.showerror("错误", "音频合并失败")

    if generated_files:
        final_msg = f"🎉 已生成 {len(generated_files)} 个单独的音频文件。\n文件将保存在脚本所在目录。"
        if merge_files:
            final_msg += f"\n已合并为文件: {merged_filename}"
        if status_callback and root_instance:
            root_instance.after(0, lambda msg=final_msg: status_callback(msg)) # 在主线程更新 GUI
        # messagebox.showinfo("完成", final_msg) # 移除提示框
        print(final_msg)
    else:
        final_msg = "提示", "没有生成任何音频文件。"
        if status_callback and root_instance:
            root_instance.after(0, lambda msg=final_msg: status_callback(msg)) # 在主线程更新 GUI
        # messagebox.showwarning("提示", "没有生成任何音频文件。") # 移除提示框

    return generated_files


async def main(): # 确保 main 函数是 async
    root = tk.Tk() # 在 main 函数中创建 root

    def start_generation_from_gui(dialogue_list, status_set_callback, custom_filename_format, generate_button, merge_option=False, merged_filename="merged_output.wav"): # 添加 generate_button 参数
        # 启动生成过程，确保在单独线程中进行
        def async_generate_wrapper(dialogue_list, status_callback, filename_format, root_instance, button, merge_files, merged_filename):
            async def _run_generation():
                await generate_individual_audios(dialogue_list, status_callback, filename_format, root_instance, merge_files, merged_filename)
                # 生成完成后，重新启用按钮，更新最终状态
                root_instance.after(0, lambda: button.config(state=tk.NORMAL))
                root_instance.after(0, lambda: status_callback("完成！所有音频文件已生成。"))

            # 适配 Python 3.13: 在新线程中创建并运行新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_run_generation())
            loop.close()
        
        threading.Thread(target=async_generate_wrapper, args=(dialogue_list, status_set_callback, custom_filename_format, root, generate_button, merge_option, merged_filename)).start()

    get_dialogue_from_gui(root, start_generation_from_gui) # 将 root 和回调函数传递给 GUI 配置函数
    root.mainloop() # 让 Tkinter GUI 持续运行
if __name__ == "__main__":
    # 直接运行 main 函数
    asyncio.run(main())