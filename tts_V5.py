import sys
import asyncio
import os
import datetime
import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
from tkinter import ttk  # Import ttk for themed widgets
import threading
import wave
import numpy as np
import tempfile
import re
import shutil 

# Suppress Tkinter image warnings
try:
    os.environ['TK_SILENCE_DEPRECATION'] = '1'
except:
    pass

# Optional audio processing libraries
try:
    import librosa
    import soundfile as sf
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    print("Warning: Audio processing libraries (librosa, soundfile) are missing. [pause_X] segments will not be generated, and WAV merging may be limited.")

# VENV check (retained for developer warning)
expected_venv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'venv311'))
python_exe_in_venv = os.path.join(expected_venv_path, 'Scripts', 'python.exe')
if expected_venv_path not in sys.executable:
    print("Warning: You do not seem to be using the project's virtual environment.")
    print(f"Recommended command: {python_exe_in_venv} {os.path.abspath(__file__)}")
    print()

import edge_tts

# --- Global Variables ---
global_stop_event = threading.Event()
is_generating = False 

# Available Voices (Keeping Chinese descriptions for internal structure)
available_voices = {
    "US English - Male": {
        "Andrew (US)": "en-US-AndrewNeural",
        "Brian (US)": "en-US-BrianNeural",
        "Christopher (US)": "en-US-ChristopherNeural",
        "Roger (US)": "en-US-RogerNeural",
        "Steffan (US)": "en-US-SteffanNeural",
        "Guy (US, Default)": "en-US-GuyNeural",
    },
    "US English - Female": {
        "Ana (US)": "en-US-AnaNeural",
        "Aria (US)": "en-US-AriaNeural",
        "Ava (US)": "en-US-AvaNeural",
        "Jenny (US, Default)": "en-US-JennyNeural",
        "Michelle (US)": "en-US-MichelleNeural",
    },
    "UK English - Male": {
        "Libby (UK)": "en-GB-LibbyNeural",
        "Ryan (UK, Default)": "en-GB-RyanNeural",
    },
    "UK English - Female": {
        "Sonia (UK)": "en-GB-SoniaNeural",
        "Maisie (UK)": "en-GB-MaisieNeural",
    },
}

# --- Audio Processing Functions (Minor updates for clarity) ---

def merge_wav_files(file_list, output_filename):
    if not file_list: return False
    try:
        if AUDIO_PROCESSING_AVAILABLE:
            import librosa 
            import soundfile as sf
            combined_audio = np.array([], dtype=np.float32)
            sample_rate = None
            for f in file_list:
                data, sr = librosa.load(f, sr=24000) 
                if sample_rate is None: sample_rate = sr
                combined_audio = np.concatenate([combined_audio, data])
            sf.write(output_filename, combined_audio, sample_rate)
            return True
        else:
            with wave.open(file_list[0], 'rb') as first_wav:
                params = first_wav.getparams()
                frames = first_wav.readframes(first_wav.getnframes())
            with wave.open(output_filename, 'wb') as out_wav:
                out_wav.setparams(params)
                out_wav.writeframes(frames)
                for f in file_list[1:]:
                    try:
                        with wave.open(f, 'rb') as w:
                            if w.getparams()[:4] == params[:4]: 
                                out_wav.writeframes(w.readframes(w.getnframes()))
                    except: continue
            return True
    except Exception as e:
        print(f"Error merging files: {e}")
        return False

def create_silence_wav(seconds, sample_rate=24000):
    if not AUDIO_PROCESSING_AVAILABLE:
        return None 
        
    num_samples = int(sample_rate*seconds)
    silence = np.zeros(num_samples, dtype=np.float32) 
    
    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        import soundfile as sf
        sf.write(tmp_file.name, silence, sample_rate)
        return tmp_file.name
    except Exception as e:
        print(f"Error creating silence file: {e}")
        os.unlink(tmp_file.name)
        return None

def get_voice_display_name(voice_id):
    for category, voices_in_category in available_voices.items():
        for name, vid in voices_in_category.items():
            if vid == voice_id:
                # Returns name part (e.g., "Ryan")
                return name.split(" (")[0]
    return voice_id

async def generate_individual_audios(dialogue_list, status_callback=None, output_dir=".", 
                                     filename_format="{index}_{speaker}.wav", root_instance=None, 
                                     merge_files=False, merged_filename="merged_output.wav", 
                                     voice_id_map=None, stop_event=None, delete_singles=True): # ADDED delete_singles
    
    global is_generating
    is_generating = True
    generated_files = []
    total = len(dialogue_list)
    pause_pattern = re.compile(r"\[pause_(\d+(\.\d+)?)\]")
    
    if not os.path.isdir(output_dir):
        try:
             os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            if status_callback and root_instance:
                 root_instance.after(0, lambda msg=f"❌ Error: Cannot create output directory {output_dir}. {e}": status_callback(msg))
            is_generating = False
            return generated_files

    for i, (speaker, text, voice_id) in enumerate(dialogue_list):
        if stop_event and stop_event.is_set():
            if status_callback and root_instance:
                root_instance.after(0, lambda msg="🚫 Generation manually stopped.": status_callback(msg))
            is_generating = False
            return generated_files

        display_name = get_voice_display_name(voice_id)
        
        if status_callback and root_instance:
            root_instance.after(0, lambda msg=f"🟡 Generating audio {i+1}/{total} (Speaker:{display_name})": status_callback(msg))

        segments = []
        last_index = 0
        
        for m in pause_pattern.finditer(text):
            start, end = m.span()
            if start>last_index: segments.append(("text", text[last_index:start]))
            segments.append(("pause", float(m.group(1))))
            last_index = end
        if last_index < len(text): segments.append(("text", text[last_index:]))

        temp_files_for_line = []
        for j, (seg_type, value) in enumerate(segments):
            if stop_event and stop_event.is_set():
                if status_callback and root_instance:
                    root_instance.after(0, lambda msg="🚫 Generation manually stopped.": status_callback(msg))
                is_generating = False
                return generated_files

            if seg_type=="text" and value.strip():
                temp_filename = os.path.join(tempfile.gettempdir(), f"tts_{os.urandom(6).hex()}_seg{j}.wav")
                
                try:
                    tts = edge_tts.Communicate(value.strip(), voice=voice_id)
                    await tts.save(temp_filename)
                    temp_files_for_line.append(temp_filename)
                except Exception as e:
                    print(f"TTS Generation Error: {e}")
                    if status_callback and root_instance:
                        root_instance.after(0, lambda msg=f"❌ Error: Audio generation failed for line {i+1} text '{value.strip()}'.": status_callback(msg))
                    for f in temp_files_for_line:
                        if os.path.exists(f): os.unlink(f)
                    temp_files_for_line = []
                    break
                    
            elif seg_type=="pause":
                silence_file = create_silence_wav(value)
                if silence_file:
                    temp_files_for_line.append(silence_file)
                elif status_callback and root_instance:
                    root_instance.after(0, lambda msg=f"⚠️ Warning: Cannot generate silence for line {i+1}. Please install librosa/soundfile.": status_callback(msg))

        
        if not temp_files_for_line: continue

        safe_speaker_name = re.sub(r'[^\w\s-]', '', display_name.replace(' ', '_'))
        output_line_file_base = filename_format.format(index=i+1, speaker=safe_speaker_name)
        output_line_file = os.path.join(output_dir, output_line_file_base)
        
        if len(temp_files_for_line)>1:
            if merge_wav_files(temp_files_for_line, output_line_file):
                generated_files.append(output_line_file)
            for f in temp_files_for_line:
                if os.path.exists(f): 
                    try: os.unlink(f)
                    except OSError: pass
        elif len(temp_files_for_line) == 1:
            old_filename = temp_files_for_line[0]
            if os.path.exists(old_filename):
                try:
                    os.rename(old_filename, output_line_file)
                    generated_files.append(output_line_file)
                except OSError:
                    try:
                        shutil.copy(old_filename, output_line_file)
                        os.unlink(old_filename)
                        generated_files.append(output_line_file)
                    except Exception as e:
                        print(f"File move/copy failed: {e}")
                        
        if status_callback and root_instance:
            root_instance.after(0, lambda msg=f"🟢 Completed audio {i+1}/{total}": status_callback(msg))

    if stop_event and stop_event.is_set():
        if status_callback and root_instance:
            root_instance.after(0, lambda msg="🚫 Generation manually stopped.": status_callback(msg))
        is_generating = False
        return generated_files

    if merge_files and generated_files:
        if status_callback and root_instance:
            root_instance.after(0, lambda msg="🔄 Merging all audio files...": status_callback(msg))
        
        final_merged_file = os.path.join(output_dir, merged_filename)
        merge_success = merge_wav_files(generated_files, final_merged_file)
        
        if merge_success and status_callback and root_instance:
            
            # --- New logic: Check delete_singles flag ---
            if delete_singles:
                files_to_keep = [final_merged_file]
                temp_list = list(generated_files) 
                for f in temp_list:
                    if f not in files_to_keep and os.path.exists(f):
                        try: 
                            os.unlink(f)
                            generated_files.remove(f)
                        except OSError: pass 
                cleanup_status = "and singles deleted"
            else:
                cleanup_status = "and singles kept"
            
            root_instance.after(0, lambda msg=f"🎉 All {total} audios generated and merged to {os.path.basename(final_merged_file)} ({cleanup_status}) in {output_dir}!": status_callback(msg))
            
    else:
        if status_callback and root_instance:
            root_instance.after(0, lambda msg=f"🎉 All {total} audios generated to directory: {output_dir}": status_callback(msg))

    is_generating = False
    return generated_files

# --- GUI Function (Major Changes for English and new Feature) ---
def get_dialogue_from_gui(root, generate_callback, stop_callback):
    
    global global_stop_event
    global is_generating
    
    # --- GUI Style Configuration ---
    root.title("TTS Dialogue Audio Generator (Edge-TTS)")
    root.geometry("850x850") 
    root.resizable(False, False)
    
    # Define Tahoma font for consistency and clarity
    FONT_TAHOMA = ("Tahoma", 10)
    FONT_TAHOMA_BOLD = ("Tahoma", 10, "bold")
    FONT_TAHOMA_STATUS = ("Tahoma", 12, "bold")
    
    style = ttk.Style(root)
    style.theme_use('clam') 
    
    # Apply Tahoma to ttk widgets
    style.configure("TLabel", font=FONT_TAHOMA)
    style.configure("TCheckbutton", font=FONT_TAHOMA)
    style.configure("TButton", font=FONT_TAHOMA_BOLD)
    style.configure("TLabelframe.Label", font=FONT_TAHOMA_BOLD)
    
    # --- Variable Setup ---
    default_output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "TTS_Output")
    output_dir_var = tk.StringVar(value=default_output_dir)
    
    speaker_voice_vars = {
        'A': tk.StringVar(value="Ryan (UK, Default)"),
        'B': tk.StringVar(value="Jenny (US, Default)"),
        'C': tk.StringVar(value="Christopher (US)"),
        'D': tk.StringVar(value="Ana (US)"),
        'E': tk.StringVar(value="Libby (UK)"),
        'F': tk.StringVar(value="Guy (US, Default)"),
    }
    
    now = datetime.datetime.now()
    default_filename = now.strftime("%Y%m%d_%H%M")
    merge_option_var = tk.BooleanVar(value=True) 
    # NEW: Control deletion of single files after successful merge
    delete_singles_var = tk.BooleanVar(value=True) 
    
    merged_filename_var = tk.StringVar(value=f"{default_filename}_merged.wav")
    filename_format_var = tk.StringVar(value="{index}_{speaker}.wav")
    status_message = tk.StringVar(value="Ready...") 

    all_voice_names = []
    voice_id_map = {}
    for category, voices_in_category in available_voices.items():
        all_voice_names.append(f"--- {category} ---") 
        for name, vid in voices_in_category.items():
            all_voice_names.append(name)
            voice_id_map[name] = vid

    # --- Layout Frames ---
    main_frame = ttk.Frame(root, padding="10 10 10 10")
    main_frame.pack(fill="both", expand=True)
    main_frame.grid_columnconfigure(0, weight=1)
    
    # --- 1. Input Area ---
    input_frame = ttk.LabelFrame(main_frame, text="✏️ Dialogue Input (Format: A: Hello [pause_2])", padding="10")
    input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
    
    # Apply Tahoma to ScrolledText
    text_area = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, width=90, height=12, 
                                          font=("Tahoma", 10)) 
    text_area.pack(fill="both", expand=True)
    text_area.insert("1.0", "A: Hello, Welcome to the listening sample test.\nB: This is not a listening test. It's just an example.\nC: What about the others?\nD: They are coming soon.\nE: I can't wait.\nF: Let's start with A and B first.")

    # --- 2. Voice Selection Area ---
    voice_frame = ttk.LabelFrame(main_frame, text="🎤 Select Speaker Voice (A, B, C, D, E, F)", padding="10")
    voice_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
    
    voice_speakers = ['A', 'B', 'C', 'D', 'E', 'F']
    
    for idx, s in enumerate(voice_speakers):
        row = idx // 2 
        col = (idx % 2) * 2 
        
        ttk.Label(voice_frame, text=f"Speaker {s} Voice:").grid(row=row, column=col, padx=(10, 5), pady=5, sticky="w")
        
        voice_combobox = ttk.Combobox(voice_frame, textvariable=speaker_voice_vars[s], 
                                     values=all_voice_names, state="readonly", width=25,
                                     font=FONT_TAHOMA) # Apply Tahoma
        voice_combobox.grid(row=row, column=col+1, padx=(0, 20), pady=5, sticky="w")
        
    # --- 3. Output Settings Area ---
    output_setting_frame = ttk.LabelFrame(main_frame, text="📁 Output Settings", padding="10")
    output_setting_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
    
    output_frame = ttk.Frame(output_setting_frame)
    output_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky="ew")
    output_frame.grid_columnconfigure(1, weight=1)
    
    ttk.Label(output_frame, text="Save Directory:").grid(row=0, column=0, padx=5, sticky="w")
    # Apply Tahoma to Entry
    ttk.Entry(output_frame, textvariable=output_dir_var, font=FONT_TAHOMA).grid(row=0, column=1, sticky="ew", padx=5)
    
    def select_output_dir():
        directory = filedialog.askdirectory(title="Select Audio Save Directory")
        if directory:
            output_dir_var.set(directory)
            
    ttk.Button(output_frame, text="Select Folder", command=select_output_dir).grid(row=0, column=2, padx=5)
    
    # Merge option
    merge_check = ttk.Checkbutton(output_setting_frame, text="Merge all audios into one file (Recommended)", 
                                  variable=merge_option_var)
    merge_check.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")
    
    # NEW: Delete singles option
    delete_singles_check = ttk.Checkbutton(output_setting_frame, text="Delete single files after successful merge", 
                                  variable=delete_singles_var)
    delete_singles_check.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")
    
    # Merged Filename
    ttk.Label(output_setting_frame, text="Merged Filename:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
    ttk.Entry(output_setting_frame, textvariable=merged_filename_var, width=40, font=FONT_TAHOMA).grid(row=3, column=1, sticky="w", padx=5)
    
    # Single Filename Format
    ttk.Label(output_setting_frame, text="Single File Format ({index}, {speaker}):").grid(row=4, column=0, padx=5, pady=5, sticky="w")
    ttk.Entry(output_setting_frame, textvariable=filename_format_var, width=40, font=FONT_TAHOMA).grid(row=4, column=1, sticky="w", padx=5)

    # --- 4. Status and Operation Area ---
    
    # Status Label
    progress_label = ttk.Label(main_frame, textvariable=status_message, 
                               font=FONT_TAHOMA_STATUS, foreground="blue") # Use custom status font
    progress_label.grid(row=3, column=0, padx=10, pady=(15, 5), sticky="ew")

    # Button Frame
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=4, column=0, padx=10, pady=(0, 10))
    
    def on_stop_button_click():
        if is_generating:
            global_stop_event.set()
            stop_button.config(state=tk.DISABLED)
            status_message.set("🚫 Requesting generation stop, please wait...")
            progress_label.config(foreground="red")
            
    stop_button = ttk.Button(button_frame, text="⏹️ STOP Generation", command=on_stop_button_click, state=tk.DISABLED)
    stop_button.pack(side=tk.RIGHT, padx=10)
    
    def on_generate_button_click():
        global is_generating
        
        if is_generating: return
        
        output_dir = output_dir_var.get()
        # Basic directory validation
        if not output_dir or not os.path.isdir(os.path.dirname(output_dir) if output_dir.endswith(('.wav', '.mp3')) else output_dir):
            messagebox.showwarning("Output Directory Error", "Please select a valid output directory!")
            return
            
        dialogue_text = text_area.get("1.0", tk.END).strip()
        if not dialogue_text: 
            messagebox.showwarning("Input Error", "Dialogue content cannot be empty!")
            return
            
        lines = [L.strip() for L in dialogue_text.splitlines() if L.strip()]
        parsed = []
        selected_voices = {s: voice_id_map.get(v.get(), "en-US-JennyNeural") 
                          for s,v in speaker_voice_vars.items()}
                          
        for line in lines:
            if ":" in line:
                s,t = line.split(":",1)
                speaker_id = s.strip().upper()
                if speaker_id in speaker_voice_vars:
                    voice_to_use = selected_voices.get(speaker_id, "en-US-JennyNeural")
                    parsed.append((speaker_id, t.strip(), voice_to_use))
                else:
                    messagebox.showwarning("Input Warning", f"Skipping invalid Speaker ID: {speaker_id}")
            
        if not parsed: 
            messagebox.showwarning("Input Error", "No valid dialogue parsed (Ensure format: Speaker: Text, and Speaker is A-F)")
            return
            
        global_stop_event.clear()
        is_generating = True
        generate_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)
        
        def progress_callback(msg):
            status_message.set(msg)
            
            if "🎉" in msg or "🚫" in msg or "❌" in msg:
                global_stop_event.clear()
                generate_button.config(state=tk.NORMAL)
                stop_button.config(state=tk.DISABLED)
                
                if "🎉" in msg:
                    progress_label.config(foreground="green")
                    messagebox.showinfo("Generation Complete", msg)
                elif "🚫" in msg:
                    progress_label.config(foreground="red")
                elif "❌" in msg:
                    progress_label.config(foreground="red")
                    messagebox.showerror("Generation Error", msg)
            elif "🟡" in msg or "🔄" in msg:
                progress_label.config(foreground="orange")
            elif "🟢" in msg:
                progress_label.config(foreground="green")
            else:
                progress_label.config(foreground="blue")
                
        progress_text = f"Total {len(parsed)} audios to generate. Starting..."
        progress_callback(progress_text)

        # Pass the new 'delete_singles_var' value to the worker function
        generate_callback(parsed, progress_callback, output_dir_var.get(), filename_format_var.get(), 
                          merge_option_var.get(), merged_filename_var.get(), voice_id_map, root, global_stop_event, 
                          delete_singles_var.get())

    generate_button = ttk.Button(button_frame, text="▶️ GENERATE Audio", style="TButton", command=on_generate_button_click)
    generate_button.pack(side=tk.LEFT, padx=10)


# Start Generation Thread
def start_gui_generation(dialogue_list, status_set_callback, output_dir, filename_format, merge_option, 
                         merged_filename, voice_id_map=None, root_instance=None, stop_event=None, delete_singles=True): # ADDED delete_singles
    def wrapper():
        asyncio.run(generate_individual_audios(dialogue_list, status_set_callback, output_dir, filename_format, 
                                               root_instance=root_instance, merge_files=merge_option, 
                                               merged_filename=merged_filename, voice_id_map=voice_id_map, 
                                               stop_event=stop_event, delete_singles=delete_singles)) # Pass the flag
    threading.Thread(target=wrapper, daemon=True).start()

# Main function
async def main():
    root = tk.Tk()
    get_dialogue_from_gui(root, start_gui_generation, global_stop_event.set)
    root.mainloop()

if __name__=="__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    default_output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "TTS_Output")
    if not os.path.exists(default_output_dir):
        try:
            os.makedirs(default_output_dir, exist_ok=True)
        except OSError:
            pass

    asyncio.run(main())