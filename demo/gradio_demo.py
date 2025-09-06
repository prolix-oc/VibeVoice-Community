"""
VibeVoice Gradio Demo - High-Quality Dialogue Generation Interface with Streaming Support
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from datetime import datetime
import threading
import numpy as np
import gradio as gr
import librosa
import resampy
import soundfile as sf
import torch
import os
import traceback

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.processor.vibevoice_tokenizer_processor import AudioNormalizer
from vibevoice.modular.streamer import AudioStreamer
from transformers.utils import logging
from transformers import set_seed

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class VibeVoiceDemo:
    def __init__(self, model_path, device="cuda", inference_steps=5):
        """Initialize the VibeVoice demo with model loading."""
        self.model_path = model_path
        self.device = device
        self.inference_steps = inference_steps
        self.is_generating = False  # Track generation state
        self.stop_generation = False  # Flag to stop generation
        self.current_streamer = None  # Track current audio streamer
        self.load_model()
        self.setup_voice_presets()
        self.load_example_scripts()  # Load example scripts
        
        # Streaming optimization parameters
        self.streaming_min_yield_interval = 30  # Seconds between streaming updates
        self.streaming_min_chunk_size_multiplier = 60  # Multiplier for minimum chunk size
        
    def load_model(self):
        """Load the VibeVoice model and processor."""
        print(f"Loading processor & model from {self.model_path}")
        # Normalize potential 'mpx'
        if self.device.lower() == "mpx":
            print("Note: device 'mpx' detected, treating it as 'mps'.")
            self.device = "mps"
        if self.device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.")
            self.device = "cpu"
        print(f"Using device: {self.device}")
        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        # Decide dtype & attention
        if self.device == "mps":
            load_dtype = torch.float32
            attn_impl_primary = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            attn_impl_primary = "flash_attention_2"
        else:
            load_dtype = torch.float32
            attn_impl_primary = "sdpa"
        print(f"Using device: {self.device}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")
        # Load model
        try:
            if self.device == "mps":
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    attn_implementation=attn_impl_primary,
                    device_map=None,
                )
                self.model.to("mps")
            elif self.device == "cuda":
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cuda",
                    attn_implementation=attn_impl_primary,
                )
            else:
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cpu",
                    attn_implementation=attn_impl_primary,
                )
        except Exception as e:
            if attn_impl_primary == 'flash_attention_2':
                print(f"[ERROR] : {type(e).__name__}: {e}")
                print(traceback.format_exc())
                fallback_attn = "sdpa"
                print(f"Falling back to attention implementation: {fallback_attn}")
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=(self.device if self.device in ("cuda", "cpu") else None),
                    attn_implementation=fallback_attn,
                )
                if self.device == "mps":
                    self.model.to("mps")
            else:
                raise e
        self.model.eval()
        
        # Use SDE solver by default
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config, 
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2'
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        
        if hasattr(self.model.model, 'language_model'):
            print(f"Language model attention: {self.model.model.language_model.config._attn_implementation}")
    
    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")
        
        # Check if voices directory exists
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return
        
        # Scan for all WAV files in the voices directory
        self.voice_presets = {}
        
        # Get all .wav files in the voices directory
        wav_files = [f for f in os.listdir(voices_dir) 
                    if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')) and os.path.isfile(os.path.join(voices_dir, f))]
        
        # Create dictionary with filename (without extension) as key
        for wav_file in wav_files:
            # Remove .wav extension to get the name
            name = os.path.splitext(wav_file)[0]
            # Create full path
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets[name] = full_path
        
        # Sort the voice presets alphabetically by name for better UI
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        
        # Filter out voices that don't exist (this is now redundant but kept for safety)
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }
        
        if not self.available_voices:
            raise gr.Error("No voice presets found. Please add .wav files to the demo/voices directory.")
        
        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        print(f"Available voices: {', '.join(self.available_voices.keys())}")
    
    def read_audio(self, audio_path, target_sr=24000):
        """Read and preprocess audio file."""
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                # Use resampy with high-quality Kaiser best resampling if available
                try:
                    import resampy
                    wav = resampy.resample(wav, sr, target_sr, filter='kaiser_best')
                except ImportError:
                    # Fallback to librosa if resampy is not available
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            return np.array([])
    
    def generate_podcast_streaming(self, 
                                 num_speakers,
                                 script,
                                 speaker_1=None,
                                 speaker_2=None,
                                 speaker_3=None,
                                 speaker_4=None,
                                 cfg_scale=1.3,
                                 **generation_params):
        try:
            
            # Reset stop flag and set generating state
            self.stop_generation = False
            self.is_generating = True
            
            # Validate inputs
            if not script.strip():
                self.is_generating = False
                raise gr.Error("Error: Please provide a script.")

            # Defend against common mistake
            script = script.replace("’", "'")
            
            if num_speakers < 1 or num_speakers > 4:
                self.is_generating = False
                raise gr.Error("Error: Number of speakers must be between 1 and 4.")
            
            # Collect selected speakers
            selected_speakers = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers]
            
            # Validate speaker selections
            for i, speaker in enumerate(selected_speakers):
                if not speaker or speaker not in self.available_voices:
                    self.is_generating = False
                    raise gr.Error(f"Error: Please select a valid speaker for Speaker {i+1}.")
            
            # Build initial log
            log = f"🎙️ Generating podcast with {num_speakers} speakers\n"
            log += f"📊 Parameters: CFG Scale={cfg_scale}, Inference Steps={self.model.ddpm_inference_steps}\n"
            log += f"⏱️ Streaming: Interval={self.streaming_min_yield_interval}s, Chunk Multiplier={self.streaming_min_chunk_size_multiplier}\n"
            log += f"🎭 Speakers: {', '.join(selected_speakers)}\n"
            
            # Check for stop signal
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            # Load voice samples
            voice_samples = []
            for speaker_name in selected_speakers:
                audio_path = self.available_voices[speaker_name]
                audio_data = self.read_audio(audio_path)
                if len(audio_data) == 0:
                    self.is_generating = False
                    raise gr.Error(f"Error: Failed to load audio for {speaker_name}")
                voice_samples.append(audio_data)
            
            # Parse script to assign speaker ID's
            lines = script.strip().split('\n')
            formatted_script_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if line already has speaker format
                if line.startswith('Speaker ') and ':' in line:
                    formatted_script_lines.append(line)
                else:
                    # Auto-assign to speakers in rotation
                    speaker_id = len(formatted_script_lines) % num_speakers
                    formatted_script_lines.append(f"Speaker {speaker_id}: {line}")
            
            formatted_script = '\n'.join(formatted_script_lines)
            log += f"📝 Formatted script with {len(formatted_script_lines)} turns\n\n"
            log += "🔄 Processing with VibeVoice (streaming mode)...\n"
            
            # Check for stop signal before processing
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            start_time = time.time()
            
            inputs = self.processor(
                text=[formatted_script],
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Move tensors to device
            target_device = self.device if self.device in ("cuda", "mps") else "cpu"
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(target_device)
            
            # Create audio streamer
            audio_streamer = AudioStreamer(
                batch_size=1,
                stop_signal=None,
                timeout=None
            )
            
            # Store current streamer for potential stopping
            self.current_streamer = audio_streamer
            
            # Start generation in a separate thread
            generation_thread = threading.Thread(
                target=self._generate_with_streamer,
                args=(inputs, cfg_scale, audio_streamer)
            )
            generation_thread.start()
            
            # Wait for generation to actually start producing audio
            time.sleep(0.1)  # Minimal wait time to reduce overhead

            # Check for stop signal after thread start
            if self.stop_generation:
                audio_streamer.end()
                generation_thread.join(timeout=5.0)  # Wait up to 5 seconds for thread to finish
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return

            # Collect audio chunks as they arrive
            sample_rate = 24000
            all_audio_chunks = []  # For final statistics
            pending_chunks = []  # Buffer for accumulating small chunks
            chunk_count = 0
            last_yield_time = time.time()
            min_yield_interval = self.streaming_min_yield_interval  # Use optimized parameter
            min_chunk_size = sample_rate * self.streaming_min_chunk_size_multiplier  # Use optimized parameter
            
            # Get the stream for the first (and only) sample
            audio_stream = audio_streamer.get_stream(0)
            
            has_yielded_audio = False
            has_received_chunks = False  # Track if we received any chunks at all
            
            for audio_chunk in audio_stream:
                # Check for stop signal in the streaming loop
                if self.stop_generation:
                    audio_streamer.end()
                    break
                    
                chunk_count += 1
                has_received_chunks = True  # Mark that we received at least one chunk
                
                # Convert tensor to numpy
                if torch.is_tensor(audio_chunk):
                    # Convert bfloat16 to float32 first, then to numpy
                    if audio_chunk.dtype == torch.bfloat16:
                        audio_chunk = audio_chunk.float()
                    audio_np = audio_chunk.cpu().numpy().astype(np.float32)
                else:
                    audio_np = np.array(audio_chunk, dtype=np.float32)
                
                # Ensure audio is 1D and properly normalized
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.squeeze()
                
                # Convert to 16-bit for Gradio
                audio_16bit = convert_to_16_bit_wav(audio_np)
                
                # Store for final statistics
                all_audio_chunks.append(audio_16bit)
                
                # Add to pending chunks buffer
                pending_chunks.append(audio_16bit)
                
                # Calculate pending audio size
                pending_audio_size = sum(len(chunk) for chunk in pending_chunks)
                current_time = time.time()
                time_since_last_yield = current_time - last_yield_time
                
                # Decide whether to yield
                should_yield = False
                if not has_yielded_audio and pending_audio_size >= min_chunk_size:
                    # First yield: wait for minimum chunk size
                    should_yield = True
                    has_yielded_audio = True
                elif has_yielded_audio and (pending_audio_size >= min_chunk_size or time_since_last_yield >= min_yield_interval):
                    # Subsequent yields: either enough audio or enough time has passed
                    should_yield = True
                
                if should_yield and pending_chunks:
                    # Concatenate and yield only the new audio chunks
                    new_audio = np.concatenate(pending_chunks)
                    new_duration = len(new_audio) / sample_rate
                    total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                    
                    log_update = log + f"🎵 Streaming: {total_duration:.1f}s generated (chunk {chunk_count})\n"
                    
                    # Yield streaming audio chunk and keep complete_audio as None during streaming
                    yield (sample_rate, new_audio), None, log_update, gr.update(visible=True)
                    
                    # Clear pending chunks after yielding
                    pending_chunks = []
                    last_yield_time = current_time
            
            # Yield any remaining chunks
            if pending_chunks:
                final_new_audio = np.concatenate(pending_chunks)
                total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                log_update = log + f"🎵 Streaming final chunk: {total_duration:.1f}s total\n"
                yield (sample_rate, final_new_audio), None, log_update, gr.update(visible=True)
                has_yielded_audio = True  # Mark that we yielded audio
            
            # Wait for generation to complete (with timeout to prevent hanging)
            generation_thread.join(timeout=1.0)  # Minimal timeout to reduce overhead

            # If thread is still alive after timeout, force end
            if generation_thread.is_alive():
                print("Warning: Generation thread did not complete within timeout")
                audio_streamer.end()
                generation_thread.join(timeout=5.0)

            # Clean up
            self.current_streamer = None
            self.is_generating = False
            
            generation_time = time.time() - start_time
            
            # Check if stopped by user
            if self.stop_generation:
                yield None, None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            # Debug logging
            # print(f"Debug: has_received_chunks={has_received_chunks}, chunk_count={chunk_count}, all_audio_chunks length={len(all_audio_chunks)}")
            
            # Check if we received any chunks but didn't yield audio
            if has_received_chunks and not has_yielded_audio and all_audio_chunks:
                # We have chunks but didn't meet the yield criteria, yield them now
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate
                
                final_log = log + f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
                final_log += f"🎵 Final audio duration: {final_duration:.2f} seconds\n"
                final_log += f"📊 Total chunks: {chunk_count}\n"
                final_log += "✨ Generation successful! Complete audio is ready.\n"
                final_log += "💡 Not satisfied? You can regenerate or adjust the CFG scale for different results."
                
                # Yield the complete audio
                yield None, (sample_rate, complete_audio), final_log, gr.update(visible=False)
                return
            
            if not has_received_chunks:
                error_log = log + f"\n❌ Error: No audio chunks were received from the model. Generation time: {generation_time:.2f}s"
                yield None, None, error_log, gr.update(visible=False)
                return
            
            if not has_yielded_audio:
                error_log = log + f"\n❌ Error: Audio was generated but not streamed. Chunk count: {chunk_count}"
                yield None, None, error_log, gr.update(visible=False)
                return

            # Prepare the complete audio
            if all_audio_chunks:
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate
                
                final_log = log + f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
                final_log += f"🎵 Final audio duration: {final_duration:.2f} seconds\n"
                final_log += f"📊 Total chunks: {chunk_count}\n"
                final_log += "✨ Generation successful! Complete audio is ready in the 'Complete Audio' tab.\n"
                final_log += "💡 Not satisfied? You can regenerate or adjust the CFG scale for different results."
                
                # Final yield: Clear streaming audio and provide complete audio
                yield None, (sample_rate, complete_audio), final_log, gr.update(visible=False)
            else:
                final_log = log + "❌ No audio was generated."
                yield None, None, final_log, gr.update(visible=False)

        except gr.Error as e:
            # Handle Gradio-specific errors (like input validation)
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ Input Error: {str(e)}"
            print(error_msg)
            yield None, None, error_msg, gr.update(visible=False)
            
        except Exception as e:
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            yield None, None, error_msg, gr.update(visible=False)
    
    def _generate_with_streamer(self, inputs, cfg_scale, audio_streamer):
        """Helper method to run generation with streamer in a separate thread."""
        try:
            # Check for stop signal before starting generation
            if self.stop_generation:
                audio_streamer.end()
                return
                
            # Define a stop check function that can be called from generate
            def check_stop_generation():
                return self.stop_generation
                
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    'do_sample': False,
                },
                audio_streamer=audio_streamer,
                stop_check_fn=check_stop_generation,  # Pass the stop check function
                verbose=False,  # Disable verbose in streaming mode
                refresh_negative=True,
            )
            
        except Exception as e:
            print(f"Error in generation thread: {e}")
            traceback.print_exc()
            # Make sure to end the stream on error
            audio_streamer.end()
    
    def stop_audio_generation(self):
        """Stop the current audio generation process."""
        self.stop_generation = True
        if self.current_streamer is not None:
            try:
                self.current_streamer.end()
            except Exception as e:
                print(f"Error stopping streamer: {e}")
        print("🛑 Audio generation stop requested")
    
    def load_example_scripts(self):
        """Load example scripts from the text_examples directory."""
        examples_dir = os.path.join(os.path.dirname(__file__), "text_examples")
        self.example_scripts = []
        
        # Check if text_examples directory exists
        if not os.path.exists(examples_dir):
            print(f"Warning: text_examples directory not found at {examples_dir}")
            return
        
        # Get all .txt files in the text_examples directory
        txt_files = sorted([f for f in os.listdir(examples_dir) 
                          if f.lower().endswith('.txt') and os.path.isfile(os.path.join(examples_dir, f))])
        
        for txt_file in txt_files:
            file_path = os.path.join(examples_dir, txt_file)
            
            import re
            # Check if filename contains a time pattern like "45min", "90min", etc.
            time_pattern = re.search(r'(\d+)min', txt_file.lower())
            if time_pattern:
                minutes = int(time_pattern.group(1))
                if minutes > 15:
                    print(f"Skipping {txt_file}: duration {minutes} minutes exceeds 15-minute limit")
                    continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    script_content = f.read().strip()
                
                # Remove empty lines and lines with only whitespace
                script_content = '\n'.join(line for line in script_content.split('\n') if line.strip())
                
                if not script_content:
                    continue
                
                # Parse the script to determine number of speakers
                num_speakers = self._get_num_speakers_from_script(script_content)
                
                # Add to examples list as [num_speakers, script_content]
                self.example_scripts.append([num_speakers, script_content])
                print(f"Loaded example: {txt_file} with {num_speakers} speakers")
                
            except Exception as e:
                print(f"Error loading example script {txt_file}: {e}")
        
        if self.example_scripts:
            print(f"Successfully loaded {len(self.example_scripts)} example scripts")
        else:
            print("No example scripts were loaded")
    
    def _get_num_speakers_from_script(self, script):
        """Determine the number of unique speakers in a script."""
        import re
        speakers = set()
        
        lines = script.strip().split('\n')
        for line in lines:
            # Use regex to find speaker patterns
            match = re.match(r'^Speaker\s+(\d+)\s*:', line.strip(), re.IGNORECASE)
            if match:
                speaker_id = int(match.group(1))
                speakers.add(speaker_id)
        
        # If no speakers found, default to 1
        if not speakers:
            return 1
        
        # Return the maximum speaker ID + 1 (assuming 0-based indexing)
        # or the count of unique speakers if they're 1-based
        max_speaker = max(speakers)
        min_speaker = min(speakers)
        
        if min_speaker == 0:
            return max_speaker + 1
        else:
            # Assume 1-based indexing, return the count
            return len(speakers)
    

def create_demo_interface(demo_instance):
    """Create the Gradio interface with streaming support."""
    
    # Custom CSS for high-end aesthetics with both light and dark mode support
    custom_css = """
    /* Base theme variables */
    :root {
        --background-primary: #f8fafc;
        --background-secondary: #e2e8f0;
        --background-card: rgba(255, 255, 255, 0.8);
        --border-color: rgba(226, 232, 240, 0.8);
        --text-primary: #1e293b;
        --text-secondary: #374151;
        --header-gradient-start: #667eea;
        --header-gradient-end: #764ba2;
        --card-gradient-start: #e2e8f0;
        --card-gradient-end: #cbd5e1;
        --button-gradient-green-start: #059669;
        --button-gradient-green-end: #0d9488;
        --button-gradient-red-start: #ef4444;
        --button-gradient-red-end: #dc2626;
        --button-gradient-gray-start: #64748b;
        --button-gradient-gray-end: #475569;
        --audio-player-bg: #f1f5f9;
        --audio-player-bg-end: #e2e8f0;
        --complete-audio-bg: #f0fdf4;
        --complete-audio-bg-end: #dcfce7;
        --queue-status-bg: #f0f9ff;
        --queue-status-bg-end: #e0f2fe;
    }
    
    /* Dark mode variables */
    @media (prefers-color-scheme: dark) {
        :root {
            --background-primary: #0f172a;
            --background-secondary: #1e293b;
            --background-card: rgba(30, 41, 59, 0.8);
            --border-color: rgba(51, 65, 85, 0.8);
            --text-primary: #f1f5f9;
            --text-secondary: #e2e8f0;
            --header-gradient-start: #4f46e5;
            --header-gradient-end: #7c3aed;
            --card-gradient-start: #334155;
            --card-gradient-end: #1e293b;
            --button-gradient-green-start: #10b981;
            --button-gradient-green-end: #0d9488;
            --button-gradient-red-start: #f87171;
            --button-gradient-red-end: #ef4444;
            --button-gradient-gray-start: #94a3b8;
            --button-gradient-gray-end: #64748b;
            --audio-player-bg: #1e293b;
            --audio-player-bg-end: #334155;
            --complete-audio-bg: #065f46;
            --complete-audio-bg-end: #047857;
            --queue-status-bg: #0c4a6e;
            --queue-status-bg-end: #0369a1;
        }
    }
    
    /* Gradio dark theme override */
    .dark {
        --background-primary: #0f172a;
        --background-secondary: #1e293b;
        --background-card: rgba(30, 41, 59, 0.8);
        --border-color: rgba(51, 65, 85, 0.8);
        --text-primary: #f1f5f9;
        --text-secondary: #e2e8f0;
        --header-gradient-start: #4f46e5;
        --header-gradient-end: #7c3aed;
        --card-gradient-start: #334155;
        --card-gradient-end: #1e293b;
        --button-gradient-green-start: #10b981;
        --button-gradient-green-end: #0d9488;
        --button-gradient-red-start: #f87171;
        --button-gradient-red-end: #ef4444;
        --button-gradient-gray-start: #94a3b8;
        --button-gradient-gray-end: #64748b;
        --audio-player-bg: #1e293b;
        --audio-player-bg-end: #334155;
        --complete-audio-bg: #065f46;
        --complete-audio-bg-end: #047857;
        --queue-status-bg: #0c4a6e;
        --queue-status-bg-end: #0369a1;
    }
    
    /* Main container */
    .gradio-container {
        background: var(--background-primary);
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling - more unified appearance */
    .main-header {
        background: linear-gradient(90deg, var(--header-gradient-start) 0%, var(--header-gradient-end) 100%);
        padding: 2.25rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Card styling - more unified appearance */
    .settings-card, .generation-card {
        background: var(--background-card);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 1.75rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Speaker selection styling */
    .speaker-grid {
        display: grid;
        gap: 1.25rem;
        margin-bottom: 1.25rem;
        position: relative;
    }
    
    .speaker-item {
        background: linear-gradient(135deg, var(--card-gradient-start) 0%, var(--card-gradient-end) 100%);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 16px;
        padding: 1.25rem;
        color: var(--text-secondary);
        font-weight: 500;
        position: relative;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Custom dropdown container */
    .custom-dropdown-container {
        position: relative;
        width: 100%;
    }
    
    /* Custom dropdown selector */
    .custom-dropdown-selector {
        background: var(--background-card);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        cursor: pointer;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
        transition: all 0.2s ease;
        width: 100%;
        font-family: inherit;
        font-size: inherit;
        color: var(--text-primary);
    }
    
    .custom-dropdown-selector:hover {
        border-color: rgba(148, 163, 184, 0.6);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.12);
    }
    
    .custom-dropdown-selector.active {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
    }
    
    .custom-dropdown-arrow {
        transition: transform 0.2s ease;
        color: var(--text-secondary);
    }
    
    .custom-dropdown-selector.active .custom-dropdown-arrow {
        transform: rotate(180deg);
    }
    
    /* Custom dropdown menu */
    .custom-dropdown-menu {
        position: fixed;
        background: var(--background-primary);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
        z-index: 10000;
        max-height: 250px;
        overflow-y: auto;
        display: none;
        min-width: 200px;
    }
    
    .custom-dropdown-menu.show {
        display: block;
    }
    
    .custom-dropdown-option {
        padding: 10px 16px;
        cursor: pointer;
        transition: all 0.15s ease;
        color: var(--text-primary);
        border-radius: 6px;
        margin: 2px 4px;
    }
    
    .custom-dropdown-option:hover {
        background-color: rgba(148, 163, 184, 0.15);
    }
    
    .custom-dropdown-option.selected {
        background-color: rgba(102, 126, 234, 0.15);
        font-weight: 500;
    }
    
    /* Hidden Gradio dropdown */
    .hidden-gradio-dropdown {
        display: none !important;
    }
    
    /* Streaming indicator */
    .streaming-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.1); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Queue status styling */
    .queue-status {
        background: linear-gradient(135deg, var(--queue-status-bg) 0%, var(--queue-status-bg-end) 100%);
        border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        text-align: center;
        font-size: 0.9rem;
        color: #0369a1;
    }
    
    /* Buttons */
    .generate-btn {
        background: linear-gradient(135deg, var(--button-gradient-green-start) 0%, var(--button-gradient-green-end) 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .generate-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    .stop-btn {
        background: linear-gradient(135deg, var(--button-gradient-red-start) 0%, var(--button-gradient-red-end) 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stop-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    .random-btn {
        background: linear-gradient(135deg, var(--button-gradient-gray-start) 0%, var(--button-gradient-gray-end) 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .random-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        background: linear-gradient(135deg, var(--button-gradient-gray-end) 0%, #334155 100%);
    }
    
    /* Audio player styling */
    .audio-output {
        background: linear-gradient(135deg, var(--audio-player-bg) 0%, var(--audio-player-bg-end) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }
    
    .complete-audio-section {
        margin-top: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, var(--complete-audio-bg) 0%, var(--complete-audio-bg-end) 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
    }
    
    /* Text areas */
    .script-input, .log-output {
        background: var(--background-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .script-input::placeholder {
        color: var(--text-secondary) !important;
    }
    
    /* Sliders */
    .slider-container {
        background: var(--background-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Labels and text */
    .gradio-container label {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
    }
    
    .gradio-container .markdown {
        color: var(--text-primary) !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2rem; }
        .settings-card, .generation-card { padding: 1rem; }
    }
    
    /* Streaming indicator */
    .streaming-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.1); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Queue status styling */
    .queue-status {
        background: linear-gradient(135deg, var(--queue-status-bg) 0%, var(--queue-status-bg-end) 100%);
        border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        text-align: center;
        font-size: 0.9rem;
        color: #0369a1;
    }
    
    /* Buttons */
    .generate-btn {
        background: linear-gradient(135deg, var(--button-gradient-green-start) 0%, var(--button-gradient-green-end) 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .generate-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    .stop-btn {
        background: linear-gradient(135deg, var(--button-gradient-red-start) 0%, var(--button-gradient-red-end) 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stop-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    .random-btn {
        background: linear-gradient(135deg, var(--button-gradient-gray-start) 0%, var(--button-gradient-gray-end) 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .random-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        background: linear-gradient(135deg, var(--button-gradient-gray-end) 0%, #334155 100%);
    }
    
    /* Audio player styling */
    .audio-output {
        background: linear-gradient(135deg, var(--audio-player-bg) 0%, var(--audio-player-bg-end) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }
    
    .complete-audio-section {
        margin-top: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, var(--complete-audio-bg) 0%, var(--complete-audio-bg-end) 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
    }
    
    /* Text areas */
    .script-input, .log-output {
        background: var(--background-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .script-input::placeholder {
        color: var(--text-secondary) !important;
    }
    
    /* Sliders */
    .slider-container {
        background: var(--background-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Labels and text */
    .gradio-container label {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
    }
    
    .gradio-container .markdown {
        color: var(--text-primary) !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2rem; }
        .settings-card, .generation-card { padding: 1rem; }
    }
    """
    
    # JavaScript for custom dropdown functionality
    custom_dropdown_js = """
    function initializeCustomDropdowns() {
        // Keep track of open dropdowns
        let openDropdown = null;
        
        // Function to position dropdown menu
        function positionDropdownMenu(selector, menu) {
            const rect = selector.getBoundingClientRect();
            const viewportWidth = window.innerWidth;
            const viewportHeight = window.innerHeight;
            
            // Position menu 2px below selector
            menu.style.left = rect.left + 'px';
            menu.style.top = (rect.bottom + 2) + 'px';
            menu.style.minWidth = rect.width + 'px';
            
            // Force reflow to get accurate dimensions
            menu.offsetHeight;
            
            // Adjust if menu goes off right edge
            if (rect.left + menu.offsetWidth > viewportWidth) {
                menu.style.left = (viewportWidth - menu.offsetWidth - 10) + 'px';
            }
            
            // Adjust if menu goes off bottom edge
            if (rect.bottom + menu.offsetHeight > viewportHeight) {
                menu.style.top = (rect.top - menu.offsetHeight - 2) + 'px';
            }
        }
        
        // Function to create custom dropdown
        function createCustomDropdown(container, gradioDropdown) {
            // Hide original Gradio dropdown
            gradioDropdown.classList.add('hidden-gradio-dropdown');
            
            // Create custom dropdown container
            const customContainer = document.createElement('div');
            customContainer.className = 'custom-dropdown-container';
            
            // Create selector
            const selector = document.createElement('div');
            selector.className = 'custom-dropdown-selector';
            
            // Create selected text
            const selectedText = document.createElement('span');
            selectedText.className = 'custom-dropdown-selected';
            
            // Create arrow
            const arrow = document.createElement('span');
            arrow.className = 'custom-dropdown-arrow';
            arrow.innerHTML = '▼';
            
            // Assemble selector
            selector.appendChild(selectedText);
            selector.appendChild(arrow);
            
            // Create menu
            const menu = document.createElement('div');
            menu.className = 'custom-dropdown-menu';
            
            // Get options from Gradio dropdown
            const gradioOptions = gradioDropdown.querySelectorAll('ul.options li');
            const options = [];
            
            gradioOptions.forEach((gradioOption, index) => {
                const option = document.createElement('div');
                option.className = 'custom-dropdown-option';
                option.textContent = gradioOption.textContent;
                option.dataset.value = gradioOption.dataset.value || gradioOption.textContent;
                option.dataset.index = index;
                
                // Select first option by default
                if (index === 0) {
                    selectedText.textContent = gradioOption.textContent;
                    option.classList.add('selected');
                }
                
                // Add click handler
                option.addEventListener('click', function(e) {
                    e.stopPropagation();
                    
                    // Update selected text
                    selectedText.textContent = this.textContent;
                    
                    // Update Gradio dropdown by triggering click on corresponding option
                    gradioOptions[this.dataset.index].click();
                    
                    // Update selected class
                    menu.querySelectorAll('.custom-dropdown-option').forEach(opt => {
                        opt.classList.remove('selected');
                    });
                    this.classList.add('selected');
                    
                    // Close menu
                    menu.classList.remove('show');
                    selector.classList.remove('active');
                    openDropdown = null;
                });
                
                menu.appendChild(option);
                options.push(option);
            });
            
            // Add click handler to selector
            selector.addEventListener('click', function(e) {
                e.stopPropagation();
                
                // Close any other open dropdown
                if (openDropdown && openDropdown !== menu) {
                    openDropdown.classList.remove('show');
                    openDropdown.previousElementSibling.classList.remove('active');
                }
                
                // Toggle this dropdown
                menu.classList.toggle('show');
                selector.classList.toggle('active');
                
                if (menu.classList.contains('show')) {
                    openDropdown = menu;
                    positionDropdownMenu(selector, menu);
                } else {
                    openDropdown = null;
                }
            });
            
            // Assemble custom dropdown
            customContainer.appendChild(selector);
            customContainer.appendChild(menu);
            
            // Insert before Gradio dropdown
            container.insertBefore(customContainer, gradioDropdown);
        }
        
        // Initialize all speaker dropdowns
        function initDropdowns() {
            document.querySelectorAll('.speaker-item .gradio-dropdown').forEach(function(gradioDropdown) {
                const container = gradioDropdown.parentElement;
                if (container && !container.querySelector('.custom-dropdown-container')) {
                    createCustomDropdown(container, gradioDropdown);
                }
            });
        }
        
        // Try to initialize immediately and after a delay
        initDropdowns();
        setTimeout(initDropdowns, 500);
        setTimeout(initDropdowns, 1000);
        
        // Also initialize when the DOM is updated
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'childList') {
                    initDropdowns();
                }
            });
        });
        
        observer.observe(document.body, { childList: true, subtree: true });
        
        // Close dropdowns when clicking outside
        document.addEventListener('click', function(e) {
            if (!e.target.closest('.custom-dropdown-selector') && !e.target.closest('.custom-dropdown-menu')) {
                document.querySelectorAll('.custom-dropdown-menu.show').forEach(function(menu) {
                    menu.classList.remove('show');
                });
                document.querySelectorAll('.custom-dropdown-selector.active').forEach(function(selector) {
                    selector.classList.remove('active');
                });
                openDropdown = null;
            }
        });
        
        // Handle window resize
        window.addEventListener('resize', function() {
            document.querySelectorAll('.custom-dropdown-menu.show').forEach(function(menu) {
                const selector = menu.previousElementSibling;
                if (selector) {
                    positionDropdownMenu(selector, menu);
                }
            });
        });
        
        return 'Custom dropdowns initialized';
    }
    """
    
    with gr.Blocks(
        title="VibeVoice - AI Podcast Generator",
        css=custom_css,
        theme=gr.themes.Default(),
        js=custom_dropdown_js
    ) as interface:
        
        # Load the JavaScript function
        interface.load(js="initializeCustomDropdowns")
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎙️ Vibe Podcasting </h1>
            <p>Generating Long-form Multi-speaker AI Podcast with VibeVoice</p>
        </div>
        """)
        
        with gr.Row():
            # Left column - Settings
            with gr.Column(scale=1, elem_classes="settings-card"):
                gr.Markdown("### 🎛️ **Podcast Settings**")
                
                # Number of speakers
                num_speakers = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=2,
                    step=1,
                    label="Number of Speakers",
                    elem_classes="slider-container"
                )
                
                # Speaker selection
                gr.Markdown("### 🎭 **Speaker Selection**")
                
                available_speaker_names = list(demo_instance.available_voices.keys())
                # default_speakers = available_speaker_names[:4] if len(available_speaker_names) >= 4 else available_speaker_names
                default_speakers = ['en-Alice_woman', 'en-Carter_man', 'en-Frank_man', 'en-Maya_woman']

                speaker_selections = []
                for i in range(4):
                    default_value = default_speakers[i] if i < len(default_speakers) else None
                    speaker = gr.Dropdown(
                        choices=available_speaker_names,
                        value=default_value,
                        label=f"Speaker {i+1}",
                        visible=(i < 2),  # Initially show only first 2 speakers
                        elem_classes="speaker-item"
                    )
                    speaker_selections.append(speaker)
                
                # Advanced settings
                gr.Markdown("### ⚙️ **Advanced Settings**")
                
                # Sampling parameters (contains all generation settings)
                with gr.Accordion("Generation Parameters", open=False):
                    cfg_scale = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.3,
                        step=0.05,
                        label="CFG Scale (Guidance Strength)",
                        # info="Higher values increase adherence to text",
                        elem_classes="slider-container"
                    )
                    
                    # Add inference steps parameter
                    inference_steps = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Inference Steps",
                        info="Number of DDPM steps (lower = faster, higher = better quality)",
                        elem_classes="slider-container"
                    )
                    
                    # Add streaming optimization parameters
                    streaming_min_yield_interval = gr.Slider(
                        minimum=5,
                        maximum=60,
                        value=30,
                        step=5,
                        label="Streaming Interval (seconds)",
                        info="Time between streaming updates",
                        elem_classes="slider-container"
                    )
                    
                    streaming_min_chunk_size_multiplier = gr.Slider(
                        minimum=30,
                        maximum=120,
                        value=60,
                        step=10,
                        label="Min Chunk Size Multiplier",
                        info="Controls minimum audio chunk size for streaming",
                        elem_classes="slider-container"
                    )
                    
                    # Add audio processing parameters
                    speech_tok_compress_ratio = gr.Slider(
                        minimum=800,
                        maximum=6400,
                        value=3200,
                        step=100,
                        label="Speech Token Compression Ratio",
                        info="Higher values = more compression, faster processing",
                        elem_classes="slider-container"
                    )
                    
                    target_dB_FS = gr.Slider(
                        minimum=-30,
                        maximum=-10,
                        value=-18,
                        step=1,
                        label="Target dB FS (Volume Leveling)",
                        info="Target volume level for audio normalization (-18 dB FS is default)",
                        elem_classes="slider-container"
                    )
                    
                    db_normalize = gr.Checkbox(
                        label="Enable Audio Normalization",
                        value=True,
                        info="Normalize input audio volume levels"
                    )
                
            # Right column - Generation
            with gr.Column(scale=2, elem_classes="generation-card"):
                gr.Markdown("### 📝 **Script Input**")
                
                script_input = gr.Textbox(
                    label="Conversation Script",
                    placeholder="""Enter your podcast script here. You can format it as:

Speaker 0: Welcome to our podcast today!
Speaker 1: Thanks for having me. I'm excited to discuss...

Or paste text directly and it will auto-assign speakers.""",
                    lines=12,
                    max_lines=20,
                    elem_classes="script-input"
                )
                
                # Button row with Random Example on the left and Generate on the right
                with gr.Row():
                    # Random example button (now on the left)
                    random_example_btn = gr.Button(
                        "🎲 Random Example",
                        size="lg",
                        variant="secondary",
                        elem_classes="random-btn",
                        scale=1  # Smaller width
                    )
                    
                    # Generate button (now on the right)
                    generate_btn = gr.Button(
                        "🚀 Generate Podcast",
                        size="lg",
                        variant="primary",
                        elem_classes="generate-btn",
                        scale=2  # Wider than random button
                    )
                
                # Stop button
                stop_btn = gr.Button(
                    "🛑 Stop Generation",
                    size="lg",
                    variant="stop",
                    elem_classes="stop-btn",
                    visible=False
                )
                
                # Streaming status indicator
                streaming_status = gr.HTML(
                    value="""
                    <div style="background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
                                border: 1px solid rgba(34, 197, 94, 0.3); 
                                border-radius: 8px; 
                                padding: 0.75rem; 
                                margin: 0.5rem 0;
                                text-align: center;
                                font-size: 0.9rem;
                                color: #166534;">
                        <span class="streaming-indicator"></span>
                        <strong>LIVE STREAMING</strong> - Audio is being generated in real-time
                    </div>
                    """,
                    visible=False,
                    elem_id="streaming-status"
                )
                
                # Output section
                gr.Markdown("### 🎵 **Generated Podcast**")
                
                # Streaming audio output (outside of tabs for simpler handling)
                audio_output = gr.Audio(
                    label="Streaming Audio (Real-time)",
                    type="numpy",
                    elem_classes="audio-output",
                    streaming=True,  # Enable streaming mode
                    autoplay=True,
                    show_download_button=False,  # Explicitly show download button
                    visible=True
                )
                
                # Complete audio output (non-streaming)
                complete_audio_output = gr.Audio(
                    label="Complete Podcast (Download after generation)",
                    type="numpy",
                    elem_classes="audio-output complete-audio-section",
                    streaming=False,  # Non-streaming mode
                    autoplay=False,
                    show_download_button=True,  # Explicitly show download button
                    visible=False  # Initially hidden, shown when audio is ready
                )
                
                gr.Markdown("""
                *💡 **Streaming**: Audio plays as it's being generated (may have slight pauses)  
                *💡 **Complete Audio**: Will appear below after generation finishes*
                """)
                
                # Generation log
                log_output = gr.Textbox(
                    label="Generation Log",
                    lines=8,
                    max_lines=15,
                    interactive=False,
                    elem_classes="log-output"
                )
        
        def update_speaker_visibility(num_speakers):
            updates = []
            for i in range(4):
                updates.append(gr.update(visible=(i < num_speakers)))
            return updates
        
        num_speakers.change(
            fn=update_speaker_visibility,
            inputs=[num_speakers],
            outputs=speaker_selections
        ).then(
            js="initializeCustomDropdowns",
            inputs=[],
            outputs=[]
        )
        
        # Main generation function with streaming
        def generate_podcast_wrapper(num_speakers, script, *speakers_and_params):
            """Wrapper function to handle the streaming generation call."""
            try:
                # Extract speakers and parameters
                speakers = speakers_and_params[:4]  # First 4 are speaker selections
                cfg_scale = speakers_and_params[4]   # CFG scale
                inference_steps = speakers_and_params[5]  # Inference steps
                streaming_interval = speakers_and_params[6]  # Streaming interval
                chunk_size_multiplier = speakers_and_params[7]  # Chunk size multiplier
                compress_ratio = speakers_and_params[8]  # Compression ratio
                target_db_fs = speakers_and_params[9]  # Target dB FS
                normalize_audio = speakers_and_params[10]  # Audio normalization flag
                
                # Update demo instance with new parameters
                demo_instance.model.set_ddpm_inference_steps(num_steps=inference_steps)
                demo_instance.streaming_min_yield_interval = streaming_interval
                demo_instance.streaming_min_chunk_size_multiplier = chunk_size_multiplier
                
                # Store parameters for use in generation (don't modify processor directly)
                generation_params = {
                    'compress_ratio': compress_ratio,
                    'target_db_fs': target_db_fs,
                    'normalize_audio': normalize_audio
                }
                
                # Clear outputs and reset visibility at start
                yield None, gr.update(value=None, visible=False), "🎙️ Starting generation...", gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
                
                # The generator will yield multiple times
                final_log = "Starting generation..."
                
                for streaming_audio, complete_audio, log, streaming_visible in demo_instance.generate_podcast_streaming(
                    num_speakers=int(num_speakers),
                    script=script,
                    speaker_1=speakers[0],
                    speaker_2=speakers[1],
                    speaker_3=speakers[2],
                    speaker_4=speakers[3],
                    cfg_scale=cfg_scale,
                    **generation_params
                ):
                    final_log = log
                    
                    # Check if we have complete audio (final yield)
                    if complete_audio is not None:
                        # Final state: clear streaming, show complete audio
                        yield None, gr.update(value=complete_audio, visible=True), log, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                    else:
                        # Streaming state: update streaming audio only
                        if streaming_audio is not None:
                            yield streaming_audio, gr.update(visible=False), log, streaming_visible, gr.update(visible=False), gr.update(visible=True)
                        else:
                            # No new audio, just update status
                            yield None, gr.update(visible=False), log, streaming_visible, gr.update(visible=False), gr.update(visible=True)

            except Exception as e:
                error_msg = f"❌ A critical error occurred in the wrapper: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                # Reset button states on error
                yield None, gr.update(value=None, visible=False), error_msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        def stop_generation_handler():
            """Handle stopping generation."""
            demo_instance.stop_audio_generation()
            # Return values for: log_output, streaming_status, generate_btn, stop_btn
            return "🛑 Generation stopped.", gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        # Add a clear audio function
        def clear_audio_outputs():
            """Clear both audio outputs before starting new generation."""
            return None, gr.update(value=None, visible=False)

        # Connect generation button with streaming outputs
        generate_btn.click(
            fn=clear_audio_outputs,
            inputs=[],
            outputs=[audio_output, complete_audio_output],
            queue=False
        ).then(  # Immediate UI update to hide Generate, show Stop (non-queued)
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            inputs=[],
            outputs=[generate_btn, stop_btn],
            queue=False
        ).then(
            fn=generate_podcast_wrapper,
            inputs=[num_speakers, script_input] + speaker_selections + [cfg_scale, inference_steps, streaming_min_yield_interval, streaming_min_chunk_size_multiplier, speech_tok_compress_ratio, target_dB_FS, db_normalize],
            outputs=[audio_output, complete_audio_output, log_output, streaming_status, generate_btn, stop_btn],
            queue=True  # Enable Gradio's built-in queue
        )
        
        # Connect stop button
        stop_btn.click(
            fn=stop_generation_handler,
            inputs=[],
            outputs=[log_output, streaming_status, generate_btn, stop_btn],
            queue=False  # Don't queue stop requests
        ).then(
            # Clear both audio outputs after stopping
            fn=lambda: (None, None),
            inputs=[],
            outputs=[audio_output, complete_audio_output],
            queue=False
        )
        
        # Function to randomly select an example
        def load_random_example():
            """Randomly select and load an example script."""
            import random
            
            # Get available examples
            if hasattr(demo_instance, 'example_scripts') and demo_instance.example_scripts:
                example_scripts = demo_instance.example_scripts
            else:
                # Fallback to default
                example_scripts = [
                    [2, "Speaker 0: Welcome to our AI podcast demonstration!\nSpeaker 1: Thanks for having me. This is exciting!"]
                ]
            
            # Randomly select one
            if example_scripts:
                selected = random.choice(example_scripts)
                num_speakers_value = selected[0]
                script_value = selected[1]
                
                # Return the values to update the UI
                return num_speakers_value, script_value
            
            # Default values if no examples
            return 2, ""
        
        # Connect random example button
        random_example_btn.click(
            fn=load_random_example,
            inputs=[],
            outputs=[num_speakers, script_input],
            queue=False  # Don't queue this simple operation
        )
        
        # Add usage tips
        gr.Markdown("""
        ### 💡 **Usage Tips**
        
        - Click **🚀 Generate Podcast** to start audio generation
        - **Live Streaming** tab shows audio as it's generated (may have slight pauses)
        - **Complete Audio** tab provides the full, uninterrupted podcast after generation
        - During generation, you can click **🛑 Stop Generation** to interrupt the process
        - The streaming indicator shows real-time generation progress
        """)
        
        # Add example scripts
        gr.Markdown("### 📚 **Example Scripts**")
        
        # Use dynamically loaded examples if available, otherwise provide a default
        if hasattr(demo_instance, 'example_scripts') and demo_instance.example_scripts:
            example_scripts = demo_instance.example_scripts
        else:
            # Fallback to a simple default example if no scripts loaded
            example_scripts = [
                [1, "Speaker 1: Welcome to our AI podcast demonstration! This is a sample script showing how VibeVoice can generate natural-sounding speech."]
            ]
        
        gr.Examples(
            examples=example_scripts,
            inputs=[num_speakers, script_input],
            label="Try these example scripts:"
        )

        # --- Risks & limitations (footer) ---
        gr.Markdown(
            """
## Risks and limitations

While efforts have been made to optimize it through various techniques, it may still produce outputs that are unexpected, biased, or inaccurate. VibeVoice inherits any biases, errors, or omissions produced by its base model (specifically, Qwen2.5 1.5b in this release).
Potential for Deepfakes and Disinformation: High-quality synthetic speech can be misused to create convincing fake audio content for impersonation, fraud, or spreading disinformation. Users must ensure transcripts are reliable, check content accuracy, and avoid using generated content in misleading ways. Users are expected to use the generated content and to deploy the models in a lawful manner, in full compliance with all applicable laws and regulations in the relevant jurisdictions. It is best practice to disclose the use of AI when sharing AI-generated content.
            """,
            elem_classes="generation-card",  # 可选：复用卡片样式
        )
    return interface


def convert_to_16_bit_wav(data):
    # Check if data is a tensor and move to cpu
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    # Ensure data is numpy array
    data = np.array(data)

    # Normalize to range [-1, 1] if it's not already
    if np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))
    
    # Scale to 16-bit integer range
    data = (data * 32767).astype(np.int16)
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Gradio Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/tmp/vibevoice-model",
        help="Path to the VibeVoice model directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")),
        help="Device for inference: cuda | mps | cpu",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=10,
        help="Number of inference steps for DDPM (not exposed to users)",
    )
    parser.add_argument(
        "--streaming_min_yield_interval",
        type=int,
        default=30,
        help="Minimum interval between streaming updates in seconds",
    )
    parser.add_argument(
        "--streaming_min_chunk_size_multiplier",
        type=int,
        default=60,
        help="Multiplier for minimum chunk size (sample_rate * multiplier)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share the demo publicly via Gradio",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the demo on",
    )
    
    return parser.parse_args()


def main():
    """Main function to run the demo."""
    args = parse_args()
    
    set_seed(42)  # Set a fixed seed for reproducibility

    print("🎙️ Initializing VibeVoice Demo with Streaming Support...")
    
    # Initialize demo instance
    demo_instance = VibeVoiceDemo(
        model_path=args.model_path,
        device=args.device,
        inference_steps=args.inference_steps
    )
    
    # Set streaming optimization parameters
    demo_instance.streaming_min_yield_interval = args.streaming_min_yield_interval
    demo_instance.streaming_min_chunk_size_multiplier = args.streaming_min_chunk_size_multiplier
    
    # Create interface
    interface = create_demo_interface(demo_instance)
    
    print(f"🚀 Launching demo on port {args.port}")
    print(f"📁 Model path: {args.model_path}")
    print(f"🎭 Available voices: {len(demo_instance.available_voices)}")
    print(f"🔴 Streaming mode: ENABLED")
    print(f"🔒 Session isolation: ENABLED")
    
    # Launch the interface
    try:
        interface.queue(
            max_size=20,  # Maximum queue size
            default_concurrency_limit=1  # Process one request at a time
        ).launch(
            share=args.share,
            # server_port=args.port,
            server_name="0.0.0.0",
            show_error=True,
            show_api=False  # Hide API docs for cleaner interface
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Server error: {e}")
        raise


if __name__ == "__main__":
    main()