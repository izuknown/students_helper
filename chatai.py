'''
import openai
import gradio as gr
import CONFIG
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment
import math
import time

openai.api_key = CONFIG.openai_api_key

#CODE version 1 - Goal. Introduce a chat feature so that a user can ask questions to chatgpt about the summary 

messages = [{"role": "system", "content": "You are friendly and knowledgable university professor."}]
full_transcript = ''

def extract_audio(input_file, output_file):
    video_clip = VideoFileClip(input_file)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_file)

def transcribe_and_extract(audio_file):
    global full_transcript

    if audio_file is not None:
        # Determine the file type and handle accordingly
        file_extension = audio_file.name.split(".")[-1].lower()
        extracted_audio_file = f"{audio_file.name}, extracted_audio.wav"

        if file_extension == "mp4":
            # Extract audio from MP4
            extract_audio(audio_file.name, extracted_audio_file)
        elif file_extension in ["wav", "mp3"]:
            # Use the uploaded audio directly
            extracted_audio_file = audio_file.name
        else:
            print(f"Unsupported file type: {file_extension}")
            return

        # Load the audio file
        audio = AudioSegment.from_file(extracted_audio_file)

        # Split the audio into 1-minute chunks
        chunk_size = 60 * 1000  # 1 minute in milliseconds
        num_chunks = math.ceil(len(audio) / chunk_size)

        # Initialize variable for full transcript
        full_transcript = ""

        # Iterate through audio chunks and transcribe
        for i in range(num_chunks):
            start_time = i * chunk_size
            end_time = (i + 1) * chunk_size
            current_chunk = audio[start_time:end_time]

            # Export the current chunk
            current_chunk.export("current_chunk.wav", format="wav")

            # Transcribe the current chunk using Whisper
            with open("current_chunk.wav", "rb") as current_chunk_file:
                transcript = openai.Audio.transcribe(
                    model="whisper-1",
                    file=current_chunk_file,
                    response_format="text"
                )
                print(transcript)
                full_transcript += transcript

        return full_transcript

def chatai(file_data, text_data):

    global messages

    def split_into_chunks(content, chunk_size, chunks):
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i+chunk_size]
            chunks.append(chunk)
        print (chunks)
        return chunks

    if file_data is not None:
        try:
            content = transcribe_and_extract(file_data)
            chunks = []
            split_into_chunks(content, 10000, chunks)
            test = chunks[0]
            messages.append({"role": "user", "content": "Please reformat the text with Headings so that I am able to use it as revision material"})
            #messages.append({"role": "user", "content": "Additionally, based on the material can you provide some suggestions for YouTube videos I could watch to deepen my understanding of the subject matter."})
            messages.append({"role": "user", "content": f"The text I want you to summarize is {test}"})

            # Retry mechanism with delay
            retry_count = 0
            max_retries = 3
            while retry_count < max_retries:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                    )
                    system_response = response["choices"][0]["message"]["content"]
                    messages.append({"role": "assistant", "content": system_response})
                    print(system_response)
                    break  # Break the loop on success
                except openai.error.Timeout as e:
                    retry_count += 1
                    print(f"Timeout error, retrying ({retry_count}/{max_retries})...")
                    time.sleep(2)  # Add a delay before retrying
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return f"Error processing file: {str(e)}"
    elif text_data is not None:
        test = text_data  # Adjust the length based on your requirements
        messages.append({"role": "user", "content": text_data})
        retry_count = 0
        max_retries = 3
        while retry_count < max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                )
                system_response = response["choices"][0]["message"]["content"]
                messages.append({"role": "assistant", "content": system_response})
                print(system_response)
                break  # Break the loop on success
            except openai.error.Timeout as e:
                retry_count += 1
                print(f"Timeout error, retrying ({retry_count}/{max_retries})...")
                time.sleep(2)  # Add a delay before retrying
    else:
        raise ValueError("Invalid input. Either 'file_data' or 'text_data' must be provided.")

    chat_transcript = ""
    for message in messages:
        if message["role"] != 'system':
            chat_transcript += message['role'] + ":" + message['content'] + "\n\n"
    return chat_transcript

demo = gr.Interface(fn=chatai, inputs=[gr.File(), gr.Text()], outputs=[gr.Text()])
demo.launch()
'''
import openai
import gradio as gr
import CONFIG
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment
import math
import time
import re

openai.api_key = CONFIG.openai_api_key

# CODE version 1 - Goal. Introduce a chat feature so that a user can ask questions to chatgpt about the summary

messages = [{"role": "system", "content": "You are a friendly and knowledgeable university professor."}]
full_transcript = ''

def extract_audio(input_file, output_file):
    video_clip = VideoFileClip(input_file)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_file)

def transcribe_and_extract(audio_file):
    global full_transcript

    if audio_file is not None:
        # Determine the file type and handle accordingly
        file_extension = audio_file.name.split(".")[-1].lower()
        extracted_audio_file = f"{audio_file.name}, extracted_audio.wav"

        if file_extension == "mp4":
            # Extract audio from MP4
            extract_audio(audio_file.name, extracted_audio_file)
        elif file_extension in ["wav", "mp3"]:
            # Use the uploaded audio directly
            extracted_audio_file = audio_file.name
        else:
            print(f"Unsupported file type: {file_extension}")
            return

        # Load the audio file
        audio = AudioSegment.from_file(extracted_audio_file)

        # Split the audio into 1-minute chunks
        chunk_size = 60 * 1000  # 1 minute in milliseconds
        num_chunks = math.ceil(len(audio) / chunk_size)

        # Initialize variable for full transcript
        full_transcript = ""

        # Iterate through audio chunks and transcribe
        for i in range(num_chunks):
            start_time = i * chunk_size
            end_time = (i + 1) * chunk_size
            current_chunk = audio[start_time:end_time]

            # Export the current chunk
            current_chunk.export("current_chunk.wav", format="wav")

            # Transcribe the current chunk using Whisper
            with open("current_chunk.wav", "rb") as current_chunk_file:
                transcript = openai.Audio.transcribe(
                    model="whisper-1",
                    file=current_chunk_file,
                    response_format="text"
                )
                print(transcript)
                full_transcript += transcript

        return full_transcript

def reformat_text(transcript):
    # Perform reformatting with headings and content from transcript
    sections = re.split(r'\n\n', transcript)  # Split into sections based on double line breaks

    formatted_text = ""
    for i, section in enumerate(sections):
        heading = f"Section {i + 1}"  # You can customize the heading as needed
        formatted_text += f"## {heading}\n\n"
        formatted_text += f"{section}\n\n"

    return formatted_text

def chatai(file_data, text_data):
    global messages

    def split_into_chunks(content, chunk_size, chunks):
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            chunks.append(chunk)
        print(chunks)
        return chunks

    if file_data is not None:
        try:
            content = transcribe_and_extract(file_data)
            chunks = []
            split_into_chunks(content, 15000, chunks)
            test = chunks[0]
            messages.append({"role": "user", "content": "Please add Headings so that I am able to use it as revision material"})
            messages.append({"role": "user", "content": f"The text I want you to summarize is {test}"})

            # Retry mechanism with delay
            retry_count = 0
            max_retries = 3
            while retry_count < max_retries:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                    )
                    system_response = response["choices"][0]["message"]["content"]
                    messages.append({"role": "assistant", "content": system_response})
                    print(system_response)

                    # Reformat the transcribed text with headings and content
                    reformatted_text = reformat_text(full_transcript)
                    print(reformatted_text)
                    break  # Break the loop on success
                except openai.error.Timeout as e:
                    retry_count += 1
                    print(f"Timeout error, retrying ({retry_count}/{max_retries})...")
                    time.sleep(2)  # Add a delay before retrying
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return f"Error processing file: {str(e)}"
    elif text_data is not None:
        test = text_data  # Adjust the length based on your requirements
        messages.append({"role": "user", "content": text_data})
        retry_count = 0
        max_retries = 3
        while retry_count < max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                )
                system_response = response["choices"][0]["message"]["content"]
                messages.append({"role": "assistant", "content": system_response})
                print(system_response)

                # Reformat the user-provided text with headings and content
                reformatted_text = reformat_text(system_response)
                print(reformatted_text)
                break  # Break the loop on success
            except openai.error.Timeout as e:
                retry_count += 1
                print(f"Timeout error, retrying ({retry_count}/{max_retries})...")
                time.sleep(2)  # Add a delay before retrying
    else:
        raise ValueError("Invalid input. Either 'file_data' or 'text_data' must be provided.")

    chat_transcript = ""
    for message in messages:
        if message["role"] != 'system':
            chat_transcript += message['role'] + ":" + message['content'] + "\n\n"
    return chat_transcript

demo = gr.Interface(fn=chatai, inputs=[gr.File(), gr.Text()], outputs=[gr.Text()])
demo.launch()
