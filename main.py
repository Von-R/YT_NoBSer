import subprocess
import tempfile
import os
from youtube_transcript_api import YouTubeTranscriptApi
import re
import logging
import openai
from transformers import GPT2Tokenizer

# TEST:
TEST = False
# Initialize a dictionary to store the final transcripts
final_transcripts = {}

# List of chunk sizes to test
chunk_sizes = [200, 300, 400, 500]

logging.basicConfig(level=logging.DEBUG)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

""" 
Overview:
This script will take a YouTube video URL as input and return a condensed transcript of the video.
        "        use the YouTube API to extract the transcript of the video and then use the OpenAI API to condense the 
                 transcript.
        "        then write the condensed transcript to a text file in the same directory as the script.

Instructions: 
Download above packages
Store your openai API key in a file called keys.txt in the same directory as this script.
Providing a second key for the "editor" is optional but will improve performance at the cost of increased API calls (2x)
Run the script and paste the URL of the video you want to summarize when prompted
Let the machine beep boop for a while
The program will notify you that the condensed transcript is ready
Check the directory for a file called "transcript.txt" and open it
Go forth and prosper
"""

CHUNK_SIZE = 375  # maximum number of tokens sent to API in one chunk for processing

# Initialize the API key variable: DONE AUTOMATICALLY
transcriber_key = None

# Read the keys.txt file
with open("keys.txt", "r") as file:
    lines = file.readlines()

# Loop through each line to find the key
for line in lines:
    if "openai_API_key_1" in line:
        transcriber_key = line.split("=")[1].strip()
    if "openai_API_key_2" in line:
        editor_key = line.split("=")[1].strip()

# Check if the key was found
if transcriber_key:
    logging.info(f"Transcriber key: {transcriber_key}")
else:
    print("Transcriber API key not found. Fatal error.")
    exit(1)
if editor_key:
    logging.info(f"Editor key: {editor_key}")
else:
    logging.info("Editor key not found")

chatGPT_prompt_reformat = ("You will be given a raw transcript of the spoken content of a video. The transcript is "
                           "not formatted in any way. Your task is to add punctuation and capitalization to the "
                           "transcript to make it easier to read, and to create complete sentences from the raw text, "
                           "paragraph breaks as necessary, etc. The end result should be a properly formatted text."
                           "Here's the raw transcript:\n")

chatGPT_prompt_condense = ("The following transcript has poor information density and many filler words, phrases and "
                           "non-essential information. Please condense and paraphrase the transcript to focus solely "
                           "on the key elements of the content."
                           "Keep the content as purely informational as possible."
                           "My aim is to distill this transcript into an information-dense summary that captures the "
                           "essential wisdom and actionable insights. Here's the formatted transcript:\n")

chatGPT_prompt_editor = ("You are an editor. The writer's task is to condense and paraphrase the transcript to focus "
                         "solely on the key elements of the content and to keep the content as purely informational "
                         "as possible."
                         "perserving essential wisdom and actionable insights. You will be sent the writer's work: "
                         "your job is to behave like an editor, review their work and"
                         "further revise it as necessary to meet those goals. This is the piece you will be editing: \n")


def extract_youtube_video_id(url):
    """
    Extracts the video ID from a YouTube URL.
    """
    # Regular expression to find YouTube video ID
    youtube_id_match = re.search(r'v=([0-9A-Za-z_-]{11})', url)
    youtube_id_match = youtube_id_match or re.search(r'be/([0-9A-Za-z_-]{11})', url)
    youtube_id_match = youtube_id_match or re.search(r'embed/([0-9A-Za-z_-]{11})', url)

    video_id = (youtube_id_match.group(1) if youtube_id_match
                else None)

    return video_id


# Test the function with different kinds of YouTube URLs
test_urls = [
    "https://www.youtube.com/watch?v=abcdefghijk",
    "https://youtu.be/abcdefghijk",
    "https://www.youtube.com/embed/abcdefghijk",
    "https://www.youtube.com/watch?time_continue=1&v=abcdefghijk",
    "https://www.youtube.com/watch?v=abcdefghijk&feature=emb_title",
]

video_ids = [extract_youtube_video_id(url) for url in test_urls]
logging.info("Test: extract_youtube_video_id" + str(video_ids))


def write_transcript_to_current_folder(transcript, file_name):
    try:
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Create the full path for the new text file
        file_path = os.path.join(current_dir, file_name)

        # Write the transcript to the text file
        with open(file_path, 'w') as file:
            file.write(transcript)

        return f"Transcript successfully written to {file_path}."

    except Exception as e:
        return f"An error occurred: {e}"


def split_transcript_into_chunks(transcript, max_tokens=350):
    # Initialize the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Tokenize the transcript
    tokens = tokenizer.tokenize(transcript)

    # Initialize variables
    chunks = []
    chunk = []
    token_count = 0

    for token in tokens:
        # Check if adding the next token will exceed the maximum token limit
        if token_count + len(tokenizer.tokenize(token)) > max_tokens:
            # Convert tokens to string and add to the list of chunks
            chunks.append(tokenizer.convert_tokens_to_string(chunk))
            # Reset the current chunk and token count
            chunk = []
            token_count = 0

        # Add token to the current chunk and update the token count
        chunk.append(token)
        token_count += len(tokenizer.tokenize(token))

    # Add the last chunk if it's not empty
    if chunk:
        chunks.append(tokenizer.convert_tokens_to_string(chunk))

    return chunks


def main_function(CHUNK_SIZE=375):
    YT_url = input("Please paste URL of the video: ")
    video_id = extract_youtube_video_id(YT_url)
    transcript_raw = YouTubeTranscriptApi.get_transcript(video_id)

    # Raw transcript
    transcription_concat = ' '.join([entry['text'] for entry in transcript_raw])
    transcript_raw_chunks = split_transcript_into_chunks(transcription_concat, CHUNK_SIZE)

    # Feed raw transcript to OpenAI API to format and delimit sections, by logically distinct context if possible
    # Initial call, with instructions
    openai.Completion.create(
        engine="text-davinci-002",  # You can choose other engines as well
        prompt=chatGPT_prompt_reformat,
        api_key=transcriber_key
    )

    formatted_transcript = []

    # Subsequent calls, appended
    for chunk in transcript_raw_chunks:
        if formatted_transcript:
            prompt = ("Context so far: " + ' '.join(formatted_transcript) + "\nChunk to reformat: " + chunk)
        else:
            prompt = ("Chunk to reformat: " + chunk)
        response = openai.Completion.create(
            engine="text-davinci-002",  # You can choose other engines as well
            prompt=prompt,
            api_key=transcriber_key
        )
        generated_text = response.choices[0].text.strip()
        formatted_transcript.append(generated_text)

    logging.info(formatted_transcript)

    transcript_formatted_chunks = split_transcript_into_chunks(formatted_transcript, CHUNK_SIZE)

    # Feed formatted transcript to OpenAI API to condense in chunks, attaching condensed transcript thus far to
    # preserve context
    condensed_transcript = []
    # Initial call, with instructions to condense formatted transcript
    openai.Completion.create(
        engine="text-davinci-002",  # You can choose other engines as well
        prompt=chatGPT_prompt_condense,
        api_key=transcriber_key
    )

    for chunk in transcript_formatted_chunks:
        if condensed_transcript:
            prompt = "Context so far: " + ' '.join(condensed_transcript) + "\nChunk to condense: " + chunk
        else:
            prompt = "Chunk to condense: " + chunk

        response = openai.Completion.create(
            engine="text-davinci-002",  # You can choose other engines as well
            prompt=prompt,
            api_key=transcriber_key
        )
        generated_text = response.choices[0].text.strip()
        # Condensed chunk is reviewed by editor, who will further refine the chunk
        if editor_key:
            editor_response = openai.Completion.create(
                engine="text-davinci-002",  # You can choose other engines as well
                prompt=prompt,
                api_key=editor_key
            )
            generated_text = editor_response.choices[0].text.strip()
        # Chunk is then attached to the condensed transcript
        condensed_transcript.append(generated_text)
        final_transcript = ' '.join(condensed_transcript)
    # Rinse and repeat until the entire transcript is condensed

    # Write condensed transcript to text file
    file_name = f"final_transcript_chunk_size_{CHUNK_SIZE}.txt"
    with open(file_name, 'w') as file:
        file.write(final_transcript)

    print("Ding! Transcript is ready.")


# Manually examine each transcript and discern which chunk size is best
# Or feed to chatGPT and see if it can do the same?
if TEST:
    for chunks in chunk_sizes:
        main_function(chunks)
else:
    main_function()