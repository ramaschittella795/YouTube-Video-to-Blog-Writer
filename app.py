import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, VideoUnavailable
import re

# 1. Load environment variables
load_dotenv()

# 2. Initialize OpenAI LLM
llm = ChatOpenAI(
    model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-4-0125-preview"),
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# 3. Helper: Extract video ID from URL (robust for both desktop and mobile)
def extract_video_id(url):
    """
    Extract the video ID from both full YouTube URLs and shortened youtu.be links.
    """
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube URL format.")

# 4. Helper: Fetch transcript with error handling
def fetch_transcript_from_url(url):
    """
    Fetch transcript from a given YouTube URL.
    Handles missing transcripts or unavailable videos.
    """
    video_id = extract_video_id(url)
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = " ".join([entry['text'] for entry in transcript_list])
        return full_transcript
    except NoTranscriptFound:
        raise Exception("No transcript available for this video.")
    except VideoUnavailable:
        raise Exception("The video is unavailable.")
    except Exception as e:
        raise Exception(f"Failed to fetch transcript: {e}")

# 5. Streamlit App UI
st.title("🎥 YouTube Video to Blog Writer ✍️")
st.write("Paste a YouTube video URL below, choose your preferred blog length, and get a nice blog post!")

# Input YouTube URL
youtube_url = st.text_input("Enter YouTube Video URL")

# Select desired word count
word_count = st.slider("Select Blog Word Count:", min_value=100, max_value=500, step=50, value=200)

# Button to trigger blog generation
if st.button("Generate Blog"):
    if youtube_url:
        with st.spinner("Fetching video and writing blog..."):
            try:
                # Fetch transcript
                transcript = fetch_transcript_from_url(youtube_url)

                # Define Prompt dynamically
                blog_prompt = PromptTemplate(
                    input_variables=["transcript"],
                    template=f"""
You are a talented blog writer.

Based on the following YouTube video transcript:

{{transcript}}

Write a blog post summarizing the main ideas of the video.
Keep it engaging, easy to understand, and approximately {word_count} words.
Structure it into an **introduction**, **main body**, and **conclusion** clearly.

Avoid copying sentences directly from the transcript. Rewrite in your own words.
"""
                )

                # Create and run Chain
                blog_chain = LLMChain(
                    llm=llm,
                    prompt=blog_prompt
                )
                result = blog_chain.invoke({"transcript": transcript})
                blog_post = result["text"]

                # Display blog
                st.success("Here's your blog post!")
                st.markdown("---")
                st.markdown("### 📄 Blog Post")
                st.markdown(blog_post)
                st.markdown("---")

                # Download functionality
                blog_filename = "youtube_blog_post.md"
                with open(blog_filename, "w", encoding="utf-8") as f:
                    f.write(blog_post)

                with open(blog_filename, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="Download Blog Post",
                        data=f,
                        file_name=blog_filename,
                        mime="text/markdown"
                    )

            except ValueError as ve:
                st.error(f"Error: {ve}")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a valid YouTube URL.")
