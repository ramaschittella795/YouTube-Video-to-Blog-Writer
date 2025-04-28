# All imports
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
##from langchain.chains import RunnableSequence
import os
from dotenv import load_dotenv
##from crewai_tools import YoutubeVideoSearchTool
from youtube_transcript_api import YouTubeTranscriptApi
import re

# Load environment
load_dotenv()

# Setup LLM
llm = ChatOpenAI(
    model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-4-0125-preview"),
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Setup YouTube tool
def fetch_transcript_from_url(url):
    # Extract the video ID from the URL
    video_id_match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    if not video_id_match:
        raise ValueError("Invalid YouTube URL format.")
    video_id = video_id_match.group(1)
    
    # Fetch the transcript
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    
    # Join the transcript text
    full_transcript = " ".join([entry['text'] for entry in transcript_list])
    return full_transcript

# --- UI must be at top level ---

st.title("🎥 YouTube Video to Blog Writer ✍️")
st.write("Paste a YouTube video URL below, choose your preferred blog length, and get a nice blog post!")

youtube_url = st.text_input("Enter YouTube Video URL")
word_count = st.slider("Select Blog Word Count:", min_value=100, max_value=500, step=50, value=200)

# Button to trigger
if st.button("Generate Blog"):
    if youtube_url:
        with st.spinner("Fetching video and writing blog..."):
            try:
                transcript = fetch_transcript_from_url(youtube_url)
                st.write(transcript)

                # Prompt Template
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

                # Create chain
                blog_chain = LLMChain(
                    llm=llm,
                    prompt=blog_prompt
                )
##                blog_chain = RunnableSequence([blog_prompt | llm])

                # Run chain
                result = blog_chain.invoke({"transcript": transcript})
                blog_post = result["text"]

                st.success("🎉 Here's your blog post!")
                st.markdown("---")
                st.markdown("### 📄 Blog Post")
                st.markdown(blog_post)
                st.markdown("---")

                # Download Button
                blog_filename = "youtube_blog_post.md"
                with open(blog_filename, "w", encoding="utf-8") as f:
                    f.write(blog_post)

                with open(blog_filename, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="📥 Download Blog Post",
                        data=f,
                        file_name=blog_filename,
                        mime="text/markdown"
                    )

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a valid YouTube URL.")
