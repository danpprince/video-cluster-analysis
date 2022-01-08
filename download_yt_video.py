import pytube

YT_URL = "https://www.youtube.com/watch?v=IMPbKVb8y8s"

if __name__ == "__main__":
    yt = pytube.YouTube(YT_URL)

    # Get the highest resolution mp4 stream
    yt_stream = yt.streams.filter(file_extension="mp4", progressive=True).order_by("resolution").last()

    yt_stream.download()