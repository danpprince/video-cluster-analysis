import pytube
import sys


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        raise ValueError("Expected YouTube URL as argument")

    yt_url = sys.argv[1]
    yt = pytube.YouTube(yt_url)

    # Get the highest resolution mp4 stream
    yt_stream = (
        yt.streams.filter(file_extension="mp4", progressive=True)
        .order_by("resolution")
        .last()
    )

    yt_stream.download("videos/")
