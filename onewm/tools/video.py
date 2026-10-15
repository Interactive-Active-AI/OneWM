"""Optional visualization encoding; never used as inference feedback."""

from pathlib import Path


def write_mp4(path, frames, fps=5):
    import av

    if fps <= 0:
        raise ValueError("fps must be positive")
    with av.open(str(Path(path)), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = int(frames.shape[2]), int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        for array in frames:
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(array, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
