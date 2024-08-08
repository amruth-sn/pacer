import os
import ffmpeg

input_file = '/mnt/c/users/amrut/onedrive/desktop/projects/speed/pacer/womens_10k.mp4'

def split_video_into_frames(input_file):
    # Create the 'frames' directory if it does not exist
    output_dir = os.path.join(os.path.dirname(input_file), 'frames')
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the output pattern with the new directory and file format
    output_pattern = os.path.join(output_dir, 'frame_%04d.png')
    
    # Run FFmpeg to split the video into frames at 20 fps
    ffmpeg.input(input_file).output(output_pattern, vf='fps=20').run()

split_video_into_frames(input_file)