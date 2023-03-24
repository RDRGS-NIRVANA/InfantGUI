import librosa
import moviepy.editor as mp
import os

video_path=r'C:\Users\Agnes_Yang\Desktop\InfantGUI\UI\test.mp4'
output_path = os.path.join(r"C:\Users\Agnes_Yang\Desktop\InfantGUI\audio\output_data\\" + video_path.split("\\")[-1][:-4] + ".wav")
my_clip = mp.VideoFileClip(video_path)
my_clip.audio.write_audiofile(output_path)
