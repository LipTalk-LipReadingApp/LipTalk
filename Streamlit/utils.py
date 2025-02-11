import tensorflow as tf
from typing import List
import cv2
import os

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path:str) -> List[float]:
    path = bytes.decode(path.numpy())
    cap = cv2.VideoCapture(path)
    frames = []
    frame_number=0
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[140:186, 110:250, :])
        frame_number+=1
        if frame_number==75:
            break
    cap.release()

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    frames = tf.cast((frames - mean), tf.float32) / std
    return frames


def load_alignment(path:str):
    file_name = path.split("/")[-1].split(".")[0]
    alignment_path = os.path.join("../Streamlit/Videos/Alignments", f"{file_name}.align") 
    with open(alignment_path, "r") as f:    #opens and read lines in alignments
      lines = f.readlines()
    tokens = []
    for line in lines:
      line = line.split()   #splits the lines
      if line[2] != "sil":
        tokens = [*tokens, " ", line[2]] #appends the alignments to tokens if not silence (sil)    
    return tf.strings.reduce_join(tf.reshape(tf.strings.unicode_split(tokens, input_encoding="UTF-8"), (-1))).numpy().decode('utf-8')