import glob

from PIL import Image


def make_gif(frame_folder, name):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*"), reverse=True)]
    frame_one = frames[0]
    frame_one.save("assets/"+name, format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)
    
print("Dear Fulvio, you should lock the screen otherwise I will be randomly adding prints to your code :D For your amusement!")
if __name__ == "__main__":
    make_gif("assets/diff2", "diff2.gif")