from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import argparse

font = cv2.FONT_HERSHEY_SIMPLEX
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crowd Counting')
    parser.add_argument(
        'InputVideo',
        type=str,
        help='Path to Model. Model should be on the same path.'
    )
    parser.add_argument(
        'OutputVideo',
        type=str,
        help='Directory of Model checkpoint folder. Checkpoint should be on the same directory.'
    )
parser.add_argument(
    'countFile',
    type=str,
    help='Path to the test feed. Video file should be on the same path.'
)
args = parser.parse_args()

vid_path = args.InputVideo
out_path = args.OutputVideo
countFile = args.countFile

feed_vid = cv2.VideoCapture(vid_path)
success = True

fps = feed_vid.get(cv2.CAP_PROP_FPS)

fps = np.int32(fps)
print("Frames Per Second:",fps,"\n")

counter = 0
tm = 0
counts = []

out = open(countFile, "r")
for line in out: 
    counts.append(line)

def pipeline(img):
    global counter, tm
    counter += 1
    if tm>=len(counts):
        tm = len(counts)-1
    txt ='Crowd Count:'+ counts[tm]
    #print(txt)
    if counter <= fps:
        cv2.putText(img, txt[:-1], (36, 36), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, txt[:-1], (36, 36), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        counter = 0
        tm += 1
    return img


video_output = out_path #'./test_videos_output/project_video_output.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip(vid_path)
out_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
out_clip.write_videofile(video_output, audio=False)

"""
feed_vid = cv2.VideoCapture(vid_path)
success = True
fourcc = cv2.VideoWriter_fourcc(*'H264')
if success:
    success, im = feed_vid.read()
    print("success-", success)

    fps = feed_vid.get(cv2.CAP_PROP_FPS)

    fps = np.int32(fps)
    print("Frames Per Second:",fps,"\n")

    counter = 0
    tm = 1
    if success:
        video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'H264'), fps, (im.shape[1], im.shape[0]))

    out = open(countFile, "r")

    while success:
        counter +=1
        img = np.copy(im)
        txt = out.readline(tm)
        if counter <= fps:
            cv2.putText(img, txt, (0, 0), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, txt, (0, 0), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            counter = 0
            tm += 1
        video.write(img)
        success, im = feed_vid.read()
"""
out.close()
#video.release()
feed_vid.release()
cv2.destroyAllWindows()

