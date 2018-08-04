from moviepy.editor import VideoFileClip
import numpy as np

import argparse
import tensorflow as tf

import cv2

import os
import re


from pyheatmap.heatmap import HeatMap
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imread, imresize

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

def heatmap(den, img):
    den_resized = np.zeros((den.shape[0] * 4, den.shape[1] * 4))
    for i in range(den_resized.shape[0]):
        for j in range(den_resized.shape[1]):
            den_resized[i][j] = den[int(i / 4)][int(j / 4)] / 16
    den = den_resized

    count = np.sum(den)

    data2=np.asarray(den*25)

    histogram = np.sum(data2, axis=0)
    histogram[histogram<0] = 0
    histogram = np.around(histogram)

    x_range = [np.min(histogram), np.max(histogram)]
    plt.xlabel("Position along the frame width")
    plt.ylabel("Number of People")
    plt.plot(list(range(data2.shape[1])), histogram)
    plt.savefig('./test/count_plot.jpg')
    den = den * 10 / np.max(den)
    w = img.shape[1]
    h = img.shape[0]

    data = []

    for j in range(len(den)):
        for i in range(len(den[0])):
            for k in range(int(den[j][i])):
                data.append([i + 1, j + 1])

    hm = HeatMap(data, base = './test/img.jpg')

    return hm
    
def pipeline(im):
    global counter, tm, font, sess, op_to_restore, x 
    success = True
    if success:
        if success:
            if success: 
                cv2.imwrite('./test/img.jpg',im)
                img = np.copy(im)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = np.array(img)
                img = (img - 127.5) / 128
                x_in = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
                x_in = np.float32(x_in)
                y_pred = []
                y_pred = sess.run(op_to_restore, feed_dict={x: x_in})
                sum =np.absolute(np.int32(np.sum(y_pred)))
                y_p_den = np.reshape(y_pred, (y_pred.shape[1], y_pred.shape[2]))
                hm = heatmap(y_p_den, im)
                hm.heatmap(save_as = './test/hm.jpg')
                
                                
                #Stiching

                temp = np.copy(im)
                hmap = cv2.imread('./test/hm.jpg')
                hmap = imresize(hmap, (np.int32(0.5*temp.shape[0]),np.int32(0.5*temp.shape[1])))

                im = imresize(im, (np.int32(temp.shape[0]),np.int32(0.5*temp.shape[1])))

                plotm = cv2.imread('./test/count_plot.jpg')
                os.remove('./test/count_plot.jpg')
                plotm = imresize(plotm, (hmap.shape[0],hmap.shape[1]))


                temp[:im.shape[0],:im.shape[1]]  = im

                temp[0:hmap.shape[0], im.shape[1]:im.shape[1]+hmap.shape[1]] = hmap
                temp[hmap.shape[0]:hmap.shape[0]+plotm.shape[0], im.shape[1]:im.shape[1]+plotm.shape[1]] = plotm

                img = temp
    counter += 1
    if tm>=len(counts):
        tm = len(counts)-1
    txt ='Crowd Count:'+ counts[tm]
    #print(txt)
    if counter <= fps:
        cv2.putText(img, txt[:-1], (36, 36), font, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, txt[:-1], (36, 36), font, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
        counter = 0
        tm += 1
    return img


video_output = out_path #'./test_videos_output/project_video_output.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph("./model.ckpt.meta")
    new_saver.restore(sess, tf.train.latest_checkpoint("./"))
    graph = tf.get_default_graph()
    op_to_restore = graph.get_tensor_by_name("add_12:0")
    x = graph.get_tensor_by_name('Placeholder:0')
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

