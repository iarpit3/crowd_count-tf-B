import tensorflow as tf
import numpy as np
import cv2
import argparse
import os
import re

model = None
font = cv2.FONT_HERSHEY_SIMPLEX
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crowd Counting')
    parser.add_argument(
        'model',
        type=str,
        help='Path to Model. Model should be on the same path.'
    )
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Directory of Model checkpoint folder. Checkpoint should be on the same directory.'
    )
parser.add_argument(
    'video',
    type=str,
    help='Path to the test feed. Video file should be on the same path.'
)
args = parser.parse_args()

model_path = args.model
ckpt_path = args.checkpoint
input_feed = args.video

if input_feed.find('.jpg', len(input_feed)-5, len(input_feed))!=-1 or input_feed.find('.png', len(input_feed)-5, len(input_feed))!=-1:
    with tf.Session() as sess:
        img = cv2.imread(input_feed, 0)
        img = np.array(img)
        img = (img - 127.5) / 128
        new_saver = tf.train.import_meta_graph(model_path)
        new_saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        graph = tf.get_default_graph()
        op_to_restore = graph.get_tensor_by_name("add_12:0")
        print(img.shape)
        x_in = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
        x_in = np.float32(x_in)
        y_pred = []
        x = graph.get_tensor_by_name('Placeholder:0')
        y_pred = sess.run(op_to_restore, feed_dict={x: x_in})
        sum = np.int32(np.sum(y_pred))
        if not sum - 15 <= 0:
            sum -= 15
        print(sum)
        count = "Crowd Count - " + str(sum)
        cv2.putText(img, count, (0, 0), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Crowd Count', img)
else:
    print("Entered else")
    with tf.Session() as sess:
        feed_vid = cv2.VideoCapture(input_feed)
        success = True
        if success:
            success, im = feed_vid.read()
            print("success-", success)
            new_saver = tf.train.import_meta_graph(model_path)
            new_saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
            graph = tf.get_default_graph()
            op_to_restore = graph.get_tensor_by_name("add_12:0")
            x = graph.get_tensor_by_name('Placeholder:0')
            # Find OpenCV version
            #(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
            #if int(major_ver)<3:
            fps = feed_vid.get(cv2.CAP_PROP_FPS)
            #print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
            #else:
            # fps = video.get(cv2.CAP_PROP_FPS)
            #print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)
            fps = np.int32(fps)
            print(fps)
            counter = 0
            avg = 0
            #imlist = []
            file = open("Count_per_sec_2.txt", "w")
            while success: 
                counter +=1
                img = np.copy(im)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = np.array(img)
                img = (img - 127.5) / 128
                #print(img.shape)
                x_in = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
                x_in = np.float32(x_in)
                y_pred = []
                y_pred = sess.run(op_to_restore, feed_dict={x: x_in})
                sum = np.int32(np.sum(y_pred))
                if sum - 15 <= 0:
                    sum += 15
                if counter<=fps:
                    avg += sum
                else:
                    counter =0 
                    avg = np.int32(avg/fps)
                    print(avg)
                    file.write(str(avg)+"/n")
                    avg = 0
                    avg += sum
                #count = "Crowd Count - " + str(sum)
                #counter += 1
                #cv2.putText(im, count, (0, 0), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                #imlist.append(im)
                '''if counter==10:
                    for e in imlist:
                        cv2.imshow('Crowd Count', im)
                    imlist = []
                    counter = 0'''
                success, im = feed_vid.read()
        file.close()
        feed_vid.release()
        cv2.destroyAllWindows()
