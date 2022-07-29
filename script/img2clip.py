#! /usr/bin/env python

import numpy as np
import rospy
import tf
import cv2
import clip
import torch, torchvision

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String

from cv_bridge import CvBridge
cvBridge = CvBridge()

# initialize
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

room_type = ["bedroom", "dining room", "hall", "bathroom", "living room", "laundry room", "kitchen", "stairs", "door", "corridor", "wall", "unknown"]
text_inputs = torch.cat([clip.tokenize(f"{c}") for c in room_type]).to(device)
text_features = None
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
resize = torchvision.transforms.Resize((224,224))

actions = ["move_forward", "turn_right", "turn_left"]
next_action = actions[0]


def doClip(img):

    #normalization
    img = cv2.resize(img,(224,224))
    img = (2.0*img.copy().astype(float)/255.0-1.0)

    with torch.no_grad():

        tensor = torch.from_numpy(img).to(device)
        tensor = tensor.permute(2,0,1)[None,:,:,:]
        
        image_features = model.encode_image(tensor)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T)
        similarity = similarity.softmax(dim=-1)
    
    values, indices = similarity[0].topk(5)
    print("//==============detection result==============//")
    print(similarity)
    for v, i in zip(values, indices):
        print(f'{room_type[i]}: {100.0*v.item()}')

    return image_features.cpu().numpy()[0], values, indices

def instruction_callback(msg):
    
    instruction = [msg.data,]
    text_inputs = torch.cat([clip.tokenize(f"{c}") for c in instruction]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    text_feature_arr = Float32MultiArray()
    text_feature_arr.data = text_features[0].cpu().numpy()
    text_feature_arr.layout.data_offset=0
    clip_text_pub.publish(text_feature_arr)

def image_callback(msg):
    img = cvBridge.imgmsg_to_cv2(msg, "bgr8")

    clip_emb = Float32MultiArray()
    clip_emb.data,_,_ = doClip(img)
    clip_emb.layout.data_offset=0
    clip_pub.publish(clip_emb)

if __name__ == "__main__":

    rospy.init_node('clip')
    clip_pub = rospy.Publisher('clip', Float32MultiArray, queue_size=1)
    clip_text_pub = rospy.Publisher('clip_text_feature', Float32MultiArray, queue_size=1)
    rospy.Subscriber("image_full", Image, image_callback, queue_size=1)
    rospy.Subscriber("instruction", String, instruction_callback, queue_size=1)
    rospy.spin()

        
