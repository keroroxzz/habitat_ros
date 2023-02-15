/*
Equirectangular Depth to 2D Laser Scan Converter
Author: Ariel

This is the source code for the converter transforming equirectangular depth image to 2D laser scan.
*/


#include <iostream>
#include <math.h>
#include <vector>
#include <boost/thread/thread.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <geometry_msgs/Vector3.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/LaserScan.h>

using namespace std;

class RangeToLaserScan
{
    private:
    ros::NodeHandle* n;
    ros::Subscriber sub_laser;
    ros::Publisher pub_laserscan;
    sensor_msgs::LaserScan laserscanmsg;

    string name;
    string sub_topic;
    string pub_topic;
    string frame;
    string unit;
    float far;
    float near;
    float ang_min;
    float ang_max;
    float ang_increment;
    // int res_x;
    // int res_y;
    // int target_row;

    public:
    RangeToLaserScan(ros::NodeHandle* nh)
    {
        n = nh;

        initParam();

        cout << "Initialize subscribers." << endl;
        sub_laser = nh->subscribe(sub_topic, 1, &RangeToLaserScan::r2l, this);
        
        cout << "Initialize publisher." << endl;
        pub_laserscan = nh->advertise<sensor_msgs::LaserScan>(pub_topic, 1);
    }

    ~RangeToLaserScan(){
        
    }

    void extendTopic(string &topic){
        topic = topic[0]=='/'? topic : name +"/"+ topic;
        topic = topic.substr(1);
    }

    void initParam(){
        name = ros::this_node::getName();

        cout<<"Initialize the parameter of "<<name<<endl;
        if(!n->getParam(name+"/sensor_info/far", far)){
            ROS_WARN("No far clip.");
        }
        if(!n->getParam(name+"/sensor_info/near", near)){
            ROS_WARN("No near clip.");
        }
        if(!n->getParam(name+"/frame", frame)){
            ROS_WARN("No frame.");
        }
        extendTopic(frame);
        if(!n->getParam(name+"/topic", sub_topic)){
            ROS_WARN("No subscribe topic.");
        }
        extendTopic(sub_topic);
        if(!n->getParam(name+"/laserscan_topic", pub_topic)){
            ROS_WARN("No publish topic.");
        }
        if(!n->getParam(name+"/sensor_info/ang_min", ang_min)){
            ROS_WARN("No minimum angle.");
        }
        if(!n->getParam(name+"/sensor_info/ang_max", ang_max)){
            ROS_WARN("No maximum angle.");
        }
        if(!n->getParam(name+"/sensor_info/ang_increment", ang_increment)){
            ROS_WARN("No angle increment.");
        }
        if(!n->getParam(name+"/sensor_info/unit", unit)){
            ROS_WARN("No unit.");
        }

        extendTopic(pub_topic);

        // hfov = hfov/180.0*M_PI;
        // target_row = res_y/2;

        if (unit.compare("deg") == 0){
            ang_max *= M_PI/180.0;
            ang_min *= M_PI/180.0;
            ang_increment *= M_PI/180.0;
            ROS_WARN("Input unit as degree, transform to radian...");
        }

        laserscanmsg.angle_max = ang_max;
        laserscanmsg.angle_min = ang_min;
        laserscanmsg.angle_increment = ang_increment;
        laserscanmsg.range_max = far;
        laserscanmsg.range_min = near;
        laserscanmsg.header.frame_id = frame;

    }

    // calculate the correction factor of the depth image
    float correction(int x){
        // float angle = (float(x)/res_x - 0.5) * hfov;
        float angle = ang_min + x * ang_increment;
        float vx = abs(cos(angle));
        float vy = abs(sin(angle));
        return max(vx,vy);
    }

    void r2l(const sensor_msgs::ImagePtr& msg){

        cv_bridge::CvImagePtr cv_ptr;
        try{
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
        }
        catch(cv_bridge::Exception& e){
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        vector<float> laserscan_array;

        for(int i = cv_ptr->image.cols-1;i>=0;i--){
            float distance = cv_ptr->image.at<float>(0, i)/correction(i);
            if (distance>laserscanmsg.range_max || distance<laserscanmsg.range_min)
                distance = 0.0;
            laserscan_array.push_back(distance);
        }

        cout << "Publish..." << endl;
        laserscanmsg.header.stamp = msg->header.stamp; 
        laserscanmsg.ranges = laserscan_array;
        pub_laserscan.publish(laserscanmsg);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "sick_laser");
    ros::NodeHandle n;
    RangeToLaserScan r(&n);
    ros::spin();
    return 0;
}