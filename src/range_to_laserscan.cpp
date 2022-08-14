#include <ros/ros.h>

#include <iostream>
#include <math.h>
#include <vector>
#include <boost/thread/thread.hpp>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <std_msgs/Float32MultiArray.h>
#include <geometry_msgs/Vector3.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/LaserScan.h>

#include <vector>
#include <math.h>

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
    float far;
    float near;
    float hfov;
    int res_x;
    int res_y;
    int target_row;

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
        if(!n->getParam(name+"/sensor_info/resolution/horizontal", res_x)){
            ROS_WARN("No horizontal resolution.");
        }
        if(!n->getParam(name+"/sensor_info/resolution/vertical", res_y)){
            ROS_WARN("No vertical resolution.");
        }
        if(!n->getParam(name+"/sensor_info/hfov", hfov)){
            ROS_WARN("No horizontal fov.");
        }
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
        extendTopic(pub_topic);

        hfov = hfov/180.0*M_PI;
        target_row = res_y/2;

        laserscanmsg.angle_max = hfov/2.0;
        laserscanmsg.angle_min = -hfov/2.0;
        laserscanmsg.angle_increment = hfov/res_x;
        laserscanmsg.range_max = far;
        laserscanmsg.range_min = near;
        laserscanmsg.header.frame_id = frame;
    }

    // calculate the correction factor of the depth image
    float correction(int x){
        float angle = (float(x)/res_x - 0.5) * hfov;
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

        for(int i=res_x-1;i>=0;i--){
            float distance = cv_ptr->image.at<float>(target_row, i)/correction(i);
            if (distance>laserscanmsg.range_max || distance<laserscanmsg.range_min)
                distance = 0.0;
            laserscan_array.push_back(distance);
        }

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