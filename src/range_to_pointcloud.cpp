/*
Equirectangular Depth to Point Cloud Converter
Author: Biran Tu

This is the source code for the converter transforming equirectangular depth image to point cloud.
*/

#include <iostream>
#include <math.h>
#include <vector>
#include <boost/thread/thread.hpp>
#include <cv_bridge/cv_bridge.h>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

using namespace std;

const double PI = 3.14159265359;
const double D2R = 3.14159265359/180.0;

class RangeToPointCloud
{
    private:
    ros::NodeHandle* n;
    ros::Subscriber sub_range;
    ros::Subscriber sub_info;
    ros::Publisher pub_pc;

    string name;
    string sub_topic;
    string pub_topic;
    string frame;
    int res_x=0, res_y=0;
    double vfov, hfov, far, near;

    double ***vector_map=nullptr;
    double **cosines=nullptr;

    public:
    RangeToPointCloud(ros::NodeHandle* nh)
    {
        n = nh;

        initParam();
        bakeVectorMap();

        cout << "Initialize subscribers and publishers." << endl;
        sub_range = nh->subscribe(sub_topic, 1, &RangeToPointCloud::range_cb, this);
        pub_pc = nh->advertise<sensor_msgs::PointCloud2>(pub_topic, 1);
    }

    public:
    ~RangeToPointCloud(){
        if(vector_map){
            for(int x=0; x<res_x; x++){
                for(int y=0; y<res_y; y++)
                    delete [] vector_map[x][y];
                delete [] vector_map[x];
            }
            delete [] vector_map;
        }

        if(cosines){
            for(int x=0; x<res_x; x++){
                delete [] cosines[x];
            }
            delete [] cosines;
        }
    }

    void extendTopic(string &topic){
        topic = topic[0]=='/'? topic : name +"/"+ topic;
        topic = topic.substr(1);
    }

    void initParam(){
        name = ros::this_node::getName();

        cout<<"Initialize the parameter of "<<name<<endl;
        if(!n->getParam(name+"/sensor_info/vfov", vfov)){
            ROS_WARN("No vertical fov.");
        }
        if(!n->getParam(name+"/sensor_info/hfov", hfov)){
            ROS_WARN("No horizontal fov.");
        }
        if(!n->getParam(name+"/sensor_info/resolution/vertical", res_y)){
            ROS_WARN("No vertical resolution.");
        }
        if(!n->getParam(name+"/sensor_info/resolution/horizontal", res_x)){
            ROS_WARN("No horizontal resolution.");
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
        if(!n->getParam(name+"/point_cloud_topic", pub_topic)){
            ROS_WARN("No publish topic.");
        }
        extendTopic(pub_topic);

        cout << "Initialize the cubic vector_maptor map ( "<< res_x << " by " << res_y <<" )." << endl;
    }

    void bakeVectorMap()
    {
        if(vector_map){
            for(int x=0; x<res_x; x++){
                for(int y=0; y<res_y; y++)
                    delete [] vector_map[x][y];
                delete [] vector_map[x];
            }
            delete [] vector_map;
        }


        vector_map = new double**[res_x];
        cosines = new double*[res_x];
        for(int x=0; x<res_x; x++){
            vector_map[x] = new double*[res_y];
            cosines[x] = new double[res_y];

            for(int y=0; y<res_y; y++){
                vector_map[x][y] = new double[3];

                double l = -(double(x)/res_x-0.5)*hfov*D2R;
                double s = -(double(y)/res_y-0.5)*vfov*D2R;

                double rx = cos(l)*cos(s);
                double ry = sin(l)*cos(s);
                double rz = sin(s);

                double ax = abs(rx);
                double ay = abs(ry);
                double az = abs(rz);
                
                double dot = max(max(ax,ay),az);
                
                cosines[x][y] = dot;
                vector_map[x][y][0]=rx/dot;
                vector_map[x][y][1]=ry/dot;
                vector_map[x][y][2]=rz/dot;
            }
        }
    }

    void normalizeEqRect(cv_bridge::CvImagePtr cv_ptr)
    {
        if (res_x != cv_ptr->image.cols ||
            res_y != cv_ptr->image.rows){
            ROS_WARN("The range image has size: (%d, %d), which is different to the expecting size: (%d, %d)", 
            cv_ptr->image.cols, cv_ptr->image.rows, res_x, res_y);
            return;
        }

        pcl::PointCloud<pcl::PointXYZ> pc_score;
        for(int x=0; x<res_x; x++){
            for(int y=0; y<res_y; y++){
                float depth = cv_ptr->image.at<float>(y,x);
                pcl::PointXYZ p(
                    depth*vector_map[x][y][0],
                    depth*vector_map[x][y][1],
                    depth*vector_map[x][y][2]);

                float distance = depth/cosines[x][y];
                if(distance<=far||distance>=near)
                    pc_score.points.push_back(p);
            }
        }

        sensor_msgs::PointCloud2 pc2;
        pcl::toROSMsg(pc_score, pc2);
        pc2.header.frame_id = frame;
        pc2.header.stamp = cv_ptr->header.stamp;
        pub_pc.publish(pc2);
    }

    void range_cb(const sensor_msgs::ImagePtr& msg){
        cv_bridge::CvImagePtr cv_ptr;
        try{
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
        }
        catch(cv_bridge::Exception e){ROS_ERROR("CV bridge error: %s", e.what());}
        normalizeEqRect(cv_ptr);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "velodyne_vlp_16");
    ros::NodeHandle n;
    RangeToPointCloud pw(&n);
    ros::spin();
    return 0;
}