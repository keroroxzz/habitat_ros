#include <iostream>
#include <math.h>
#include <vector>
#include <boost/thread/thread.hpp>
#include <cv_bridge/cv_bridge.h>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <habitat_ros/LiDARINFO.h>

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

    int wx=0, wy=0, wz=0;
    double ***vec=nullptr;
    habitat_ros::LiDARINFO info;

    public:
    RangeToPointCloud(ros::NodeHandle* nh)
    {
        n = nh;
        cout << "Initialize subscribers and publishers." << endl;
        sub_info = nh->subscribe("lidar_info", 10, &RangeToPointCloud::info_cb, this);
        sub_range = nh->subscribe("range", 1, &RangeToPointCloud::range_cb, this);
        pub_pc = nh->advertise<sensor_msgs::PointCloud2>("point_cloud", 1);
    }

    public:
    ~RangeToPointCloud(){
    }

    void updateCubicVecMat()
    {
        if(vec){
            for(int x=0; x<wx; x++){
                for(int y=0; y<wy; y++)
                    delete [] vec[x][y];
                delete [] vec[x];
            }
            delete [] vec;
        }
        
        cout << "Initialize the cubic vector map." << endl;

        wx = info.horizontal_res;
        wy = info.vertical_res;

        vec = new double**[wx];
        for(int x=0; x<wx; x++){
            vec[x] = new double*[wy];

            for(int y=0; y<wy; y++){
                vec[x][y] = new double[3];

                double l = -(double(x)/wx-0.5)*info.hfov*D2R;
                double s = -(double(y)/wy-0.5)*info.vfov*D2R;

                double rx = cos(l)*cos(s);
                double ry = sin(l)*cos(s);
                double rz = sin(s);

                double ax = abs(rx);
                double ay = abs(ry);
                double az = abs(rz);
                
                double dot = 1.0f;

                if(ax>ay && ax>az) dot = ax;
                else if(ay>ax && ay>az) dot = ay;
                else if(az>ax && az>ay) dot = az;
                else dot = ax;
                
                vec[x][y][0]=rx/dot;
                vec[x][y][1]=ry/dot;
                vec[x][y][2]=rz/dot;
            }
        }
    }

    void normalizeEqRect(cv_bridge::CvImagePtr cv_ptr)
    {
        if (info.horizontal_res != cv_ptr->image.cols ||
            info.vertical_res != cv_ptr->image.rows)
            ROS_DEBUG("The range image has size: (%d, %d), which is different to the expecting size: (%d, %d)", 
            cv_ptr->image.cols, cv_ptr->image.rows, int(info.horizontal_res), int(info.vertical_res));

        pcl::PointCloud<pcl::PointXYZ> pc_score;
        for(int x=0; x<wx; x++){
            for(int y=0; y<wy; y++){
                float depth = cv_ptr->image.at<float>(y,x);
                pcl::PointXYZ p(
                    depth*vec[x][y][0],
                    depth*vec[x][y][1],
                    depth*vec[x][y][2]);

                pc_score.points.push_back(p);
            }
        }

        sensor_msgs::PointCloud2 pc2;
        pcl::toROSMsg(pc_score, pc2);
        pc2.header.frame_id = "lidar";
        pc2.header.stamp = cv_ptr->header.stamp;
        pub_pc.publish(pc2);
    }

    void info_cb(const habitat_ros::LiDARINFOPtr& msg){
        info = *msg;
        updateCubicVecMat();
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
    ros::init(argc, argv, "EquRectDepth_to_3DLiDAR");
    ros::NodeHandle n;
    RangeToPointCloud pw(&n);
    ros::spin();
    return 0;
}