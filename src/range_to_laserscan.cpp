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

    public:
    RangeToLaserScan(ros::NodeHandle* nh)
    {
        n = nh;

        cout << "Initialize subscribers." << endl;
        sub_laser = nh->subscribe("laser", 1, &RangeToLaserScan::r2l, this);
        

        cout << "Initialize publisher." << endl;
        pub_laserscan = nh->advertise<sensor_msgs::LaserScan>("LaserScan", 1);
        

    }

    ~RangeToLaserScan(){
        
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
        
        // Update GUI Window
        // cv::imshow("laser_image", cv_ptr->image);
        // cv::waitKey(1);

        sensor_msgs::LaserScan laserscanmsg;
        laserscanmsg.angle_max = M_PI;
        laserscanmsg.angle_min = -M_PI;//-2.3561899662017822;
        laserscanmsg.angle_increment = M_PI/180;//0.006554075051099062;
        laserscanmsg.range_max = 30;
        laserscanmsg.range_min = 0.1;
        laserscanmsg.header.frame_id = "LaserScan";
        laserscanmsg.header.stamp = msg->header.stamp; 

        int theta;

        for(int i=359;i>=0;i--){
            if(i>315){
                theta=i-360;
            }
            else if(i<45){
                theta=i;
            }
            else if(i>=45&i<135){
                theta=i-90;
            }
            else if(i>=135&i<225){
                theta=i-180;
            }
            else if(i>=225&i<315){
                theta=i-270;
            }
            laserscan_array.push_back(cv_ptr->image.at<float>(180,i)/cos(theta*M_PI/180));
            // cout<<laserscan_array[i]<<endl;
        }
        
        laserscanmsg.ranges = laserscan_array;
        
        pub_laserscan.publish(laserscanmsg);
         
        // Output modified video stream
        

    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "range_to_laserscan");
    ros::NodeHandle n;
    RangeToLaserScan r(&n);
    ros::spin();
    return 0;
}