#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/sfm.hpp>
#include <boost/python.hpp>
using namespace boost::python;
using namespace cv;
using namespace cv::sfm;
using namespace std;
 
int load(string filename)
{
 
    Mat img = imread(filename,CV_LOAD_IMAGE_COLOR);
    //imshow("opencvtest",img);
    //waitKey(0);
    return 1;
}

BOOST_PYTHON_MODULE(opencvtest)
{
    def("load",load);
}
