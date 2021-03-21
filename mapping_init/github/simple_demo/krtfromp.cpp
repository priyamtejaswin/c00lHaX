#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/sfm.hpp>
// #include <boost/python.hpp>
// using namespace boost::python;
using namespace cv;
using namespace cv::sfm;
using namespace std;
 
int recover()
{
    Mat Pmat;
//    Pmat = (Mat_<double>(3, 4) << 1.0, 0.0, 0.0, 0.0,
//                                  0.0, 1.0, 0.0, 0.0,
//                                  0.0, 0.0, 1.0, 0.0);
    Pmat = (Mat_<double>(3, 4) << 6.38344873e-05,  2.26940704e-03,  1.27613645e-01, -9.91825342e-01,
                                  4.96158922e-04,  1.76397095e-02,  9.91824124e-01, 1.27601554e-01,
                                  3.47670250e-06,  1.82726492e-06,  2.09855812e-02, 5.78843763e-04);
                                  
    Mat K;
    Mat R;
    Vec3d t;

    KRtFromProjection(Pmat, K, R, t);

    cout << "K\n"
         << K
         << "\nR\n"
         << R
         << "\nt\n"
         << t
         << "\n";

    return 0;
}

/*
BOOST_PYTHON_MODULE(krtfromp)
{
    def("recover", recover);
}
*/

int main()
{
    recover();
}
