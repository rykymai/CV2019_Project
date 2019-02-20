#include <opencv2/opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <fstream>

using namespace std;
using namespace cv;

const int SMOOTHING_VAL = 5; // Parameter for averaging window filter, bigger is this and more smoothed is the result
const double PI = 3.141592653;

struct TransformParam
{
	TransformParam() {}
	TransformParam(double _dx, double _dy) {
		dx = _dx;
		dy = _dy;
	}

	double dx;
	double dy;
};

struct Trajectory
{
	Trajectory() {}
	Trajectory(double _x, double _y) {
		x = _x;
		y = _y;
	}

	double x;
	double y;
};

void drawArrow(Mat image, Point start, Point end, Scalar color, int arrow_magnitude, int thickness, int line_type, int shift);

int main(int argc, char **argv)
{
	// Create a txt file with the eleborated data
	ofstream out_transform("prev_to_cur_transformation.txt");
	ofstream out_trajectory("trajectory.txt");

	//Open video file and return 0 if the file don't exist
	VideoCapture cap;
	cap.open("../video_1_720.mp4");
	if (!cap.isOpened())
		return 0;

	//Create a Matrix that will be contain one frame of the video
	Mat cur, cur_grey;
	Mat prev, prev_grey;

	cap >> cur;

	Mat traj(Size(cur.cols, cur.rows), CV_8UC3, Scalar(0, 0, 0));

	//Gain constant for correct the magnetude of the arrows and the space of the grid
	int gain_dis = 10;
	float gain_tra = .5;
	int gain_sm = 10;
	int dist = 50;

	//Point that will contain the coordinate of the arrows
	Point pt2, pt3, pt3prev, pt4;
	Point center;

	center.x = cur.cols / 2;
	center.y = cur.rows / 2;

	//Setting the trajectory starting from the middle of image and normalized with the gain constant
	double x = traj.cols / 2 + traj.cols*gain_tra;
	double y = traj.rows / 2 + traj.rows*gain_tra;

	//Enable the record of the output video and the trajectory video
	VideoWriter recordOutput("../Output.avi", CV_FOURCC('D', 'I', 'V', 'X'), 30, cur.size(), true);
	VideoWriter recordTraj("../Trajectory.avi", CV_FOURCC('D', 'I', 'V', 'X'), 30, cur.size(), true);

	if (!recordOutput.isOpened())
		return -1;

	if (!recordTraj.isOpened())
		return -1;

	cap >> prev;
	cvtColor(prev, prev_grey, COLOR_BGR2GRAY);	//Convert the original video in to grayscale video

	// Now I get previous to current frame transformation for all frames
	vector <TransformParam> prev_to_cur_transform;

	int k = 1;
	int max_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
	Mat last_T;

	while (true) {
		cap >> cur;

		//Draw a grid
		traj = Scalar(0, 0, 0);
		int width = traj.size().width;
		int height = traj.size().height;

		for (int i = 0; i<height; i += dist)
			line(traj, Point(0, i), Point(width, i), cv::Scalar(255, 255, 255));

		for (int i = 0; i<width; i += dist)
			line(traj, Point(i, 0), Point(i, height), cv::Scalar(255, 255, 255));

		if (cur.data == NULL) {
			break;
		}

		cvtColor(cur, cur_grey, COLOR_BGR2GRAY);

		//Now I define the vectors that will be contain the good feature points of current end previus frame, the status and error of optical flow
		vector <Point2f> prev_corner, cur_corner;
		vector <Point2f> prev_corner2, cur_corner2;
		vector <uchar> status;
		vector <float> err;

		goodFeaturesToTrack(prev_grey, prev_corner, 200, 0.01, 15);  //Detect the good feature points. I pass the max number of point to return, the quality threshold and the max euclidean distance between two points
		calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, cur_corner, status, err);	//Compute the optical flow

		// I read all value returned by optial flow function and if the status is setted to false, I discard this value
		for (size_t i = 0; i < status.size(); i++) {
			if (status[i]) {
				prev_corner2.push_back(prev_corner[i]);
				cur_corner2.push_back(cur_corner[i]);
			}
		}

		//Compute the translation plus rotation of my 2D points
		Mat T = estimateRigidTransform(prev_corner2, cur_corner2, false);

		//In rare cases no transform is found. We'll just use the last known good transform.
		if (T.data == NULL) {
			last_T.copyTo(T);
		}

		T.copyTo(last_T);

		//Extract dx and dy from trasformation matrix
		double dx = T.at<double>(0, 2);
		double dy = T.at<double>(1, 2);

		prev_to_cur_transform.push_back(TransformParam(dx, dy));

		out_transform << k << " " << dx << " " << dy << endl;	//Write in txt file

		cur.copyTo(prev);
		cur_grey.copyTo(prev_grey);

		cout << "Frame: " << k << "/" << max_frames << " - good optical flow: " << prev_corner2.size() << endl;	//Output on console

		vector <TransformParam> smoothed_displacement;

		//Performing the averaging window for smoothing the result of displace
		for (size_t i = 0; i < prev_to_cur_transform.size(); i++)
		{
			double sum_x = 0;
			double sum_y = 0;
			int count = 0;

			for (int j = -SMOOTHING_VAL; j <= SMOOTHING_VAL; j++) {
				if (i + j >= 0 && i + j < prev_to_cur_transform.size()) {
					sum_x += prev_to_cur_transform[i + j].dx;
					sum_y += prev_to_cur_transform[i + j].dy;

					count++;
				}
			}

			double avg_x = sum_x / count;
			double avg_y = sum_y / count;

			smoothed_displacement.push_back(TransformParam(avg_x, avg_y));

			//Assign the vaule of x and y in a point moltiplied by a gain factor 
			pt2.x = center.x + (-dx*gain_dis);
			pt2.y = center.y + (-dy*gain_dis);

			pt4.x = center.x + (-avg_x*gain_sm);
			pt4.y = center.y + (-avg_y*gain_sm);
		}

		//Now i extimate the trajectory with an incremental variable that add/subtract his status every iteration with the average flow value. 
		vector <Trajectory> trajectory;
		x -= (center.x - pt4.x) / gain_sm;
		y -= (center.y - pt4.y) / gain_sm;

		trajectory.push_back(Trajectory(x, y));

		out_trajectory << (k + 1) << " " << x << " " << y << endl;

		k++;

		pt3.x = (x*gain_tra);
		pt3.y = (y*gain_tra);

		pt3prev.x = pt3.x + (dx*gain_tra);
		pt3prev.y = pt3.y + (dy*gain_tra);

		drawArrow(cur, center, pt2, Scalar(0, 0, 255), 9, 2, 8, 0);		//Draw displacement
		drawArrow(cur, center, pt4, Scalar(255, 0, 0), 9, 4, 8, 0);		//Draw displacement smooth
		drawArrow(traj, pt3prev, pt3, Scalar(0, 0, 255), 9, 3, 8, 0);	//Draw the trajectory

		//Saving the output videos
		recordOutput << cur;
		recordTraj << traj;

		//Display the output videos
		imshow("Output", cur);
		imshow("Path", traj);
		waitKey(30);
	}
	return 0;
}

void drawArrow(Mat image, Point start, Point end, Scalar color, int arrow_magnitude, int thickness, int line_type, int shift)
{
	//Draw the main line
	line(image, start, end, color, thickness, line_type, shift);

	//Compute the angle alpha
	double angle = atan2(start.y - end.y, start.x - end.x);

	//Compute the coordinates of the first segment
	start.x = (end.x + arrow_magnitude * cos(angle + PI / 4));
	start.y = (end.y + arrow_magnitude * sin(angle + PI / 4));

	//Draw the first segment
	line(image, start, end, color, thickness, line_type, shift);

	//Compute the coordinates of the second segment
	start.x = (end.x + arrow_magnitude * cos(angle - PI / 4));
	start.y = (end.y + arrow_magnitude * sin(angle - PI / 4));

	//Draw the second segment
	line(image, start, end, color, thickness, line_type, shift);
}