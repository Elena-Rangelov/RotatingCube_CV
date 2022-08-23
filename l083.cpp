
//	Elena Rangelov
//	P03
//	17.3.2022

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <string>
#include <ctime>
#include <math.h>
#include <iomanip>
#include <array>
#include <algorithm>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <cstdlib>
#include <stack>
#include <cmath>
#include <queue>

using namespace std;
using namespace cv;


int WIDTH, HEIGHT;

// ---------------------------------------------------- PART 1 -----------------------------------------------

Mat orthoRotation(double t, Mat vertex) {

	Mat x_rotation, y_rotation;
	x_rotation = (Mat_<double>(3, 3) << 1, 0, 0,  0, cos(t), -sin(t), 0, sin(t), cos(t));
	y_rotation = (Mat_<double>(3, 3) << cos(t), 0, sin(t), 0, 1, 0, -sin(t), 0, cos(t));

	Mat new_vertex = x_rotation * y_rotation * vertex;

	return new_vertex;
	
}

Mat perProjection(Mat vertex) {

	double view = 5.0;
	double eye = 10.0;

	double t = -view / (vertex.at<double>(0, 0) - eye);

	double u = vertex.at<double>(1, 0) * t;
	double v = vertex.at<double>(2, 0) * t;

	Mat ret = (Mat_<double>(2, 1) << u, v);
	return ret;

}



void part1() {

	int HEIGHT = 600;
	int WIDTH = 800;
	int CUBE_SIZE = 100;
	Mat SCALE_MATRIX;
	SCALE_MATRIX = (Mat_<double>(3, 3) << CUBE_SIZE, 0, 0,	0, CUBE_SIZE, 0,  0, 0, CUBE_SIZE);


	Mat img;
	Mat vertices;
	Mat v;
	Mat cp = Mat::zeros(2, 1, CV_64F);
	Mat cp1;
	vector<Point> rotations (8, Point());

	vertices = (Mat_<double>(8, 3) << 1, 1, 1,   1, -1, 1,   -1, -1, 1,   -1, 1, 1,   -1, 1, -1,   -1, -1, -1,   1, -1, -1,   1, 1, -1);

	ofstream output("coordinates.txt");

	VideoWriter writer("rotation.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(WIDTH, HEIGHT));
	Mat frame(WIDTH, HEIGHT, CV_8UC3, Scalar(0, 0, 0));
	double theta = 0;
	double speed = M_PI / 180 * 3;

	for (int frames = 0; frames < 60; frames++) {

		img = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
		

		for (int i = 0; i < 8; i++) {

			v = (Mat_<double>(3, 1) << vertices.at<double>(i, 0), vertices.at<double>(i, 1), vertices.at<double>(i, 2));

			cp = orthoRotation(theta, v);
			cp1 = cp;
			cp1 = SCALE_MATRIX * cp1;

			cp1 = perProjection(cp);

			if (frames < 4)
				output << "(" << cp.at<double>(0) << "," << cp.at<double>(1) << "),";
			if (i == 7)
				output << endl;

			rotations[i] = (Point((cp1.at<double>(0) * CUBE_SIZE) + WIDTH / 2, (cp1.at<double>(1) * CUBE_SIZE) + HEIGHT / 2));

			if (i > 0)
				line(img, rotations[i], rotations[i-1], Scalar(255, 255, 255));
			if (i == 7)
				line(img, rotations[7], rotations[0], Scalar(255, 255, 255));
			if (i == 6)
				line(img, rotations[6], rotations[1], Scalar(255, 255, 255));
			if (i == 5)
				line(img, rotations[5], rotations[2], Scalar(255, 255, 255));
			if (i == 7)
				line(img, rotations[7], rotations[4], Scalar(255, 255, 255));
			if (i == 3)
				line(img, rotations[3], rotations[0], Scalar(255, 255, 255));


		}

		writer.write(img);
		theta += speed;
	}

	vertices = (Mat_<double>(4, 3) << 1/sqrt(3), 0, 0,    0, 0, 2/sqrt(6),    -sqrt(3)/6, -1.0/2.0, 0,    -sqrt(3) / 6, 1.0 / 2.0, 0 );

	for (int frames = 0; frames < 60; frames++) {

		img = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);


		for (int i = 0; i < 4; i++) {

			v = (Mat_<double>(3, 1) << vertices.at<double>(i, 0), vertices.at<double>(i, 1), vertices.at<double>(i, 2));

			cp = orthoRotation(theta, v);
			cp1 = cp;
			cp1 = SCALE_MATRIX * cp1;

			cp1 = perProjection(cp);

			if (frames < 4)
				output << "(" << cp.at<double>(0) << "," << cp.at<double>(1) << "),";
			if (i == 7)
				output << endl;

			rotations[i] = (Point((cp1.at<double>(0) * CUBE_SIZE) + WIDTH / 2, (cp1.at<double>(1) * CUBE_SIZE) + HEIGHT / 2));

			if (i > 0)
				line(img, rotations[i], rotations[i - 1], Scalar(255, 255, 255));
			if (i == 3){
				line(img, rotations[3], rotations[0], Scalar(255, 255, 255));
				line(img, rotations[3], rotations[1], Scalar(255, 255, 255));
			}
			if (i == 2)
				line(img, rotations[2], rotations[0], Scalar(255, 255, 255));

		}

		writer.write(img);
		theta += speed;
	}


	writer.release();
};

void part3() {

	int HEIGHT = 600;
	int WIDTH = 800;
	int CUBE_SIZE = 100;
	Mat SCALE_MATRIX;
	SCALE_MATRIX = (Mat_<double>(3, 3) << CUBE_SIZE, 0, 0, 0, CUBE_SIZE, 0, 0, 0, CUBE_SIZE);


	Mat img;
	Mat vertices;
	Mat v;
	Mat cp = Mat::zeros(3, 1, CV_64F);
	Mat cp1;
	vector<Point> rotations(8, Point());
	Vec3f A = { 300, 300, 300 };
	Vec3f E = { 800, 800, 800 };
	Vec3f N = { 1, 1, 1 };

	vertices = (Mat_<double>(8, 3) << 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1);

	ofstream output("coordinates.txt");

	VideoWriter writer("rotation.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 60, Size(WIDTH, HEIGHT));
	Mat frame(WIDTH, HEIGHT, CV_8UC3, Scalar(0, 0, 0));
	double theta = 0;
	double speed = M_PI / 180;

	Vec3f p0 = { 0, 0, 0 };
	Vec3f v1 = { 1, 1, 1 };
	Vec3f v2 = { 1, 1, -1 };
	Vec3f v3 = { 1, -1, 1 };

	Vec3f pv0 = ((A - E).dot(N) / ((p0 - E).dot(N))) * ((p0 - E)) + E;
	Vec3f pv1 = ((A - E).dot(N) / ((v1 - E).dot(N))) * ((v1 - E)) + E;
	Vec3f pv2 = ((A - E).dot(N) / ((v2 - E).dot(N))) * ((v2 - E)) + E;
	Vec3f pv3 = ((A - E).dot(N) / ((v3 - E).dot(N))) * ((v3 - E)) + E;

	Vec3f va = pv1-pv2;
	Vec3f vb = pv1-pv3;
	
	Vec3f w1 = va;
	Vec3f w2 = vb;

	w2 = w2 - w2.dot(w1) / w1.dot(w1) * w1;
	w1 = normalize(w1);
	w2 = normalize(w2);

	for (int frames = 0; frames < 360; frames++) {

		img = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);

		for (int i = 0; i < 8; i++) {

			v = (Mat_<double>(3, 1) << vertices.at<double>(i, 0), vertices.at<double>(i, 1), vertices.at<double>(i, 2));

			cp = orthoRotation(theta, v);
			cp1 = cp;
			cp1 = SCALE_MATRIX * cp1;
			double x = cp1.at<double>(0, 0);
			double y = cp1.at<double>(1, 0);
			double z = cp1.at<double>(2, 0);

			Vec3f p = { float(x), float(y), float(z) };


			Vec3f proj_p = ((A - E).dot(N) / ((p - E).dot(N))) * (p - E) + E;

			Mat m = (Mat_<double>(2, 2) << w1[0], w2[0], w1[1], w2[1]);
			m = m.inv();
			Mat b = (Mat_<double>(2, 1) << proj_p[0] - pv0[0],
				proj_p[1] - pv0[1]); 


			Mat s = m * b;

			if (frames < 4)
				output << "(" << s.at<double>(0) << "," << s.at<double>(1) << "),";
			if (i == 7)
				output << endl;

			rotations[i] = (Point((s.at<double>(0, 0)) + WIDTH / 2, (s.at<double>(1, 0)) + HEIGHT / 2));

			if (i > 0)
				line(img, rotations[i], rotations[i - 1], Scalar(255, 255, 255));
			if (i == 7)
				line(img, rotations[7], rotations[0], Scalar(255, 255, 255));
			if (i == 6)
				line(img, rotations[6], rotations[1], Scalar(255, 255, 255));
			if (i == 5)
				line(img, rotations[5], rotations[2], Scalar(255, 255, 255));
			if (i == 7)
				line(img, rotations[7], rotations[4], Scalar(255, 255, 255));
			if (i == 3)
				line(img, rotations[3], rotations[0], Scalar(255, 255, 255));


		}

		writer.write(img);
		theta += speed;
	}


	vertices = (Mat_<double>(4, 3) << 1 / sqrt(3), 0, 0, 0, 0, 2 / sqrt(6), -sqrt(3) / 6, -1.0 / 2.0, 0, -sqrt(3) / 6, 1.0 / 2.0, 0);

	p0 = { 0, 0, 0 };
	v1 = { float(1 / sqrt(3)), 0, 0 };
	v2 = { 0, 0, float(2 / sqrt(6)) };
	v3 = { float(- sqrt(3) / 6), float( - 1.0 / 2.0), 0};

	pv0 = ((A - E).dot(N) / ((p0 - E).dot(N))) * ((p0 - E)) + E;
	pv1 = ((A - E).dot(N) / ((v1 - E).dot(N))) * ((v1 - E)) + E;
	pv2 = ((A - E).dot(N) / ((v2 - E).dot(N))) * ((v2 - E)) + E;
	pv3 = ((A - E).dot(N) / ((v3 - E).dot(N))) * ((v3 - E)) + E;

	va = pv1 - pv2;
	vb = pv1 - pv3;

	w1 = va;
	w2 = vb;

	w2 = w2 - w2.dot(w1) / w1.dot(w1) * w1;
	w1 = normalize(w1);
	w2 = normalize(w2);

	for (int frames = 0; frames < 360; frames++) {

		img = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);

		for (int i = 0; i < 4; i++) {

			v = (Mat_<double>(3, 1) << vertices.at<double>(i, 0), vertices.at<double>(i, 1), vertices.at<double>(i, 2));

			cp = orthoRotation(theta, v);
			cp1 = cp;
			cp1 = SCALE_MATRIX * cp1;
			double x = cp1.at<double>(0, 0);
			double y = cp1.at<double>(1, 0);
			double z = cp1.at<double>(2, 0);

			Vec3f p = { float(x), float(y), float(z) };

			Vec3f proj_p = ((A - E).dot(N) / ((p - E).dot(N))) * (p - E) + E;

			Mat m = (Mat_<double>(2, 2) << w1[0], w2[0], w1[1], w2[1]);
			m = m.inv();
			Mat b = (Mat_<double>(2, 1) << proj_p[0] - pv0[0],
				proj_p[1] - pv0[1]);

			Mat s = m * b;

			if (frames < 4)
				output << "(" << s.at<double>(0) << "," << s.at<double>(1) << "),";
			if (i == 7)
				output << endl;

			rotations[i] = (Point((s.at<double>(0, 0)) + WIDTH / 2, (s.at<double>(1, 0)) + HEIGHT / 2));

			if (i > 0)
				line(img, rotations[i], rotations[i - 1], Scalar(255, 255, 255));
			if (i == 3) {
				line(img, rotations[3], rotations[0], Scalar(255, 255, 255));
				line(img, rotations[3], rotations[1], Scalar(255, 255, 255));
			}
			if (i == 2)
				line(img, rotations[2], rotations[0], Scalar(255, 255, 255));


		}

		writer.write(img);
		theta += speed;
	}

	writer.release();

}


int main(int argc, char* argv[]) {

	part3();

}