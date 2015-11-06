/* Function for inputting and outputting the images. */


#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <math.h>
#include <stack>
#include "color.h"

typedef unsigned char uchar;

using namespace cv;
using namespace std; 

#define ALPHA 20
#define METHOD 2  // 0 for Jacobi, 1 for Gauss-Seidel
#define JAC 0
#define GAU 1
#define SOR 2
#define WEIGHT 1.7

/* Assigns Ix, Iy, It gradient values computed from imgA and imgB. */
void compute_derivatives(const Mat_<float>& imgA, const Mat_<float>& imgB, 
			 Mat_<float>& Ix, Mat_<float>& Iy, Mat_<float>& It) {
  
	int channels = 1; //assume grayscale images

	int nRows = imgA.rows; //assume same size of imgA and imgB
	int nCols = imgA.cols * channels;

	Ix = Mat::zeros(imgA.size(), CV_32F);
	Iy = Mat::zeros(imgA.size(), CV_32F);
	It = Mat::zeros(imgA.size(), CV_32F);

	float *p_a1, *p_a2, *p_b1, *p_b2;
	float *p_ix, *p_iy, *p_it;
	
	p_a1 = (float*)imgA.ptr<float>(0); //cast is needed (const uchar* to uchar*)
	p_b1 = (float*)imgB.ptr<float>(0);
	p_a2 = p_a1 + nCols;
	p_b2 = p_b1 + nCols;
	p_ix = Ix.ptr<float>(0);
	p_iy = Iy.ptr<float>(0);
	p_it = It.ptr<float>(0);
	for ( int j = 0; j < nCols*(nRows-1); ++j)
	{       
		//calculate spatiotemporal averages like Horn & Schunck				//up at right-hand side x-boundary)
		p_ix[j] = 0.25 * (p_a1[j+1] - p_a1[j]+p_a2[j+1]-p_a2[j]
				+p_b1[j+1]-p_b1[j]+p_b2[j+1]-p_b2[j]);
		p_iy[j] = 0.25 * (p_a2[j]-p_a1[j]+p_a2[j+1]-p_a1[j+1]
				+p_b2[j]-p_b1[j]+p_b2[j+1]-p_b1[j+1]);
		p_it[j] = 0.25 * (p_b1[j] - p_a1[j]+p_b1[j+1]-p_a1[j+1]
				+p_b2[j]-p_a2[j]+p_b2[j+1]-p_a2[j+1]);
		
	}
}


void iterative_computation(Mat_<float>& u, Mat_<float>& v, const Mat_<float>& Ix, 
			   const Mat_<float>& Iy, const Mat_<float>& It) {
	if (METHOD == 0) { // Jacobi Methd
	    Mat_<float> f =  (Mat_<float>(3,3) << 1.0/12.0, 1.0/6.0, 1.0/12.0, 1.0/6.0, 0.0, 1.0/6.0, 
					  1.0/12.0, 1.0/6.0, 1.0/12.0);
	    Mat_<float> avg_u, avg_v;
	    filter2D(u, avg_u, -1 , f, Point(-1, -1), 0, BORDER_DEFAULT );
	    filter2D(v, avg_v, -1 , f, Point(-1, -1), 0, BORDER_DEFAULT );
	    Mat_<float> d1 = Ix.mul(avg_u) + Iy.mul(avg_v) + It;
	    Mat_<float> d2 = Mat::ones(u.size(), CV_32F) * ALPHA * ALPHA + Ix.mul(Ix) + Iy.mul(Iy);
	    Mat_<float> r = d1.mul(1 / d2);
	    u = avg_u - Ix.mul(r);
	    v = avg_v - Iy.mul(r);
	}
	else if (METHOD == 1) { // Gauss-Seidel method 
	    for (int i = 1; i < u.rows - 1; i ++) {
		for (int j = 1; j < u.cols - 1; j ++) {
			float avg_u = 1.f / 6.f * (u.at<float>(i - 1, j) + u.at<float>(i, j + 1) + 
					       u.at<float>(i + 1, j) + u.at<float>(i, j - 1)) +
				      1.f / 12.f * (u.at<float>(i - 1, j -1) + u.at<float>(i - 1, j + 1) + 
						u.at<float>(i + 1, j - 1) + u.at<float>(i + 1, j + 1));
			float avg_v = 1.f / 6.f * (v.at<float>(i - 1, j) + v.at<float>(i, j + 1) + 
					       v.at<float>(i + 1, j) + v.at<float>(i, j - 1)) +
				      1.f / 12.f * (v.at<float>(i - 1, j -1) + v.at<float>(i - 1, j + 1) + 
						v.at<float>(i + 1, j - 1) + v.at<float>(i + 1, j + 1));
			float ix = Ix.at<float>(i, j);
			float iy = Iy.at<float>(i, j);
			float it = It.at<float>(i, j);
			float r = (ix * avg_u + iy * avg_v + it) / 
				  (ALPHA * ALPHA + ix * ix + iy * iy);
			
			u.at<float>(i, j) = avg_u - ix * r;
			v.at<float>(i, j) = avg_v - iy * r;
		}
	    }
	} 

	else if (METHOD == 2) { // Successive overrelaxation method 
	    for (int i = 1; i < u.rows - 1; i ++) {
		for (int j = 1; j < u.cols - 1; j ++) {
			float avg_u = 1.f / 6.f * (u.at<float>(i - 1, j) + u.at<float>(i, j + 1) + 
					       u.at<float>(i + 1, j) + u.at<float>(i, j - 1)) +
				      1.f / 12.f * (u.at<float>(i - 1, j -1) + u.at<float>(i - 1, j + 1) + 
						u.at<float>(i + 1, j - 1) + u.at<float>(i + 1, j + 1));
			float avg_v = 1.f / 6.f * (v.at<float>(i - 1, j) + v.at<float>(i, j + 1) + 
					       v.at<float>(i + 1, j) + v.at<float>(i, j - 1)) +
				      1.f / 12.f * (v.at<float>(i - 1, j -1) + v.at<float>(i - 1, j + 1) + 
						v.at<float>(i + 1, j - 1) + v.at<float>(i + 1, j + 1));
			float ix = Ix.at<float>(i, j);
			float iy = Iy.at<float>(i, j);
			float it = It.at<float>(i, j);
			float r = (ix * avg_u + iy * avg_v + it) / 
				  (ALPHA * ALPHA + ix * ix + iy * iy);
			
			u.at<float>(i, j) = (1 - WEIGHT) * u.at<float>(i, j) + WEIGHT * (avg_u - ix * r);
			v.at<float>(i, j) = (1 - WEIGHT) * v.at<float>(i, j) + WEIGHT * (avg_v - iy * r);
		}
	    }
	}			   
}  

stack<pair<Mat, Mat> > compute_gaussian_pyramids(const Mat_<float>& imgA, 
						  const Mat_<float>& imgB, 
						  int nl, float ds) {
        int ds2=2; //ignore ds (better define it as int)
        
	stack<pair<Mat, Mat> > gp;
	gp.push(make_pair(imgA, imgB));
	/* TODO: compute the matrices for successive layers */
	for (int l=0;l<nl-1;l++){
	  Mat Ablur, Bblur;
	  GaussianBlur( gp.top().first, Ablur, Size( int(ds2*2+1), int(ds2*2+1) ), 0, 0);
	  GaussianBlur( gp.top().second, Bblur, Size( int(ds2*2+1), int(ds2*2+1) ), 0, 0);
	  Mat Ads, Bds;
	  Ads=Mat_<float>(floor(gp.top().first.rows/ds2), floor(gp.top().first.cols/ds2), CV_32F);
	  Bds=Mat_<float>(floor(gp.top().first.rows/ds2), floor(gp.top().first.cols/ds2), CV_32F);
	  for(int i=0;i<Ads.rows;i++){
	    for(int j=0;j<Ads.cols;j++){
	      Ads.at<float>(i,j)=Ablur.at<float>(i*ds2,j*ds2);
	      Bds.at<float>(i,j)=Bblur.at<float>(i*ds2,j*ds2);
	    }
	  }
	  gp.push(make_pair(Ads,Bds));
	}  
	return gp;
}

void bound(int& x, int& y, const Mat& img) {
	x = min(x, img.rows - 1);
	y = min(y, img.cols - 1);
}

inline int bilinear_interpolate(float x, float y, const Mat& img) {
	if (x >= img.rows - 1.f) x = img.rows - 1.01;
	if (y >= img.cols - 1.f) y = img.cols - 1.01;
	
	int x1 = floor(x);
	int x2 = x1 + 1;
	int y1 = floor(y);
	int y2 = y1 + 1;
	 	
	float r1 = (x2 - x) * img.at<float>(x1, y1) + (x - x1) * img.at<float>(x2, y1);
	float r2 = (x2 - x) * img.at<float>(x1, y2) + (x - x1) * img.at<float>(x2, y2);	
	return int((y2 - y) * r1 + (y - y1) * r2);
}

/* Compute the warped image of img using motion vector fields from previous layer of pyramid. */
Mat compute_warp(const Mat& img, Mat& u, Mat& v, float ds) {
	Mat_<float> r(img.size());
	Mat_<float> u_new(img.size());
	Mat_<float> v_new(img.size());
	for (int i = 0; i < r.rows; i ++) {
	    for (int j = 0; j < r.cols; j ++) {
		float di = i * ds;
		float dj = j * ds;
		int ux = bilinear_interpolate(di, dj, u) / ds;
		int vx = bilinear_interpolate(di, dj, v) / ds;
		int ni = min(max(i + vx, 0), img.rows - 1);
		int nj = min(max(j + ux, 0), img.cols - 1);
		r.at<float>(i, j) = img.at<float>(ni, nj);
		u_new.at<float>(i, j) = ux;
		v_new.at<float>(i, j) = vx;
	    }
	}
	u = u_new;
	v = v_new;
	return r;
}

void showImg(const string& s, const Mat& img) {
	Mat i;
	img.convertTo(i, CV_8UC3);
	imshow(s, i);
	cvWaitKey(0);
}

Mat color_map(const Mat& u, const Mat& v) {
	/* Combine u, v to form colored flow map. */
	uchar* pix = (uchar*)malloc(3 * sizeof(uchar));
	float maxrad = -1;
	for (int i = 0; i < u.rows; i ++) {
	    for (int j = 0; j < u.cols; j ++) {
		float up = u.at<float>(i, j);
		float vp = v.at<float>(i, j);
		maxrad = max(maxrad, sqrt(up * up + vp * vp));
	    }
	}	
	Mat Mflow = Mat::ones(u.size(), CV_32FC3);
	for (int i = 0; i < u.rows; i ++) {
	    for (int j = 0; j < u.cols; j ++) {	    
		computeColor(u.at<float>(i, j) / maxrad, v.at<float>(i, j) / maxrad, pix);
 		Mflow.at<Vec3f>(i, j)[0] = (float)pix[0];
		Mflow.at<Vec3f>(i, j)[1] = (float)pix[1];
		Mflow.at<Vec3f>(i, j)[2] = (float)pix[2];
	    }
	}
	return Mflow;
}

Mat optical_flow(const Mat_<float>& ImgA, const Mat_<float>& ImgB, 
		 int num_it, float threshold) {

	// Compute Gaussin Pyramid 
	int nl = 5;
	float ds = 0.5;
	stack<pair<Mat, Mat> > gp = compute_gaussian_pyramids(ImgA, ImgB, nl, ds);
	
	Mat_<float> u = Mat::zeros(ImgA.size(), CV_32F);
	Mat_<float> v = Mat::zeros(ImgA.size(), CV_32F);
	while (!gp.empty()) {
	    Mat imgA = (gp.top()).first;
	    Mat imgB = (gp.top()).second;
	    
	    // Warp the first image. 
	    Mat_<float> imgBw = imgB.clone();
	   
	    /* Compute warping here from u and v. */
	    imgBw = compute_warp(imgB, u, v, ds);

	    /* Compute the derivatives. */
	    Mat_<float> Ix, Iy, It;
	    compute_derivatives(imgBw, imgA, Ix, Iy, It); // papers       
	    Mat_<float> du = Mat::zeros(imgA.size(), CV_32F);
	    Mat_<float> dv = Mat::zeros(imgA.size(), CV_32F);
	    for (int i = 0; i < 500; i ++) 
		  iterative_computation(du, dv, Ix, Iy, It);

	    u = u - du;
	    v = v - dv;
	    gp.pop();
	}
	
	Mat Mflow = color_map(u, v);
	return Mflow;
}

int main(int argc, char** argv)
{
	// Load two images and allocate other structures
	Mat A = imread("frame10.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat B = imread("frame11.png", CV_LOAD_IMAGE_GRAYSCALE);
	
	Mat GA, GB;
	GaussianBlur( A, GA, Size( 9, 9 ), 0, 0); 
	GaussianBlur( B, GB, Size( 9, 9 ), 0, 0);

	// Convert to float matrix.
	Mat_<float> imgA, imgB;
	GA.convertTo(imgA, CV_32F);
	GB.convertTo(imgB, CV_32F);
	
	if (imgA.size() != imgB.size()) 
	    cerr << "Input images do not have matching dimensions." << endl;
	
	Size img_sz = imgA.size();
	
	// num_it being the maximum number of iterations to do.
	int num_it = 10;
	float threshold = 5;
	Mat imgC = optical_flow(imgA, imgB, num_it, threshold);
	Mat C;	
 	imgC.convertTo(C, CV_8UC3);
	
// 	namedWindow( "ImageA", WINDOW_AUTOSIZE );
// 	namedWindow( "ImageB", WINDOW_AUTOSIZE );
// 	namedWindow( "OpticalFlow", WINDOW_AUTOSIZE );
 
//	imshow( "ImageA", A );
//	imshow( "ImageB", B );
//	imshow( "OpticalFlow", imgC );
	imshow( "test", C );
	string r;
	if (METHOD == 0) {
	      r = "result_Jacobi.png";
	}
	else if (METHOD == 1) {
	      r = "result_Gauss_Seidel.png";
	}     
	else if (METHOD == 2){
	      r = "result_SOR.png";
	}   
	imwrite(r, C);
	cvWaitKey(0);
	
	return 0;
}

