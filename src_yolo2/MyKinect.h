#pragma once
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <Kinect.h>

class kinect
{
public:
	int init();
	int run();
	void release();

	int iDepthWidth, iDepthHeight;
	int iColorWidth, iColorHeight;
	int	iWidth , iHeight;

	cv::Mat	imgColor, DepthImg, BodyDepthImg;
	
private:
	IKinectSensor* pSensor;
	IBodyIndexFrameSource* pFrameSource;
	IFrameDescription* pFrameDescription;
	IDepthFrameSource* pDepthFrameSource;
	IFrameDescription* pDepthFrameDescription;
	IColorFrameSource* pColorFrameSource;
	IFrameDescription* pColorFrameDescription;
	IColorFrameReader* pColorFrameReader;
	IDepthFrameReader* pDepthFrameReader;
	IBodyIndexFrameReader* pFrameReader;
	ICoordinateMapper* pCoordinateMapper;
	unsigned int uColorBufferSize = 0;
};
