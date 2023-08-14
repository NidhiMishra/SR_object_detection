#include "Mykinect.h"
#include "image.h"
#include <vector>


#define camType 1 // 1--kinect; 0--computer cam.

using namespace std;
using namespace cv;

kinect Mykinect;
extern "C" void kinectCam()
{
	Mykinect.init();
}

extern "C" image readCam(int flag)
{
	image im;
	im.data = 0;
	im.h = 0;
	im.w = 0;
	im.c = 0;
	int i, j, k;
	Mat Colorframe, Depthframe;
	flag = -1;
	if (camType == 0)
	{
		/*		VideoCapture cap(0);
		//cap.set(CV_CAP_PROP_FRAME_WIDTH, 1080);
		//cap.set(CV_CAP_PROP_FRAME_HEIGHT, 960);
		cap >> frame;
		uchar *p = frame.data;

		//Mat frameChannel[3];
		//split(frame, frameChannel);

		im.data = 0;
		im.h = frame.rows;
		im.w = frame.cols;
		im.c = frame.channels();
		im.data = (float*)calloc(im.h*im.w*im.c, sizeof(float));

		for (k = 0; k < im.c; ++k){
		for (j = 0; j < im.h; ++j){
		for (i = 0; i < im.w; ++i){
		int dst_index = i + im.w*j + im.w*im.h*k;
		int src_index = im.c - 1 - k + im.c*i + im.c*im.w*j;
		if (frame.isContinuous())
		{
		im.data[dst_index] = (float)(frame.at<uchar>(src_index)) / 255.;
		}
		else
		{
		printf("data Error！！！");
		}

		}
		}
		}*/
	}
	else if (camType == 1)
	{
		//while (flag == -1)
		//{
		flag = Mykinect.run();
		//}
		//if (flag == 0)
		//{
		Mykinect.imgColor.copyTo(Colorframe);
		//imshow("调用摄像头", Colorframe);

		//Mykinect.DepthImg.copyTo(Depthframe);
		/*Depthframe.convertTo(Depthframe, CV_8U, 1 / 255.0);
		vector<Mat> DepthframeChannels;
		for (int i = 0; i < 4; i++)
		{
		DepthframeChannels.push_back(Depthframe);
		}
		merge(DepthframeChannels, Depthframe);
		Mat show = Depthframe.clone();
		resize(Depthframe, show, Size(Depthframe.cols*1.437, Depthframe.rows*1.437));
		Mat showImageDepth = show.clone();

		// 用resize函数把宽、高分辨率都缩小为原来的1/2
		Mat showImage = Colorframe.clone();
		resize(Colorframe, showImage, Size(Colorframe.cols / 2, Colorframe.rows / 2));

		// 矩形Rect，函数原型是 Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
		// 前两个参数表示矩形左上角的坐标，后两个参数表示矩形的宽和高
		Rect rect(145, 0, 702, 538);

		// 偏移量为（33，-145），即彩色（i，j）对应深度（i+33，j-145）
		// opencv坐标对应关系是（x，y）对应（列号col，行号row）
		// 对应坐标的深度与彩色像素按照 0.6:0.4 比例相加混合到新的Mat showImage内
		int x = 0;// 33,
		int y = -125;// -145;
		for (int i = 0; i <540; i++)
		for (int j = 145; j < 960 - 114; j++)
		showImage.at<Vec4b>(i, j) = showImageDepth.at<Vec4b>(i + x, j + y)*0.6 + showImage.at<Vec4b>(i, j)*0.4;

		// 用矩形指定showImage的感兴趣区域（不感兴趣区域应该是配准效果不好的区域）
		Mat image_roi = showImage(rect);
		// 显示感兴趣区域，也是配准后的结果
		imshow("Image_Roi", image_roi);

		//imshow("调用摄像头", show);*/
		//UINT16 *depthData = new UINT16[424 * 512];

		//hr = m_pDepthFrame->CopyFrameDataToArray(424 * 512, depthData);



		//vector<Mat> frameChannel;
		//frameChannel.resize(3);

		Mat frameChannel[4];
		split(Colorframe, frameChannel);

		

		/*for (i = 0; i < 3; i++)
		{
		frameChannel[i] = test[i];
		}

		//frameChannel.erase(frameChannel.begin() + 3);
		Mat mergeImage;
		merge(frameChannel, mergeImage);

		//Rect rect(145, 500, 702, 538);
		Mat imageResize;
		//mergeImage(rect).copyTo(imageResize);
		mergeImage.copyTo(imageResize);*/
		//mergeImage.copyTo(imageResize);
		//imshow("调用摄像头", image2);

		//uchar *p = imageResize.data;
		im.data = 0;
		im.h = frameChannel[0].rows;//imageResize.rows;
		im.w = frameChannel[0].cols;//imageResize.cols;
		im.c = 3;// imageResize.channels();
		im.data = (float*)calloc(im.h*im.w*im.c, sizeof(float));

		for (k = 0; k < im.c; ++k){
			for (j = 0; j < im.h; ++j){
				for (i = 0; i < im.w; ++i){
					int dst_index = i + im.w*j + im.w*im.h*k;
					int src_index = im.c - 1 - k + im.c*i + im.c*im.w*j;
					im.data[dst_index] = (float)frameChannel[2 - k].at<unsigned char>(j, i) / 255.;
				}
			}
		}

	}


	return im;
}

extern "C" void releaseCam()
{
	Mykinect.release();
}