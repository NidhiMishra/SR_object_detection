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
		printf("data Error������");
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
		//imshow("��������ͷ", Colorframe);

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

		// ��resize�����ѿ��߷ֱ��ʶ���СΪԭ����1/2
		Mat showImage = Colorframe.clone();
		resize(Colorframe, showImage, Size(Colorframe.cols / 2, Colorframe.rows / 2));

		// ����Rect������ԭ���� Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
		// ǰ����������ʾ�������Ͻǵ����꣬������������ʾ���εĿ�͸�
		Rect rect(145, 0, 702, 538);

		// ƫ����Ϊ��33��-145��������ɫ��i��j����Ӧ��ȣ�i+33��j-145��
		// opencv�����Ӧ��ϵ�ǣ�x��y����Ӧ���к�col���к�row��
		// ��Ӧ�����������ɫ���ذ��� 0.6:0.4 ������ӻ�ϵ��µ�Mat showImage��
		int x = 0;// 33,
		int y = -125;// -145;
		for (int i = 0; i <540; i++)
		for (int j = 145; j < 960 - 114; j++)
		showImage.at<Vec4b>(i, j) = showImageDepth.at<Vec4b>(i + x, j + y)*0.6 + showImage.at<Vec4b>(i, j)*0.4;

		// �þ���ָ��showImage�ĸ���Ȥ���򣨲�����Ȥ����Ӧ������׼Ч�����õ�����
		Mat image_roi = showImage(rect);
		// ��ʾ����Ȥ����Ҳ����׼��Ľ��
		imshow("Image_Roi", image_roi);

		//imshow("��������ͷ", show);*/
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
		//imshow("��������ͷ", image2);

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