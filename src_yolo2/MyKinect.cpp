#include <iostream>
#include <opencv2\opencv.hpp> 
#include "MyKinect.h"

using namespace std;
using namespace cv;

int kinect::init()
{
	// 1.Get default Sensor
	std::cout << "Try to get default sensor" << endl;
	pSensor = nullptr;
	if (GetDefaultKinectSensor(&pSensor) != S_OK)
	{
		std::cerr << "Get Sensor failed" << endl;
		return -1;
	}
	// 1.Open sensor
	std::cout << "Try to open sensor" << endl;
	if (pSensor->Open() != S_OK)
	{
		std::cerr << "Can't open sensor" << endl;
		return -1;
	}

	// 2.Get BodyIndex frame source
	std::cout << "Try to get body index source" << endl;
	 pFrameSource = nullptr;
	if (pSensor->get_BodyIndexFrameSource(&pFrameSource) != S_OK)
	{
		std::cerr << "Can't get body index frame source" << endl;
		return -1;
	}
	// 2. Get BodyIndex frame description
	std::cout << "Try to get body index frame description" << endl;
	iWidth = 0; iHeight = 0;
	pFrameDescription = nullptr;
	if (pFrameSource->get_FrameDescription(&pFrameDescription) == S_OK)
	{
		pFrameDescription->get_Width(&iWidth);
		pFrameDescription->get_Height(&iHeight);
	}
	// 2. Get BodyIndex frame reader
	std::cout << "Try to get body index frame reader" << endl;
	//IBodyIndexFrameReader* pFrameReader = nullptr;
	if (pFrameSource->OpenReader(&pFrameReader) != S_OK)
	{
		std::cerr << "Can't get body index frame reader" << endl;
		return -1;
	}

	// 3. Get depth frame source
	std::cout << "Try to get depth source" << endl;
	pDepthFrameSource = nullptr;
	if (pSensor->get_DepthFrameSource(&pDepthFrameSource) != S_OK)
	{
		std::cerr << "Can't get depth frame source" << endl;
		return -1;
	}
	// 3.Get depth frame description
	std::cout << "Try to get depth frame description" << endl;
	//int iDepthWidth = 0, iDepthHeight = 0;
	 pDepthFrameDescription = nullptr;
	if (pDepthFrameSource->get_FrameDescription(&pDepthFrameDescription) == S_OK)
	{
		pDepthFrameDescription->get_Width(&iDepthWidth);
		pDepthFrameDescription->get_Height(&iDepthHeight);
	}
	// 3.Get depth frame reader
	std::cout << "Try to get depth frame reader" << endl;
	//IDepthFrameReader* pDepthFrameReader = nullptr;
	if (pDepthFrameSource->OpenReader(&pDepthFrameReader) != S_OK)
	{
		std::cerr << "Can't get depth frame reader" << endl;
		return -1;
	}


	//4. Get frame source
	std::cout << "Try to get color source" << endl;
	pColorFrameSource = nullptr;
	if (pSensor->get_ColorFrameSource(&pColorFrameSource) != S_OK)
	{
		std::cerr << "Can't get color frame source" << endl;
		return -1;
	}

	// 4.Get frame description
	//int iColorWidth = 0, iColorHeight = 0;
	
	UINT uColorPointNum = 0;
	std::cout << "get color frame description" << endl;
	pColorFrameDescription = nullptr;
	if (pColorFrameSource->get_FrameDescription(&pColorFrameDescription) == S_OK)
	{
		pColorFrameDescription->get_Width(&iColorWidth);
		pColorFrameDescription->get_Height(&iColorHeight);

		uColorPointNum = iColorWidth * iColorHeight;
		uColorBufferSize = uColorPointNum * 4 * sizeof(BYTE);
	}
	pColorFrameDescription->Release();
	pColorFrameDescription = nullptr;

	//4. Get frame reader
	//IColorFrameReader* pColorFrameReader = nullptr;
	std::cout << "Try to get color frame reader" << endl;
	if (pColorFrameSource->OpenReader(&pColorFrameReader) != S_OK)
	{
		std::cerr << "Can't get color frame reader" << endl;
		return -1;
	}

	// 4.release Frame source
	std::cout << "Release frame source" << endl;
	pColorFrameSource->Release();
	pColorFrameSource = nullptr;

	//5. Coordinate Mapper
	std::cout << "Try to get CoordinateMapper" << endl;
	//ICoordinateMapper* pCoordinateMapper = nullptr;
	if (pSensor->get_CoordinateMapper(&pCoordinateMapper) != S_OK)
	{
		std::cerr << "get_CoordinateMapper failed" << endl;
		return -1;
	}


	imgColor.create(iColorHeight, iColorWidth, CV_8UC4);
	DepthImg.create(iHeight, iWidth, CV_16UC1);
	BodyDepthImg.create(iHeight, iWidth, CV_16UC1);


}


int kinect::run()
{
	int flag = -1;
	
	IBodyIndexFrame* pFrame = nullptr;
	IDepthFrame* pDepthFrame = nullptr;

	// Read color frame
	IColorFrame* pColorFrame = nullptr;
	if (pColorFrameReader->AcquireLatestFrame(&pColorFrame) == S_OK)
	{
		pColorFrame->CopyConvertedFrameDataToArray(uColorBufferSize, imgColor.data, ColorImageFormat_Bgra);
		pColorFrame->Release();
		pColorFrame = nullptr;
		//flag = 0;
	}	

	if ((pFrameReader->AcquireLatestFrame(&pFrame) == S_OK) && (pDepthFrameReader->AcquireLatestFrame(&pDepthFrame) == S_OK))
	{

		UINT uSize = 0;
		BYTE* pBuffer = nullptr;
		UINT32 nDepthFrameBufferSize = 0;
		UINT16* pDepthBuffer = nullptr;

		if ((pFrame->AccessUnderlyingBuffer(&uSize, &pBuffer) == S_OK) && (pDepthFrame->AccessUnderlyingBuffer(&nDepthFrameBufferSize, &pDepthBuffer) == S_OK))
		{
			for (int y = 0; y < iDepthHeight; ++y)
			{
				for (int x = 0; x < iDepthWidth; ++x)
				{
					int uBodyIdx = pBuffer[x + y * iDepthWidth];
					DepthImg.at<ushort>(y, x) = pDepthBuffer[x + y * iDepthWidth];
					if (uBodyIdx < 6)
					{
						BodyDepthImg.at<ushort>(y, x) = 65535;
					}
					else
						BodyDepthImg.at<ushort>(y, x) = 0;

				}
			}
			// release frame
			pFrame->Release();
			pDepthFrame->Release();
			//cerr << "Kinect init success" << endl;
			flag = 0;
		}
	}
	return flag;
}

void kinect::release()
{
	// 3.Release depth Frame 
	pDepthFrameReader->Release();
	pDepthFrameReader = nullptr;

	pDepthFrameDescription->Release();
	pDepthFrameDescription = nullptr;

	pDepthFrameSource->Release();
	pDepthFrameSource = nullptr;

	// 2.Release BodyIndex frame 
	pFrameReader->Release();
	pFrameReader = nullptr;

	pFrameDescription->Release();
	pFrameDescription = nullptr;

	pFrameSource->Release();
	pFrameSource = nullptr;

	// 1. Close Sensor
	cout << "close sensor" << endl;
	pSensor->Close();

	// 1. Release Sensor
	cout << "Release sensor" << endl;
	pSensor->Release();
	pSensor = nullptr;

}