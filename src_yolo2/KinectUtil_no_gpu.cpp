#include "KinectUtil_with_cam.h"
#include "util.h"
#include "utils.h"

#include <thread>
#include <chrono>
#include <omp.h>
#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#include <windows.h>
#include <Kinect.h>// Kinect Header files
#include "detection_layer.h"
extern "C" {
#include "yolo.h"
#include "parser.h"
#include "network.h"
#include "test_detector.h"
}

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
//char *cfgfile = "G:/IMI-PROJECTS/i2p_Nadine_Robot/development/i2p_perception/i2p_object_detection/data/cfg/tiny-coco.cfg";
//char *weightfile = "G:/IMI-PROJECTS/i2p_Nadine_Robot/development/i2p_perception/i2p_object_detection/data/weights/tiny-coco.weights";
//char *filename = "G:/IMI-PROJECTS/i2p_Nadine_Robot/development/i2p_perception/i2p_object_detection/data/img/dog.jpg";
//char *cfgfile = "data/cfg/tiny-coco.cfg";
//char *weightfile = "data/weights/tiny-coco.weights";
//char *filename = "data/img/dog.jpg";
//float thresh = 0.15;

using namespace std;

KinectUtil::KinectUtil()
{
}

KinectUtil::KinectUtil(char *datacfg, char *namelist, char *cfgfile, char *weightfile, float num_thresh, ProtectedClient<imi::ObjectDetectionServiceClient>* mclient) {
	objectClient = mclient;
	initialize(datacfg, namelist, cfgfile, weightfile);
	thresh = num_thresh;
	isUseThrift = true;
}

KinectUtil::KinectUtil(char *datacfg, char *namelist, char *cfgfile, char *weightfile, float num_thresh) {
	initialize(datacfg, namelist, cfgfile, weightfile);
	thresh = num_thresh;
	isUseThrift = false;
}

KinectUtil::~KinectUtil()
{
	finalize();
}

// Processing
void KinectUtil::run()
{
	// Main Loop
	while (true) {
		update();
		drawDepth();
		show();
		// Key Check
		const int key = cv::waitKey(10);
		if (key == VK_ESCAPE) {
			break;
		}
	}
}

// Initialize
void KinectUtil::initialize(char *datacfg, char *namelist, char *cfgfile, char *weightfile)
{
	cv::setUseOptimized(true);
	initializeSensor();// Initialize Sensor
	initializeDepth();// Initialize Depth
	initializeColor();//Initialize Color
	initializeBodyFrame();//Initialize Body Skeleton

	i_RgbTodepthForshow.create(1080, 1920, CV_8UC1);//show 8bit
	i_RgbTodepth.create(1080, 1920, CV_16UC1);//calculate the distance 16bit
	m_pDepthCoordinates = new DepthSpacePoint[1920 * 1080];


	maxObjectNum = 100;

	hasNew = false;

	// Wait a Few Seconds until begins to Retrieve Data from Sensor ( about 2000-[ms] )
	std::this_thread::sleep_for(std::chrono::seconds(2));

	net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	l = net.layers[net.n - 1];
	set_batch_network(&net, 1);

	alphabet = load_alphabet();

	options = read_data_cfg(datacfg);
	name_list = option_find_str(options, "names", namelist);
	names = get_labels(name_list);
}

// Initialize Sensor
inline void KinectUtil::initializeSensor()
{
	// Open Sensor
	ERROR_CHECK(GetDefaultKinectSensor(&kinect));

	ERROR_CHECK(kinect->Open());

	// Check Open
	BOOLEAN isOpen = FALSE;
	ERROR_CHECK(kinect->get_IsOpen(&isOpen));
	if (!isOpen) {
		throw std::runtime_error("failed IKinectSensor::get_IsOpen( &isOpen )");
	}

	// Retrieve Coordinate Mapper
	ERROR_CHECK(kinect->get_CoordinateMapper(&coordinateMapper));
}

// Initialize Depth
inline void KinectUtil::initializeDepth()
{
	// Open Depth Reader
	ComPtr<IDepthFrameSource> depthFrameSource;
	ERROR_CHECK(kinect->get_DepthFrameSource(&depthFrameSource));
	ERROR_CHECK(depthFrameSource->OpenReader(&depthFrameReader));

	// Retrieve Depth Description
	ComPtr<IFrameDescription> depthFrameDescription;
	ERROR_CHECK(depthFrameSource->get_FrameDescription(&depthFrameDescription));
	ERROR_CHECK(depthFrameDescription->get_Width(&depthWidth)); // 512
	ERROR_CHECK(depthFrameDescription->get_Height(&depthHeight)); // 424
	ERROR_CHECK(depthFrameDescription->get_BytesPerPixel(&depthBytesPerPixel)); // 2

	// Retrieve Depth Reliable Range
	UINT16 minReliableDistance;
	UINT16 maxReliableDistance;
	ERROR_CHECK(depthFrameSource->get_DepthMinReliableDistance(&minReliableDistance)); // 500
	ERROR_CHECK(depthFrameSource->get_DepthMaxReliableDistance(&maxReliableDistance)); // 4500
	std::cout << "Depth Reliable Range : " << minReliableDistance << " - " << maxReliableDistance << std::endl;

	// Allocation Depth Buffer
	depthBuffer.resize(depthWidth * depthHeight);
}

// Initialize Color
inline void KinectUtil::initializeColor()
{
	// Open Color Reader
	ComPtr<IColorFrameSource> colorFrameSource;
	ERROR_CHECK(kinect->get_ColorFrameSource(&colorFrameSource));
	ERROR_CHECK(colorFrameSource->OpenReader(&colorFrameReader));

	// Retrieve Color Description
	ComPtr<IFrameDescription> colorFrameDescription;
	ERROR_CHECK(colorFrameSource->CreateFrameDescription(ColorImageFormat::ColorImageFormat_Bgra, &colorFrameDescription));
	ERROR_CHECK(colorFrameDescription->get_Width(&colorWidth)); // 1920
	ERROR_CHECK(colorFrameDescription->get_Height(&colorHeight)); // 1080
	ERROR_CHECK(colorFrameDescription->get_BytesPerPixel(&colorBytesPerPixel)); // 4

	// Allocation Color Buffer
	colorBuffer.resize(colorWidth * colorHeight * colorBytesPerPixel);
}

// Initialize Body Index
inline void KinectUtil::initializeBodyIndex()
{
	ComPtr<IBodyIndexFrameSource> pBodyIndexFrameSource;
	ERROR_CHECK(kinect->get_BodyIndexFrameSource(&pBodyIndexFrameSource));
	ERROR_CHECK(pBodyIndexFrameSource->OpenReader(&BodyIndexReader));
}

// Initialize Body Frame
inline void KinectUtil::initializeBodyFrame()
{
	ComPtr<IBodyFrameSource> pBodyFrameSource;
	ERROR_CHECK(kinect->get_BodyFrameSource(&pBodyFrameSource));
	ERROR_CHECK(pBodyFrameSource->OpenReader(&BodyFrameReader));
}

// Finalize
void KinectUtil::finalize()
{
	cv::destroyAllWindows();

	// Close Sensor
	if (kinect != nullptr) {
		kinect->Close();
	}
}

// Update Data
void KinectUtil::update()
{
	// Update Depth
	updateDepth();
	// Update Color
	updateColor();
	// Update Body Frame
	updateBodyFrame();
}
//Update Body Frame
inline void KinectUtil::updateBodyFrame()
{
	HRESULT ret = BodyFrameReader->AcquireLatestFrame(&bodyFrame);
	if (FAILED(ret)) {
		return;
	}
	ERROR_CHECK(bodyFrame->GetAndRefreshBodyData(_countof(ppBodies), ppBodies));
	m_vecKinectBodies.clear();

	int nBodyCount = BODY_COUNT;

#pragma omp parallel for
	for (int i = 0; i < nBodyCount; i++)
	{
		IBody* pBody = ppBodies[i];
		if (pBody)
		{
			BOOLEAN bTracked = false;
			ret = pBody->get_IsTracked(&bTracked);
			if (SUCCEEDED(ret) && bTracked)
			{
				Joint joints[JointType_Count];
				HandState leftHandState = HandState_Unknown;
				HandState rightHandState = HandState_Unknown;

				pBody->get_HandLeftState(&leftHandState);
				pBody->get_HandRightState(&rightHandState);

				ret = pBody->GetJoints(_countof(joints), joints);
				if (SUCCEEDED(ret))
				{
					cv::Point ptJoints[JointType_Count];
					cv::Vec3f vJoints[JointType_Count];
					TrackingState nStates[JointType_Count];

					for (int j = 0; j < _countof(joints); ++j)
					{
						vJoints[joints[j].JointType] = cv::Vec3f(joints[j].Position.X, joints[j].Position.Y, joints[j].Position.Z);
						//ptJoints[joints[j].JointType] = ProjectToPixel(vJoints[joints[j].JointType]);
						nStates[joints[j].JointType] = joints[j].TrackingState;
					}
					KinectBody Tempbody(vJoints, nStates);
					m_vecKinectBodies.push_back(Tempbody);
				}
			}
		}
	}
}

// Update Depth
inline void KinectUtil::updateDepth()
{
	// Retrieve Depth Frame

	const HRESULT ret = depthFrameReader->AcquireLatestFrame(&depthFrame);
	if (FAILED(ret)) {
		return;
	}

	// Retrieve Depth Data
	ERROR_CHECK(depthFrame->CopyFrameDataToArray(static_cast<UINT>(depthBuffer.size()), &depthBuffer[0]));
}

// Update Color
inline void KinectUtil::updateColor()
{
	// Retrieve Color Frame

	const HRESULT ret = colorFrameReader->AcquireLatestFrame(&colorFrame);
	if (FAILED(ret)) {
		return;
	}

	// Convert Format ( YUY2 -> BGRA )
	ERROR_CHECK(colorFrame->CopyConvertedFrameDataToArray(static_cast<UINT>(colorBuffer.size()), &colorBuffer[0], ColorImageFormat::ColorImageFormat_Bgra));
}

// Draw Depth
inline void KinectUtil::drawDepth()
{
	// Retrieve Mapped Coordinates
	std::vector<DepthSpacePoint> depthSpace(colorWidth * colorHeight);
	ERROR_CHECK(coordinateMapper->MapColorFrameToDepthSpace(depthBuffer.size(), &depthBuffer[0], depthSpace.size(), &depthSpace[0]));

	// Mapping Depth to Color Resolution
	std::vector<UINT16> buffer(colorWidth * colorHeight);
	std::vector<UINT8> buffer8bit(colorWidth * colorHeight);

#pragma omp parallel for
	for (int colorY = 0; colorY < colorHeight; colorY++) {
		unsigned int colorOffset = colorY * colorWidth;
		for (int colorX = 0; colorX < colorWidth; colorX++) {
			unsigned int colorIndex = colorOffset + colorX;
			m_pDepthCoordinates[colorIndex] = depthSpace[colorIndex];
			int depthX = static_cast<int>(depthSpace[colorIndex].X + 0.5f);
			int depthY = static_cast<int>(depthSpace[colorIndex].Y + 0.5f);
			if ((0 <= depthX) && (depthX < depthWidth) && (0 <= depthY) && (depthY < depthHeight)) {
				unsigned int depthIndex = depthY * depthWidth + depthX;
				buffer[colorIndex] = depthBuffer[depthIndex];
				buffer8bit[colorIndex] = static_cast<BYTE>(depthBuffer[depthIndex] >> 5);
			}
		}
	}

	// Create cv::Mat from Coordinate Buffer
	//depthMat = cv::Mat(colorHeight, colorWidth, CV_16UC1, &buffer[0]).clone();
	cv::Mat dMat = cv::Mat(colorHeight, colorWidth, CV_16UC1, &buffer[0]).clone();
	cv::Mat dMat8bit = cv::Mat(colorHeight, colorWidth, CV_8UC1, &buffer8bit[0]).clone();
	if (!dMat.empty())
		depthMat = dMat;
	if (!dMat8bit.empty())
		depthMat8bit = dMat8bit;
	i_RgbTodepth = depthMat;
	i_RgbTodepthForshow = depthMat8bit;
}

// Show Data
void KinectUtil::show()
{
	cv::Mat cMat = cv::Mat(colorHeight, colorWidth, CV_8UC4, &colorBuffer[0]);
	if (!cMat.empty())
		colorMat = cMat;
	detection();
}

// Show Depth
inline void KinectUtil::showDepth()
{
	if (depthMat.empty()) {
		return;
	}

	// Scaling ( 0-8000 -> 255-0 )
	cv::Mat scaleMat;
	depthMat.convertTo(scaleMat, CV_8U, -255.0 / 8000.0, 255.0);
	//cv::applyColorMap( scaleMat, scaleMat, cv::COLORMAP_BONE );

	// Show Image
	cv::imshow("Depth", scaleMat);
}

// Show Color
inline void KinectUtil::showColor()
{
	if (colorMat.empty()) {
		return;
	}

	// Resize Image
	cv::Mat resizeMat;
	const double scale = 0.5;
	cv::resize(colorMat, resizeMat, cv::Size(), scale, scale);

	// Show Image
	cv::imshow("Color", resizeMat);
}

void KinectUtil::checkObjects(std::vector<::imi::ObjectInfo> & Objects) {
	initializeChecking();
	updateNew(Objects);
	current_objects = Objects;
}

void KinectUtil::initializeChecking() {
	for (std::vector<::imi::ObjectInfo>::iterator current_obj = current_objects.begin();
		current_obj != current_objects.end(); ++current_obj) {
		current_obj->mchecked = false;
	}
}

void KinectUtil::updateNew(std::vector<::imi::ObjectInfo> & detectedObjects) {
	hasNew = false;
	if (detectedObjects.size() != current_objects.size()) {
		hasNew = true;
	}
	else {
		for (std::vector<::imi::ObjectInfo>::iterator detectedObj = detectedObjects.begin();
			detectedObj != detectedObjects.end(); ++detectedObj) {
			bool isNewFlag = true;
			for (std::vector<::imi::ObjectInfo>::iterator current_obj = current_objects.begin();
				current_obj != current_objects.end(); ++current_obj) {
				// find if any in unchecked objects.
				if (!current_obj->mchecked && detectedObj->label == current_obj->label) {
					// update matched objects
					current_obj->mchecked = true;
					isNewFlag = false;
					break;
				}
			}
			// add new object to current objects
			if (isNewFlag) {
				hasNew = true;
			}
		}
	}
}

void KinectUtil::write_infor_to_txt(object *RecObects, int *objectNumPerFrame)
{
	string pathRst = "test.txt";


	ofstream fCtr(pathRst.c_str(), ios::out);
	if (fCtr)
	{
/*		fCtr << "objNumber = " << *objectNumPerFrame << endl;

		for (int i = 0; i < *objectNumPerFrame; i++)
		{
			fCtr << endl;
			fCtr << "x = " << RecObects[i].x << endl;
			fCtr << "y = " << RecObects[i].y << endl;
			fCtr << "w = " << RecObects[i].w << endl;
			fCtr << "h = " << RecObects[i].h << endl;
			//fCtr << "CameraX = " << RecObects[i].CameraX << endl;
			//fCtr << "CameraY = " << RecObects[i].CameraY << endl;
			//fCtr << "CameraZ = " << RecObects[i].CameraZ << endl;
			fCtr << "name = " << RecObects[i].name << endl;
			fCtr << "prob = " << RecObects[i].prob << endl;
			fCtr << "objClass = " << RecObects[i].objClass << endl;
			//fCtr << "eventMessage = " << RecObects[i].eventMessage << endl;

		}

*/
		fCtr << "objNumber = " << *objectNumPerFrame << endl;
		for (int i = 0; i < *objectNumPerFrame; i++)
		{
			fCtr << "name = " << RecObects[i].name << endl;
			fCtr << "CameraX = " << RecObects[i].CameraX << endl;
			fCtr << "CameraY = " << RecObects[i].CameraY << endl;
			fCtr << "CameraZ = " << RecObects[i].CameraZ << endl;
		}
		fCtr.close();
	}
	/*
	fstream _file;
	_file.open(pathRst, ios::in);
	while (_file)
	{
	_file.close();
	_file.open(pathRst, ios::in);
	}*/
}


inline void KinectUtil::detection()
{
	if (colorMat.empty() || depthMat.empty() || depthFrame == NULL) {
		return;
	}

	//userRGB2Depth();//map RGB to Depth.

	IplImage testImg = colorMat;

	object *RecObects = new object[maxObjectNum];

	objectNumPerFrame = 0;
	std::vector< ::imi::ObjectInfo> detectedObjects; // Data to be transimited

	image im = ipl_to_image(&testImg);
	//Detect objects for per image.
	test_detector_img(names, alphabet, net, im, thresh, RecObects, &objectNumPerFrame);

	objectFilter(RecObects, &objectNumPerFrame, 2);

	//for suggestion system: I think you forget sth.//
	//object_reminder(im, names, alphabet, l.classes, RecObects, &objectNumPerFrame);	
	caculateXYZinCameraSpace(RecObects, objectNumPerFrame);

	if (objectNumPerFrame > 0)
	{
		write_infor_to_txt(RecObects, &objectNumPerFrame);
	}

	object_show(im, names, alphabet, l.classes, RecObects, &objectNumPerFrame);
	//show_image(im, "predictions");
	im = showImg(im);
	free_image(im);

	for (int i = 0; i < objectNumPerFrame; i++)
	{
		std::cout << "Send message to I2P!!" << RecObects[i].name << "\n";
		if (strlen(RecObects[i].name)<2 || RecObects[i].x<0)
			continue;

		//caculateXYZinCameraSpace(&RecObects[i], 1);
		std::cout << "corrdinates: " << RecObects[i].CameraX << " " << RecObects[i].CameraY << " " << RecObects[i].CameraZ << std::endl;

		// add data to detected object
		::imi::ObjectInfo detectedObj = ::imi::ObjectInfo();
		std::cout << RecObects[i].name << std::endl;
		detectedObj.__set_label(RecObects[i].name); // set label
		detectedObj.__set_prob(RecObects[i].prob); // set probability 
		detectedObj.coordinate.__set_x(RecObects[i].CameraX);
		detectedObj.coordinate.__set_y(RecObects[i].CameraY);
		detectedObj.coordinate.__set_z(RecObects[i].CameraZ);
		detectedObj.isNew = true;
		detectedObj.mchecked = false;
		detectedObj.duplicated = false;
		detectedObjects.push_back(detectedObj);
	}
	//checkObjects(detectedObjects); // check current objects
	// Transmit recongized objects to Reactive Layer
	if (isUseThrift && objectClient->ensureConnection()) // send iff there are new objects
	{
		try
		{
			objectClient->getClient()->objectRecognized(detectedObjects);

		}
		catch (apache::thrift::TException &tx)
		{
			std::cerr << "EXCEPTION opening the network conn: " << tx.what() << "\n";
			objectClient->receiveNetworkException();
		}
		catch (...) {
			std::cerr << "Unexpected EXCEPTION!!!" << "\n";
			objectClient->receiveNetworkException();
		}
	}
	delete[] RecObects;

}

inline image KinectUtil::showImg(image im)
{
	cv::Mat im_Mat(im.h, im.w, CV_8UC4);
#pragma omp parallel for
	for (int y = 0; y < im.h; ++y){
		for (int x = 0; x < im.w; ++x){
			for (int k = 0; k < im.c; ++k){
				im_Mat.at<cv::Vec4b>(y, x)[k] = (unsigned char)(get_pixel(im, x, y, k) * 255);
			}
		}
	}
	cv::Mat tmp_Mat = DrawSkeletonFrame(im_Mat);
	IplImage showImg = tmp_Mat;
	image showIm = ipl_to_image(&showImg);
	show_image(showIm, "predictions");
	//free_image(showIm);
	return showIm;
}

inline float KinectUtil::GetImgAvg(cv::Mat img)
{
	int sum = 0;
	int cols = img.cols, rows = img.rows;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			sum += img.at<ushort>(i, j);
		}
	}
	return sum / (cols*rows);
}

inline float KinectUtil::GetImgAvg(cv::Mat img, int thr)
{
	int sum = 0;
	int sumAll = 0;
	int cols = img.cols, rows = img.rows;
	int idx = 0;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			sumAll += img.at<ushort>(i, j);;
			if (img.at<ushort>(i, j) < thr && img.at<ushort>(i, j) > 0)
			{
				sum += img.at<ushort>(i, j);
				idx++;
			}

		}
	}
	float res = 0;
	if (idx != 0)
		res = sum / idx;
	else
		res = sumAll / (cols * rows);
	return res;
}
inline void KinectUtil::caculateXY(object b, float *x, float *y, int thr)
{
	int colorWidth = i_RgbTodepth.cols;
	int colorHeight = i_RgbTodepth.rows;
	int idx = 0;
	int xSum = 0, ySum = 0;

	int DepthLeft = max(0, (b.x - b.w / 2.)*colorWidth);
	int DepthRight = min(colorWidth, (b.x + b.w / 2.)*colorWidth);
	int DepthTop = max(0, (b.y - b.h / 2.)*colorHeight);
	int DepthBot = min(colorHeight, (b.y + b.h / 2.)*colorHeight);

#pragma omp parallel for
	for (int colorY = DepthTop; colorY < DepthBot; colorY++) {
		unsigned int colorOffset = colorY * colorWidth;
		for (int colorX = DepthLeft; colorX < DepthRight; colorX++) {
			unsigned int colorIndex = colorOffset + colorX;
			int depthX = static_cast<int>(m_pDepthCoordinates[colorIndex].X + 0.5f);
			int depthY = static_cast<int>(m_pDepthCoordinates[colorIndex].Y + 0.5f);
			//(i_RgbTodepthForshow.at<UINT8>(colorIndex / colorWidth, colorIndex % colorWidth) < thr) &&
			if ((i_RgbTodepthForshow.at<UINT8>(colorIndex / colorWidth, colorIndex % colorWidth) < thr) && (0 <= depthX) && (depthX < depthWidth) && (0 <= depthY) && (depthY < depthHeight)) {
				xSum += depthX;
				ySum += depthY;
				idx++;
			}
		}
	}
	if (idx > 0)
	{
		*x = xSum / idx;
		*y = ySum / idx;
	}
	else
	{
		*x = 0;
		*y = 0;
	}


}

inline void KinectUtil::caculateXYZinCameraSpace(object *RecObects, int objectNumPerFrame)
{
#pragma omp parallel for
	for (int i = 0; i < objectNumPerFrame; i++)
	{
		object b = RecObects[i];

		//int DepthLeft = max(0, (b.x - b.w / 2.)*colorMat.cols);
		//int DepthRight = min(colorMat.cols, (b.x + b.w / 2.)*colorMat.cols);
		//int DepthTop = max(0, (b.y - b.h / 2.)*colorMat.rows);
		//int DepthBot = min(colorMat.rows, (b.y + b.h / 2.)*colorMat.rows);

		int DepthLeft = max(0, (b.x - b.w * 3 / 4.)*colorWidth);
		int DepthRight = min(colorWidth, (b.x + b.w * 3 / 4.)*colorWidth);
		int DepthTop = max(0, (b.y - b.h / 2.)*colorHeight);
		int DepthBot = min(colorHeight, (b.y + b.h / 2.)*colorHeight);

		//calculate average depth
		cv::Mat imageROI8Bit = i_RgbTodepthForshow(cv::Rect(DepthLeft, DepthTop, DepthRight - DepthLeft, DepthBot - DepthTop));
		int thr = otsuThreshold(imageROI8Bit);
		cv::Mat imageROI = i_RgbTodepth(cv::Rect(DepthLeft, DepthTop, DepthRight - DepthLeft, DepthBot - DepthTop));
		thr = thr * 32;

		float AvgDepth = GetImgAvg(imageROI, thr);

		CameraSpacePoint cp;
		DepthSpacePoint d;
		int depthX, depthY;

		caculateXY(b, &d.X, &d.Y, thr);
		coordinateMapper->MapDepthPointToCameraSpace(d, AvgDepth, &cp);

		RecObects[i].CameraX = cp.X;
		RecObects[i].CameraY = cp.Y;
		RecObects[i].CameraZ = cp.Z;
	}

}

inline int KinectUtil::otsuThreshold(cv::Mat frame)
{
	int GrayScale = 256;
	int width = frame.cols;
	int height = frame.rows;
	int* pixelCount = new int[GrayScale];// [GrayScale];
	float* pixelPro = new float[GrayScale];
	int i, j, pixelSum = width * height, threshold = 0;
	//uchar* data = (uchar*)frame->imageData;

	for (i = 0; i < GrayScale; i++)
	{
		pixelCount[i] = 0;
		pixelPro[i] = 0;
	}

	//统计灰度级中每个像素在整幅图像中的个数  
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			pixelCount[frame.at<UINT8>(i, j)]++;
		}
	}
	if (pixelCount[0] > pixelSum * 0.85)
		return 0;
	//计算每个像素在整幅图像中的比例  
	pixelSum = pixelSum - pixelCount[0];

	pixelCount[0] = 0;
	for (i = 0; i < GrayScale; i++)
	{
		pixelPro[i] = (float)pixelCount[i] / pixelSum;
	}

	//遍历灰度级[1,255]  
	float w0, w1, u0tmp, u1tmp, u0, u1, u,
		deltaTmp, deltaMax = 0;
	for (i = 1; i < GrayScale; i++)
	{
		w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = 0;
		for (j = 1; j < GrayScale; j++)
		{
			if (j <= i)   //背景部分  
			{
				w0 += pixelPro[j];
				u0tmp += j * pixelPro[j];
			}
			else   //前景部分  
			{
				w1 += pixelPro[j];
				u1tmp += j * pixelPro[j];
			}
		}
		u0 = u0tmp / w0;
		u1 = u1tmp / w1;
		u = u0tmp + u1tmp;
		deltaTmp =
			w0 * pow((u0 - u), 2) + w1 * pow((u1 - u), 2);
		if (deltaTmp > deltaMax)
		{
			deltaMax = deltaTmp;
			threshold = i;
		}
	}
	return threshold;
}


KinectBody::KinectBody(cv::Vec3f vJoints[JointType_Count], TrackingState nStates[JointType_Count])
{
	//	memcpy(Joints2D, ptJoints, sizeof(cv::Point) * JointType_Count);
	memcpy(Joints3D, vJoints, sizeof(cv::Vec3f) * JointType_Count);
	memcpy(JointStates, nStates, sizeof(TrackingState)* JointType_Count);

	IsMainBody = true;
}


cv::Mat KinectUtil::DrawSkeletonFrame(cv::Mat Img)
{
	cv::Mat mtxRGB(Img.clone());
	for (int i = 0; i < m_vecKinectBodies.size(); i++)
		DrawBody(mtxRGB, m_vecKinectBodies[i]);

	//cv::imshow("co", mtxRGB);
	return mtxRGB;
}

void KinectUtil::DrawBody(cv::Mat &mtxRGBCanvas, KinectBody& body)
{
	// Torso
	DrawBone(mtxRGBCanvas, body, JointType_Head, JointType_Neck);
	DrawBone(mtxRGBCanvas, body, JointType_Neck, JointType_SpineShoulder);
	DrawBone(mtxRGBCanvas, body, JointType_SpineShoulder, JointType_SpineMid);
	DrawBone(mtxRGBCanvas, body, JointType_SpineMid, JointType_SpineBase);
	DrawBone(mtxRGBCanvas, body, JointType_SpineShoulder, JointType_ShoulderRight);
	DrawBone(mtxRGBCanvas, body, JointType_SpineShoulder, JointType_ShoulderLeft);
	DrawBone(mtxRGBCanvas, body, JointType_SpineBase, JointType_HipRight);
	DrawBone(mtxRGBCanvas, body, JointType_SpineBase, JointType_HipLeft);

	// Right Arm    
	DrawBone(mtxRGBCanvas, body, JointType_ShoulderRight, JointType_ElbowRight);
	DrawBone(mtxRGBCanvas, body, JointType_ElbowRight, JointType_WristRight);
	DrawBone(mtxRGBCanvas, body, JointType_WristRight, JointType_HandRight);
	DrawBone(mtxRGBCanvas, body, JointType_HandRight, JointType_HandTipRight);
	DrawBone(mtxRGBCanvas, body, JointType_WristRight, JointType_ThumbRight);

	// Left Arm
	DrawBone(mtxRGBCanvas, body, JointType_ShoulderLeft, JointType_ElbowLeft);
	DrawBone(mtxRGBCanvas, body, JointType_ElbowLeft, JointType_WristLeft);
	DrawBone(mtxRGBCanvas, body, JointType_WristLeft, JointType_HandLeft);
	DrawBone(mtxRGBCanvas, body, JointType_HandLeft, JointType_HandTipLeft);
	DrawBone(mtxRGBCanvas, body, JointType_WristLeft, JointType_ThumbLeft);

	// Right Leg
	DrawBone(mtxRGBCanvas, body, JointType_HipRight, JointType_KneeRight);
	DrawBone(mtxRGBCanvas, body, JointType_KneeRight, JointType_AnkleRight);
	DrawBone(mtxRGBCanvas, body, JointType_AnkleRight, JointType_FootRight);

	// Left Leg
	DrawBone(mtxRGBCanvas, body, JointType_HipLeft, JointType_KneeLeft);
	DrawBone(mtxRGBCanvas, body, JointType_KneeLeft, JointType_AnkleLeft);
	DrawBone(mtxRGBCanvas, body, JointType_AnkleLeft, JointType_FootLeft);

	// draw the joints
	for (int i = 0; i < JointType_Count; i++)
	{
		cv::Point pt0 = ProjectDepthToColorPixel(body.Joints3D[i]);
		if (pt0.x != -std::numeric_limits<int>::infinity() && pt0.y != -std::numeric_limits<int>::infinity())
		{
			cv::Point ptRGB(1.0 * pt0.x / colorWidth * mtxRGBCanvas.cols, 1.0 * pt0.y / colorHeight * mtxRGBCanvas.rows);
			cv::circle(mtxRGBCanvas, ptRGB, 6, cv::Scalar(0, 255, 0), -1);

		}
	}

	// highlight the middle spine to determine whether or not activate gesture recognition
	cv::Point ptSignal = ProjectDepthToColorPixel(body.Joints3D[JointType_SpineMid]);
	if (ptSignal.x != -std::numeric_limits<int>::infinity() && ptSignal.y != -std::numeric_limits<int>::infinity())
	{
		cv::Point ptRGB(1.0 * ptSignal.x / colorWidth * mtxRGBCanvas.cols, 1.0 * ptSignal.y / colorHeight * mtxRGBCanvas.rows);
		cv::circle(mtxRGBCanvas, ptRGB, 8, cv::Scalar(0, 0, 255), -1);
	}
}

void KinectUtil::DrawBone(cv::Mat &mtxRGBCanvas, const KinectBody& body, JointType joint0, JointType joint1)
{
	TrackingState joint0State = body.JointStates[joint0];
	TrackingState joint1State = body.JointStates[joint1];

	// If we can't find either of these joints, exit
	if ((joint0State == TrackingState_NotTracked) || (joint1State == TrackingState_NotTracked))
	{
		return;
	}

	// Don't draw if both points are inferred
	if ((joint0State == TrackingState_Inferred) && (joint1State == TrackingState_Inferred))
	{
		return;
	}

	// We assume all drawn bones are inferred unless BOTH joints are tracked
	cv::Point pt0 = ProjectDepthToColorPixel(body.Joints3D[joint0]);
	cv::Point pt1 = ProjectDepthToColorPixel(body.Joints3D[joint1]);
	if (pt0.x != -std::numeric_limits<int>::infinity() && pt0.y != -std::numeric_limits<int>::infinity()
		&& pt1.x != -std::numeric_limits<int>::infinity() && pt1.y != -std::numeric_limits<int>::infinity())
	{
		cv::Point ptRGB0(1.0 * pt0.x / colorWidth * mtxRGBCanvas.cols, 1.0 * pt0.y / colorHeight * mtxRGBCanvas.rows);
		cv::Point ptRGB1(1.0 * pt1.x / colorWidth * mtxRGBCanvas.cols, 1.0 * pt1.y / colorHeight * mtxRGBCanvas.rows);
		if ((joint0State == TrackingState_Tracked) && (joint1State == TrackingState_Tracked))
		{
			if (body.IsMainBody == true)
				cv::line(mtxRGBCanvas, ptRGB0, ptRGB1, cv::Scalar(0, 0, 255), 4);
			else
				cv::line(mtxRGBCanvas, ptRGB0, ptRGB1, cv::Scalar(255, 0, 0), 4);
		}
		else
		{
			if (body.IsMainBody == true)
				cv::line(mtxRGBCanvas, ptRGB0, ptRGB1, cv::Scalar(0, 0, 255), 2);
			else
				cv::line(mtxRGBCanvas, ptRGB0, ptRGB1, cv::Scalar(255, 0, 0), 2);
		}
	}
}

cv::Point KinectUtil::ProjectDepthToColorPixel(cv::Vec3f v)
{
	CameraSpacePoint v0;
	ColorSpacePoint pt0;
	v0.X = v[0];
	v0.Y = v[1];
	v0.Z = v[2];
	coordinateMapper->MapCameraPointToColorSpace(v0, &pt0);

	if (pt0.X < 0 || pt0.Y < 0 || pt0.X >= 1920 || pt0.Y >= 1080)
		pt0 = pt0;

	if (pt0.X != -std::numeric_limits<float>::infinity() && pt0.Y != -std::numeric_limits<float>::infinity())
		return cv::Point(pt0.X, pt0.Y);
	else
		return cv::Point(-std::numeric_limits<int>::infinity(), -std::numeric_limits<int>::infinity());
}