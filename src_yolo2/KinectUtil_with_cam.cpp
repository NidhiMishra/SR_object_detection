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
extern void plane_segmentation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, UINT16 *depthBuffer, UINT16 depthHeight, UINT16 depthWidth);

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

#ifdef i2p
KinectUtil::KinectUtil(char *datacfg, char *namelist, char *cfgfile, char *weightfile, float num_thresh, ProtectedClient<imi::ObjectDetectionServiceClient>* mclient) {
	objectClient = mclient;
	initialize(datacfg, namelist, cfgfile, weightfile);
	thresh = num_thresh;
	isUseThrift = true;
}
#endif

KinectUtil::KinectUtil(char *datacfg, char *namelist, char *cfgfile, char *weightfile, float num_thresh) {
	initialize(datacfg, namelist, cfgfile, weightfile);
	thresh = num_thresh;
	isUseThrift = false;
}

KinectUtil::~KinectUtil()
{
	//release_image_cv(disp_main);
	//cvReleaseImage(&disp_main);
	finalize();
}

// Processing
void KinectUtil::run(objectDetectionEvent objEvent)
{
	// Main Loop
	while (true) {
		update();
		drawDepth();
		show(objEvent);
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
	initializeBodyIndex();//Initialize Body Idx

	i_RgbTodepthForshow.create(1080, 1920, CV_8UC1);//show 8bit
	i_RgbTodepth.create(1080, 1920, CV_16UC1);//calculate the distance 16bit
	i_RgbTodepthForGrasping.create(1080, 1920, CV_16UC1);//for grasping
	m_pDepthCoordinates = new DepthSpacePoint[1920 * 1080];
	
	frame = 0;
	trackingInterval = 5;
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

	eventFlagForDemo.eventBackhome = 0;
	eventFlagForDemo.eventTakeBagOff = 0;
	eventFlagForDemo.eventTakeABook = 0;
	eventFlagForDemo.eventTakeACup = 0;
	eventFlagForDemo.eventTakeABottle = 0;
	eventFlagForDemo.eventSeatInChair = 0;
	eventFlagForDemo.eventForgetBottle = 0;
	eventFlagForDemo.eventDrinkWater = 0;
	eventFlagForDemo.eventRemindWater = 0;
	eventFlagForDemo.eventPersonLeaving = 0;

	objectFlagForDemoWhatitis.bottleflag = 0;
	objectFlagForDemoWhatitis.bowlflag = 0;
	objectFlagForDemoWhatitis.cupflag = 0;
	objectFlagForDemoWhatitis.phoneflag = 0;
	objectFlagForDemoWhatitis.wineglassflag = 0;
	objectFlagForDemoWhatitis.bookflag = 0;


	tmpDemoflag = 0;
	frameIdx = 0;
	demoIdx = 0;
	controlflag = 0;

	//disp_main = cvCreateImage(cvSize(1920, 1080), IPL_DEPTH_8U, 4);
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
	depthBufferGrasping.resize(depthWidth * depthHeight);
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
	// Update Body Index
	updateBodyIndex();
}

//Extra Functions from ImageProcessor.h
inline void GetDistinctColor(int n, unsigned char &R, unsigned char &G, unsigned char &B)
{
	n = n % 56;
	static int nColourValues[56] =
	{
		0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0xFF00FF, 0x00FFFF, 0x800000,
		0x008000, 0x000080, 0x808000, 0x800080, 0x008080, 0x808080, 0xC00000,
		0x00C000, 0x0000C0, 0xC0C000, 0xC000C0, 0x00C0C0, 0xC0C0C0, 0x004040,
		0x400000, 0x004000, 0x000040, 0x404000, 0x400040, 0x404040, 0x200000,
		0x002000, 0x000020, 0x202000, 0x200020, 0x002020, 0x202020, 0x600000,
		0x006000, 0x000060, 0x606000, 0x600060, 0x006060, 0x606060, 0xA00000,
		0x00A000, 0x0000A0, 0xA0A000, 0xA000A0, 0x00A0A0, 0xA0A0A0, 0xE00000,
		0x00E000, 0x0000E0, 0xE0E000, 0xE000E0, 0x00E0E0, 0xE0E0E0, 0x000000
	};
	R = (nColourValues[n] & 0xFF0000) >> 16;
	G = (nColourValues[n] & 0x00FF00) >> 8;
	B = nColourValues[n] & 0x0000FF;
}

//Update Body Index
inline void KinectUtil::updateBodyIndex()
{
	int nWidth, nHeight;
	UINT nBufferSize = 0;
	BYTE *pBuffer = NULL;
	HRESULT hr = BodyIndexReader->AcquireLatestFrame(&bodyIndexFrame);
	if (SUCCEEDED(hr))	hr = bodyIndexFrame->get_FrameDescription(&pFrameDescription);
	if (SUCCEEDED(hr))	hr = pFrameDescription->get_Width(&nWidth);
	if (SUCCEEDED(hr))	hr = pFrameDescription->get_Height(&nHeight);
	if (SUCCEEDED(hr))	hr = bodyIndexFrame->AccessUnderlyingBuffer(&nBufferSize, &pBuffer);

	// make sure we've received valid data and update the body index frame
	if (SUCCEEDED(hr) && pBuffer && (nWidth == depthWidth) && (nHeight == depthHeight))
	{
		//Foreground_points.clear();
		static cv::Mat mtxLabels(depthHeight,depthWidth, CV_32SC1);
		static cv::Mat mtxIdx(depthHeight, depthWidth, CV_8UC1);
		static cv::Mat mtxLabelColors(depthHeight, depthWidth, CV_8UC3);
		for (int i = 0; i < nWidth * nHeight; i++)
		{
			unsigned char R, G, B;
			GetDistinctColor(pBuffer[i], R, G, B);

			if (pBuffer[i] != 0xff)
			{
				*((int*)mtxLabels.data + i) = pBuffer[i]+1;
				*((cv::Vec3b*)mtxLabelColors.data + i) = cv::Vec3b(B, G, R);
				//*(mtxIdx.data + i) = 255;
			}
			else
			{
				*((int*)mtxLabels.data + i) = 0xff;
				*((cv::Vec3b*)mtxLabelColors.data + i) = cv::Vec3b(0, 0, 0);
				//*(mtxIdx.data + i) = 0;
			}
		}
		//bodyLabels = mtxIdx.clone();
		bodyLabels = mtxLabels.clone();
		//cv::imshow("Label", mtxLabelColors);
	}
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
						ptJoints[joints[j].JointType] = ProjectToPixel(vJoints[joints[j].JointType]);
						nStates[joints[j].JointType] = joints[j].TrackingState;
					}
					KinectBody Tempbody(ptJoints, vJoints, nStates);
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

	desk_seg(1.0);//1.0m,get the depth map for grapsing, remove  background and desk
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
	std::vector<UINT16> bufferForGrasping(colorWidth * colorHeight);
	std::vector<UINT8> buffer8bit(colorWidth * colorHeight);
	std::vector<UINT32> bufferPersonIdx(colorWidth * colorHeight);

#pragma omp parallel for
	for (int colorY = 0; colorY < colorHeight; colorY++) {
		unsigned int colorOffset = colorY * colorWidth;
		for (int colorX = 0; colorX < colorWidth; colorX++) {
			unsigned int colorIndex = colorOffset + colorX;
			bufferPersonIdx[colorIndex] = 255;
			m_pDepthCoordinates[colorIndex] = depthSpace[colorIndex];
			int depthX = static_cast<int>(depthSpace[colorIndex].X + 0.5f);
			int depthY = static_cast<int>(depthSpace[colorIndex].Y + 0.5f);
			if ((0 <= depthX) && (depthX < depthWidth) && (0 <= depthY) && (depthY < depthHeight)) {
				unsigned int depthIndex = depthY * depthWidth + depthX;
				buffer[colorIndex] = depthBuffer[depthIndex];
				buffer8bit[colorIndex] = static_cast<BYTE>(depthBuffer[depthIndex] >> 5);
				bufferPersonIdx[colorIndex] = bodyLabels.at<UINT32>(depthY, depthX);
				bufferForGrasping[colorIndex] = depthBufferGrasping[depthIndex];
			}
		}
	}

	// Create cv::Mat from Coordinate Buffer
	//depthMat = cv::Mat(colorHeight, colorWidth, CV_16UC1, &buffer[0]).clone();
	cv::Mat dMat = cv::Mat(colorHeight, colorWidth, CV_16UC1, &buffer[0]).clone();
	cv::Mat dMatForGrasping = cv::Mat(colorHeight, colorWidth, CV_16UC1, &bufferForGrasping[0]).clone();
	cv::Mat dMat8bit = cv::Mat(colorHeight, colorWidth, CV_8UC1, &buffer8bit[0]).clone();
	cv::Mat dMatPersonIdx = cv::Mat(colorHeight, colorWidth, CV_32SC1, &bufferPersonIdx[0]).clone();
	if (!dMat.empty())
		depthMat = dMat;
	if (!dMat8bit.empty())
		depthMat8bit = dMat8bit;
	i_RgbTodepth = depthMat;
	i_RgbTodepthForshow = depthMat8bit;
	i_PersonIdx = dMatPersonIdx;
	i_RgbTodepthForGrasping = dMatForGrasping;
//	cv::Mat showMat;
	//cv::resize(dMat8bit, showMat, cv::Size(colorWidth / 2, colorHeight/2), (0, 0), (0, 0), cv::INTER_LINEAR);
//	cv::imshow("Label", showMat);
}

// Show Data
/*void KinectUtil::show()
{
	cv::Mat cMat = cv::Mat(colorHeight, colorWidth, CV_8UC4, &colorBuffer[0]);
	if (!cMat.empty())
		colorMat = cMat;
	detection();
}*/

void KinectUtil::show(objectDetectionEvent objEvent)
{
	cv::Mat cMat = cv::Mat(colorHeight, colorWidth, CV_8UC4, &colorBuffer[0]);
	if (!cMat.empty())
		colorMat = cMat;
	detection(objEvent);
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

#ifdef i2p
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
#endif

void KinectUtil::write_infor_to_txt(object *RecObects, int *objectNumPerFrame)
{
	string pathRst = "\\\\imi-storage\\04_SharedBox\\_Rui Yang\\Programming\\Object_Coordinates.txt";
	
		ofstream fCtr(pathRst.c_str(), ios::out);
		if (fCtr)
		{
			fCtr << "objNumber = " << *objectNumPerFrame << endl;

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

void KinectUtil::write_infor_to_txt_grasp(object *RecObects, int *objectNumPerFrame)
{
	//string pathRst = "\\\\imi-storage\\04_SharedBox\\_zwfang\\objectRecognition\\test.txt";
	//string pathRst = "\\\\IMI-7736-FANGZH\\shareFile\\test.file";
	string pathRst = "\\\\imi-storage\\04_SharedBox\\_Rui Yang\\Programming\\Object_Coordinates.txt";

	ofstream fCtr(pathRst.c_str(), ios::out);
	if (fCtr)
	{
		
		//fCtr.setf(ios::fixed);
		for (int i = 0; i < *objectNumPerFrame; i++)
		{
			fCtr << "(" << setprecision(3) << -RecObects[i].CameraX * 100 << " " << RecObects[i].CameraZ * 100 << " " << RecObects[i].CameraY * 100 << ")[" << 0 <<
				" " << RecObects[i].CameraWidth * 100 << " " << RecObects[i].CameraHeight * 100 << "]{" << RecObects[i].name << "}";
			//				fCtr << "(" << fixed << RecObects[i].CameraX << " " << RecObects[i].CameraY << " " << RecObects[i].CameraZ << ")[" << 0 <<
			//					" " << RecObects[i].CameraWidth << " " << RecObects[i].CameraHeight << "]{" << RecObects[i].name << "}";

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

void KinectUtil::write_infor_to_txt_general_obj(object *RecObects, int objectNumPerFrame)
{
	string pathRst = "C:/xampp/htdocs/Reader/Object_predictions.txt";

	object *cateObects = new object[maxObjectNum];
	int *cnt_each_cate = new int[maxObjectNum]();

	int flag = 0;
	unsigned char cateCount = 0;
	for (int i = 0; i < objectNumPerFrame; i++)
	{
		flag = 0;
		for (int j = 0; j < cateCount; j++)
		{
			if (strcmp(RecObects[i].name, cateObects[j].name) == 0)
			{
				cnt_each_cate[j]++;
				flag = 1;
			}
		}
		if (flag == 0)
		{
			cateObects[cateCount] = RecObects[i];
			cnt_each_cate[cateCount]++;
			cateCount++;
		}
	}

	ofstream fCtr(pathRst.c_str(), ios::out);
	if (fCtr && objectNumPerFrame > 0)
	{
		//fCtr << "General\\";
		/*for (int i = 0; i < cateCount; i++)
		{
			fCtr << cateObects[i].name << '-' << to_string(cnt_each_cate[i]) << endl;
		}
		//fCtr << RecObects[objectNumPerFrame-1].name << endl;
		*/
		for (int i = 0; i < objectNumPerFrame; i++)
		{
			//fCtr << RecObects[i].name << '-' << to_string(RecObects[i].CameraX) << '-' << to_string(RecObects[i].CameraY) << '-' << to_string(RecObects[i].CameraZ) << endl;
			//if (strcmp(RecObects[i].name, "person") != 0)
				fCtr << RecObects[i].name << endl;
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
	delete[] cateObects;
	delete[] cnt_each_cate;
}

/*void KinectUtil::write_infor_to_txt_handheld_obj(object *RecObects, int objectNumPerFrame)
{
	string pathRst = "Handheld_Object_Inf.txt";

	ofstream fCtr(pathRst.c_str(), ios::out);
	if (fCtr && objectNumPerFrame > 0)
	{
		int idx = 0;
		float maxprob = 0;
		for (int i = 0; i < objectNumPerFrame; i++)
		{
			if (RecObects[i].prob > maxprob)
			{
				idx = i;
				maxprob = RecObects[i].prob;
			}
		}
		fCtr << RecObects[idx].name;
		fCtr << endl;

		fCtr.close();
	}

}*/

void KinectUtil::write_infor_to_txt_left_handheld_obj(object *RecObects, int objectNumPerFrame)
{
	string pathRst = "C:/xampp/htdocs/Reader/leftHandheld_Object_predictions.txt";

	ofstream fCtr(pathRst.c_str(), ios::out);
	if (fCtr && objectNumPerFrame > 0)
	{
		int idx = 0;
		float maxprob = 0;
		for (int i = 0; i < objectNumPerFrame; i++)
		{
			if (RecObects[i].prob > maxprob)
			{
				idx = i;
				maxprob = RecObects[i].prob;
			}
		}
		fCtr << RecObects[idx].name;
		fCtr << endl;

		fCtr.close();
	}
}

void KinectUtil::write_infor_to_txt_right_handheld_obj(object *RecObects, int objectNumPerFrame)
{
	string pathRst = "C:/xampp/htdocs/Reader/rightHandheld_Object_predictions.txt";

	ofstream fCtr(pathRst.c_str(), ios::out);
	if (fCtr && objectNumPerFrame > 0)
	{
		int idx = 0;
		float maxprob = 0;
		for (int i = 0; i < objectNumPerFrame; i++)
		{
			if (RecObects[i].prob > maxprob)
			{
				idx = i;
				maxprob = RecObects[i].prob;
			}
		}
		fCtr << RecObects[idx].name;
		fCtr << endl;

		fCtr.close();
	}
}

void KinectUtil::write_infor_to_txt_carried_obj(object *RecObects, int objectNumPerFrame)
{
	string pathRst = "Carried_Object_Inf.txt";

	ofstream fCtr(pathRst.c_str(), ios::out);
	if (fCtr && objectNumPerFrame > 0)
	{
		//fCtr << "General\\";
		for (int i = 0; i < objectNumPerFrame-1; i++)
		{
			fCtr << RecObects[i].name << ",";
		}
		fCtr << RecObects[objectNumPerFrame - 1].name << endl;

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

inline void KinectUtil::InitialTracker(object *RecObects, int objectNumPerFrame)
{
	trackers.clear();
	tracking_objects.clear();
	int imgWidth = colorMat.cols;
	int imgHeight = colorMat.rows;
	for (int i = 0; i < objectNumPerFrame; i++)
	{
		KCF_Tracker tracker;
		BBox_c bb;
		bb.cx = RecObects[i].x * imgWidth;
		bb.cy = RecObects[i].y * imgHeight;
		bb.w = RecObects[i].w * imgWidth;
		bb.h = RecObects[i].h * imgHeight;
		tracker.init(colorMat, bb);
		trackers.push_back(tracker);
		tracking_objects.push_back(RecObects[i]);
	}
}

inline void KinectUtil::test_tracker_img(object *RecObects, int *objectNumPerFrame)
{
	*objectNumPerFrame = trackers.size();
	int N = *objectNumPerFrame;
	cv::Mat im = colorMat;
	int imgWidth = im.cols;
	int imgHeight = im.rows;
	#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		KCF_Tracker tracker = trackers[i];
		tracker.track(im);
		BBox_c bb = tracker.getBBox();
		tracking_objects[i].w = bb.w / imgWidth;
		tracking_objects[i].h = bb.h / imgHeight;
		tracking_objects[i].x = bb.cx / imgWidth;
		tracking_objects[i].y = bb.cy / imgHeight;
		RecObects[i] = tracking_objects[i];
	}
}

inline void KinectUtil::voice(string str)
{//"..\\x64\\Release\\voice.exe \"I want to have a rest!\""
	WinExec(str.c_str(), SW_HIDE);
}

string KinectUtil::object2str(object *RecObects, int objectNumPerFrame, objectDetectionEvent objEvent)
{
	string str = "..\\x64\\Release\\voice.exe \"";
	if (objEvent == objectDetectionEvent::Demo_what)
	{
		object tmpObj;
		float maxPro = 0;
		for (int i = 0; i < objectNumPerFrame; i++)
		{
			if (RecObects[i].prob > maxPro)
			{
				maxPro = RecObects[i].prob;
				tmpObj = RecObects[i];
			}
		}
		if (strcmp(tmpObj.name, "cup") == 0 && objectFlagForDemoWhatitis.cupflag == 0)
		{
			str = str + "I see you take a cup. would you like a cup of coffe?";
			objectFlagForDemoWhatitis.cupflag = 1;
			objectFlagForDemoWhatitis.bookflag = 0;
			objectFlagForDemoWhatitis.bottleflag = 0;
			objectFlagForDemoWhatitis.phoneflag = 0;
			objectFlagForDemoWhatitis.wineglassflag = 0;

		}
		else if (strcmp(tmpObj.name, "bottle") == 0 && objectFlagForDemoWhatitis.bottleflag == 0)
		{
			str = str + "The bottle is empty. I will call somebody to give you a new one.";
			objectFlagForDemoWhatitis.cupflag = 0;
			objectFlagForDemoWhatitis.bookflag = 0;
			objectFlagForDemoWhatitis.bottleflag = 1;
			objectFlagForDemoWhatitis.phoneflag = 0;
			objectFlagForDemoWhatitis.wineglassflag = 0;
		}
		else if (strcmp(tmpObj.name, "book") == 0 && objectFlagForDemoWhatitis.bookflag == 0)
		{
			str = str + "You take a book. Reading is a good habit.";
			objectFlagForDemoWhatitis.cupflag = 0;
			objectFlagForDemoWhatitis.bookflag = 1;
			objectFlagForDemoWhatitis.bottleflag = 0;
			objectFlagForDemoWhatitis.phoneflag = 0;
			objectFlagForDemoWhatitis.wineglassflag = 0;
		}
		else if (strcmp(tmpObj.name, "wine glass") == 0 && objectFlagForDemoWhatitis.wineglassflag == 0)
		{
			str = str + "You take a wine glass. Do you have anything to celebrate?";
			objectFlagForDemoWhatitis.cupflag = 0;
			objectFlagForDemoWhatitis.bookflag = 0;
			objectFlagForDemoWhatitis.bottleflag = 0;
			objectFlagForDemoWhatitis.phoneflag = 0;
			objectFlagForDemoWhatitis.wineglassflag = 1;
		}
		else if (strcmp(tmpObj.name, "cell phone") == 0 && objectFlagForDemoWhatitis.phoneflag == 0)
		{
			str = str + "You take a cell phone. would you want to call somebody?";
			objectFlagForDemoWhatitis.cupflag = 0;
			objectFlagForDemoWhatitis.bookflag = 0;
			objectFlagForDemoWhatitis.bottleflag = 0;
			objectFlagForDemoWhatitis.phoneflag = 1;
			objectFlagForDemoWhatitis.wineglassflag = 0;
		}
		str = str + "\"";
		return str;
	}
	
}

inline void KinectUtil::detection(objectDetectionEvent objEvent)
{
	if (colorMat.empty() || depthMat.empty() || depthFrame == NULL) {
		return;
	}
	frame++;

	//userRGB2Depth();//map RGB to Depth.
	object *RecObects = new object[maxObjectNum];

	objectNumPerFrame = 0;
#ifdef i2p
	std::vector< ::imi::ObjectInfo> detectedObjects; // Data to be transimited
#endif
	IplImage testImg = colorMat;
	image im = ipl_to_image(&testImg);
		
	//Detect objects for per image.
	if (objEvent == objectDetectionEvent::General){

	}
	else if (objEvent == objectDetectionEvent::Grasp)
	{		
		

	}
	else if (objEvent == objectDetectionEvent::Demo_what)//demo for handheld object detection.
	{
		int Id = 255;
		int personCount = 0;
		int handLeftIdx = 7;
		int handRightIdx = 11;
		int headIdx = 3;
		std::vector<cv::Vec3f> handLeftAll_3D, handRightALL_3D, headALL_3D;
		std::vector<cv::Point> handLeftAll_2D, handRightALL_2D, headALL_2D;
		std::vector<TrackingState> handLeftALL_nStates, handRightALL_nStates, headALL_nStates;
		int selectedBodyIdx = -1;
		float minDistance = 10000;
		int bodyNum = m_vecKinectBodies.size();
		int flagHand = 0, flagLeftHand = 0, flagRightHand = 0;

		object *RecObectsLeftHand = new object[maxObjectNum];
		int objNumLeftHand = 0;
		object *RecObectsRightHand = new object[maxObjectNum];
		int objNumRightHand = 0;

		image imLocalLeftHand, imLocalRightHand;
		//test_detector_img(names, alphabet, net, im, thresh, RecObects, &objectNumPerFrame);
		//objectFilterUsingObjectCategory(RecObects, &objectNumPerFrame, 1);
		//object_show(im, names, alphabet, l.classes, RecObects, &objectNumPerFrame);
		//objectNumPerFrame = 0;
		if (bodyNum > 0)
		{
			for (int idx = 0; idx < bodyNum; idx++)
			{
				KinectBody Tempbody = m_vecKinectBodies[idx];

				TrackingState nStates[JointType_Count];
				cv::Vec3f vJoints[JointType_Count];
				cv::Point ptJoints[JointType_Count];
				memcpy(vJoints, Tempbody.Joints3D, sizeof(cv::Vec3f) * JointType_Count);
				memcpy(ptJoints, Tempbody.Joints2D, sizeof(cv::Point) * JointType_Count);
				memcpy(nStates, Tempbody.JointStates, sizeof(TrackingState) * JointType_Count);

				cv::Vec3f vJointsHandLeft = vJoints[handLeftIdx];
				handLeftAll_3D.push_back(vJointsHandLeft);
				cv::Point ptJointsHandLeft = ptJoints[handLeftIdx];
				handLeftAll_2D.push_back(ptJointsHandLeft);
				TrackingState nStateHandLeft = nStates[handLeftIdx];
				handLeftALL_nStates.push_back(nStateHandLeft);

				cv::Vec3f vJointsHandRight = vJoints[handRightIdx];
				handRightALL_3D.push_back(vJointsHandRight);
				cv::Point ptJointsHandRight = ptJoints[handRightIdx];
				handRightALL_2D.push_back(ptJointsHandRight);
				TrackingState nStateHandRight = nStates[handRightIdx];
				handRightALL_nStates.push_back(nStateHandRight);

				cv::Vec3f vJointsHead = vJoints[headIdx];
				headALL_3D.push_back(vJointsHead);
				cv::Point ptJointsHead = ptJoints[headIdx];
				headALL_2D.push_back(ptJointsHead);
				TrackingState nStateHead = nStates[headIdx];
				headALL_nStates.push_back(nStateHead);
				

				if (nStateHead == TrackingState_Tracked && (vJointsHead[2] * vJointsHead[2] + vJointsHead[1] * vJointsHead[1] + vJointsHead[0] * vJointsHead[0] < minDistance))
				{
					minDistance = vJointsHead[2] * vJointsHead[2] + vJointsHead[1] * vJointsHead[1] + vJointsHead[0] * vJointsHead[0];
					selectedBodyIdx = idx;
					Id = i_PersonIdx.at<INT32>(headALL_2D[selectedBodyIdx].y, headALL_2D[selectedBodyIdx].x);
				}
			}		

			cv::Point selectedPosition;
			selectedPosition.x = -500;
			selectedPosition.y = -500;
			float jointDistance = 0;
			if (handLeftALL_nStates[selectedBodyIdx] != TrackingState_Inferred && handRightALL_nStates[selectedBodyIdx] == TrackingState_Inferred)
			{
				//selectedPosition = handLeftAll_2D[selectedBodyIdx];
				//jointDistance = handLeftAll_3D[selectedBodyIdx][2];
				flagLeftHand = 1;
			}
			else if (handLeftALL_nStates[selectedBodyIdx] == TrackingState_Inferred && handRightALL_nStates[selectedBodyIdx] != TrackingState_Inferred)
			{
				//selectedPosition = handRightALL_2D[selectedBodyIdx];
				//jointDistance = handRightALL_3D[selectedBodyIdx][2];
				flagRightHand = 1;
			}
			else if (handLeftALL_nStates[selectedBodyIdx] != TrackingState_Inferred && handRightALL_nStates[selectedBodyIdx] != TrackingState_Inferred)
			{
				/*if (handLeftAll_3D[selectedBodyIdx][2] <= handRightALL_3D[selectedBodyIdx][2])
				{
					selectedPosition = handLeftAll_2D[selectedBodyIdx];
					jointDistance = handLeftAll_3D[selectedBodyIdx][2];
				}
				else if (handLeftAll_3D[selectedBodyIdx][2] > handRightALL_3D[selectedBodyIdx][2])
				{
					selectedPosition = handRightALL_2D[selectedBodyIdx];
					jointDistance = handRightALL_3D[selectedBodyIdx][2];
				}*/
				flagLeftHand = 1;
				flagRightHand = 1;
			}
			
			if (flagLeftHand == 1)
			{
				selectedPosition = handLeftAll_2D[selectedBodyIdx];
				jointDistance = handLeftAll_3D[selectedBodyIdx][2];

				float rate = 1 / jointDistance;
				int width = int(544 * rate);
				int height = int(544 * rate);
				int Left = max(200, selectedPosition.x - width / 2);//remove the black block,because RGB image is larger than Depth image
				int Top = max(1, selectedPosition.y - height / 2);
				int Right = min(selectedPosition.x + width / 2, colorMat.cols - 200);//remove the black block
				int Down = min(selectedPosition.y + height / 2, colorMat.rows - 1);
				width = Right - Left;
				height = Down - Top;
				if (selectedPosition.y != -500 && selectedPosition.x != -500 && width > 20 && height > 20 && Right > 0 && Down > 0 && Left < colorMat.cols - 1 && Top < colorMat.rows - 1)
				{
					flagHand = 1;
					cv::Mat localImg = colorMat(cv::Rect(Left, Top, width, height));
					cv::Mat localImgDepth = i_RgbTodepthForshow(cv::Rect(Left, Top, width, height));
					localImg = colorImgFilterbyDistance(localImg, jointDistance + 0.3, localImgDepth);

					IplImage testLocalImg = localImg;
					imLocalLeftHand = ipl_to_image(&testLocalImg);

					float t = 0.15;

					test_detector_img(names, alphabet, net, imLocalLeftHand, t, RecObectsLeftHand, &objNumLeftHand);

					for (int idx = 0; idx < objNumLeftHand; idx++)
					{
						RecObectsLeftHand[idx].x = (RecObectsLeftHand[idx].x * width + Left) / colorMat.cols;
						RecObectsLeftHand[idx].y = (RecObectsLeftHand[idx].y * height + Top) / colorMat.rows;
						RecObectsLeftHand[idx].w = RecObectsLeftHand[idx].w * width / colorMat.cols;
						RecObectsLeftHand[idx].h = RecObectsLeftHand[idx].h * height / colorMat.rows;
					}
					//free_image(imLocal);

					//cv::Mat showImLocal;
					//cv::resize(localImg, showImLocal, cv::Size(localImg.rows / 2, localImg.cols / 2), (0, 0), (0, 0), cv::INTER_LINEAR);
					//cv::imshow("handRegion",showImLocal);
				}
				else
					flagHand = 0;
			}
			if (flagRightHand == 1)
			{
				selectedPosition = handRightALL_2D[selectedBodyIdx];
				jointDistance = handRightALL_3D[selectedBodyIdx][2];

				float rate = 1 / jointDistance;
				int width = int(544 * rate);
				int height = int(544 * rate);
				int Left = max(200, selectedPosition.x - width / 2);//remove the black block,because RGB image is larger than Depth image
				int Top = max(1, selectedPosition.y - height / 2);
				int Right = min(selectedPosition.x + width / 2, colorMat.cols - 200);//remove the black block
				int Down = min(selectedPosition.y + height / 2, colorMat.rows - 1);
				width = Right - Left;
				height = Down - Top;
				if (selectedPosition.y != -500 && selectedPosition.x != -500 && width > 20 && height > 20 && Right > 0 && Down > 0 && Left < colorMat.cols - 1 && Top < colorMat.rows - 1)
				{
					flagHand = 1;
					cv::Mat localImg = colorMat(cv::Rect(Left, Top, width, height));
					cv::Mat localImgDepth = i_RgbTodepthForshow(cv::Rect(Left, Top, width, height));
					localImg = colorImgFilterbyDistance(localImg, jointDistance + 0.3, localImgDepth);

					IplImage testLocalImg = localImg;
					imLocalRightHand = ipl_to_image(&testLocalImg);

					float t = 0.15;

					test_detector_img(names, alphabet, net, imLocalRightHand, t, RecObectsRightHand, &objNumRightHand);

					for (int idx = 0; idx < objNumRightHand; idx++)
					{
						RecObectsRightHand[idx].x = (RecObectsRightHand[idx].x * width + Left) / colorMat.cols;
						RecObectsRightHand[idx].y = (RecObectsRightHand[idx].y * height + Top) / colorMat.rows;
						RecObectsRightHand[idx].w = RecObectsRightHand[idx].w * width / colorMat.cols;
						RecObectsRightHand[idx].h = RecObectsRightHand[idx].h * height / colorMat.rows;
					}
				}
				else
					flagHand = 0;
			}
		}			

		//Record the objects in the hand
		if (flagHand == 1 && objNumLeftHand > 0 && flagLeftHand == 1)
		{
			objectFilterUsingObjectCategory(RecObectsLeftHand, &objNumLeftHand, objEvent);
			objectBelong2Person(RecObectsLeftHand, objNumLeftHand);
			objectFilterSpecialID(RecObectsLeftHand, &objNumLeftHand, Id);
			object_show(im, names, alphabet, l.classes, RecObectsLeftHand, &objNumLeftHand);
			write_infor_to_txt_left_handheld_obj(RecObectsLeftHand, objNumLeftHand);
		}
		//else
		//	write_infor_to_txt_left_handheld_obj(RecObects, objectNumPerFrame);
		if (flagHand == 1 && objNumRightHand > 0 && flagRightHand == 1)
		{
			objectFilterUsingObjectCategory(RecObectsRightHand, &objNumRightHand, objEvent);
			objectBelong2Person(RecObectsRightHand, objNumRightHand);
			objectFilterSpecialID(RecObectsRightHand, &objNumRightHand, Id);
			object_show(im, names, alphabet, l.classes, RecObectsRightHand, &objNumRightHand);
			write_infor_to_txt_right_handheld_obj(RecObectsRightHand, objNumRightHand);
		}

		//Record the objects in the room
		objectNumPerFrame = 0;
		test_detector_img(names, alphabet, net, im, thresh, RecObects, &objectNumPerFrame);
		//objectFilterUsingObjectCategory(RecObects, &objectNumPerFrame, 0);
		object_show(im, names, alphabet, l.classes, RecObects, &objectNumPerFrame);
		write_infor_to_txt_general_obj(RecObects, objectNumPerFrame);
		//Record the objects carried by the person
		objectFilterUsingObjectCategory(RecObects, &objectNumPerFrame, objEvent);
		objectBelong2Person(RecObects, objectNumPerFrame);
		objectFilterSpecialID(RecObects, &objectNumPerFrame, Id);
		write_infor_to_txt_carried_obj(RecObects, objectNumPerFrame);

		skeletonShow(im, selectedBodyIdx);//255 for all; 0-5 for special body; others for empty

		if (flagHand == 1)
		{
			if (flagLeftHand == 1)
			{
				float width = 0, height = 0;
				if (imLocalLeftHand.w > imLocalLeftHand.h)
				{
					height = 272;
					width = float(imLocalLeftHand.w) / (float(imLocalLeftHand.h) / 272);
				}
				else
				{
					width = 272;
					height = float(imLocalLeftHand.h) / (float(imLocalLeftHand.w) / 272);
				}
				image resized = resize_image(imLocalLeftHand, width, height);
				//embed_image(resized, im, 1, 1);
				free_image(resized);
				free_image(imLocalLeftHand);
			}
			if (flagRightHand == 1)
			{
				float width = 0, height = 0;
				if (imLocalRightHand.w > imLocalRightHand.h)
				{
					height = 272;
					width = float(imLocalRightHand.w) / (float(imLocalRightHand.h) / 272);
				}
				else
				{
					width = 272;
					height = float(imLocalRightHand.h) / (float(imLocalRightHand.w) / 272);
				}
				image resized = resize_image(imLocalRightHand, width, height);
				//embed_image(resized, im, colorMat.cols - width - 1, 1);
				free_image(resized);
				free_image(imLocalRightHand);
			}
		}
		
		//image resized_im = resize_image(im, colorMat.cols*2, colorMat.rows*2);

		show_image(im, "predictions");

		//free_image(resized_im);
		/*cv::Mat im_Mat(im.h, im.w, CV_8UC4);

		for (int y = 0; y < im.h; ++y){
			for (int x = 0; x < im.w; ++x){
				for (int k = 0; k < im.c; ++k){
					im_Mat.at<cv::Vec4b>(y, x)[k] = (unsigned char)(get_pixel(im, x, y, k) * 255);
				}
			}
		}
		// Resize Image
		cv::Mat resizeMat;
		const double scale = 1;
		cv::resize(im_Mat, resizeMat, cv::Size(), scale, scale);

		// Show Image
		cv::imshow("Color", resizeMat);
		*/
	
		delete[] RecObectsLeftHand;
		delete[] RecObectsRightHand;
		
	}
	//object_vote_mutilframe(RecObects, &objectNumPerFrame);	
	//objectBelong2Person(RecObects, objectNumPerFrame);	
	
	/*
	if (objectNumPerFrame > 0 && objEvent != objectDetectionEvent::Grasp)
	{		
		write_infor_to_txt(RecObects, &objectNumPerFrame);
	}	
	else if (objectNumPerFrame > 0 && objEvent == objectDetectionEvent::Grasp)
	{
		write_infor_to_txt_grasp(RecObects, &objectNumPerFrame);
	}
	*/

	//object_show(im, names, alphabet, l.classes, RecObects, &objectNumPerFrame);
	//show_image(im, "predictions");
	//im = showImg(im);//skeleton is added;
	free_image(im);	
	
#ifdef i2p
	for (int i = 0; i < objectNumPerFrame; i++)
	{
		std::cout << "Send message to I2P!!" << RecObects[i].name<<"\n";
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
#endif
	delete [] RecObects;

}


void KinectUtil::objectDetectionLocal(object *RecObects, int *objectNumPerFrame, int width, int height, cv::Point selectedPosition)
{
	int Left = max(200, selectedPosition.x - width / 2);//remove the black block,because RGB image is larger than Depth image
	int Top = max(1, selectedPosition.y - height / 2);
	int Right = min(selectedPosition.x + width / 2, colorMat.cols - 200);//remove the black block
	int Down = min(selectedPosition.y + height / 2, colorMat.rows - 1);

	cv::Mat localImg = colorMat(cv::Rect(Left, Top, width, height));

	IplImage testLocalImg = localImg;
	image imLocal = ipl_to_image(&testLocalImg);

	test_detector_img(names, alphabet, net, imLocal, thresh, RecObects, objectNumPerFrame);

	for (int idx = 0; idx < *objectNumPerFrame; idx++)
	{
		RecObects[idx].x = (RecObects[idx].x * width + Left) / colorMat.cols;
		RecObects[idx].y = (RecObects[idx].y * height + Top) / colorMat.rows;
		RecObects[idx].w = RecObects[idx].w * width / colorMat.cols;
		RecObects[idx].h = RecObects[idx].h * height / colorMat.rows;
	}
	free_image(imLocal);
}

inline void KinectUtil::skeletonShow(image im, int Id)
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
	cv::Mat tmp_Mat = DrawSkeletonFrame(im_Mat, Id);
#pragma omp parallel for
	for (int y = 0; y < im.h; ++y){
		for (int x = 0; x < im.w; ++x){
			for (int k = 0; k < im.c; ++k){
				set_pixel(im, x, y, k, float(tmp_Mat.at<cv::Vec4b>(y, x)[k]) / 255);
			}
		}
	}
//	IplImage showImg = tmp_Mat;
//	image showIm = ipl_to_image(&showImg);
//	show_image(showIm, "predictions");
	//free_image(showIm);
//	return showIm;
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
inline void KinectUtil::caculateXY(object b, float *centreX, float *centreY, float *topX, float *topY, float *bottomX, float *bottomY, float *leftX, float *leftY, float *rightX, float *rightY, int thr)
{
	int colorWidth = i_RgbTodepth.cols;
	int colorHeight = i_RgbTodepth.rows;
	int idx = 0;
	int centreXSum = 0, centreYSum = 0;
	int leftXSum = 0, leftYSum = 0;
	int rightXSum = 0, rightYSum = 0;
	int topXSum = 0, topYSum = 0;
	int bottomXSum = 0, bottomYSum = 0;

	int leftIdx = 0, rightIdx = 0, topIdx = 0, bottomIdx = 0;

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
				centreXSum += depthX;
				centreYSum += depthY;
				idx++;
			}
			if ((0 <= depthX) && (depthX < depthWidth) && (0 <= depthY) && (depthY < depthHeight))
			{
				if (colorY == DepthTop) {
					topXSum += depthX;
					topYSum += depthY;
					topIdx++;
				}
				if (colorY == DepthBot - 1) {
					bottomXSum += depthX;
					bottomYSum += depthY; 
					bottomIdx++;
				}
				if (colorX == DepthLeft) {
					leftXSum += depthX; 
					leftYSum += depthY;
					leftIdx++;
				}
				if (colorX == DepthRight - 1) {
					rightXSum += depthX;
					rightYSum += depthY;
					rightIdx++;
				}
			}
		}
	}
	if (idx > 0)
	{
		*centreX = (float)centreXSum / idx;
		*centreY = (float)centreYSum / idx;
	}
	else
	{
		*centreX = 0;
		*centreY = 0;
	}
	if (topIdx != 0)
	{
		*topX = (float)topXSum / topIdx;
		*topY = (float)topYSum / topIdx;
	}
	else
	{
		*topX = 0;
		*topY = 0;
	}
	if (bottomIdx != 0)
	{
		*bottomX = (float)bottomXSum / bottomIdx;
		*bottomY = (float)bottomYSum / bottomIdx;
	}
	else
	{
		*bottomX = 0;
		*bottomY = 0;
	}
	if (leftIdx != 0)
	{
		*leftX = (float)leftXSum / leftIdx;
		*leftY = (float)leftYSum / leftIdx;
	}
	else
	{
		*leftX = 0;
		*leftY = 0;
	}
	if (rightIdx != 0)
	{
		*rightX = (float)rightXSum / rightIdx;
		*rightY = (float)rightYSum / rightIdx;
	}
	else
	{
		*rightX = 0;
		*rightY = 0;
	}
}

inline void KinectUtil::show_grasp(image im, char **names, image **alphabet, int classes, object *RecObects, int objectNumPerFrame)
{
	image im_backup = copy_image(im);
	for (int i = 0; i < im.h; i++)
		for (int j = 0; j < im.w; j++)
		{
			set_pixel(im, j, i, 0, 1);
			set_pixel(im, j, i, 1, 1);
			set_pixel(im, j, i, 2, 1);
		}

	for (int i = 0; i < objectNumPerFrame; i++)
	{
		object b = RecObects[i];
		int DepthLeft = max(0, (b.x - b.w / 2.)*colorWidth);
		int DepthRight = min(colorWidth, (b.x + b.w / 2.)*colorWidth);
		int DepthTop = max(0, (b.y - b.h / 2.)*colorHeight);
		int DepthBot = min(colorHeight, (b.y + b.h / 2.)*colorHeight);
		cv::Mat imageROI = i_RgbTodepthForGrasping(cv::Rect(DepthLeft, DepthTop, DepthRight - DepthLeft, DepthBot - DepthTop));
		int thr = 255 * 32;
		labelOBjPixels(im, im_backup, imageROI, b, thr);

	}
	object_show_grasp(im, names, alphabet, l.classes, RecObects, &objectNumPerFrame);
}


inline void KinectUtil::caculateXYZinCameraSpace(object *RecObects, int objectNumPerFrame, image im, unsigned char flag, objectDetectionEvent objEvent)
{
	image im_backup = copy_image(im);
	//clear the image
	if (flag == 1)
	{
		for (int i = 0; i < im.h; i++)
			for (int j = 0; j < im.w; j++)
			{
				set_pixel(im, j, i, 0, 0.4);
				set_pixel(im, j, i, 1, 0.4);
				set_pixel(im, j, i, 2, 0.4);
			}
	}
#pragma omp parallel for
	for (int i = 0; i < objectNumPerFrame; i++)
	{
		object b = RecObects[i];

		int DepthLeft = max(0, (b.x - b.w / 2.)*colorWidth);
		int DepthRight = min(colorWidth, (b.x + b.w / 2.)*colorWidth);
		int DepthTop = max(0, (b.y - b.h / 2.)*colorHeight);
		int DepthBot = min(colorHeight, (b.y + b.h / 2.)*colorHeight);
		float thr = 0;
		float AvgDepth = 0;
		//for grasping
		if (objEvent == objectDetectionEvent::Grasp)
		{
			cv::Mat imageROI = i_RgbTodepthForGrasping(cv::Rect(DepthLeft, DepthTop, DepthRight - DepthLeft, DepthBot - DepthTop));
			thr = 255 * 32;
			if (flag == 1)
			{
				labelOBjPixels(im, im_backup, imageROI, b, thr);
			}
			AvgDepth = GetImgAvg(imageROI, thr);
			//end grasping
		}
		else if (objEvent == objectDetectionEvent::Demo_what)
		{
			//calculate average depth
			cv::Mat imageROI8Bit = i_RgbTodepthForshow(cv::Rect(DepthLeft, DepthTop, DepthRight - DepthLeft, DepthBot - DepthTop));
			thr = otsuThreshold(imageROI8Bit);
			cv::Mat imageROI = i_RgbTodepth(cv::Rect(DepthLeft, DepthTop, DepthRight - DepthLeft, DepthBot - DepthTop));
			thr = thr * 32;
			AvgDepth = GetImgAvg(imageROI, thr) - 16;//16 ==1.6cm
		}

		CameraSpacePoint pCentre, pTopCentre, pBottomCentre, pLeftCentre, pRightCentre;
		DepthSpacePoint dCentre, dTopCentre, dBottomCentre, dLeftCentre, dRightCentre;
		int depthX, depthY;
		float depthW, depthH;

		caculateXY(b, &dCentre.X, &dCentre.Y,
			&dTopCentre.X, &dTopCentre.Y,
			&dBottomCentre.X, &dBottomCentre.Y,
			&dLeftCentre.X, &dLeftCentre.Y, 
			&dRightCentre.X, &dRightCentre.Y, thr);

		coordinateMapper->MapDepthPointToCameraSpace(dCentre, AvgDepth, &pCentre);
		coordinateMapper->MapDepthPointToCameraSpace(dTopCentre, AvgDepth, &pTopCentre);
		coordinateMapper->MapDepthPointToCameraSpace(dBottomCentre, AvgDepth, &pBottomCentre);
		coordinateMapper->MapDepthPointToCameraSpace(dLeftCentre, AvgDepth, &pLeftCentre);
		coordinateMapper->MapDepthPointToCameraSpace(dRightCentre, AvgDepth, &pRightCentre);

		RecObects[i].CameraX = pCentre.X;
		RecObects[i].CameraY = pCentre.Y;
		RecObects[i].CameraZ = pCentre.Z;
		if (pCentre.X == -std::numeric_limits<float>::infinity() || pCentre.Y == -std::numeric_limits<float>::infinity() || pCentre.Z == -std::numeric_limits<float>::infinity()
			|| pCentre.X == std::numeric_limits<float>::infinity() || pCentre.Y == std::numeric_limits<float>::infinity() || pCentre.Z == std::numeric_limits<float>::infinity())
		{
			RecObects[i].CameraX = 0; RecObects[i].CameraY = 0; RecObects[i].CameraZ = -1;
		}
			//RecObects[i].CameraWidth = abs(pLeftCentre.X - pRightCentre.X);
		//RecObects[i].CameraHeight = abs(pTopCentre.Y - pBottomCentre.Y);
		RecObects[i].CameraWidth = sqrt((pLeftCentre.X - pRightCentre.X) * (pLeftCentre.X - pRightCentre.X)
			+ (pLeftCentre.Y - pRightCentre.Y) * (pLeftCentre.Y - pRightCentre.Y)) - 0.02;
		RecObects[i].CameraHeight = sqrt((pTopCentre.X - pBottomCentre.X) * (pTopCentre.X - pBottomCentre.X)
			+ (pTopCentre.Y - pBottomCentre.Y) * (pTopCentre.Y - pBottomCentre.Y));
	}
	free_image(im_backup);
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
			pixelCount[frame.at<UINT8>(i,j)]++;
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

inline void KinectUtil::objectBelong2Person(object *RecObects, int objectNumPerFrame)
{
//#pragma omp parallel for
	for (int i = 0; i < objectNumPerFrame; i++)
	{
		object b = RecObects[i];
		//if (strcmp(b.name, "person") == 0)continue;
		int DepthLeft = max(0, (b.x - b.w / 2.)*colorWidth);
		int DepthRight = min(colorWidth, (b.x + b.w / 2.)*colorWidth);
		int DepthTop = max(0, (b.y - b.h / 2.)*colorHeight);
		int DepthBot = min(colorHeight, (b.y + b.h / 2.)*colorHeight);

		//calculate average depth
		//cv::Mat imageROI8Bit = i_RgbTodepthForshow(cv::Rect(DepthLeft, DepthTop, DepthRight - DepthLeft, DepthBot - DepthTop));
		//int thr = otsuThreshold(imageROI8Bit);
		cv::Mat imageROIPersonIdx = i_PersonIdx(cv::Rect(DepthLeft, DepthTop, DepthRight - DepthLeft, DepthBot - DepthTop));

		int personIdx[BODY_COUNT] = { 255, 255, 255, 255, 255, 255 };
		int personPixNum[BODY_COUNT] = { 0, 0, 0, 0, 0, 0 };
		for (int j = 0; j < BODY_COUNT; j++)
		{
			personIdx[j] = 0;
			personPixNum[j] = 0;
		}
		unsigned char personCount = 0;
		float NForground = 0;
		unsigned char flag = 0;
		for (int j = 0; j < imageROIPersonIdx.rows; j++)
			for (int k = 0; k < imageROIPersonIdx.cols; k++)
			{
				if (imageROIPersonIdx.at<INT32>(j, k) <= 6 && imageROIPersonIdx.at<INT32>(j, k) >= 1)
					{
						flag = 0;
						for (int pIdx = 0; pIdx < personCount; pIdx++)
						{
							if (imageROIPersonIdx.at<INT32>(j, k) == personIdx[pIdx])
							{
								personPixNum[pIdx]++;
								flag = 1;
							}
						}
						if (flag == 0)
						{
							personIdx[personCount] = imageROIPersonIdx.at<INT32>(j, k);
							personPixNum[personCount]++;
							personCount++;
						}
					}
					//if (imageROIPersonIdx.at<UINT8>(j, k) == 255)
					//	NPensonoIdx++;
				//}
			}
		float tmpMax = 0;
		int maxIdx = 0;
		for (int j = 0; j < personCount; j++)
		{
			if (personPixNum[j] > tmpMax)
			{
				tmpMax = personPixNum[j];
				maxIdx = j;
			}
		}

		if (tmpMax / (imageROIPersonIdx.rows * imageROIPersonIdx.cols) > 0.5)
		{
			RecObects[i].flagBelong2Person = 1;
			RecObects[i].bodyId = personIdx[maxIdx];
		}
		else{
			RecObects[i].flagBelong2Person = 0;
			RecObects[i].bodyId = 255;
		}
			
	}
}

KinectBody::KinectBody(cv::Point ptJoints[JointType_Count], cv::Vec3f vJoints[JointType_Count], TrackingState nStates[JointType_Count])
{
	memcpy(Joints2D, ptJoints, sizeof(cv::Point) * JointType_Count);
	memcpy(Joints3D, vJoints, sizeof(cv::Vec3f) * JointType_Count);
	memcpy(JointStates, nStates, sizeof(TrackingState)* JointType_Count);

	IsMainBody = true;
}

// calculate 3D point projected on RGB image plane
cv::Point KinectUtil::ProjectToPixel(cv::Vec3f v)
{
	ColorSpacePoint colorPoint = { 0 };
	CameraSpacePoint vCamPoint;
	vCamPoint.X = v[0];
	vCamPoint.Y = v[1];
	vCamPoint.Z = v[2];
	coordinateMapper->MapCameraPointToColorSpace(vCamPoint, &colorPoint);

	//	CameraIntrinsics pCamParams;
	//	m_pCoordinateMapper->GetDepthCameraIntrinsics(&pCamParams);

	int nPointX = colorPoint.X;
	int nPointY = colorPoint.Y;
	return cv::Point(nPointX, nPointY);
}


cv::Mat KinectUtil::DrawSkeletonFrame(cv::Mat Img, int Id)
{
	cv::Mat mtxRGB(Img.clone());
	if (Id == 255)
		for (int i = 0; i < m_vecKinectBodies.size(); i++)
			DrawBody(mtxRGB, m_vecKinectBodies[i]);
	else if (Id >= 0 && Id <= 5)
		DrawBody(mtxRGB, m_vecKinectBodies[Id]);

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

cv::Mat KinectUtil::colorImgFilterbyDistance(cv::Mat colorMat, float distance, cv::Mat RGB2DepthMat)
{
	cv::Vec4b pixel;
	pixel[0] = 255;
	pixel[1] = 255;
	pixel[2] = 255;
	pixel[3] = 255;
	cv::Mat tmp = colorMat.clone();


	for (int i = 0; i < tmp.rows; i++)
		for (int j = 0; j < tmp.cols; j++)
		{
			if (RGB2DepthMat.at<unsigned char>(i, j) <= 500 / 32 || RGB2DepthMat.at<unsigned char>(i, j) >= distance * 1000 / 32)
				tmp.at<cv::Vec4b>(i, j) = pixel;
		}
	
	//cv::Mat showMat;
	//cv::resize(tmp, showMat, cv::Size(colorWidth / 2, colorHeight / 2));
	//cv::imshow("Label1", showMat);

	return tmp;
}

void KinectUtil::labelOBjPixels(image im, image im_backup, cv::Mat imageROI8Bit, object b, int thr)
{
	float color[3];
	color[0] = b.boxRGB[0];
	color[1] = b.boxRGB[1];
	color[2] = b.boxRGB[2];

	int DepthLeft = max(0, (b.x - b.w / 2.)*colorWidth);
	int DepthRight = min(colorWidth, (b.x + b.w / 2.)*colorWidth);
	int DepthTop = max(0, (b.y - b.h / 2.)*colorHeight);
	int DepthBot = min(colorHeight, (b.y + b.h / 2.)*colorHeight);

/*	for (int i = 0; i < imageROI8Bit.rows; i++)
		for (int j = 0; j < imageROI8Bit.cols; j++)
		{
			if (imageROI8Bit.at<UINT16>(i, j) > 0)//label the objects
			{
				set_pixel(im, DepthLeft + j, DepthTop + i, 0, color[0]);
				set_pixel(im, DepthLeft + j, DepthTop + i, 1, color[1]);
				set_pixel(im, DepthLeft + j, DepthTop + i, 2, color[2]);
			}
			else//remove background
			{
				//set_pixel(im, DepthLeft + j, DepthTop + i, 0, 255);
				//set_pixel(im, DepthLeft + j, DepthTop + i, 1, 255);
				//set_pixel(im, DepthLeft + j, DepthTop + i, 2, 255);
			}
		}
*/
	for (int i = 0; i < imageROI8Bit.rows; i++)
		for (int j = 0; j < imageROI8Bit.cols; j++)
		{
			if (imageROI8Bit.at<UINT16>(i, j) > 0)//label the objects
			{
				set_pixel(im, DepthLeft + j, DepthTop + i, 0, get_pixel(im_backup, DepthLeft + j, DepthTop + i, 0));
				set_pixel(im, DepthLeft + j, DepthTop + i, 1, get_pixel(im_backup, DepthLeft + j, DepthTop + i, 1));
				set_pixel(im, DepthLeft + j, DepthTop + i, 2, get_pixel(im_backup, DepthLeft + j, DepthTop + i, 2));
			}
		}
}

void KinectUtil::desk_seg(float thr)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

	//cloud->width = static_cast<uint32_t>(depthWidth);
	//cloud->height = static_cast<uint32_t>(depthHeight);
	//cloud->is_dense = false;

	//cloud->points.resize(cloud->height * cloud->width);

	// kill the points whose distance is larger than thr * 1000
	//std::vector<UINT16> depthBufferTmp;
	//depthBufferTmp.resize(depthWidth * depthHeight);
	for (int i = 0; i < depthHeight * depthWidth; i++)
	{
		if (depthBuffer[i] > thr * 1000)
			depthBufferGrasping[i] = 0;
		else
			depthBufferGrasping[i] = depthBuffer[i];
	}


	//pcl::PointXYZ* pt = &cloud->points[0];
	for (int y = 0; y < depthHeight; y++){
		for (int x = 0; x < depthWidth; x++){
			pcl::PointXYZ point;

			DepthSpacePoint depthSpacePoint = { static_cast<float>(x), static_cast<float>(y) };
			UINT16 depth = depthBufferGrasping[y * depthWidth + x];
			if (depth == 0) continue;

			// Coordinate Mapping Depth to Camera Space, and Setting PointCloud XYZ  
			CameraSpacePoint cameraSpacePoint = { 0.0f, 0.0f, 0.0f };
			coordinateMapper->MapDepthPointToCameraSpace(depthSpacePoint, depth, &cameraSpacePoint);
			point.x = cameraSpacePoint.X;
			point.y = cameraSpacePoint.Y;
			point.z = cameraSpacePoint.Z;

			//*pt = point;
			cloud->points.push_back(point);
		}
	}

	plane_segmentation(cloud, &depthBufferGrasping[0], depthHeight, depthWidth);
/*	cv::Mat dMat(depthHeight, depthWidth, CV_8UC1);
	for (int i = 0; i < depthHeight; i++)
		for (int j = 0; j < depthWidth; j++)
			dMat.at<unsigned char>(i, j) = static_cast<BYTE>(depthBufferTmp[i * depthWidth + j] >> 5);
	cv::imshow("depth", dMat);*/
	//return cloud;
}