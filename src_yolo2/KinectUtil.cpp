#include "KinectUtil.h"
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
using namespace std;
//char *cfgfile = "G:/IMI-PROJECTS/i2p_Nadine_Robot/development/i2p_perception/i2p_object_detection/data/cfg/tiny-coco.cfg";
//char *weightfile = "G:/IMI-PROJECTS/i2p_Nadine_Robot/development/i2p_perception/i2p_object_detection/data/weights/tiny-coco.weights";
//char *filename = "G:/IMI-PROJECTS/i2p_Nadine_Robot/development/i2p_perception/i2p_object_detection/data/img/dog.jpg";
//char *cfgfile = "data/cfg/tiny-coco.cfg";
//char *weightfile = "data/weights/tiny-coco.weights";
//char *filename = "data/img/dog.jpg";
//float thresh = 0.15;

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
	initializeColor();
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
	updateColor();
}

// Update Depth
inline void KinectUtil::updateDepth()
{
	// Retrieve Depth Frame
	ComPtr<IDepthFrame> depthFrame;
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
	ComPtr<IColorFrame> colorFrame;
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

#pragma omp parallel for
	for (int colorY = 0; colorY < colorHeight; colorY++) {
		unsigned int colorOffset = colorY * colorWidth;
		for (int colorX = 0; colorX < colorWidth; colorX++) {
			unsigned int colorIndex = colorOffset + colorX;
			int depthX = static_cast<int>(depthSpace[colorIndex].X + 0.5f);
			int depthY = static_cast<int>(depthSpace[colorIndex].Y + 0.5f);
			if ((0 <= depthX) && (depthX < depthWidth) && (0 <= depthY) && (depthY < depthHeight)) {
				unsigned int depthIndex = depthY * depthWidth + depthX;
				buffer[colorIndex] = depthBuffer[depthIndex];
			}
		}
	}

	// Create cv::Mat from Coordinate Buffer
	//depthMat = cv::Mat(colorHeight, colorWidth, CV_16UC1, &buffer[0]).clone();
	cv::Mat dMat = cv::Mat(colorHeight, colorWidth, CV_16UC1, &buffer[0]).clone();
	if (!dMat.empty())
		depthMat = dMat;
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
	//string pathRst = "\\\\imi-storage\\04_SharedBox\\_zwfang\\objectRecognition\\test.txt";
	//string pathRst = "\\\\IMI-7736-FANGZH\\shareFile\\test.file";
	string pathRst = "..\\..\\..\\res\\Objects.txt";

	ofstream fCtr(pathRst.c_str(), ios::out);
	if (fCtr)
	{
		if (*objectNumPerFrame == 0)
		{
			fCtr << "there is nothing in this room!";
		}
		else if(*objectNumPerFrame == 1)
		{
			fCtr << "i can see " << RecObects[0].name << "." << endl;
		}
		else
		{
			std::vector<int> objectID;
			std::vector<string> objectName;
			int objectNum = 0;
			unsigned char flag = 0;
			for (int i = 0; i < *objectNumPerFrame; i++)
			{
				flag = 0;
				for (int j = 0; j < objectNum; j++)
				{
					if (objectID[j] == RecObects[i].objClass)
					{
						flag = 1;
					}
				}
				if (flag == 0)
				{
					objectName.push_back(RecObects[i].name);
					objectID.push_back(RecObects[i].objClass);
					objectNum++;
				}
			}

			fCtr << "there are many things in this room. i can see ";
			for (int i = 0; i < objectNum - 2; i++)
			{
				fCtr << objectName[i] << ", ";
			}
			fCtr << objectName[objectNum - 2] << " ";
			fCtr << "and " << objectName[objectNum - 1] << "." << endl;
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
	if (colorMat.empty() || depthMat.empty()) {
		return;
	}
	//precess color image
	cv::Mat resizeMat;
	const double scale = 0.5;
	cv::resize(colorMat, resizeMat, cv::Size(), scale, scale);// Resize Image

	//process depth image
	cv::Mat scaleMat;
	depthMat.convertTo(scaleMat, CV_8U, -255.0 / 1000.0, 255.0);// Scaling ( 0-8000 -> 255-0 )

	IplImage ColorImg = resizeMat;
	IplImage testImg = colorMat;
	IplImage DepthImg = scaleMat;

	object *RecObects = new object[maxObjectNum];
	objectNumPerFrame = 0;
	std::vector< ::imi::ObjectInfo> detectedObjects; // Data to be transimited

	image im = ipl_to_image(&testImg);
	//Detect objects for per image.
	test_detector_img(names, alphabet, net, im, thresh, RecObects, &objectNumPerFrame);
	objectFilterUsingObjectCategory(RecObects, &objectNumPerFrame, 1);
	write_infor_to_txt(RecObects, &objectNumPerFrame);
	//object_reminder(im, names, alphabet, l.classes, RecObects, &objectNumPerFrame);
	object_show(im, names, alphabet, l.classes, RecObects, &objectNumPerFrame);
	show_image(im, "predictions");
	free_image(im);

	for (int i = 0; i < objectNumPerFrame; i++)
	{
		if (strlen(RecObects[i].name)<2 || RecObects[i].x<0)
			continue;		

		object b = RecObects[i];
		int left = (b.x - b.w / 2.)*ColorImg.width;
		int right = (b.x + b.w / 2.)*ColorImg.width;
		int top = (b.y - b.h / 2.)*ColorImg.height;
		int bot = (b.y + b.h / 2.)*ColorImg.height;

		int DepthLeft = (b.x - b.w / 2.)*DepthImg.width;
		int DepthRight = (b.x + b.w / 2.)*DepthImg.width;
		int DepthTop = (b.y - b.h / 2.)*DepthImg.height;
		int DepthBot = (b.y + b.h / 2.)*DepthImg.height;
		/*cv::rectangle(resizeMat, cvPoint(left, top), cvPoint(right, bot), cv::Scalar(255, 0, 0), 2, 1, 0);
		cv::putText(resizeMat, RecObects[i].name, cvPoint(left,top+20), CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 0, 0));
		cv::rectangle(scaleMat, cvPoint(DepthLeft, DepthTop), cvPoint(DepthRight, DepthBot), cv::Scalar(0), 2, 1, 0);
		*/
		//calculate average depth
		//0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows
		if (DepthLeft<0 || DepthRight - DepthLeft <= 0 || DepthRight>depthMat.cols || DepthTop<0 || DepthBot - DepthTop <= 0 || DepthBot > depthMat.rows)
			continue;
		cv::Mat imageROI = depthMat(cv::Rect(DepthLeft, DepthTop, DepthRight-DepthLeft, DepthBot-DepthTop));
		float AvgDepth = GetImgAvg(imageROI);

		CameraSpacePoint cp1;
		DepthSpacePoint d;
		d.X = (int)(b.x*512);   // in pixels,512 is the width of depth image,424 is height
		d.Y = (int)(b.y*424);   // in pixels
		coordinateMapper->MapDepthPointToCameraSpace(d, AvgDepth, &cp1);
		//std::cout << "d: " << d.X << " " << d.Y<<" "<<AvgDepth<<" ";
		std::cout << "corrdinates: " << cp1.X << " " << cp1.Y << " " << cp1.Z << std::endl;


		// add data to detected object
		::imi::ObjectInfo detectedObj = ::imi::ObjectInfo();
		std::cout << RecObects[i].name << std::endl;
		detectedObj.__set_label(RecObects[i].name); // set label
		detectedObj.__set_prob(RecObects[i].prob); // set probability 
		detectedObj.coordinate.__set_x(cp1.X);
		detectedObj.coordinate.__set_y(cp1.Y);
		detectedObj.coordinate.__set_z(cp1.Z);
		detectedObj.isNew = true;
		detectedObj.mchecked = false;
		detectedObj.duplicated = false;
		detectedObjects.push_back(detectedObj);
	}

	//cv::imshow("color", resizeMat);
	cv::resize(scaleMat, scaleMat, cv::Size(), scale, scale);
	//cv::imshow("depth", scaleMat);

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
	std::cout << "\n\n" << std::endl;

	delete [] RecObects;

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