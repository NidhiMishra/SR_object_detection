#include "Process_Kinect.h"
#include <sys/timeb.h>
#include <iostream>
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
//#include "opencv2\core\mat.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2/video/tracking.hpp"

#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/opencv.hpp>

#include <direct.h>  
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits>
#include "scene_flow_impair.h"
#include "elml.hpp"
#include <fstream>

static float NaN = std::numeric_limits<float>::quiet_NaN();
int height[] = { 480, 240, 120, 60, 30, 15 };	// Pyramid size (height)
int width[] = { 640, 320, 160, 80, 40, 20 };	// Pyramid size (width)
//int height[] = { 424, 212, 106, 53, 26, 13 };	// Pyramid size (height)
//int width[] = { 512, 256, 128, 64, 32, 16 };	// Pyramid size (width)

using namespace std;

KinectBody::KinectBody(cv::Point ptJoints[JointType_Count], cv::Vec3f vJoints[JointType_Count],
	TrackingState nStates[JointType_Count], double Orientation, int view, int x_min, int x_max, int y_min, int y_max, bool valid_neck)
{
	memcpy(Joints2D, ptJoints, sizeof(cv::Point) * JointType_Count);
	memcpy(Joints3D, vJoints, sizeof(cv::Vec3f) * JointType_Count);
	memcpy(JointStates, nStates, sizeof(TrackingState)* JointType_Count);

	/*LHand2D = Joints2D[JointType_HandLeft];
	LWrist2D = Joints2D[JointType_WristLeft];
	LElbow2D = Joints2D[JointType_ElbowLeft];
	RHand2D = Joints2D[JointType_HandRight];
	RWrist2D = Joints2D[JointType_WristRight];
	RElbow2D = Joints2D[JointType_ElbowRight];

	LHand3D = Joints3D[JointType_HandLeft];
	LWrist3D = Joints3D[JointType_WristLeft];
	LElbow3D = Joints3D[JointType_ElbowLeft];
	RHand3D = Joints3D[JointType_HandRight];
	RWrist3D = Joints3D[JointType_WristRight];
	RElbow3D = Joints3D[JointType_ElbowRight];

	LHandState = ls;
	RHandState = rs;*/
	BodyOrientation = Orientation;
	BodyView = view;
	Foreground_x_lim[0] = x_min;
	Foreground_x_lim[1] = x_max;
	Foreground_y_lim[0] = y_min;
	Foreground_y_lim[1] = y_max;
	IsMainBody = true;
	neck_valid = valid_neck;
}

Process_Kinect::Process_Kinect(bool Thrift_use)
{
	isUseThrift = Thrift_use;
	Init();
}


Process_Kinect::~Process_Kinect()
{
	Close();
}

void Process_Kinect::Init()
{
	m_pKinectSensor = NULL;

	m_sOrgColor = cv::Size(1920, 1080);
	m_sOrgDepth = cv::Size(512, 424);

	//m_sOrgColor = cv::Size(640, 480);
	//m_sOrgDepth = cv::Size(640, 480);

	m_fMinDepth = 0.3;
	m_fMaxDepth = 3.0;

	//SetSize(cv::Size(512, 424));
	SetSize(cv::Size(640, 480));
	m_nFrmNum = 0;
	m_NoSkeletons = 0;
	Action_Detected = false;

	//Kinect camera parameters - Setting parameters
	double f[2] = {366, 366};	// fc
	double c[2] = { 252, 204 };		// cc
	double k[5] = { 0, 0, 0, 0, 0 };	// kc
	double al = 0;				// alpha_c
	SetProjParams(f, c, k, al);

	for (int iter = 0; iter < BODY_COUNT; iter++)
	{
		Skeleton_TrackingID[iter] = 0;
	}

	m_recognition_threshold = 0.7;

	HRESULT hr = Nui_Init();//InitializeDefaultSensor();
	if (FAILED(hr))
	{
		printf("Fail to open the camera!\n");
	}


	//---------------------------------------------------------------------------
	//change to IMI_LIBRARIES
	String face_cascade_name = "D:\\nadineRobot\\i2p_Nadine_Robot\\dependency\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml";

	//-- 1. Load the face cascades - To identify back or front view of the skeleton
	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading face\n"); };
	//---------------------------------------------------------------------------
	//for (int i = 0; i < 4; i++)
	//	m_hFrameEvents[i] = NULL;
	//ELM Initialize
	ELM_Initialize();

	//---------------------------------------------------------------------------
	//-----------Get ProtecteClient if Thrift is used----------------------------
	if (isUseThrift)
	{
		ActionManager = new ProtectedClient<imi::ActionRecognitionServiceClient>("localhost", imi::g_Inputs_constants.DEFAULT_ACTION_RECOGNITION_SERVICE_PORT);
	}
	//---------------------------------------------------------------------------
}

void Process_Kinect::ELM_Initialize(void)
{
	m_ndims = (NO_OF_GRIDS*NO_OF_ORIENTATIONS*NO_OF_CELLS * 2 * 2 * 3 ) + (NO_OF_GRIDS*NO_OF_ORIENTATIONS*NO_OF_CELLS * 2 * 3 * 3);
	m_Jointndims = (NO_OF_JOINTS*NO_OF_ORIENTATIONS * 2 * 2) + (NO_OF_JOINTS*NO_OF_ORIENTATIONS * 3);
	//m_Jointndims = (NO_OF_JOINTS*NO_OF_ORIENTATIONS * 2) + (NO_OF_JOINTS*NO_OF_ORIENTATIONS * 3);
	m_nhn = 5000;

	inW = MatrixXd::Random(m_nhn, m_ndims);
	bias = MatrixXd::Random(m_nhn, 1);
	outW = MatrixXd::Random(m_nhn, NO_OF_ACTIONS);
	//---------------------inW--------------------------
	int param_row = 0, param_col = 0;
	string line;
	ifstream myfile("Kinematic_inW_3class.txt");
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			//cout << line << '\n';
			
			inW(param_row, param_col) = stod(line);
			param_col++;
			if (param_col == m_ndims)
			{
				param_col = 0;
				param_row++;
			}
		}
		myfile.close();
	}
	else
	{
		printf("Cannot Open Kinematic inW file\n");
	}
	//--------------------------------------------------
	//---------------------bias--------------------------
	param_row = 0;
	param_col = 0;
	myfile.open("Kinematic_bias_3class.txt");
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			//cout << line << '\n';

			bias(param_row, param_col) = stod(line);
			param_col++;
			if (param_col == 1)
			{
				param_col = 0;
				param_row++;
			}
		}
		myfile.close();
	}
	else
	{
		printf("Cannot Open Kinematic bias file\n");
	}
	//--------------------------------------------------
	//---------------------outW--------------------------
	param_row = 0;
	param_col = 0;
	myfile.open("Kinematic_outW_3class.txt");
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			//cout << line << '\n';

			outW(param_row, param_col) = stod(line);
			param_col++;
			if (param_col == NO_OF_ACTIONS)
			{
				param_col = 0;
				param_row++;
			}
		}
		myfile.close();
	}
	else
	{
		printf("Cannot Open Kinematic outW file\n");
	}
	//--------------------------------------------------
	Stick_inW = MatrixXd::Random(m_nhn, m_Jointndims);
	Stick_bias = MatrixXd::Random(m_nhn, 1);
	Stick_outW = MatrixXd::Random(m_nhn, NO_OF_ACTIONS);
	//---------------------Stick_inW--------------------------
	param_row = 0;
	param_col = 0;
	myfile.open("Stick_Kinematic_inW_3class.txt");
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			//cout << line << '\n';

			Stick_inW(param_row, param_col) = stod(line);
			param_col++;
			if (param_col == m_Jointndims)
			{
				param_col = 0;
				param_row++;
			}
		}
		myfile.close();
	}
	else
	{
		printf("Cannot Open Stick Kinematic inW file\n");
	}
	//--------------------------------------------------
	//---------------------Stick_bias--------------------------
	param_row = 0;
	param_col = 0;
	myfile.open("Stick_Kinematic_bias_3class.txt");
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			//cout << line << '\n';

			Stick_bias(param_row, param_col) = stod(line);
			param_col++;
			if (param_col == 1)
			{
				param_col = 0;
				param_row++;
			}
		}
		myfile.close();
	}
	else
	{
		printf("Cannot Open Stick Kinematic bias file\n");
	}
	//--------------------------------------------------
	//---------------------Stick_outW--------------------------
	param_row = 0;
	param_col = 0;
	myfile.open("Stick_Kinematic_outW_3class.txt");
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			//cout << line << '\n';

			Stick_outW(param_row, param_col) = stod(line);
			param_col++;
			if (param_col == NO_OF_ACTIONS)
			{
				param_col = 0;
				param_row++;
			}
		}
		myfile.close();
	}
	else
	{
		printf("Cannot Open Stick Kinematic outW file\n");
	}
	//--------------------------------------------------
}

void Process_Kinect::Close(void)
{
	Nui_Clear();
}

/// <summary>
/// Initializes the default Kinect sensor
/// </summary>
/// <returns>S_OK on success else the failure code</returns>
HRESULT Process_Kinect::InitializeDefaultSensor()
{
	HRESULT hr;

	hr = GetDefaultKinectSensor(&m_pKinectSensor);
	if (FAILED(hr))
	{
		return hr;
	}

	if (m_pKinectSensor)
	{
		// Initialize Kinect and get color, body and face readers
		IDepthFrameSource* pDepthFrameSource = NULL;
		IColorFrameSource* pColorFrameSource = NULL;
		IBodyIndexFrameSource* pBodyIndexFrameSource = NULL;
		IBodyFrameSource* pBodyFrameSource = NULL;

		hr = m_pKinectSensor->Open();


		//Depth Initialize
		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_DepthFrameSource(&pDepthFrameSource);
			if (FAILED(hr))
			{
				printf("Failed to find the Depth Frame source\n");
				exit(10);
			}
		}
		if (SUCCEEDED(hr))
		{
			hr = pDepthFrameSource->OpenReader(&m_pDepthFrameReader);
			if (FAILED(hr))
			{
				printf("Failed to open Depth Reader\n");
				exit(10);
			}
		}

		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_CoordinateMapper(&m_pCoordinateMapper);
		}

		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_ColorFrameSource(&pColorFrameSource);
		}

		if (SUCCEEDED(hr))
		{
			hr = pColorFrameSource->OpenReader(&m_pColorFrameReader);
		}

		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_BodyFrameSource(&pBodyFrameSource);
		}

		if (SUCCEEDED(hr))
		{
			hr = pBodyFrameSource->OpenReader(&m_pBodyFrameReader);
		}

		//Body Index frame Initialize
		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_BodyIndexFrameSource(&pBodyIndexFrameSource);
			if (FAILED(hr))
			{
				printf("Failed to find the Body Index Frame source\n");
				exit(10);
			}
		}
		if (SUCCEEDED(hr))
		{
			hr = pBodyIndexFrameSource->OpenReader(&m_pBodyIndexReader);
			if (FAILED(hr))
			{
				printf("Failed to find the Body Index Reader\n");
				exit(10);
			}
		}

		SafeRelease(pDepthFrameSource);
		SafeRelease(pColorFrameSource);
		SafeRelease(pBodyIndexFrameSource);
		SafeRelease(pBodyFrameSource);
	}

	if (!m_pKinectSensor || FAILED(hr))
	{
		//SetStatusMessage(L"No ready Kinect found!", 10000, true);
		return E_FAIL;
	}

	return hr;
}

HRESULT Process_Kinect::Nui_Init()
{
	// init the sensor body
	HRESULT hr;
	if (m_pKinectSensor == NULL)
	{
		hr = GetDefaultKinectSensor(&m_pKinectSensor);
		if (FAILED(hr))
			return hr;
	}

	// init the event handles
	//for (int i = 0; i < 4; i++)
	//	m_hFrameEvents[i] = (WAITABLE_HANDLE)CreateEvent(NULL, FALSE, FALSE, NULL);


	// open depth, color, body index and skeleton readers
	IDepthFrameSource* pDepthFrameSource = NULL;
	IColorFrameSource* pColorFrameSource = NULL;
	IBodyIndexFrameSource* pBodyIndexFrameSource = NULL;
	IBodyFrameSource* pBodyFrameSource = NULL;
	hr = m_pKinectSensor->Open();

	//Depth Initialize
	if (SUCCEEDED(hr))
	{
		hr = m_pKinectSensor->get_DepthFrameSource(&pDepthFrameSource);
		if (FAILED(hr))
		{
			printf("Failed to find the Depth Frame source\n");
			exit(10);
		}
	}
	if (SUCCEEDED(hr))
	{
		hr = pDepthFrameSource->OpenReader(&m_pDepthFrameReader);
		if (FAILED(hr))
		{
			printf("Failed to open Depth Reader\n");
			exit(10);
		}
	}
	//if (SUCCEEDED(hr))	hr = m_pDepthFrameReader->SubscribeFrameArrived(&m_hFrameEvents[0]);

	//Color Initialize
	if (SUCCEEDED(hr))
	{
		hr = m_pKinectSensor->get_ColorFrameSource(&pColorFrameSource);
		if (FAILED(hr))
		{
			printf("Failed to find the Color Frame source\n");
			exit(10);
		}
	}
	if (SUCCEEDED(hr))
	{
		hr = pColorFrameSource->OpenReader(&m_pColorFrameReader);
		if (FAILED(hr))
		{
			printf("Failed to find the Color Frame Reader\n");
			exit(10);
		}
	}
	//if (SUCCEEDED(hr))	hr = m_pColorFrameReader->SubscribeFrameArrived(&m_hFrameEvents[1]);

	//Body Index frame Initialize
	if (SUCCEEDED(hr))
	{
		hr = m_pKinectSensor->get_BodyIndexFrameSource(&pBodyIndexFrameSource);
		if (FAILED(hr))
		{
			printf("Failed to find the Body Index Frame source\n");
			exit(10);
		}
	}
	if (SUCCEEDED(hr))
	{
		hr = pBodyIndexFrameSource->OpenReader(&m_pBodyIndexReader);
		if (FAILED(hr))
		{
			printf("Failed to find the Body Index Reader\n");
			exit(10);
		}
	}
	//if (SUCCEEDED(hr))	hr = m_pBodyIndexReader->SubscribeFrameArrived(&m_hFrameEvents[2]);

	//Coordinate mapper Initialize
	if (SUCCEEDED(hr))
	{
		hr = m_pKinectSensor->get_CoordinateMapper(&m_pCoordinateMapper);
		if (FAILED(hr))
		{
			printf("Failed to find the Coordinate Mapper\n");
			exit(10);
		}
	}

	//Body Frame Initialize
	if (SUCCEEDED(hr))	
	{ 
		hr = m_pKinectSensor->get_BodyFrameSource(&pBodyFrameSource); 
		if (FAILED(hr))
		{
			printf("Failed to find the Body Frame source\n");
			exit(10);
		}
	}
	if (SUCCEEDED(hr))
	{
		hr = pBodyFrameSource->OpenReader(&m_pBodyFrameReader);
		if (FAILED(hr))
		{
			printf("Failed to find the Body Frame Reader\n");
			exit(10);
		}
	}
	//if (SUCCEEDED(hr))	hr = m_pBodyFrameReader->SubscribeFrameArrived(&m_hFrameEvents[3]);

	SafeRelease(pDepthFrameSource);
	SafeRelease(pColorFrameSource);
	SafeRelease(pBodyIndexFrameSource);
	SafeRelease(pBodyFrameSource);

	if (!m_pKinectSensor || FAILED(hr))
	{
		cout << "No ready Kinect found!" << endl;
		return E_FAIL;
	}

	return hr;
}

void Process_Kinect::Nui_Clear()
{
	// close the event handles
	//m_pDepthFrameReader->UnsubscribeFrameArrived(m_hFrameEvents[0]);
	//m_pColorFrameReader->UnsubscribeFrameArrived(m_hFrameEvents[1]);
	//m_pBodyIndexReader->UnsubscribeFrameArrived(m_hFrameEvents[2]);
	//m_pBodyFrameReader->UnsubscribeFrameArrived(m_hFrameEvents[3]);
	//for (int i = 0; i < 4; i++)
	//{
	//	CloseHandle((HANDLE)m_hFrameEvents[i]);
	//	m_hFrameEvents[i] = NULL;
	//}
	
	m_mtxColorImg.release();
	m_mtxColorImg_Prev.release();
	m_mtxDepthImg.release();
	m_mtxDepthImg_Prev.release();
	m_mtxDepthPoints.release();
	m_mtxPseudoDepth.release();
	m_mtxLabels.release();

	m_Divergence.release();
	m_Vorticity.release();
	m_Projection.release();
	m_Rotation.release();
	m_BodyPartProjection.release();
	m_BodyPartRotation.release();


	Foreground_points.clear();
	m_vecKinectBodies.clear();
	m_Prev_vecKinectBodies.clear();
	Clear_Vectors();
	// done with body frame reader
	SafeRelease(m_pDepthFrameReader);
	SafeRelease(m_pColorFrameReader);
	SafeRelease(m_pBodyFrameReader);
	SafeRelease(m_pBodyIndexReader);
	SafeRelease(m_pCoordinateMapper);

	// close the Kinect Sensor
	if (m_pKinectSensor)
	{
		m_pKinectSensor->Close();
	}

	SafeRelease(m_pKinectSensor);
}

//Clear the vectors created for every video
void Process_Kinect::Clear_Vectors()
{
	//m_vecKinectBodies.size()
	for (int Person_Index = 0; Person_Index < BODY_COUNT; ++Person_Index)
	{
		for (int Grid_Index = 0; Grid_Index < NO_OF_GRIDS; Grid_Index++)
		{
			for (int Cell_Index = 0; Cell_Index < (2 * NO_OF_CELLS); Cell_Index++)
			{
				for (int Ori_Index = 0; Ori_Index < NO_OF_ORIENTATIONS; Ori_Index++)
				{
					if (Ori_Index == 0)
					{
						Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_UD.clear();
						Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.clear();
						Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.clear();
					}
					else if (Ori_Index == 1)
					{
						Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.clear();
						Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.clear();
						Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.clear();
					}
					else if (Ori_Index == 2)
					{
						Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.clear();
						Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.clear();
						Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.clear();
					}
				}
				//end of Ori_Index
			}
			//end of Cell_Index
		}
		//end of Grid_Index
		for (int joint_iter = 0; joint_iter < JointType_Count; joint_iter++)
		{
			Person[Person_Index].Joint_Neck_Ref[joint_iter].Joint_FB.clear();
			Person[Person_Index].Joint_Neck_Ref[joint_iter].Joint_LR.clear();
			Person[Person_Index].Joint_Neck_Ref[joint_iter].Joint_UD.clear();
		}
	}
	//end of Person_Index


}

bool Process_Kinect::Update(void)
{
	//printf("Update function\n");
	int i = 0;
	int Frame_interval = 30;
	cv::Mat prevcolorgray;
	cv::Mat currcolorgray;
	cv::Mat	Colorflow; //For Color Optical flow
	cv::Mat	Depthflow; //For Depth flow

	if (!m_pColorFrameReader || !m_pBodyFrameReader)
	{
		return false;
	}
	printf("Kinect Loaded!!!\n");

	bool Color_Rcvd, Depth_Rcvd;
	while (true)
	{
		//Color
		//Previous Color Image
		m_mtxColorImg.copyTo(m_mtxColorImg_Prev);
		//Convert to gray scale
		cvtColor(m_mtxColorImg_Prev, prevcolorgray, COLOR_BGR2GRAY);
		//cv::imshow("PrevColor", prevcolorgray);
		//waitKey(5);
		Sleep(5);
		//Current Color Image
		Color_Rcvd = Nui_NextColorFrame();
		//Convert to gray scale
		cvtColor(m_mtxColorImg, currcolorgray, COLOR_BGR2GRAY);
		//cv::imshow("Color", currcolorgray);
		//waitKey(5);

		// calculate optical flow 
		/*calcOpticalFlowFarneback(prevcolorgray, currcolorgray, Colorflow, 0.4, 1, 12, 2, 8, 1.2, 0);

		for (int y = 0; y < m_mtxColorImg_Prev.rows; y += 5) {
			for (int x = 0; x < m_mtxColorImg_Prev.cols; x += 5)
			{
				// get the flow from y, x position * 10 for better visibility
				const Point2f flowatxy = Colorflow.at<Point2f>(y, x) * 10;
				//flowatxy.x flowatxy.y
				// draw line at flow direction
				line(m_mtxColorImg_Prev, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(255, 0, 0));
				// draw initial point
				circle(m_mtxColorImg_Prev, Point(x, y), 1, Scalar(0, 0, 0), -1);
			}

		}

		// draw the results
		cv::namedWindow("Flow", WINDOW_AUTOSIZE);
		cv::imshow("Flow", m_mtxColorImg_Prev);
		cv::waitKey(5);*/
		Sleep(5);
		//Depth
		//Previous Depth Image
		m_mtxDepthImg.copyTo(m_mtxDepthImg_Prev);
		//Current Depth Image
		Depth_Rcvd = Nui_NextDepthFrame();
		/*
		//Subtract both images
		cv::subtract(m_mtxDepthImg_Prev, m_mtxDepthImg, Depthflow);
		cv::namedWindow("Depth", WINDOW_AUTOSIZE);
		//cv::imshow("Depth", m_mtxDepthImg);
		cv::imshow("Depth", Depthflow);
		waitKey(5);*/
		
		//printf("R: %d %d\n", Color_Rcvd, Depth_Rcvd);
		if ((Color_Rcvd == true) & (Depth_Rcvd == true))
		{
			if (m_FrameNum >= 2)
			{
				int sfx_min = m_sFrame.width;
				int sfx_max = 0;
				int sfy_min = m_sFrame.height;
				int sfy_max = 0;

				SFlow.Vx = Mat::zeros(m_sFrame, CV_32FC1);
				SFlow.Vy = Mat::zeros(m_sFrame, CV_32FC1);
				SFlow.Vz = Mat::zeros(m_sFrame, CV_32FC1);
				if (m_vecKinectBodies.size() != 0)
				{
					//check for x_min, x_max,  y_min, y_max
					for (int body_it = 0; body_it < m_vecKinectBodies.size(); body_it++)
					{
						if (m_vecKinectBodies[body_it].Foreground_x_lim[0] < sfx_min)
						{
							sfx_min = m_vecKinectBodies[body_it].Foreground_x_lim[0];
						}
						if (m_vecKinectBodies[body_it].Foreground_x_lim[1] > sfx_max)
						{
							sfx_max = m_vecKinectBodies[body_it].Foreground_x_lim[1];
						}
						if (m_vecKinectBodies[body_it].Foreground_y_lim[0] < sfy_min)
						{
							sfy_min = m_vecKinectBodies[body_it].Foreground_y_lim[0];
						}
						if (m_vecKinectBodies[body_it].Foreground_y_lim[1] > sfy_max)
						{
							sfy_max = m_vecKinectBodies[body_it].Foreground_y_lim[1];
						}
					}
					//PD_flow_opencv sceneflow(m_sFrame.height, prevcolorgray, m_mtxDepthImg_Prev, currcolorgray, m_mtxDepthImg);
					PD_flow_opencv sceneflow(m_sFrame.height, prevcolorgray, m_mtxDepthImg_Prev, currcolorgray, m_mtxDepthImg, sfx_min, sfx_max, sfy_min, sfy_max);
					for (unsigned int v = 0; v < m_sFrame.height; v++)
					{
						for (unsigned int u = 0; u < m_sFrame.width; u++)
						{
							SFlow.Vx.at<float>(v, u) = sceneflow.dxp[v + u*m_sFrame.height];
							SFlow.Vy.at<float>(v, u) = sceneflow.dyp[v + u*m_sFrame.height];
							SFlow.Vz.at<float>(v, u) = sceneflow.dzp[v + u*m_sFrame.height];
						}
					}
				}

				
				//For labels
				Nui_NextLabelFrame();

				m_Divergence = Mat::zeros(m_sFrame, CV_32FC1);
				m_Vorticity = Mat::zeros(m_sFrame, CV_32FC3);
				m_Projection = Mat::zeros(m_sFrame, CV_32FC1);
				m_Rotation = Mat::zeros(m_sFrame, CV_32FC3);
				m_BodyPartProjection = Mat::zeros(m_sFrame, CV_32FC1);
				m_BodyPartRotation = Mat::zeros(m_sFrame, CV_32FC3);
				//For Skeleton
				Nui_NextBodyFrame();
				Nui_DrawFrame();
				if (m_vecKinectBodies.size() != 0)
				{
					ConvertWorldMotionToBodySpace();
					Skel_Present = true;
					if (m_vecKinectBodies.size() > m_NoSkeletons)
					{
						m_NoSkeletons = m_vecKinectBodies.size();
					}
				}

				if (m_Prev_vecKinectBodies.size() != 0)
				{
					ComputeStickKinematicFeatures();
				}

				m_Divergence.release();
				m_Vorticity.release();
				m_Projection.release();
				m_Rotation.release();
				m_BodyPartProjection.release();
				m_BodyPartRotation.release();

				SFlow.Vx.release();
				SFlow.Vy.release();
				SFlow.Vz.release();
			}
			
			if (m_FrameNum >= Frame_interval)
			{
				if (m_FrameNum % Frame_interval == 0)
				{
					//printf("%d \n", m_FrameNum);
					if (Skel_Present == true)
					{
						Compute_Action_Histograms();
						Clear_Vectors();
						Skel_Present = false;
						m_NoSkeletons = 0;
						/*
						if (User_actions.size() != 0)
						{
						//Send action information to reactive layer
						if (isUseThrift)
						{
						//Ensure connection to server before sending the result
						if (ActionManager->isConnected())
						{
						if (ActionManager->ensureConnection())
						{
						printf("Actions sent to Reactive Layer!!!\n");
						//User_actions has to be sent here
						ActionManager->getClient()->send_queryActionRecogition(User_actions);
						}
						}
						}
						}
						*/
					}
					else
					{
						Action_Detected = false;
						User_actions.clear();
						printf("Could you please act in front of me!!\n");
					}
				}

				if ((m_FrameNum % Frame_interval != 0) && (m_FrameNum % 3 == 0))
				{
					if (Action_Detected)
					{
						User_actions.clear();
						imi::UserAction U;
						std::vector<imi::ActionType::type> Final_action;
						Final_action.clear();
						Final_action.push_back(Map_inttoActionType_type(9));
						U.__set_actions(Final_action);
						User_actions.push_back(U);
						if (User_actions.size() != 0)
						{
							//Send action information to reactive layer
							if (isUseThrift)
							{
								//Ensure connection to server before sending the result
								if (ActionManager->isConnected())
								{
									if (ActionManager->ensureConnection())
									{
										printf("Actions sent to Reactive Layer 1!!!\n");
										//User_actions has to be sent here
										ActionManager->getClient()->send_queryActionRecogition(User_actions);
									}
									// EnsureConnection
								}
								// isConnected
							}
							// isUseThrift
						}
						//User_actions.size()
						Final_action.clear();
						User_actions.clear();
					}
					//Action_Detected
				}
			}
		}
		
	}
	return true;
}

//Helper Functions
void Process_Kinect::ComputeStickKinematicFeatures()
{
	for (int body_no = 0; body_no < m_vecKinectBodies.size(); body_no++)
	{
		//----------------------------------------------------------
		//Velocity or Motion Unit Vector at Necks
		float Neck_x = m_vecKinectBodies[body_no].Joints3D[JointType_Neck].val[0] - m_Prev_vecKinectBodies[body_no].Joints3D[JointType_Neck].val[0];
		float Neck_y = m_vecKinectBodies[body_no].Joints3D[JointType_Neck].val[1] - m_Prev_vecKinectBodies[body_no].Joints3D[JointType_Neck].val[1];
		float Neck_z = m_vecKinectBodies[body_no].Joints3D[JointType_Neck].val[2] - m_Prev_vecKinectBodies[body_no].Joints3D[JointType_Neck].val[2];

		float Neck_V_Magnitude = sqrt(pow(Neck_x, 2) + pow(Neck_y, 2) + pow(Neck_z, 2));
		if ((Neck_x == 0) & (Neck_y == 0) & (Neck_z == 0))
		{

		}
		else
		{
			Neck_x = Neck_x / Neck_V_Magnitude;
			Neck_y = Neck_y / Neck_V_Magnitude;
			Neck_z = Neck_z / Neck_V_Magnitude;
		}
		float max = 0.f;
		float x_component, y_component, z_component;
		for (int joint_iter = 0; joint_iter < JointType_Count; joint_iter++)
		{
			//x, y, z flow components of each foreground pixel
			x_component = abs((m_vecKinectBodies[body_no].Joints3D[joint_iter].val[0] - m_vecKinectBodies[body_no].Joints3D[JointType_Neck].val[0]) - (m_Prev_vecKinectBodies[body_no].Joints3D[joint_iter].val[0] - m_Prev_vecKinectBodies[body_no].Joints3D[JointType_Neck].val[0]));
			y_component = abs((m_vecKinectBodies[body_no].Joints3D[joint_iter].val[1] - m_vecKinectBodies[body_no].Joints3D[JointType_Neck].val[1]) - (m_Prev_vecKinectBodies[body_no].Joints3D[joint_iter].val[1] - m_Prev_vecKinectBodies[body_no].Joints3D[JointType_Neck].val[1]));
			z_component = abs((m_vecKinectBodies[body_no].Joints3D[joint_iter].val[2] - m_vecKinectBodies[body_no].Joints3D[JointType_Neck].val[2]) - (m_Prev_vecKinectBodies[body_no].Joints3D[joint_iter].val[2] - m_Prev_vecKinectBodies[body_no].Joints3D[JointType_Neck].val[2]));

			if (max < x_component)
			{
				max = x_component;
			}
			if (max < y_component)
			{
				max = y_component;
			}
			if (max < z_component)
			{
				max = z_component;
			}
		}

		double joint_Orientation;
		float Stick_P, Stick_Rx, Stick_Ry, Stick_Rz;
		//----------------------------------------------------------
		for (int joint_iter = 0; joint_iter < JointType_Count; joint_iter++)
		{
			//x, y, z flow components of each foreground pixel
			x_component = ((m_vecKinectBodies[body_no].Joints3D[joint_iter].val[0] - m_vecKinectBodies[body_no].Joints3D[JointType_Neck].val[0]) - (m_Prev_vecKinectBodies[body_no].Joints3D[joint_iter].val[0] - m_Prev_vecKinectBodies[body_no].Joints3D[JointType_Neck].val[0])) / max;
			y_component = ((m_vecKinectBodies[body_no].Joints3D[joint_iter].val[1] - m_vecKinectBodies[body_no].Joints3D[JointType_Neck].val[1]) - (m_Prev_vecKinectBodies[body_no].Joints3D[joint_iter].val[1] - m_Prev_vecKinectBodies[body_no].Joints3D[JointType_Neck].val[1])) / max;
			z_component = ((m_vecKinectBodies[body_no].Joints3D[joint_iter].val[2] - m_vecKinectBodies[body_no].Joints3D[JointType_Neck].val[2]) - (m_Prev_vecKinectBodies[body_no].Joints3D[joint_iter].val[2] - m_Prev_vecKinectBodies[body_no].Joints3D[JointType_Neck].val[2])) / max;

			//Kinematic feature - computed with respect to Neck
			//Stick_Projection - kinematic feature computed for each stick joint
			joint_Orientation = atan2(double(y_component), double(x_component)) * 180 / PI;
			//joint_Orientation,body_no,m_vecKinectBodies[body_no].BodyView,m_vecKinectBodies[body_no].BodyOrientation,Stick_P,Stick_Rx,Stick_Ry,Stick_Rz
			Stick_P = (x_component * Neck_x) + (y_component * Neck_y) + (z_component * Neck_z);

			//Stick_Rotation - kinematic feature computed for each foreground pixel
			Stick_Rx = (Neck_y * z_component) - (Neck_z * y_component);
			Stick_Ry = -((Neck_x * z_component) - (Neck_z * x_component));
			Stick_Rz = (Neck_x * y_component) - (Neck_y * x_component);

			Up_Down_Joint(joint_Orientation, body_no, joint_iter, m_vecKinectBodies[body_no].BodyView, m_vecKinectBodies[body_no].BodyOrientation, Stick_P, Stick_Rx, Stick_Ry, Stick_Rz);
			Left_Right_Joint(joint_Orientation, body_no, joint_iter, m_vecKinectBodies[body_no].BodyView, m_vecKinectBodies[body_no].BodyOrientation, Stick_P, Stick_Rx, Stick_Ry, Stick_Rz);
			Gaze_Joint(joint_Orientation, body_no, joint_iter, m_vecKinectBodies[body_no].BodyView, m_vecKinectBodies[body_no].BodyOrientation, Stick_P, Stick_Rx, Stick_Ry, Stick_Rz);
		}
	}
}

void Process_Kinect::PixeltoBodyPartLabel(std::vector<KinectBody>& vecKinectBodies)
{
	// reinitialize the tracking id if nobody is detected
	if (vecKinectBodies.size() != 0)
	{
		// For size of skeletons
		for (int i = 0; i < vecKinectBodies.size(); i++)
		{

			//Rows and columns..
			for (int r = vecKinectBodies[i].Foreground_x_lim[0]; r <= vecKinectBodies[i].Foreground_x_lim[1]; r++)
			{
				for (int c = vecKinectBodies[i].Foreground_y_lim[0]; c < vecKinectBodies[i].Foreground_y_lim[1]; c++)
				{
					double fValue = m_mtxLabels.at<float>(r, c);
				}
				//end of c
			}
			//end of r
		}
		//end of i
	}
	//end of if size
	
}

//bool Process_Kinect::Update(void)
//{
//	HANDLE handles[4];
//	for (int i = 0; i < 4; i++)
//		handles[i] = reinterpret_cast<HANDLE>(m_hFrameEvents[i]);

//	int nEventIdx = WaitForMultipleObjects(4, handles, TRUE, 5);
//	if (nEventIdx != WAIT_FAILED && nEventIdx != WAIT_TIMEOUT)
//	{
		/* !! very important note here, the following commented code generally cannot
		read all data correctly. By debuging the Kinect sample code, we find that
		FAILED(hr) is true for almost all times, and they adopt the strategy to always
		query whether the data is ready. I think there are some bits to indicate whether
		the different frames are ready, which are constantly updated and overwritten.
		You can only get the data at the samll instant that it is set to be ready, which
		means you have to constantly query. Also consider the synchronization problem
		between different frames, i.e. depth, color, label and body, you can hardly
		succeed with the following commented code to retrieve the data */
		/*	bool bDepth = Nui_NextDepthFrame();
		bool bColor = Nui_NextColorFrame();
		bool bLabel = Nui_NextLabelFrame();
		bool bBody = Nui_NextBodyFrame();*/

		// constant query mode, which works quite well
//		while (!Nui_NextDepthFrame());
//		while (!Nui_NextColorFrame());
//		while (!Nui_NextLabelFrame());
//		while (!Nui_NextBodyFrame());
//		m_nFrmNum++;

//		Nui_DrawFrame();
//		ShowDepth();

		// determine the main body for interaction
//		int nMainIdx = Nui_FindMainBody(m_vecKinectBodies);
//		if (nMainIdx == -1)
//			return false;
//		else
//		{
//			m_mtxHandMask.create(m_mtxDepthImg.size(), CV_32SC1);
//			bool bRefine = Nui_FindHandMask(m_mtxDepthImg, m_vecKinectBodies[nMainIdx], m_mtxHandMask, m_bHandActive);
//			return bRefine;
//		}
//	}
//	else
//		return false;
//}

bool Process_Kinect::Nui_NextColorFrame(void)
{
	if (!m_pColorFrameReader)
		return false;

	IColorFrame* pColorFrame = NULL;
	IFrameDescription* pFrameDescription = NULL;
	int nWidth, nHeight;
	//printf("Here1\n");
	//printf("Here1\n");
	HRESULT hr = m_pColorFrameReader->AcquireLatestFrame(&pColorFrame);

	if (SUCCEEDED(hr))	hr = pColorFrame->get_FrameDescription(&pFrameDescription);
	if (SUCCEEDED(hr))	hr = pFrameDescription->get_Width(&nWidth);
	if (SUCCEEDED(hr))	hr = pFrameDescription->get_Height(&nHeight);
	//printf("Here2\n");
	//printf("Here2\n");
	// make sure we've received valid data and update the color frame
	if (SUCCEEDED(hr) && (nWidth == m_sOrgColor.width) && (nHeight == m_sOrgColor.height))
	{
		m_FrameNum++;
		//printf("Success %d\n",m_FrameNum);
		static cv::Mat mtxColor(m_sOrgColor, CV_8UC4);
		UINT nBufferSize = m_sOrgColor.width * m_sOrgColor.height * sizeof(RGBQUAD);
		hr = pColorFrame->CopyConvertedFrameDataToArray(nBufferSize, reinterpret_cast<BYTE*>(mtxColor.data), ColorImageFormat_Bgra);
		cv::resize(mtxColor, m_mtxColorImg, m_mtxColorImg.size());

		SafeRelease(pFrameDescription);
	}
	//printf("Here3\n");
	//printf("Here3\n");
	SafeRelease(pColorFrame);
	if (FAILED(hr))
		return false;

	return true;
}

bool Process_Kinect::Nui_NextDepthFrame(void)
{
	if (!m_pDepthFrameReader)		return false;
		

	IDepthFrame* pDepthFrame = NULL;
	IFrameDescription* pFrameDescription = NULL;
	USHORT nDepthMinReliableDistance = 0;
	USHORT nDepthMaxReliableDistance = USHRT_MAX;
	int nWidth, nHeight;
	UINT nBufferSize = 0;
	UINT16 *pBuffer = NULL;
	//printf("Here4\n");
	//printf("Here4\n");
	HRESULT hr = m_pDepthFrameReader->AcquireLatestFrame(&pDepthFrame);
	if (FAILED(hr))
	{
		return false;
	}
	//printf("Here5\n");
	//printf("Here5\n");
	if (SUCCEEDED(hr))	hr = pDepthFrame->get_FrameDescription(&pFrameDescription);
	if (SUCCEEDED(hr))	hr = pFrameDescription->get_Width(&nWidth);
	if (SUCCEEDED(hr))	hr = pFrameDescription->get_Height(&nHeight);
	if (SUCCEEDED(hr))	hr = pDepthFrame->AccessUnderlyingBuffer(&nBufferSize, &pBuffer);
	

	// filtering the depth map with predefined or reliable distances
	if (SUCCEEDED(hr))	hr = pDepthFrame->get_DepthMinReliableDistance(&nDepthMinReliableDistance);
	if (SUCCEEDED(hr))	hr = pDepthFrame->get_DepthMaxReliableDistance(&nDepthMaxReliableDistance);
	// make sure we've received valid data and update the depth frame
	if (SUCCEEDED(hr) && pBuffer && (nWidth == m_sOrgDepth.width) && (nHeight == m_sOrgDepth.height))
	{
		//	m_fMinDepth = nDepthMinReliableDistance / 1000.0f;
		//	m_fMaxDepth = nDepthMaxReliableDistance / 1000.0f;
		static cv::Mat mtxDepth(m_sOrgDepth, CV_32FC1);
		static cv::Mat mtxPseudoDepth(m_sOrgDepth, CV_8UC3);
		float* ptr = (float*)mtxDepth.data;
		for (int i = 0; i < nWidth * nHeight; i++)
		{
			float fDepth = (*(pBuffer + i)) / 1000.0f;
			if ((fDepth >= m_fMinDepth) && (fDepth <= m_fMaxDepth))
				ptr[i] = fDepth;
			else
				ptr[i] = 0.0;
		}
		cv::resize(mtxDepth, m_mtxDepthImg, m_mtxDepthImg.size());

		SafeRelease(pFrameDescription);
	}

	SafeRelease(pDepthFrame);
	//printf("Here6\n");
	//printf("Here6\n");
	if (FAILED(hr))
	{
		return false;
	}

	return true;
}

bool Process_Kinect::Nui_NextLabelFrame(void)
{
	IBodyIndexFrame* pBodyIndexFrame = NULL;
	IFrameDescription* pFrameDescription = NULL;
	int nWidth, nHeight;
	UINT nBufferSize = 0;
	BYTE *pBuffer = NULL;

	HRESULT hr = m_pBodyIndexReader->AcquireLatestFrame(&pBodyIndexFrame);

	if (SUCCEEDED(hr))	hr = pBodyIndexFrame->get_FrameDescription(&pFrameDescription);
	if (SUCCEEDED(hr))	hr = pFrameDescription->get_Width(&nWidth);
	if (SUCCEEDED(hr))	hr = pFrameDescription->get_Height(&nHeight);
	if (SUCCEEDED(hr))	hr = pBodyIndexFrame->AccessUnderlyingBuffer(&nBufferSize, &pBuffer);
	
	// make sure we've received valid data and update the body index frame
	if (SUCCEEDED(hr) && pBuffer && (nWidth == m_sOrgDepth.width) && (nHeight == m_sOrgDepth.height))
	{
		Foreground_points.clear();
		static cv::Mat mtxLabels(m_sOrgDepth, CV_32SC1);
		static cv::Mat mtxLabelColors(m_sOrgDepth, CV_8UC3);
		for (int i = 0; i < nWidth * nHeight; i++)
		{
			unsigned char R, G, B;
			GetDistinctColor(pBuffer[i], R, G, B);

			if (pBuffer[i] != 0xff)
			{
				*((int*)mtxLabels.data + i) = pBuffer[i];
				*((cv::Vec3b*)mtxLabelColors.data + i) = cv::Vec3b(B, G, R);
			}
			else
			{
				*((int*)mtxLabels.data + i) = 0;
				*((cv::Vec3b*)mtxLabelColors.data + i) = cv::Vec3b(0, 0, 0);
			}
		}

		
		//!! be very careful when resizing the images of integer type, bug of opencv
		cv::resize(mtxLabels, mtxLabels, mtxLabels.size(), 0, 0, cv::INTER_NEAREST);
		cv::resize(mtxLabels, m_mtxLabels, m_mtxLabels.size(), 0, 0, cv::INTER_NEAREST);
		for (int r = 0; r < m_mtxLabels.rows; r++)
		{
			for (int c = 0; c < m_mtxLabels.cols; c++)
			{
				double fValue = m_mtxLabels.at<float>(r, c);
				if (fValue != 0)
				{
					std::vector<int> Fg_pt(5, 0);
					Fg_pt[0] = r;
					Fg_pt[1] = c;
					Foreground_points.push_back(Fg_pt);
				}
			}
			//end of c
		}
		//end of r
		cv::imshow("Label", mtxLabelColors);

		SafeRelease(pFrameDescription);
	}
	SafeRelease(pBodyIndexFrame);

	if (FAILED(hr))
		return false;

	return true;
}


bool Process_Kinect::Nui_NextBodyFrame(void)
{
	if (!m_pBodyFrameReader)
		return false;

	IBodyFrame* pBodyFrame = NULL;
	IBody* ppBodies[BODY_COUNT] = { 0 };

	HRESULT hr = m_pBodyFrameReader->AcquireLatestFrame(&pBodyFrame);
	if (SUCCEEDED(hr))
	{
		hr = pBodyFrame->GetAndRefreshBodyData(_countof(ppBodies), ppBodies);

		if (SUCCEEDED(hr))
		{
			m_Prev_vecKinectBodies.clear();
			for (int body_no = 0; body_no < m_vecKinectBodies.size(); body_no++)
			{
				//Create the Kinectbody structure for each skeleton
				m_Prev_vecKinectBodies.push_back(m_vecKinectBodies[body_no]);
			}
			m_vecKinectBodies.clear();
			double Skel_No_Fgpts[6][6] = { 0 }, Skel_Fg_Row[6][6] = { 0 }, Skel_Fg_Col[6][6] = { 0 };
			float Body_center_Vx[6][6] = { 0 }, Body_center_Vy[6][6] = { 0 }, Body_center_Vz[6][6] = { 0 };
			int Skel_Bodypartindex = 0;

			//-----------------------------------------------------------------------------
			//Replace Left or lost Skeleton ID with zero.
			//Will be replaced by new one later in the loop.
			uint64_t Tracking_ID;
			bool FoundID_Prevframe[BODY_COUNT] = { false };
			for (int i = 0; i < BODY_COUNT; ++i)
			{
				IBody* pBody = ppBodies[i];
				if (pBody)
				{
					BOOLEAN bTracked = false;
					hr = pBody->get_IsTracked(&bTracked);
					if (SUCCEEDED(hr) && bTracked)
					{
						pBody->get_TrackingId(&(Tracking_ID));
						for (int ID_iter = 0; ID_iter < BODY_COUNT; ++ID_iter)
						{
							if (Skeleton_TrackingID[ID_iter] == Tracking_ID)
							{
								FoundID_Prevframe[ID_iter] = true;
								break;
							}
						}
					}
				}
			}
			for (int i = 0; i < BODY_COUNT; ++i)
			{
				if (FoundID_Prevframe[i] == false)
				{
					Skeleton_TrackingID[i] = 0;
					//call for action recognition if vectors are populated and clear all vectors
				}
			}
			//-----------------------------------------------------------------------------
			for (int i = 0; i < BODY_COUNT; ++i)
			{
				IBody* pBody = ppBodies[i];
				if (pBody)
				{
					BOOLEAN bTracked = false;
					hr = pBody->get_IsTracked(&bTracked);
					if (SUCCEEDED(hr) && bTracked)
					{
						Joint joints[JointType_Count];
						double Orientation, leftdepth=0, rightdepth=0;
						int view = 0;
						int x_min, y_min, x_max = 0, y_max = 0;
						//int left_tracked = 0, right_tracked = 0;
						/*
						HandState leftHandState = HandState_Unknown;
						HandState rightHandState = HandState_Unknown;

						pBody->get_HandLeftState(&leftHandState);
						pBody->get_HandRightState(&rightHandState);*/
						hr = pBody->GetJoints(_countof(joints), joints);
						if (SUCCEEDED(hr))
						{
							cv::Point ptJoints[JointType_Count];
							cv::Vec3f vJoints[JointType_Count];
							TrackingState	nStates[JointType_Count];
							for (int j = 0; j < _countof(joints); ++j)
							{
								vJoints[joints[j].JointType] = cv::Vec3f(joints[j].Position.X, joints[j].Position.Y, joints[j].Position.Z);
								ptJoints[joints[j].JointType] = ProjectToPixel(vJoints[joints[j].JointType]);
								nStates[joints[j].JointType] = joints[j].TrackingState;
								if (j == 0)
								{
									x_min = ptJoints[joints[0].JointType].x;
									y_min = ptJoints[joints[0].JointType].y;
								}
							
								if (ptJoints[joints[j].JointType].x > 0 && (ptJoints[joints[j].JointType].x <=m_sFrame.width))
								{
									if (ptJoints[joints[j].JointType].x > x_max)
									{
										x_max = ptJoints[joints[j].JointType].x;
									}
									if (ptJoints[joints[j].JointType].x < x_min)
									{
										x_min = ptJoints[joints[j].JointType].x;
									}
								}

								if (ptJoints[joints[j].JointType].y > 0 && (ptJoints[joints[j].JointType].y <= m_sFrame.height))
								{
									if (ptJoints[joints[j].JointType].y > y_max)
									{
										y_max = ptJoints[joints[j].JointType].y;
									}
									if (ptJoints[joints[j].JointType].y < y_min)
									{
										y_min = ptJoints[joints[j].JointType].y;
									}
								}
							}
							
							/* Compute Body orientation or posture
							The spine base (0) and Neck (2) points are used for measuring this orientation
							Upright or non-upright posture can be identified.
							*/
							Orientation = atan2(double(joints[0].Position.Y - joints[2].Position.Y), double(joints[0].Position.X - joints[2].Position.X)) * 180 / PI;
							
							if ((Orientation >= -135) & (Orientation <= -45))
							{
								Orientation = 1;
							}
							else if ((Orientation >= 45) & (Orientation <= 135))
							{
								Orientation = 2;
							}
							else if ((Orientation > -45) & (Orientation < 45))
							{
								Orientation = 4;
							}
							else if ((Orientation >= -180) & (Orientation < -135))
							{
								Orientation = 3;
							}
							else if ((Orientation > 135) & (Orientation <= 180))
							{
								Orientation = 3;
							}
							
							// Adjust foreground minimum to account for pixels - Head pixels above head joint
							if ((x_min - EXTEND_FG >= 0) && (x_min - EXTEND_FG <= m_sFrame.width))
							{
								x_min = x_min - EXTEND_FG;
							}
							else
							{
								x_min = 0;
							}

							if ((x_max + EXTEND_FG <= m_sFrame.width) && (x_max + EXTEND_FG >= 0))
							{
								x_max = x_max + EXTEND_FG;
							}
							else
							{
								x_max = m_sFrame.width;
							}

							if ((y_min - EXTEND_FG >= 0) && (y_min - EXTEND_FG <= m_sFrame.height))
							{
								y_min = y_min - EXTEND_FG;
							}
							else
							{
								y_min = 0;
							}

							if ((y_max + EXTEND_FG <= m_sFrame.height) && (y_max + EXTEND_FG >= 0))
							{
								y_max = y_max + EXTEND_FG;
							}
							else
							{
								y_max = m_sFrame.height;
							}

							/*Compute person's view based on joint's Z position
							Left depth  -  left shouler - shoulder center
							Right side  -  right shouler - shoulder center
							Threshold is anything more than 0.1  or -0.1
							if (leftdepth - rightdepth) > positive thresh -> right facing
							if (leftdepth - rightdepth) < negative thresh -> left facing
							*/
							leftdepth = joints[JointType_ShoulderLeft].Position.Z - joints[JointType_SpineShoulder].Position.Z;
							rightdepth = joints[JointType_ShoulderRight].Position.Z - joints[JointType_SpineShoulder].Position.Z;
							if (abs(leftdepth - rightdepth) < 0.1)
							{
								int frontal_face;
								cv::Mat frontal_frame;
								//Front or back to be decided based on face
								m_mtxColorImg(cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min)).copyTo(frontal_frame);
								frontal_face = detectFaces(frontal_frame);
								if (frontal_face != 0)
								{
									view = 2;
								}
								else
								{
									view = 1;
								}
							}
							else
							{
								if ((leftdepth - rightdepth) > 0)
								{
									//Right facing view
									view = 4;
								}
								else
								{
									//Left facing view
									view = 3;
								}
							}
							//printf("Joint : %f %d\n", (leftdepth - rightdepth),view);
							//printf("1x_min %d x_max %d y_min %d y_max %d\n", x_min,x_max,y_min,y_max);
							//  x limits are for width or column
							bool valid_neck = false;
							if (ptJoints[JointType_Neck].y > 0 && (ptJoints[JointType_Neck].y <= m_sFrame.height))
							{
								if (ptJoints[JointType_Neck].x > 0 && (ptJoints[JointType_Neck].x <= m_sFrame.width))
								{
									valid_neck = true;
								}
							}
							//Create the Kinectbody structure for each skeleton
							KinectBody Tempbody(ptJoints, vJoints, nStates, Orientation, view, x_min, x_max, y_min, y_max, valid_neck);
							//KinectBody kbTmp(ptJoints, vJoints, nStates, leftHandState, rightHandState);
							pBody->get_TrackingId(&(Tempbody.BodyIndex));
							m_vecKinectBodies.push_back(Tempbody);
							
							//printf("2: Skeleton %d\n", i);
							//printf("test uint64_t : %" PRIu64 "\n", Tempbody.BodyIndex);
							bool Track_ID_found = false, first_zero = false;
							int Skel_ID;

							for (int ID_iter = 0; ID_iter < BODY_COUNT; ++ID_iter)
							{
								if (Skeleton_TrackingID[ID_iter] == Tempbody.BodyIndex)
								{
									Track_ID_found = true;
									Skel_ID = ID_iter;
									break;
								}
								if (first_zero == false)
								{
									if (Skeleton_TrackingID[ID_iter] == 0)
									{
										first_zero = true;
										Skel_ID = ID_iter;
									}
								}
							}
							Skeleton_TrackingID[Skel_ID] = Tempbody.BodyIndex;
							/*
							1) Track_ID_found = true -> Skeleton tracked previously
							2) Track_ID_found = false and first_zero = true, 
							newly seen skeleton insert into first_zero location - for initial population
							3) Track_ID_found = false and first_zero = false,
							   when 6 skeletons are there, one skeleton leaves and new one enters
								Removed skeleton ID must be replaced with new one 
								Complete Action recognition for previous skeleton ID and clear all vectors for the same
							*/
							//printf("Skel ID %d \n", Skel_ID);
							// Check for neck's validity

							if (ptJoints[JointType_Neck].y > 0 && (ptJoints[JointType_Neck].y <= m_sFrame.height))
							{
								if (ptJoints[JointType_Neck].x > 0 && (ptJoints[JointType_Neck].x <= m_sFrame.width))
								{
									//Velocity or Motion Unit Vector at Necks
									float Neck_x = SFlow.Vx.at<float>(ptJoints[JointType_Neck].y, ptJoints[JointType_Neck].x);
									float Neck_y = SFlow.Vy.at<float>(ptJoints[JointType_Neck].y, ptJoints[JointType_Neck].x);
									float Neck_z = SFlow.Vz.at<float>(ptJoints[JointType_Neck].y, ptJoints[JointType_Neck].x);

									float Neck_V_Magnitude = sqrt(pow(Neck_x, 2) + pow(Neck_y, 2) + pow(Neck_z, 2));
									if ((Neck_x == 0) & (Neck_y == 0) & (Neck_z == 0))
									{

									}
									else
									{
										Neck_x = Neck_x / Neck_V_Magnitude;
										Neck_y = Neck_y / Neck_V_Magnitude;
										Neck_z = Neck_z / Neck_V_Magnitude;
									}
									int dist = 0, dist_min = 0, bodypart_index, row_index, col_index;
									for (int fg_index = 0; fg_index < Foreground_points.size(); fg_index++)
									{
										if ((Foreground_points[fg_index][1] >= x_min) && (Foreground_points[fg_index][1] <= x_max) && (Foreground_points[fg_index][0] >= y_min) && (Foreground_points[fg_index][0] <= y_max))
										{
											//Skeleton index for each point.
											Foreground_points[fg_index][2] = i;
											//Index in m_vecKinectBodies
											//Foreground_points[fg_index][4] = m_vecKinectBodies.size()-1;
											Foreground_points[fg_index][4] = Skel_ID;

											int min_flag = 0;
											//Body part label for each joint
											for (int joint_index = 0; joint_index < _countof(joints); ++joint_index)
											{
												if (ptJoints[joints[joint_index].JointType].x > 0 && (ptJoints[joints[joint_index].JointType].x <= m_sFrame.width) && ptJoints[joints[joint_index].JointType].y > 0 && (ptJoints[joints[joint_index].JointType].y <= m_sFrame.height))
												{
													// Assume first one as minimum distance
													if (min_flag == 0)
													{
														dist_min = sqrt(pow(Foreground_points[fg_index][1] - ptJoints[joints[joint_index].JointType].x, 2) +
															pow(Foreground_points[fg_index][0] - ptJoints[joints[joint_index].JointType].y, 2));
														min_flag = 1;
													}
													dist = sqrt(pow(Foreground_points[fg_index][1] - ptJoints[joints[joint_index].JointType].x, 2) +
														pow(Foreground_points[fg_index][0] - ptJoints[joints[joint_index].JointType].y, 2));

													if (dist < dist_min)
													{
														dist_min = dist;
														bodypart_index = joint_index;
													}
												}
											}

											//Foreground_points[fg_index][3] = bodypart_index; //Store specific body parts
											//bodypart_index
											//Head - 2(Neck), 3(Head)
											//Torso -  0(Spinebase), 1(Spinemid), 20(Spineshoulder), 16(HipRight), 12(HipLeft), 4 (Shoulderleft), 8(Shoulderright)
											//LeftHand - 5(Elbowleft), 6(Wristleft), 7(Handleft), 21(HandTipleft), 22(Thumbleft)
											//RightHand - 9(Elbowright), 10(Wristright), 11(Handleft), 23(HandTipright), 24(Thumbright)
											//LeftLeg - 13(KneeLeft), 14(AnkleLeft), 15(FootLeft)
											//RightLeg - 17(KneeRight), 18(AnkleRight), 19(FootRight)
											if (bodypart_index == 2 || bodypart_index == 3)
											{
												Skel_Bodypartindex = 0; // Head
											}
											else if (bodypart_index == 0 || bodypart_index == 1 || bodypart_index == 20 || bodypart_index == 16 || bodypart_index == 12 || bodypart_index == 4 || bodypart_index == 8)
											{
												Skel_Bodypartindex = 1; // Torso
											}
											else if (bodypart_index == 5 || bodypart_index == 6 || bodypart_index == 7 || bodypart_index == 21 || bodypart_index == 22)
											{
												Skel_Bodypartindex = 2; // LeftHand
											}
											else if (bodypart_index == 9 || bodypart_index == 10 || bodypart_index == 11 || bodypart_index == 23 || bodypart_index == 24)
											{
												Skel_Bodypartindex = 3; // RightHand
											}
											else if (bodypart_index == 13 || bodypart_index == 14 || bodypart_index == 15)
											{
												Skel_Bodypartindex = 4; // LeftLeg
											}
											else if (bodypart_index == 17 || bodypart_index == 18 || bodypart_index == 19)
											{
												Skel_Bodypartindex = 5; // RightLeg
											}
											Skel_No_Fgpts[i][Skel_Bodypartindex]++;
											Skel_Fg_Row[i][Skel_Bodypartindex] = Skel_Fg_Row[i][Skel_Bodypartindex] + Foreground_points[fg_index][0];
											Skel_Fg_Col[i][Skel_Bodypartindex] = Skel_Fg_Col[i][Skel_Bodypartindex] + Foreground_points[fg_index][1];
											Foreground_points[fg_index][3] = Skel_Bodypartindex; // Abstracted body part index

											//x, y, z flow components of each foreground pixel
											float x_component = SFlow.Vx.at<float>(Foreground_points[fg_index][0], Foreground_points[fg_index][1]);
											float y_component = SFlow.Vy.at<float>(Foreground_points[fg_index][0], Foreground_points[fg_index][1]);
											float z_component = SFlow.Vz.at<float>(Foreground_points[fg_index][0], Foreground_points[fg_index][1]);

											//Kinematic feature - computed with respect to Neck
											//Projection - kinematic feature computed for each foreground pixel
											float & Kinematic_Feature_P = m_Projection.at<float>(Foreground_points[fg_index][0], Foreground_points[fg_index][1]);
											Kinematic_Feature_P = (x_component * Neck_x) + (y_component * Neck_y) + (z_component * Neck_z);

											//Rotation - kinematic feature computed for each foreground pixel
											Vec3f & Kinematic_Feature_R = m_Rotation.at<Vec3f>(Foreground_points[fg_index][0], Foreground_points[fg_index][1]);
											Kinematic_Feature_R[0] = (Neck_y * z_component) - (Neck_z * y_component);
											Kinematic_Feature_R[1] = -((Neck_x * z_component) - (Neck_z * x_component));
											Kinematic_Feature_R[2] = (Neck_x * y_component) - (Neck_y * x_component);

											//Kinematic feature - computed with respect to Neighbourhood
											if (Foreground_points[fg_index][0] + 1 >= m_Divergence.rows)
											{
												row_index = 0;
											}
											else
											{
												row_index = Foreground_points[fg_index][0] + 1;
											}
											if (Foreground_points[fg_index][1] + 1 >= m_Divergence.cols)
											{
												col_index = 0;
											}
											else
											{
												col_index = Foreground_points[fg_index][1] + 1;
											}
											float du_dx = x_component - SFlow.Vx.at<float>(row_index, Foreground_points[fg_index][1]);
											float du_dy = x_component - SFlow.Vx.at<float>(Foreground_points[fg_index][0], col_index);

											float dv_dx = y_component - SFlow.Vy.at<float>(row_index, Foreground_points[fg_index][1]);
											float dv_dy = y_component - SFlow.Vy.at<float>(Foreground_points[fg_index][0], col_index);

											float dw_dx = z_component - SFlow.Vz.at<float>(row_index, Foreground_points[fg_index][1]);
											float dw_dy = z_component - SFlow.Vz.at<float>(Foreground_points[fg_index][0], col_index);

											//Divergence - kinematic feature computed for each foreground pixel - Assume dw_dz = 0
											float & Kinematic_Feature_D = m_Divergence.at<float>(Foreground_points[fg_index][0], Foreground_points[fg_index][1]);
											Kinematic_Feature_D = du_dx + dv_dy;

											//Vorticity - kinematic feature computed for each foreground pixel - Assume du_dz, dv_dz = 0
											Vec3f & Kinematic_Feature_V = m_Vorticity.at<Vec3f>(Foreground_points[fg_index][0], Foreground_points[fg_index][1]);
											Kinematic_Feature_V[0] = dw_dy;
											Kinematic_Feature_V[1] = -(dw_dx);
											Kinematic_Feature_V[2] = dv_dx - du_dy;
										}
									}
									//end of for fg_index
									float Body_Vx = 0, Body_Vy = 0, Body_Vz = 0, Body_V_Magnitude = 0;
									for (int Skel_Fgiter = 0; Skel_Fgiter < BODY_COUNT; Skel_Fgiter++)
									{
										for (int Skel_Fgiter_c = 0; Skel_Fgiter_c < BODY_COUNT; Skel_Fgiter_c++)
										{
											if (Skel_No_Fgpts[Skel_Fgiter][Skel_Fgiter_c] != 0)
											{
												Skel_Fg_Row[Skel_Fgiter][Skel_Fgiter_c] = (int)Skel_Fg_Row[Skel_Fgiter][Skel_Fgiter_c] / Skel_No_Fgpts[Skel_Fgiter][Skel_Fgiter_c];
												Skel_Fg_Col[Skel_Fgiter][Skel_Fgiter_c] = (int)Skel_Fg_Col[Skel_Fgiter][Skel_Fgiter_c] / Skel_No_Fgpts[Skel_Fgiter][Skel_Fgiter_c];

												//x, y, z flow components of each foreground pixel
												Body_Vx = SFlow.Vx.at<float>(Skel_Fg_Row[Skel_Fgiter][Skel_Fgiter_c], Skel_Fg_Col[Skel_Fgiter][Skel_Fgiter_c]);
												Body_Vy = SFlow.Vy.at<float>(Skel_Fg_Row[Skel_Fgiter][Skel_Fgiter_c], Skel_Fg_Col[Skel_Fgiter][Skel_Fgiter_c]);
												Body_Vz = SFlow.Vz.at<float>(Skel_Fg_Row[Skel_Fgiter][Skel_Fgiter_c], Skel_Fg_Col[Skel_Fgiter][Skel_Fgiter_c]);
												Body_V_Magnitude = sqrt(pow(Body_Vx, 2) + pow(Body_Vy, 2) + pow(Body_Vz, 2));

												if ((Body_Vx == 0) & (Body_Vy == 0) & (Body_Vz == 0))
												{

												}
												else
												{
													Body_center_Vx[Skel_Fgiter][Skel_Fgiter_c] = Body_Vx / Body_V_Magnitude;
													Body_center_Vy[Skel_Fgiter][Skel_Fgiter_c] = Body_Vy / Body_V_Magnitude;
													Body_center_Vz[Skel_Fgiter][Skel_Fgiter_c] = Body_Vz / Body_V_Magnitude;
												}

											}//end of if Skel_No_Fgpts
										}//end of for Skel_Fgiter_c
									}//end of for Skel_Fgiter

									//Compute Body part referenced kinematic features
									for (int fg_index = 0; fg_index < Foreground_points.size(); fg_index++)
									{
										//BodyPart Center pixel velocities - (Body_center_Vx[Foreground_points[fg_index][2]][Foreground_points[fg_index][3]], Body_center_Vy[Foreground_points[fg_index][2]][Foreground_points[fg_index][3]], Body_center_Vz[Foreground_points[fg_index][2]][Foreground_points[fg_index][3]])
										//x, y, z flow components of each foreground pixel
										float x_component = SFlow.Vx.at<float>(Foreground_points[fg_index][0], Foreground_points[fg_index][1]);
										float y_component = SFlow.Vy.at<float>(Foreground_points[fg_index][0], Foreground_points[fg_index][1]);
										float z_component = SFlow.Vz.at<float>(Foreground_points[fg_index][0], Foreground_points[fg_index][1]);

										//Kinematic feature - computed with respect to Neck
										//Projection - kinematic feature computed for each foreground pixel
										float & Kinematic_Feature_BPP = m_BodyPartProjection.at<float>(Foreground_points[fg_index][0], Foreground_points[fg_index][1]);
										Kinematic_Feature_BPP = (x_component * Body_center_Vx[Foreground_points[fg_index][2]][Foreground_points[fg_index][3]]) + (y_component * Body_center_Vy[Foreground_points[fg_index][2]][Foreground_points[fg_index][3]]) + (z_component * Body_center_Vz[Foreground_points[fg_index][2]][Foreground_points[fg_index][3]]);


										//Rotation - kinematic feature computed for each foreground pixel
										Vec3f & Kinematic_Feature_BPR = m_BodyPartRotation.at<Vec3f>(Foreground_points[fg_index][0], Foreground_points[fg_index][1]);
										Kinematic_Feature_BPR[0] = (Body_center_Vy[Foreground_points[fg_index][2]][Foreground_points[fg_index][3]] * z_component) - (Body_center_Vz[Foreground_points[fg_index][2]][Foreground_points[fg_index][3]] * y_component);
										Kinematic_Feature_BPR[1] = -((Body_center_Vx[Foreground_points[fg_index][2]][Foreground_points[fg_index][3]] * z_component) - (Body_center_Vz[Foreground_points[fg_index][2]][Foreground_points[fg_index][3]] * x_component));
										Kinematic_Feature_BPR[2] = (Body_center_Vx[Foreground_points[fg_index][2]][Foreground_points[fg_index][3]] * y_component) - (Body_center_Vy[Foreground_points[fg_index][2]][Foreground_points[fg_index][3]] * x_component);
									}//end of fg_index

								}// Check  Neck x validity

							}// Check  Neck y validity
						}
						// end of if succeeded
					}
					// end of if succeeded
				}
				// end of if pBody
			}
			// end of for i =0
		}
		// end of if succeeded
		for (int i = 0; i < _countof(ppBodies); ++i)
		{
			SafeRelease(ppBodies[i]);
		}
	}
	// end of if succeeded
	SafeRelease(pBodyFrame);

	if (FAILED(hr))
		return false;

	return true;
}

/** @function detectAndDisplay */
int  Process_Kinect::detectFaces(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	/*printf("%d \n", faces.size());

	for (size_t i = 0; i < faces.size(); i++)
	{
	Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
	ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
	}
	//-- Show what you got
	imshow("Result", frame);*/

	return faces.size();
}

/*
int Process_Kinect::Nui_FindMainBody(std::vector<KinectBody>& vecKinectBodies)
{
	static UINT64 nTrackingID = -1;

	// reinitialize the tracking id if nobody is detected
	if (vecKinectBodies.size() == 0)
	{
		nTrackingID = -1;

		return -1;
	}
	else
	{
		// initialize the tracking id if anybody is detected
		if (nTrackingID == -1)
		{
			// find the one nearest to the camera
			double fMinDist = 1e10;
			int nMainIdx = -1;
			for (int i = 0; i < vecKinectBodies.size(); i++)
			{
				vecKinectBodies[i].IsMainBody = false;
				double fDist = cv::norm(vecKinectBodies[i].Joints3D[JointType_SpineMid]);
				if (fDist < fMinDist)
				{
					fMinDist = fDist;
					nMainIdx = i;
				}
			}
			vecKinectBodies[nMainIdx].IsMainBody = true;
			nTrackingID = vecKinectBodies[nMainIdx].BodyIndex;

			return nMainIdx;
		}
		else
		{
			// track the body with the same tracking id
			int nMainIdx = -1;
			for (int i = 0; i < vecKinectBodies.size(); i++)
			{
				if (vecKinectBodies[i].BodyIndex == nTrackingID)
				{
					vecKinectBodies[i].IsMainBody = true;
					nMainIdx = i;
				}
				else
					vecKinectBodies[i].IsMainBody = false;
			}

			return nMainIdx;
		}
	}
}
*/
void Process_Kinect::Nui_DrawFrame(void)
{
	cv::Mat mtxRGB(m_mtxColorImg.clone());
	for (int i = 0; i < m_vecKinectBodies.size(); i++)
		Nui_DrawBody(mtxRGB, m_vecKinectBodies[i]);

	cv::imshow("co", mtxRGB);
}

void Process_Kinect::Nui_DrawBody(cv::Mat &mtxRGBCanvas, KinectBody& body)
{
	// Torso
	Nui_DrawBone(mtxRGBCanvas, body, JointType_Head, JointType_Neck);
	Nui_DrawBone(mtxRGBCanvas, body, JointType_Neck, JointType_SpineShoulder);
	Nui_DrawBone(mtxRGBCanvas, body, JointType_SpineShoulder, JointType_SpineMid);
	Nui_DrawBone(mtxRGBCanvas, body, JointType_SpineMid, JointType_SpineBase);
	Nui_DrawBone(mtxRGBCanvas, body, JointType_SpineShoulder, JointType_ShoulderRight);
	Nui_DrawBone(mtxRGBCanvas, body, JointType_SpineShoulder, JointType_ShoulderLeft);
	Nui_DrawBone(mtxRGBCanvas, body, JointType_SpineBase, JointType_HipRight);
	Nui_DrawBone(mtxRGBCanvas, body, JointType_SpineBase, JointType_HipLeft);

	// Right Arm    
	Nui_DrawBone(mtxRGBCanvas, body, JointType_ShoulderRight, JointType_ElbowRight);
	Nui_DrawBone(mtxRGBCanvas, body, JointType_ElbowRight, JointType_WristRight);
	Nui_DrawBone(mtxRGBCanvas, body, JointType_WristRight, JointType_HandRight);
	Nui_DrawBone(mtxRGBCanvas, body, JointType_HandRight, JointType_HandTipRight);
	Nui_DrawBone(mtxRGBCanvas, body, JointType_WristRight, JointType_ThumbRight);

	// Left Arm
	Nui_DrawBone(mtxRGBCanvas, body, JointType_ShoulderLeft, JointType_ElbowLeft);
	Nui_DrawBone(mtxRGBCanvas, body, JointType_ElbowLeft, JointType_WristLeft);
	Nui_DrawBone(mtxRGBCanvas, body, JointType_WristLeft, JointType_HandLeft);
	Nui_DrawBone(mtxRGBCanvas, body, JointType_HandLeft, JointType_HandTipLeft);
	Nui_DrawBone(mtxRGBCanvas, body, JointType_WristLeft, JointType_ThumbLeft);

	// Right Leg
	Nui_DrawBone(mtxRGBCanvas, body, JointType_HipRight, JointType_KneeRight);
	Nui_DrawBone(mtxRGBCanvas, body, JointType_KneeRight, JointType_AnkleRight);
	Nui_DrawBone(mtxRGBCanvas, body, JointType_AnkleRight, JointType_FootRight);

	// Left Leg
	Nui_DrawBone(mtxRGBCanvas, body, JointType_HipLeft, JointType_KneeLeft);
	Nui_DrawBone(mtxRGBCanvas, body, JointType_KneeLeft, JointType_AnkleLeft);
	Nui_DrawBone(mtxRGBCanvas, body, JointType_AnkleLeft, JointType_FootLeft);

	// draw the joints
	for (int i = 0; i < JointType_Count; i++)
	{
		cv::Point pt0 = Nui_ProjectDepthToColorPixel(body.Joints3D[i]);
		if (pt0.x != -std::numeric_limits<int>::infinity() && pt0.y != -std::numeric_limits<int>::infinity())
		{
			cv::Point ptRGB(1.0 * pt0.x / m_sOrgColor.width * mtxRGBCanvas.cols, 1.0 * pt0.y / m_sOrgColor.height * mtxRGBCanvas.rows);
			cv::circle(mtxRGBCanvas, ptRGB, 6, cv::Scalar(0, 255, 0), -1);
			
		}
	}

	// highlight the middle spine to determine whether or not activate gesture recognition
	cv::Point ptSignal = Nui_ProjectDepthToColorPixel(body.Joints3D[JointType_SpineMid]);
	if (ptSignal.x != -std::numeric_limits<int>::infinity() && ptSignal.y != -std::numeric_limits<int>::infinity())
	{
		cv::Point ptRGB(1.0 * ptSignal.x / m_sOrgColor.width * mtxRGBCanvas.cols, 1.0 * ptSignal.y / m_sOrgColor.height * mtxRGBCanvas.rows);
		cv::circle(mtxRGBCanvas, ptRGB, 8, cv::Scalar(0, 0, 255), -1);
	}

	// draw the left hand
	/*cv::Point ptLHand = Nui_ProjectDepthToColorPixel(body.LHand3D);
	if (ptLHand.x != -std::numeric_limits<int>::infinity() && ptLHand.y != -std::numeric_limits<int>::infinity())
	{
		cv::Point ptLHandRGB(1.0 * ptLHand.x / m_sOrgColor.width * mtxRGBCanvas.cols, 1.0 * ptLHand.y / m_sOrgColor.height * mtxRGBCanvas.rows);
		switch (body.LHandState)
		{
		case HandState_Closed:
			cv::circle(mtxRGBCanvas, ptLHandRGB, 15, cv::Scalar(255, 0, 0), -1);
			break;
		case HandState_Open:
			cv::circle(mtxRGBCanvas, ptLHandRGB, 15, cv::Scalar(0, 255, 0), -1);
			break;
		case HandState_Lasso:
			cv::circle(mtxRGBCanvas, ptLHandRGB, 15, cv::Scalar(0, 0, 255), -1);
			break;
		}
	}

	// draw the right hand
	cv::Point ptRHand = Nui_ProjectDepthToColorPixel(body.RHand3D);
	if (ptRHand.x != -std::numeric_limits<int>::infinity() && ptRHand.y != -std::numeric_limits<int>::infinity())
	{
		cv::Point ptRHandRGB(1.0 * ptRHand.x / m_sOrgColor.width * mtxRGBCanvas.cols, 1.0 * ptRHand.y / m_sOrgColor.height * mtxRGBCanvas.rows);
		switch (body.RHandState)
		{
		case HandState_Closed:
			cv::circle(mtxRGBCanvas, ptRHandRGB, 15, cv::Scalar(255, 0, 0), -1);
			break;
		case HandState_Open:
			cv::circle(mtxRGBCanvas, ptRHandRGB, 15, cv::Scalar(0, 255, 0), -1);
			break;
		case HandState_Lasso:
			cv::circle(mtxRGBCanvas, ptRHandRGB, 15, cv::Scalar(0, 0, 255), -1);
			break;
		}
	}*/
}

void Process_Kinect::Nui_DrawBone(cv::Mat &mtxRGBCanvas, const KinectBody& body, JointType joint0, JointType joint1)
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
	cv::Point pt0 = Nui_ProjectDepthToColorPixel(body.Joints3D[joint0]);
	cv::Point pt1 = Nui_ProjectDepthToColorPixel(body.Joints3D[joint1]);
	if (pt0.x != -std::numeric_limits<int>::infinity() && pt0.y != -std::numeric_limits<int>::infinity()
		&& pt1.x != -std::numeric_limits<int>::infinity() && pt1.y != -std::numeric_limits<int>::infinity())
	{
		cv::Point ptRGB0(1.0 * pt0.x / m_sOrgColor.width * mtxRGBCanvas.cols, 1.0 * pt0.y / m_sOrgColor.height * mtxRGBCanvas.rows);
		cv::Point ptRGB1(1.0 * pt1.x / m_sOrgColor.width * mtxRGBCanvas.cols, 1.0 * pt1.y / m_sOrgColor.height * mtxRGBCanvas.rows);
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

cv::Point Process_Kinect::Nui_ProjectDepthToColorPixel(cv::Vec3f v)
{
	CameraSpacePoint v0;
	ColorSpacePoint pt0;
	v0.X = v[0];
	v0.Y = v[1];
	v0.Z = v[2];
	m_pCoordinateMapper->MapCameraPointToColorSpace(v0, &pt0);

	if (pt0.X < 0 || pt0.Y < 0 || pt0.X >= 1920 || pt0.Y >= 1080)
		pt0 = pt0;

	if (pt0.X != -std::numeric_limits<float>::infinity() && pt0.Y != -std::numeric_limits<float>::infinity())
		return cv::Point(pt0.X, pt0.Y);
	else
		return cv::Point(-std::numeric_limits<int>::infinity(), -std::numeric_limits<int>::infinity());
}

// calculate 3D point projected on depth image plane
cv::Point Process_Kinect::ProjectToPixel(cv::Vec3f v)
{
	DepthSpacePoint depthPoint = { 0 };
	CameraSpacePoint vCamPoint;
	vCamPoint.X = v[0];
	vCamPoint.Y = v[1];
	vCamPoint.Z = v[2];
	m_pCoordinateMapper->MapCameraPointToDepthSpace(vCamPoint, &depthPoint);

	//	CameraIntrinsics pCamParams;
	//	m_pCoordinateMapper->GetDepthCameraIntrinsics(&pCamParams);

	int nPointX = (depthPoint.X * m_sFrame.width) / m_sOrgDepth.width;
	int nPointY = (depthPoint.Y * m_sFrame.height) / m_sOrgDepth.height;
	return cv::Point(nPointX, nPointY);
}

cv::Vec3f Process_Kinect::BackProjectPixel(cv::Point xp, double z)
{
	DepthSpacePoint depthPoint;
	CameraSpacePoint vCamPoint;
	depthPoint.X = xp.x;
	depthPoint.Y = xp.y;
	m_pCoordinateMapper->MapDepthPointToCameraSpace(depthPoint, z * 1000, &vCamPoint);

	return cv::Vec3f(vCamPoint.X, vCamPoint.Y, vCamPoint.Z);
}


/*
bool Process_Kinect::Nui_FindHandMask(cv::Mat &mtxDepthImg, const KinectBody& mainbody, cv::Mat &mtxMask, bool bActive[2])
{
	// threshold the input depth image	
	mtxMask.setTo(cv::Scalar(255));
	int xmin, xmax, ymin, ymax;
	xmin = ymin = _INT_MAX;
	xmax = ymax = _INT_MIN;
	for (int i = 0; i < mtxDepthImg.rows; i++)
	{
		for (int j = 0; j < mtxDepthImg.cols; j++)
		{
			double fValue = mtxDepthImg.at<float>(i, j);
			if (fValue >= m_fMinDepth && fValue <= m_fMaxDepth)
			{
				xmin = j < xmin ? j : xmin;
				xmax = j > xmax ? j : xmax;
				ymin = i < ymin ? i : ymin;
				ymax = i > ymax ? i : ymax;
				mtxMask.at<int>(i, j) = 0;
			}
		}
	}

	cv::Point2f ptPalmCenter[2] = { mainbody.Joints2D[JointType_HandLeft], mainbody.Joints2D[JointType_HandRight] };
	cv::Vec3f vPalmCenter[2] = { mainbody.Joints3D[JointType_HandLeft], mainbody.Joints3D[JointType_HandRight] };
	cv::Point2f ptArmCenter[2] = { mainbody.Joints2D[JointType_WristLeft], mainbody.Joints2D[JointType_WristRight] };

	// determine the active hands and whether further refinement is needed
	bool bRefine[2] = { false, false };
	double d0x[2], d0y[2];
	for (int i = 0; i < 2; i++)
	{
		//	if (ptPalmCenter[i].y > ptArmCenter[i].y + 10)
		if (ptPalmCenter[i].y > mainbody.Joints2D[JointType_SpineMid].y)
			bActive[i] = false;
		else
		{
			bActive[i] = true;

			// if the distance between the palm center and arm center is too small, no refinement 
			// is needed as the arm region are actually not included in the foreground
			cv::Vec2f vDist(ptArmCenter[i].x - ptPalmCenter[i].x, ptArmCenter[i].y - ptPalmCenter[i].y);
			d0x[i] = ptArmCenter[i].x - ptPalmCenter[i].x;
			d0y[i] = ptArmCenter[i].y - ptPalmCenter[i].y;
			if (sqrt(cv::norm(vDist)) > 1.0)
				bRefine[i] = true;
		}
	}
	if (!bActive[0] && !bActive[1])
		return false;

	// apply the hand joint positions to refine the depth image
	for (int i = ymin; i < ymax + 1; i++)
	{
		for (int j = xmin; j < xmax + 1; j++)
		{
			double fDepth = mtxDepthImg.at<float>(i, j);
			if (fDepth != 0.0)
			{
				double fMinDist = 1e10;
				int nIndex = 0;
				double dcx[2], dcy[2];
				bool bInlier = false;
				for (int k = 0; k < 2; k++)
				{
					if (bActive[k] && fDepth - vPalmCenter[k][2] < 0.04 && fDepth - vPalmCenter[k][2] > -0.09)
					{
						bInlier = true;
						dcx[k] = j - ptPalmCenter[k].x;
						dcy[k] = i - ptPalmCenter[k].y;
						double fDist = sqrtf(dcx[k] * dcx[k] + dcy[k] * dcy[k]);
						if (fDist < fMinDist)
						{
							fMinDist = fDist;
							nIndex = k;
						}
					}
				}
				if (bInlier)
				{
					mtxMask.at<int>(i, j) = nIndex;

					double fThreshold = 30 * 0.95 / vPalmCenter[nIndex][2];
					if (fMinDist > fThreshold * 1.5)
					{
						mtxMask.at<int>(i, j) = 255;
					}*/
					/*		else if (bRefine[nIndex] && d0x[nIndex] * dcx[nIndex] + d0y[nIndex] * dcy[nIndex] > 0)
					{
					if (fMinDist > fThreshold * 1.3)
					{
					mtxMask.at<int>(i, j) = 255;
					}
					}*/
				/*}
				else
				{
					mtxMask.at<int>(i, j) = 255;
				}
			}
		}
	}
	return true;
}*/

//Compute Unnormalized WH for each cell in each direction
std::vector<double> Unnormalized_Cell_Histogram(std::vector<float> Cell_Storage)
{
	//Return Value - Includes both UWH and WH
	// WH  - Indices - 3, 4 - Normalized positive and negative values
	std::vector<double> Hist_Return;
	Hist_Return.clear();
	double Return_Value[2] = { 0, 0 };

	for (int Storage_Index = 0; Storage_Index < Cell_Storage.size(); Storage_Index++)
	{
		if (Cell_Storage[Storage_Index] > 0)
		{
			Return_Value[0] += (Cell_Storage[Storage_Index]);
		}
		else if (Cell_Storage[Storage_Index] < 0)
		{
			Return_Value[1] += (Cell_Storage[Storage_Index]);
		}
	}



	Hist_Return.push_back(Return_Value[0]);
	Hist_Return.push_back(Return_Value[1]);
	return Hist_Return;
}

//Compute UWH and WH for each cell in each direction
std::vector<double> Cell_Direction_Histogram(std::vector<float> Cell_Storage)
{
	//Return Value - Includes both UWH and WH
	// UWH - Indices - 0, 1, 2 - No. of positive, negative and zeros
	// WH  - Indices - 3, 4 - Normalized positive and negative values
	std::vector<double> Hist_Return;
	Hist_Return.clear();
	double Return_Value[5] = { 0, 0, 0, 0, 0 };
	double norm = 0;

	for (int Storage_Index = 0; Storage_Index < Cell_Storage.size(); Storage_Index++)
	{
		if (Cell_Storage[Storage_Index] > 0)
		{
			Return_Value[0]++;
		}
		else if (Cell_Storage[Storage_Index] < 0)
		{
			Return_Value[1]++;
		}
		else if (Cell_Storage[Storage_Index] == 0)
		{
			Return_Value[2]++;
		}
		norm = norm + pow(Cell_Storage[Storage_Index], 2);
	}
	norm = sqrt(norm);


	if (norm != 0)
	{
		for (int Storage_Index = 0; Storage_Index < Cell_Storage.size(); Storage_Index++)
		{
			if (Cell_Storage[Storage_Index] > 0)
			{
				Return_Value[3] += (Cell_Storage[Storage_Index] / norm);
			}
			else if (Cell_Storage[Storage_Index] < 0)
			{
				Return_Value[4] += (Cell_Storage[Storage_Index] / norm);
			}
		}
	}


	if (Cell_Storage.size() != 0)
	{
		//Return Value for UWH
		Return_Value[0] = Return_Value[0] / Cell_Storage.size();
		Return_Value[1] = Return_Value[1] / Cell_Storage.size();
		Return_Value[2] = Return_Value[2] / Cell_Storage.size();
	}

	Hist_Return.push_back(Return_Value[0]);
	Hist_Return.push_back(Return_Value[1]);
	Hist_Return.push_back(Return_Value[2]);
	Hist_Return.push_back(Return_Value[3]);
	Hist_Return.push_back(Return_Value[4]);
	return Hist_Return;
}

//Compute Pose-invariant action histograms using Kinematic motion features in Body centric space
void Process_Kinect::Compute_Action_Histograms()
{
	if (m_NoSkeletons != 0)
	{

		int UWH_Counter = 0;
		int WH_Counter = 0;

		/* Features main index counter*/
		int UB_Counter = 0;
		int UN_Counter = 0;
		int UW_Counter = 0;
		int WB_Counter = 0;
		int WN_Counter = 0;
		int WW_Counter = 0;

		int JointUWH_Counter = 0;
		int JointWH_Counter = 0;
		int JointUNWH_Counter = 0;
		int person_offset = 0;
		/*-----------------------------*/
		int UWHJoint_Counter = 0;
		int WHJoint_Counter = 0;
		int UNWHJoint_Counter = 0;
		std::vector<double> Without_Hist;
		std::vector<double> Neck_Hist;
		std::vector<double> Body_Hist;
		std::vector<double> Joint_Hist;
		std::vector<double> UJoint_Hist;

		//Allocate space for train and test features and labels
		double *testX, *testJointsX;
		//BODY_COUNT
		testX = (double *)malloc(sizeof(double)*m_ndims*m_NoSkeletons);
		testJointsX = (double *)malloc(sizeof(double)*m_Jointndims*m_NoSkeletons);

		//BODY_COUNT
		for (int Person_Index = 0; Person_Index < m_NoSkeletons; ++Person_Index)
		{
			UWH_Counter = 0;
			WH_Counter = 0;
			UWHJoint_Counter = 0;
			WHJoint_Counter = 0;
			UNWHJoint_Counter = 0;

			/* Features main index counter*/
			JointUWH_Counter = 0;
			JointWH_Counter = 0;
			JointUNWH_Counter = 0;

			UB_Counter = 0;
			UN_Counter = 0;
			UW_Counter = 0;
			WB_Counter = 0;
			WN_Counter = 0;
			WW_Counter = 0;
			person_offset = Person_Index * m_ndims;
			/*-----------------------------*/
			for (int Grid_Index = 0; Grid_Index < NO_OF_GRIDS; Grid_Index++)
			{
				for (int Cell_Index = 0; Cell_Index < (2 * NO_OF_CELLS); Cell_Index++)
				{
					for (int Ori_Index = 0; Ori_Index < NO_OF_ORIENTATIONS; Ori_Index++)
					{
						if (Ori_Index == 0)
						{
							//Up Down
							Without_Hist.clear();
							Without_Hist = Cell_Direction_Histogram(Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_UD);

							testX[person_offset + UW_OFFSET + UW_Counter] = Without_Hist[0];
							UW_Counter++;
							testX[person_offset + UW_OFFSET + UW_Counter] = Without_Hist[1];
							UW_Counter++;
							testX[person_offset + UW_OFFSET + UW_Counter] = Without_Hist[2];
							UW_Counter++;

							testX[person_offset + WW_OFFSET + WW_Counter] = Without_Hist[3];
							WW_Counter++;
							testX[person_offset + WW_OFFSET + WW_Counter] = Without_Hist[4];
							WW_Counter++;

							//testX[] = Without_Hist[0];
							Neck_Hist.clear();
							Neck_Hist = Cell_Direction_Histogram(Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD);

							testX[person_offset + UN_OFFSET + UN_Counter] = Neck_Hist[0];
							UN_Counter++;
							testX[person_offset + UN_OFFSET + UN_Counter] = Neck_Hist[1];
							UN_Counter++;
							testX[person_offset + UN_OFFSET + UN_Counter] = Neck_Hist[2];
							UN_Counter++;

							testX[person_offset + WN_OFFSET + WN_Counter] = Neck_Hist[3];
							WN_Counter++;
							testX[person_offset + WN_OFFSET + WN_Counter] = Neck_Hist[4];
							WN_Counter++;

							Body_Hist.clear();
							Body_Hist = Cell_Direction_Histogram(Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD);

							testX[person_offset + UB_Counter] = Body_Hist[0];
							UB_Counter++;
							testX[person_offset + UB_Counter] = Body_Hist[1];
							UB_Counter++;
							testX[person_offset + UB_Counter] = Body_Hist[2];
							UB_Counter++;

							testX[person_offset + WB_OFFSET + WB_Counter] = Body_Hist[3];
							WB_Counter++;
							testX[person_offset + WB_OFFSET + WB_Counter] = Body_Hist[4];
							WB_Counter++;
						}
						else if (Ori_Index == 1)
						{
							//Left Right
							Without_Hist.clear();
							Without_Hist = Cell_Direction_Histogram(Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR);

							testX[person_offset + UW_OFFSET + UW_Counter] = Without_Hist[0];
							UW_Counter++;
							testX[person_offset + UW_OFFSET + UW_Counter] = Without_Hist[1];
							UW_Counter++;
							testX[person_offset + UW_OFFSET + UW_Counter] = Without_Hist[2];
							UW_Counter++;

							testX[person_offset + WW_OFFSET + WW_Counter] = Without_Hist[3];
							WW_Counter++;
							testX[person_offset + WW_OFFSET + WW_Counter] = Without_Hist[4];
							WW_Counter++;

							Neck_Hist.clear();
							Neck_Hist = Cell_Direction_Histogram(Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR);

							testX[person_offset + UN_OFFSET + UN_Counter] = Neck_Hist[0];
							UN_Counter++;
							testX[person_offset + UN_OFFSET + UN_Counter] = Neck_Hist[1];
							UN_Counter++;
							testX[person_offset + UN_OFFSET + UN_Counter] = Neck_Hist[2];
							UN_Counter++;

							testX[person_offset + WN_OFFSET + WN_Counter] = Neck_Hist[3];
							WN_Counter++;
							testX[person_offset + WN_OFFSET + WN_Counter] = Neck_Hist[4];
							WN_Counter++;

							Body_Hist.clear();
							Body_Hist = Cell_Direction_Histogram(Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR);

							testX[person_offset + UB_Counter] = Body_Hist[0];
							UB_Counter++;
							testX[person_offset + UB_Counter] = Body_Hist[1];
							UB_Counter++;
							testX[person_offset + UB_Counter] = Body_Hist[2];
							UB_Counter++;

							testX[person_offset + WB_OFFSET + WB_Counter] = Body_Hist[3];
							WB_Counter++;
							testX[person_offset + WB_OFFSET + WB_Counter] = Body_Hist[4];
							WB_Counter++;
						}
						else if (Ori_Index == 2)
						{
							//Forward Backward
							Without_Hist.clear();
							Without_Hist = Cell_Direction_Histogram(Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB);

							testX[person_offset + UW_OFFSET + UW_Counter] = Without_Hist[0];
							UW_Counter++;
							testX[person_offset + UW_OFFSET + UW_Counter] = Without_Hist[1];
							UW_Counter++;
							testX[person_offset + UW_OFFSET + UW_Counter] = Without_Hist[2];
							UW_Counter++;

							testX[person_offset + WW_OFFSET + WW_Counter] = Without_Hist[3];
							WW_Counter++;
							testX[person_offset + WW_OFFSET + WW_Counter] = Without_Hist[4];
							WW_Counter++;

							Neck_Hist.clear();
							Neck_Hist = Cell_Direction_Histogram(Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB);

							testX[person_offset + UN_OFFSET + UN_Counter] = Neck_Hist[0];
							UN_Counter++;
							testX[person_offset + UN_OFFSET + UN_Counter] = Neck_Hist[1];
							UN_Counter++;
							testX[person_offset + UN_OFFSET + UN_Counter] = Neck_Hist[2];
							UN_Counter++;

							testX[person_offset + WN_OFFSET + WN_Counter] = Neck_Hist[3];
							WN_Counter++;
							testX[person_offset + WN_OFFSET + WN_Counter] = Neck_Hist[4];
							WN_Counter++;

							Body_Hist.clear();
							Body_Hist = Cell_Direction_Histogram(Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB);

							testX[person_offset + UB_Counter] = Body_Hist[0];
							UB_Counter++;
							testX[person_offset + UB_Counter] = Body_Hist[1];
							UB_Counter++;
							testX[person_offset + UB_Counter] = Body_Hist[2];
							UB_Counter++;

							testX[WB_OFFSET + WB_Counter] = Body_Hist[3];
							WB_Counter++;
							testX[WB_OFFSET + WB_Counter] = Body_Hist[4];
							WB_Counter++;
						}

						Action_Features[Person_Index].UWH.UW_Hist[UWH_Counter] = Without_Hist[0];
						Action_Features[Person_Index].UWH.UN_Hist[UWH_Counter] = Neck_Hist[0];
						Action_Features[Person_Index].UWH.UB_Hist[UWH_Counter] = Body_Hist[0];
						UWH_Counter++;

						Action_Features[Person_Index].UWH.UW_Hist[UWH_Counter] = Without_Hist[1];
						Action_Features[Person_Index].UWH.UN_Hist[UWH_Counter] = Neck_Hist[1];
						Action_Features[Person_Index].UWH.UB_Hist[UWH_Counter] = Body_Hist[1];
						UWH_Counter++;

						Action_Features[Person_Index].UWH.UW_Hist[UWH_Counter] = Without_Hist[2];
						Action_Features[Person_Index].UWH.UN_Hist[UWH_Counter] = Neck_Hist[2];
						Action_Features[Person_Index].UWH.UB_Hist[UWH_Counter] = Body_Hist[2];
						UWH_Counter++;

						Action_Features[Person_Index].WH.WW_Hist[WH_Counter] = Without_Hist[3];
						Action_Features[Person_Index].WH.WN_Hist[WH_Counter] = Neck_Hist[3];
						Action_Features[Person_Index].WH.WB_Hist[WH_Counter] = Body_Hist[3];
						WH_Counter++;

						Action_Features[Person_Index].WH.WW_Hist[WH_Counter] = Without_Hist[4];
						Action_Features[Person_Index].WH.WN_Hist[WH_Counter] = Neck_Hist[4];
						Action_Features[Person_Index].WH.WB_Hist[WH_Counter] = Body_Hist[4];
						WH_Counter++;

					}
					//end of Ori_Index
				}
				//end of Cell_Index
			}
			//end of Grid_Index

			person_offset = Person_Index * m_Jointndims;
			for (int joint_iter = 0; joint_iter < NO_OF_JOINTS; joint_iter++)
			{
				for (int Ori_Index = 0; Ori_Index < NO_OF_ORIENTATIONS; Ori_Index++)
				{
					if (Ori_Index == 0)
					{
						//Up Down
						Joint_Hist.clear();
						Joint_Hist = Cell_Direction_Histogram(Person[Person_Index].Joint_Neck_Ref[joint_iter].Joint_UD);

						//Unnormalized version
						UJoint_Hist.clear();
						UJoint_Hist = Unnormalized_Cell_Histogram(Person[Person_Index].Joint_Neck_Ref[joint_iter].Joint_UD);

						testJointsX[person_offset + JointUWH_Counter] = Joint_Hist[0];
						JointUWH_Counter++;
						testJointsX[person_offset + JointUWH_Counter] = Joint_Hist[1];
						JointUWH_Counter++;
						testJointsX[person_offset + JointUWH_Counter] = Joint_Hist[2];
						JointUWH_Counter++;

						testJointsX[person_offset + WJOINT_OFFSET + JointWH_Counter] = Joint_Hist[3];
						JointWH_Counter++;
						testJointsX[person_offset + WJOINT_OFFSET + JointWH_Counter] = Joint_Hist[4];
						JointWH_Counter++;

						//Unnormalized version
						testJointsX[person_offset + UNWJOINT_OFFSET + JointUNWH_Counter] = UJoint_Hist[0];
						JointUNWH_Counter++;
						testJointsX[person_offset + UNWJOINT_OFFSET + JointUNWH_Counter] = UJoint_Hist[1];
						JointUNWH_Counter++;
					}
					else if (Ori_Index == 1)
					{
						//Left Right
						Joint_Hist.clear();
						Joint_Hist = Cell_Direction_Histogram(Person[Person_Index].Joint_Neck_Ref[joint_iter].Joint_LR);

						//Unnormalized version
						UJoint_Hist.clear();
						UJoint_Hist = Unnormalized_Cell_Histogram(Person[Person_Index].Joint_Neck_Ref[joint_iter].Joint_LR);

						testJointsX[person_offset + JointUWH_Counter] = Joint_Hist[0];
						JointUWH_Counter++;
						testJointsX[person_offset + JointUWH_Counter] = Joint_Hist[1];
						JointUWH_Counter++;
						testJointsX[person_offset + JointUWH_Counter] = Joint_Hist[2];
						JointUWH_Counter++;

						testJointsX[person_offset + WJOINT_OFFSET + JointWH_Counter] = Joint_Hist[3];
						JointWH_Counter++;
						testJointsX[person_offset + WJOINT_OFFSET + JointWH_Counter] = Joint_Hist[4];
						JointWH_Counter++;

						//Unnormalized version
						testJointsX[person_offset + UNWJOINT_OFFSET + JointUNWH_Counter] = UJoint_Hist[0];
						JointUNWH_Counter++;
						testJointsX[person_offset + UNWJOINT_OFFSET + JointUNWH_Counter] = UJoint_Hist[1];
						JointUNWH_Counter++;

					}
					else if (Ori_Index == 2)
					{
						//Forward Backward
						Joint_Hist.clear();
						Joint_Hist = Cell_Direction_Histogram(Person[Person_Index].Joint_Neck_Ref[joint_iter].Joint_FB);

						//Unnormalized version
						UJoint_Hist.clear();
						UJoint_Hist = Unnormalized_Cell_Histogram(Person[Person_Index].Joint_Neck_Ref[joint_iter].Joint_FB);

						testJointsX[person_offset + JointUWH_Counter] = Joint_Hist[0];
						JointUWH_Counter++;
						testJointsX[person_offset + JointUWH_Counter] = Joint_Hist[1];
						JointUWH_Counter++;
						testJointsX[person_offset + JointUWH_Counter] = Joint_Hist[2];
						JointUWH_Counter++;

						testJointsX[person_offset + WJOINT_OFFSET + JointWH_Counter] = Joint_Hist[3];
						JointWH_Counter++;
						testJointsX[person_offset + WJOINT_OFFSET + JointWH_Counter] = Joint_Hist[4];
						JointWH_Counter++;

						//Unnormalized version
						testJointsX[person_offset + UNWJOINT_OFFSET + JointUNWH_Counter] = UJoint_Hist[0];
						JointUNWH_Counter++;
						testJointsX[person_offset + UNWJOINT_OFFSET + JointUNWH_Counter] = UJoint_Hist[1];
						JointUNWH_Counter++;
					}
					for (int ch = 0; ch < Joint_Hist.size(); ch++)
					{
						if (ch > 2)
						{
							Action_Features[Person_Index].WH.WJoint_Hist[WHJoint_Counter] = Joint_Hist[ch];
							WHJoint_Counter++;
						}
						else
						{
							Action_Features[Person_Index].UWH.UJoint_Hist[UWHJoint_Counter] = Joint_Hist[ch];
							UWHJoint_Counter++;
						}
					}
					for (int ch = 0; ch < UJoint_Hist.size(); ch++)
					{
						Action_Features[Person_Index].WH.UNWJoint_Hist[UNWHJoint_Counter] = UJoint_Hist[ch];
						UNWHJoint_Counter++;
						//UJoint_Hist[ch] << "\n";
					}
				}//end of Ori_Index
			}// end of joint_iter

		}
		//end of Person_Index
		//-------------------------------------------------------------------
		MatrixXd mScores, mStickScores;
		int code;
		//BODY_COUNT
		// launch prediction - Foreground points
		code = elmPredict(testX, m_ndims, m_NoSkeletons,
			mScores,
			inW, bias, outW);

		if (code != 0)
			printf("Failed to predict class scores.\n");
		//-------------------------------------------------------------------
		float max_Score = 0;
		int act_class = 0;
		for (int score_col = 0; score_col < mScores.cols(); score_col++)
		{
			for (int score_row = 0; score_row < mScores.rows(); score_row++)
			{
				if (score_row == 0)
				{
					max_Score = mScores(score_row, score_col);
					act_class = score_row + 1;
				}
				else
				{
					if (max_Score < mScores(score_row, score_col))
					{
						max_Score = mScores(score_row, score_col);
						act_class = score_row + 1;
					}
				}
				//printf("%f ", mScores(score_row, score_col));
			}
			//printf("Chk: %d", act_class);
			//printf("\n");
		}
		//-------------------------------------------------------------------
		//BODY_COUNT
		// launch prediction - Stick pose
		code = elmPredict(testJointsX, m_Jointndims, m_NoSkeletons,
			mStickScores,
			Stick_inW, Stick_bias, Stick_outW);

		if (code != 0)
			printf("Failed to predict class scores.\n");

		//-------------------------------------------------------------------
		max_Score = 0;
		act_class = 0;
		for (int score_col = 0; score_col < mStickScores.cols(); score_col++)
		{
			for (int score_row = 0; score_row < mStickScores.rows(); score_row++)
			{
				if (score_row == 0)
				{
					max_Score = mStickScores(score_row, score_col);
					act_class = score_row + 1;
				}
				else
				{
					if (max_Score < mStickScores(score_row, score_col))
					{
						max_Score = mStickScores(score_row, score_col);
						act_class = score_row + 1;
					}
				}
				//printf("%f ", mStickScores(score_row, score_col));
			}
			//printf("Chk: %d", act_class);
			//printf("\n");
		}
		//-------------------------------------------------------------------
		User_actions.clear();
		std::vector<double> score_vector;
		std::vector<double> sort_vector;
		std::vector<double> Final_score;

		std::vector<double> Temp_vector;
		//std::vector<double> Final_action;

		std::vector<double> Temp_vector_score;
		std::vector<imi::ActionType::type> Final_action;
		std::vector<imi::ActionType::type> Final_action_score;
		vector<double>::iterator it;
		int FGClassifier_BC[NO_OF_ACTIONS] = { 0, 0, 0 }; // , 0, 0, 0, 0, 0, 0};
		int StickClassifier_BC[NO_OF_ACTIONS] = { 0, 0, 0 }; // , 0, 0, 0, 0, 0, 0};
		for (int score_col = 0; score_col < mScores.cols(); score_col++)
		{
			printf("Sample: %d\n", score_col);
			score_vector.clear();
			sort_vector.clear();
			Final_score.clear();
			//Push into the score and sort vectors
			for (int score_row = 0; score_row < mScores.rows(); score_row++)
			{
				score_vector.push_back(mScores(score_row, score_col));
				sort_vector.push_back(mScores(score_row, score_col));
			}
			// sort according to scores
			std::sort(sort_vector.begin(), sort_vector.end());

			//Compute Borda count
			for (int vec_iter = sort_vector.size(); vec_iter > 0; vec_iter--)
			{
				it = find(score_vector.begin(), score_vector.end(), sort_vector[vec_iter - 1]);
				if (it != score_vector.end())
				{
					//std::cout << (it - score_vector.begin() + 1) << " ";
				}

				FGClassifier_BC[it - score_vector.begin()] = vec_iter - 1;
			}

			// Normalize score range from 0 to 1 and then multiply with Borda count to obtain final score
			//sort_vector(0)  min
			//sort_vector(sort_vector.size() - 1) - sort_vector(0)  max
			for (int vec_iter = 0; vec_iter < score_vector.size(); vec_iter++)
			{
				double push = (score_vector[vec_iter] - sort_vector[0]) / (sort_vector[sort_vector.size() - 1] - sort_vector[0]);
				//Final_score.push_back(score_vector[vec_iter] * FGClassifier_BC[vec_iter]);
				Final_score.push_back(push * FGClassifier_BC[vec_iter]);
			}
			printf("\n");
			score_vector.clear();
			sort_vector.clear();
			//Push into the score and sort vectors
			for (int score_row = 0; score_row < mStickScores.rows(); score_row++)
			{
				score_vector.push_back(mStickScores(score_row, score_col));
				sort_vector.push_back(mStickScores(score_row, score_col));
			}
			// sort according to scores
			std::sort(sort_vector.begin(), sort_vector.end());

			//Compute Borda count
			for (int vec_iter = sort_vector.size(); vec_iter > 0; vec_iter--)
			{
				it = find(score_vector.begin(), score_vector.end(), sort_vector[vec_iter - 1]);
				if (it != score_vector.end())
				{
					//std::cout << (it - score_vector.begin() + 1) << " ";
				}

				StickClassifier_BC[it - score_vector.begin()] = vec_iter - 1;
			}
			printf("\n");

			// Normalize score range from 0 to 1 and then multiply with Borda count to obtain final score
			//sort_vector(0)  min
			//sort_vector(sort_vector.size() - 1) - sort_vector(0)  max
			for (int vec_iter = 0; vec_iter < score_vector.size(); vec_iter++)
			{
				double push = (score_vector[vec_iter] - sort_vector[0]) / (sort_vector[sort_vector.size() - 1] - sort_vector[0]);
				//Final_score[vec_iter] = Final_score[vec_iter] + (score_vector[vec_iter] * StickClassifier_BC[vec_iter]);
				Final_score[vec_iter] = (0.5 * Final_score[vec_iter]) + (push * StickClassifier_BC[vec_iter]);
				//printf("%f ", Final_score[vec_iter]);
				Temp_vector_score.push_back(Final_score[vec_iter]);
			}
			//printf("\n");

			printf("Action Based on Weighted Scores: ");
			// sort according to weighted scores
			std::sort(Temp_vector_score.begin(), Temp_vector_score.end(), std::greater<double>());
			auto last_score = std::unique(Temp_vector_score.begin(), Temp_vector_score.end());
			Temp_vector_score.erase(last_score, Temp_vector_score.end());
			for (int temp_iter = 0; temp_iter < Temp_vector_score.size(); temp_iter++)
			{
				for (int BC_iter = 0; BC_iter < NO_OF_ACTIONS; BC_iter++)
				{
					if (Final_score[BC_iter] == Temp_vector_score[temp_iter])
					{
						if (temp_iter == 0)
						{
							Final_action_score.push_back(Map_inttoActionType_small(BC_iter + 1));
						}
						printf("%d ", BC_iter + 1);
					}
				}
			}
			printf("\n");
			
			/*
			float Final_BC[NO_OF_ACTIONS] = { 0, 0, 0 };// , 0, 0, 0, 0, 0, 0};
			int max_BC = 0;
			for (int BC_iter = 0; BC_iter < NO_OF_ACTIONS; BC_iter++)
			{
				//printf("%d %d %d %d\n", (BC_iter + 1), FGClassifier_BC[BC_iter], StickClassifier_BC[BC_iter], (FGClassifier_BC[BC_iter] + StickClassifier_BC[BC_iter]));
				Final_BC[BC_iter] = (FGClassifier_BC[BC_iter] + StickClassifier_BC[BC_iter]);
				if (BC_iter == 0)
				{
					max_BC = Final_BC[BC_iter];
				}
				else
				{
					if (max_BC < Final_BC[BC_iter])
					{
						max_BC = Final_BC[BC_iter];
					}
				}
			}

			Temp_vector.clear();
			Final_action.clear();

			
			for (int BC_iter = 0; BC_iter < NO_OF_ACTIONS; BC_iter++)
			{
				Final_BC[BC_iter] = Final_BC[BC_iter] / max_BC;
				if (Final_BC[BC_iter] >= m_recognition_threshold)
				{
					Temp_vector.push_back(Final_BC[BC_iter]);
				}
			}
			printf("Actions detected: ");
			// sort according to scores
			std::sort(Temp_vector.begin(), Temp_vector.end(), std::greater<double>());
			auto last = std::unique(Temp_vector.begin(), Temp_vector.end());
			Temp_vector.erase(last, Temp_vector.end());
			for (int temp_iter = 0; temp_iter < Temp_vector.size(); temp_iter++)
			{
				for (int BC_iter = 0; BC_iter < NO_OF_ACTIONS; BC_iter++)
				{
					if (Final_BC[BC_iter] == Temp_vector[temp_iter])
					{
						printf("%d ", BC_iter + 1);
						Final_action.push_back(Map_inttoActionType_type(BC_iter + 1));
					}
				}
			}
			*/
			printf("\n");
			imi::UserAction U;
			U.__set_actions(Final_action_score);
			User_actions.push_back(U);
			Temp_vector.clear();
			Final_action.clear();
			Temp_vector_score.clear();
			Final_action_score.clear();
		}
		
		if (User_actions.size() != 0)
		{
			Action_Detected = true;
			//Send only when thrift is used to send action recognition result.
			if (isUseThrift)
			{
				//Ensure connection to server before sending the result
				if (ActionManager->isConnected())
				{
					if (ActionManager->ensureConnection())
					{
						printf("Actions sent to Reactive Layer 2!!!\n");
						//User_actions has to be sent here
						ActionManager->getClient()->send_queryActionRecogition(User_actions);
					}
				}
			}
		}
		
		free(testX);
		free(testJointsX);
	}
}
//Convert int to imi::ActionType - for smaller set of actions - hand shake, none, walking
imi::ActionType::type Process_Kinect::Map_inttoActionType_small(int Recognized_Action)
{
	switch (Recognized_Action) {

	case 1: return(imi::ActionType::type::OTHER); // Action: Other or none

	case 2: return(imi::ActionType::type::HAND_SHAKE); // Action: Hand Shake

	case 3: return(imi::ActionType::type::WALKING); // Action: Walking

	default: return(imi::ActionType::type::OTHER); // Action: Others
	}
}

//Convert int to imi::ActionType
imi::ActionType::type Process_Kinect::Map_inttoActionType_type(int Recognized_Action)
{
	switch (Recognized_Action) {

	case 1: return(imi::ActionType::type::CHECK_WATCH); // Action: Check Watch

	case 2: return(imi::ActionType::type::DRINKING); // Action: Drink

	case 3: return(imi::ActionType::type::EATING); // Action: Eat

	case 4: return(imi::ActionType::type::GIVING); // Action: Give

	case 5: return(imi::ActionType::type::HAND_SHAKE); // Action: Hand Shake

	case 6: return(imi::ActionType::type::TAKE_PHOTO); // Action: Take Photo

	case 7: return(imi::ActionType::type::WALKING); // Action: Walk

	case 8: return(imi::ActionType::type::ANSWER_PHONE); // Action: Answer Phone

	case 9: return(imi::ActionType::type::OTHER); // Action: Others

	default: return(imi::ActionType::type::OTHER); // Action: Others
	}
}
//Converts Kinematic motion features to Body centric space
void Process_Kinect::ConvertWorldMotionToBodySpace()
{
	int Cell_Numbering[4][4][6] = { { { 1, 2, 3, 4, 5, 6 }, { 2, 1, 4, 3, 6, 5 }, { 2, 1, 4, 3, 6, 5 }, { 1, 2, 3, 4, 5, 6 } },
	{ { 6, 5, 4, 3, 2, 1 }, { 5, 6, 3, 4, 1, 2 }, { 5, 6, 3, 4, 1, 2 }, { 6, 5, 4, 3, 2, 1 } },
	{ { 5, 3, 1, 6, 4, 2 }, { 6, 4, 2, 5, 3, 1 }, { 6, 4, 2, 5, 3, 1 }, { 5, 3, 1, 6, 4, 2 } },
	{ { 2, 4, 6, 1, 3, 5 }, { 1, 3, 5, 2, 4, 6 }, { 1, 3, 5, 2, 4, 6 }, { 2, 4, 6, 1, 3, 5 } }
	};
	int To_Check, Grid1_Start, Grid1_End, Grid2_Start, Grid2_End, Grid3_Start, Grid3_End, Reference_point;
	int Start, End, Cell_Number_iterator, Check_Cell_iterator;
	int Grid_Index = 0;
	int Cell_Index = 0;
	int Person_Index, Orientation_State, View;
	float Push_To_Storage, x_component, y_component;
	double pixel_Orientation;
	
	// For each foreground point, convert the motion feature to human body centric space
	for (int fg_index = 0; fg_index < Foreground_points.size(); fg_index++)
	{
		Grid_Index = 0;
		Cell_Index = 0;

		//Person_Index = Foreground_points[fg_index][2];
		Person_Index = Foreground_points[fg_index][4];
		
		/*for (int body_iter = 0; body_iter < BODY_COUNT; ++body_iter)
		{
			if (Skeleton_TrackingID[Foreground_points[fg_index][4]] == m_vecKinectBodies[body_iter].BodyIndex)
			{
				Person_Index = body_iter;
			}
		}*/
		
		Orientation_State = m_vecKinectBodies[Person_Index].BodyOrientation;
		View = m_vecKinectBodies[Person_Index].BodyView;

		if ((Foreground_points[fg_index][1] >= m_vecKinectBodies[Person_Index].Foreground_x_lim[0]) && (Foreground_points[fg_index][1] <= m_vecKinectBodies[Person_Index].Foreground_x_lim[1]) && (Foreground_points[fg_index][0] >= m_vecKinectBodies[Person_Index].Foreground_y_lim[0]) && (Foreground_points[fg_index][0] <= m_vecKinectBodies[Person_Index].Foreground_y_lim[1]))
		{
			if (m_vecKinectBodies[Person_Index].neck_valid)
			{
				//Identify Grid to which Fg_pt belongs to 
				//---------------------------------------------------------------------
				if (Orientation_State <= 2)
				{
					To_Check = Foreground_points[fg_index][0];
					if (Orientation_State == 1)
					{
						Grid1_Start = m_vecKinectBodies[Person_Index].Foreground_y_lim[0];
						Grid1_End = (m_vecKinectBodies[Person_Index].Joints2D[JointType_Neck].y);

						Grid2_Start = (m_vecKinectBodies[Person_Index].Joints2D[JointType_Neck].y) + 1;
						Grid2_End = (m_vecKinectBodies[Person_Index].Joints2D[JointType_SpineBase].y);

						Grid3_Start = (m_vecKinectBodies[Person_Index].Joints2D[JointType_SpineBase].y) + 1;
						Grid3_End = m_vecKinectBodies[Person_Index].Foreground_y_lim[1];
					}
					else
					{
						Grid1_Start = (m_vecKinectBodies[Person_Index].Joints2D[JointType_Neck].y);
						Grid1_End = m_vecKinectBodies[Person_Index].Foreground_y_lim[1];

						Grid2_Start = (m_vecKinectBodies[Person_Index].Joints2D[JointType_SpineBase].y);
						Grid2_End = (m_vecKinectBodies[Person_Index].Joints2D[JointType_Neck].y) - 1;

						Grid3_Start = m_vecKinectBodies[Person_Index].Foreground_y_lim[0];
						Grid3_End = (m_vecKinectBodies[Person_Index].Joints2D[JointType_SpineBase].y) - 1;
					}
				}
				else
				{
					To_Check = Foreground_points[fg_index][1];
					if (Orientation_State == 4)
					{
						Grid1_Start = m_vecKinectBodies[Person_Index].Foreground_x_lim[0];
						Grid1_End = (m_vecKinectBodies[Person_Index].Joints2D[JointType_Neck].x);

						Grid2_Start = (m_vecKinectBodies[Person_Index].Joints2D[JointType_Neck].x) + 1;
						Grid2_End = (m_vecKinectBodies[Person_Index].Joints2D[JointType_SpineBase].x);

						Grid3_Start = (m_vecKinectBodies[Person_Index].Joints2D[JointType_SpineBase].x) + 1;
						Grid3_End = m_vecKinectBodies[Person_Index].Foreground_x_lim[1];
					}
					else
					{
						Grid1_Start = (m_vecKinectBodies[Person_Index].Joints2D[JointType_Neck].x);
						Grid1_End = m_vecKinectBodies[Person_Index].Foreground_x_lim[1];

						Grid2_Start = (m_vecKinectBodies[Person_Index].Joints2D[JointType_SpineBase].x);
						Grid2_End = (m_vecKinectBodies[Person_Index].Joints2D[JointType_Neck].x) - 1;

						Grid3_Start = m_vecKinectBodies[Person_Index].Foreground_x_lim[0];
						Grid3_End = (m_vecKinectBodies[Person_Index].Joints2D[JointType_SpineBase].x) - 1;
					}
				}
				//---------------------------------------------------------------------
				/*printf("Chk3 %d %d \n", Grid1_Start, Grid1_End);
				printf("Chk4 %d %d \n", Grid2_Start, Grid2_End);
				printf("Chk5 %d %d \n", Grid3_Start, Grid3_End);
				printf("Chk6 %d \n", To_Check);*/
				if ((To_Check >= Grid1_Start) & (To_Check <= Grid1_End))
				{
					Grid_Index = 0;
					Start = Grid1_Start;
					End = Grid1_End;
				}
				if ((To_Check >= Grid2_Start) & (To_Check <= Grid2_End))
				{
					Grid_Index = 1;
					Start = Grid2_Start;
					End = Grid2_End;
				}
				if ((To_Check >= Grid3_Start) & (To_Check <= Grid3_End))
				{
					Grid_Index = 2;
					Start = Grid3_Start;
					End = Grid3_End;
				}
				//Identify Cell Index
				//---------------------------------------------------------------------
				/*printf("Chk7: %d\n", Start);*/
				Check_Cell_iterator = ((To_Check - Start) / ceil((End - Start) / NO_OF_CELLS));
				if (Check_Cell_iterator > NO_OF_CELLS)
				{
					Check_Cell_iterator = NO_OF_CELLS;
				}
				if (Check_Cell_iterator == 0)
				{
					Check_Cell_iterator = 1;
				}
				if (Orientation_State <= 2)
				{
					Reference_point = (m_vecKinectBodies[Person_Index].Joints2D[JointType_Neck].x);
					if (Foreground_points[fg_index][1] > Reference_point)
					{
						//iter 2n				
						//Cell_Number_iterator = 2 * ((To_Check - Start) / floor((End - Start) / NO_OF_CELLS));
						Cell_Number_iterator = 2 * Check_Cell_iterator;
					}
					else
					{
						//iter 2n -1
						//Cell_Number_iterator = (2 * (To_Check - Start) / floor((End - Start) / NO_OF_CELLS)) - 1;
						Cell_Number_iterator = (2 * Check_Cell_iterator) - 1;
					}
				}
				else
				{
					Reference_point = (m_vecKinectBodies[Person_Index].Joints2D[JointType_Neck].y);
					if (Foreground_points[fg_index][0] > Reference_point)
					{
						//iter n+ NO_OF_CELLS
						//Cell_Number_iterator = (To_Check - Start) / floor((End - Start) / NO_OF_CELLS) + NO_OF_CELLS;
						Cell_Number_iterator = Check_Cell_iterator + NO_OF_CELLS;
					}
					else
					{
						//iter
						//Cell_Number_iterator = (To_Check - Start) / floor((End - Start) / NO_OF_CELLS);
						Cell_Number_iterator = Check_Cell_iterator;
					}
				}
				Cell_Index = Cell_Numbering[Orientation_State - 1][View - 1][Cell_Number_iterator - 1];
				Cell_Index = Cell_Index - 1;
				//---------------------------------------------------------------------
				x_component = SFlow.Vx.at<float>(Foreground_points[fg_index][0], Foreground_points[fg_index][1]);
				y_component = SFlow.Vy.at<float>(Foreground_points[fg_index][0], Foreground_points[fg_index][1]);
				pixel_Orientation = atan2(double(y_component), double(x_component)) * 180 / PI;

				Up_Down_Cell(pixel_Orientation, Person_Index, Grid_Index, Cell_Index, View, Orientation_State, Foreground_points[fg_index][0], Foreground_points[fg_index][1]);
				Left_Right_Cell(pixel_Orientation, Person_Index, Grid_Index, Cell_Index, View, Orientation_State, Foreground_points[fg_index][0], Foreground_points[fg_index][1]);
				Gaze_Cell(pixel_Orientation, Person_Index, Grid_Index, Cell_Index, View, Orientation_State, Foreground_points[fg_index][0], Foreground_points[fg_index][1]);
			}
		}
	}
}

//Up Down cell - Identifies Up and Down motion of the pixel based on orientation, view and populates the proper motion feature storage 
void Process_Kinect::Up_Down_Cell(double pixel_Orientation, int Person_Index, int Grid_Index, int Cell_Index, int View, int Orientation_State, int row, int col)
{

	if (Orientation_State == 1)
	{
		if ((pixel_Orientation >= -135) & (pixel_Orientation <= -45))
		{
			//Up - Push abs Push_To_Storage
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(abs(m_Divergence.at<float>(row, col)));
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(abs(m_Projection.at<float>(row, col)));
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(abs(m_BodyPartProjection.at<float>(row, col)));
		}
		else if ((pixel_Orientation >= 45) & (pixel_Orientation <= 135))
		{
			//Down - Push -1 * abs Push_To_Storage
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(-1 * abs(m_Projection.at<float>(row, col)));
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
		}
		Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(m_Vorticity.at<Vec3f>(row, col)[1]);
		Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(m_Rotation.at<Vec3f>(row, col)[1]);
		Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[1]);
	}
	else if (Orientation_State == 2)
	{
		if ((pixel_Orientation >= -135) & (pixel_Orientation <= -45))
		{
			//Down - Push -1 * abs Push_To_Storage
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(-1 * abs(m_Projection.at<float>(row, col)));
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
		}
		else if ((pixel_Orientation >= 45) & (pixel_Orientation <= 135))
		{
			//Up - Push abs Push_To_Storage
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(abs(m_Divergence.at<float>(row, col)));
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(abs(m_Projection.at<float>(row, col)));
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(abs(m_BodyPartProjection.at<float>(row, col)));
		}
		Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(m_Vorticity.at<Vec3f>(row, col)[1]);
		Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(m_Rotation.at<Vec3f>(row, col)[1]);
		Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[1]);
	}
	else if (Orientation_State == 3)
	{
		if ((pixel_Orientation > -45) & (pixel_Orientation < 45))
		{
			//Down - Push -1 * abs Push_To_Storage
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(-1 * abs(m_Projection.at<float>(row, col)));
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
		}
		else if ((pixel_Orientation >= -180) & (pixel_Orientation < -135))
		{
			//Up - Push abs Push_To_Storage
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(abs(m_Divergence.at<float>(row, col)));
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(abs(m_Projection.at<float>(row, col)));
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(abs(m_BodyPartProjection.at<float>(row, col)));
		}
		else if ((pixel_Orientation > 135) & (pixel_Orientation <= 180))
		{
			//Up - Push abs Push_To_Storage
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(abs(m_Divergence.at<float>(row, col)));
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(abs(m_Projection.at<float>(row, col)));
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(abs(m_BodyPartProjection.at<float>(row, col)));
		}
		Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(m_Vorticity.at<Vec3f>(row, col)[0]);
		Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(m_Rotation.at<Vec3f>(row, col)[0]);
		Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[0]);
	}
	else if (Orientation_State == 4)
	{
		if ((pixel_Orientation > -45) & (pixel_Orientation < 45))
		{
			//Up - Push abs Push_To_Storage
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(abs(m_Divergence.at<float>(row, col)));
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(abs(m_Projection.at<float>(row, col)));
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(abs(m_BodyPartProjection.at<float>(row, col)));
		}
		else if ((pixel_Orientation >= -180) & (pixel_Orientation < -135))
		{
			//Down - Push -1 * abs Push_To_Storage
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(-1 * abs(m_Projection.at<float>(row, col)));
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
		}
		else if ((pixel_Orientation > 135) & (pixel_Orientation <= 180))
		{
			//Down - Push -1 * abs Push_To_Storage
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(-1 * abs(m_Projection.at<float>(row, col)));
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
		}
		Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(m_Vorticity.at<Vec3f>(row, col)[0]);
		Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(m_Rotation.at<Vec3f>(row, col)[0]);
		Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_UD.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[0]);
	}
}
//Left Right Cell - Identifies Left and Right motion of the pixel based on orientation, view and populates the proper motion feature storage 
void Process_Kinect::Left_Right_Cell(double pixel_Orientation, int Person_Index, int Grid_Index, int Cell_Index, int View, int Orientation_State, int row, int col)
{
	if (Orientation_State == 1)
	{
		if (View == 1)
		{
			if ((pixel_Orientation > -45) & (pixel_Orientation < 45))
			{
				//Left - Push -1 * abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back( abs(m_BodyPartProjection.at<float>(row, col)));
			}
			else if ((pixel_Orientation >= -180) & (pixel_Orientation < -135))
			{
				//Right - Push abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
			}
			else if ((pixel_Orientation > 135) & (pixel_Orientation <= 180))
			{
				//Right - Push abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
			}
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Vorticity.at<Vec3f>(row, col)[0]);
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Rotation.at<Vec3f>(row, col)[0]);
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[0]);
		}
		else if (View == 2)
		{
			if ((pixel_Orientation > -45) & (pixel_Orientation < 45))
			{
				//Right - Push -1 * abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
			}
			else if ((pixel_Orientation >= -180) & (pixel_Orientation < -135))
			{
				//Left - Push abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_BodyPartProjection.at<float>(row, col)));
			}
			else if ((pixel_Orientation > 135) & (pixel_Orientation <= 180))
			{
				//Left - Push abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_BodyPartProjection.at<float>(row, col)));
			}
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Vorticity.at<Vec3f>(row, col)[0]);
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Rotation.at<Vec3f>(row, col)[0]);
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[0]);
		}
		else if (View >= 3)
		{
			//Only motion in z- direction
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Vorticity.at<Vec3f>(row, col)[2]);
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Rotation.at<Vec3f>(row, col)[2]);
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[2]);
		}
	}
	else if (Orientation_State == 2)
	{
		if (View == 1)
		{
			if ((pixel_Orientation > -45) & (pixel_Orientation < 45))
			{
				//Left - Push -1 * abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
			}
			else if ((pixel_Orientation >= -180) & (pixel_Orientation < -135))
			{
				//Right - Push abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_BodyPartProjection.at<float>(row, col)));
			}
			else if ((pixel_Orientation > 135) & (pixel_Orientation <= 180))
			{
				//Right - Push abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_BodyPartProjection.at<float>(row, col)));
			}
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Vorticity.at<Vec3f>(row, col)[0]);
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Rotation.at<Vec3f>(row, col)[0]);
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[0]);
		}
		else if (View == 2)
		{
			if ((pixel_Orientation > -45) & (pixel_Orientation < 45))
			{
				//Right - Push -1 * abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_BodyPartProjection.at<float>(row, col)));
			}
			else if ((pixel_Orientation >= -180) & (pixel_Orientation < -135))
			{
				//Left - Push abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
			}
			else if ((pixel_Orientation > 135) & (pixel_Orientation <= 180))
			{
				//Left - Push abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
			}
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Vorticity.at<Vec3f>(row, col)[0]);
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Rotation.at<Vec3f>(row, col)[0]);
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[0]);
		}
		else if (View >= 3)
		{
			//Only motion in z- direction
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Vorticity.at<Vec3f>(row, col)[2]);
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Rotation.at<Vec3f>(row, col)[2]);
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[2]);
		}
	}
	else if (Orientation_State == 3)
	{
		if (View == 1)
		{
			if ((pixel_Orientation >= -135) & (pixel_Orientation <= -45))
			{
				//Left - Push abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_BodyPartProjection.at<float>(row, col)));
			}
			else if ((pixel_Orientation >= 45) & (pixel_Orientation <= 135))
			{
				//Right - Push -1 * abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
			}
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Vorticity.at<Vec3f>(row, col)[1]);
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Rotation.at<Vec3f>(row, col)[1]);
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[1]);
		}
		else if (View == 2)
		{
			if ((pixel_Orientation >= -135) & (pixel_Orientation <= -45))
			{
				//Right - Push -1 * abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
			}
			else if ((pixel_Orientation >= 45) & (pixel_Orientation <= 135))
			{
				//Left - Push abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_BodyPartProjection.at<float>(row, col)));
			}
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Vorticity.at<Vec3f>(row, col)[1]);
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Rotation.at<Vec3f>(row, col)[1]);
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[1]);
		}
		else if (View >= 3)
		{
			//Only motion in z- direction
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Vorticity.at<Vec3f>(row, col)[2]);
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Rotation.at<Vec3f>(row, col)[2]);
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[2]);
		}
	}
	else if (Orientation_State == 4)
	{
		if (View == 1)
		{
			if ((pixel_Orientation >= -135) & (pixel_Orientation <= -45))
			{
				//Right - Push -1 * abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
			}
			else if ((pixel_Orientation >= 45) & (pixel_Orientation <= 135))
			{
				//Left - Push abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_BodyPartProjection.at<float>(row, col)));
			}
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Vorticity.at<Vec3f>(row, col)[1]);
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Rotation.at<Vec3f>(row, col)[1]);
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[1]);
		}
		else if (View == 2)
		{
			if ((pixel_Orientation >= -135) & (pixel_Orientation <= -45))
			{
				//Left - Push abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(abs(m_BodyPartProjection.at<float>(row, col)));
			}
			else if ((pixel_Orientation >= 45) & (pixel_Orientation <= 135))
			{
				//Right - Push -1 * abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
			}
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Vorticity.at<Vec3f>(row, col)[1]);
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Rotation.at<Vec3f>(row, col)[1]);
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[1]);
		}
		else if (View >= 3)
		{
			//Only motion in z- direction
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Vorticity.at<Vec3f>(row, col)[2]);
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_Rotation.at<Vec3f>(row, col)[2]);
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_LR.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[2]);
		}
	}
}

//Gaze Cell - Identifies Forward and Backward motion of the pixel based on orientation, view and populates the proper motion feature storage 
void Process_Kinect::Gaze_Cell(double pixel_Orientation, int Person_Index, int Grid_Index, int Cell_Index, int View, int Orientation_State, int row, int col)
{
	if (Orientation_State <= 2)
	{
		if (View <= 2)
		{
			//Only motion in z- direction
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_Vorticity.at<Vec3f>(row, col)[2]);
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_Rotation.at<Vec3f>(row, col)[2]);
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[2]);
		}
		else if (View == 3)
		{
			if ((pixel_Orientation > -45) & (pixel_Orientation < 45))
			{
				//Back - Push -1 * abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(-1 * abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
			}
			else if ((pixel_Orientation >= -180) & (pixel_Orientation < -135))
			{
				//Forward - Push abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(abs(m_BodyPartProjection.at<float>(row, col)));
			}
			else if ((pixel_Orientation > 135) & (pixel_Orientation <= 180))
			{
				//Forward - Push abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(abs(m_BodyPartProjection.at<float>(row, col)));
			}
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_Vorticity.at<Vec3f>(row, col)[0]);
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_Rotation.at<Vec3f>(row, col)[0]);
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[0]);
		}
		else if (View == 4)
		{
			if ((pixel_Orientation > -45) & (pixel_Orientation < 45))
			{
				//Forward - Push abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(abs(m_BodyPartProjection.at<float>(row, col)));
			}
			else if ((pixel_Orientation >= -180) & (pixel_Orientation < -135))
			{
				//Back - Push -1 *  abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(-1 * abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
			}
			else if ((pixel_Orientation > 135) & (pixel_Orientation <= 180))
			{
				//Back - Push -1 *  abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(-1 * abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
			}
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_Vorticity.at<Vec3f>(row, col)[0]);
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_Rotation.at<Vec3f>(row, col)[0]);
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[0]);
		}
	}
	else if (Orientation_State == 3)
	{
		if (View <= 2)
		{
			//Only motion in z- direction
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_Vorticity.at<Vec3f>(row, col)[2]);
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_Rotation.at<Vec3f>(row, col)[2]);
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[2]);
		}
		else if (View == 3)
		{
			if ((pixel_Orientation >= -135) & (pixel_Orientation <= -45))
			{
				//Back - Push -1 * abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(-1 * abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
			}
			else if ((pixel_Orientation >= 45) & (pixel_Orientation <= 135))
			{
				//Forward - Push abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(abs(m_BodyPartProjection.at<float>(row, col)));
			}
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_Vorticity.at<Vec3f>(row, col)[1]);
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_Rotation.at<Vec3f>(row, col)[1]);
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[1]);
		}
		else if (View == 4)
		{
			if ((pixel_Orientation >= -135) & (pixel_Orientation <= -45))
			{
				//Forward - Push abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(abs(m_BodyPartProjection.at<float>(row, col)));
			}
			else if ((pixel_Orientation >= 45) & (pixel_Orientation <= 135))
			{
				//Back - Push -1 * abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(-1 * abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
			}
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_Vorticity.at<Vec3f>(row, col)[1]);
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_Rotation.at<Vec3f>(row, col)[1]);
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[1]);
		}
	}
	else if (Orientation_State == 4)
	{
		if (View <= 2)
		{
			//Only motion in z- direction
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_Vorticity.at<Vec3f>(row, col)[2]);
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_Rotation.at<Vec3f>(row, col)[2]);
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[2]);
		}
		else if (View == 3)
		{
			if ((pixel_Orientation >= -135) & (pixel_Orientation <= -45))
			{
				//Forward - Push abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(abs(m_BodyPartProjection.at<float>(row, col)));
			}
			else if ((pixel_Orientation >= 45) & (pixel_Orientation <= 135))
			{
				//Back - Push -1 * abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(-1 * abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
			}
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_Vorticity.at<Vec3f>(row, col)[1]);
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_Rotation.at<Vec3f>(row, col)[1]);
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[1]);
		}
		else if (View == 4)
		{
			if ((pixel_Orientation >= -135) & (pixel_Orientation <= -45))
			{
				//Back - Push -1 * abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(-1 * abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(-1 * abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(-1 * abs(m_BodyPartProjection.at<float>(row, col)));
			}
			else if ((pixel_Orientation >= 45) & (pixel_Orientation <= 135))
			{
				//Forward - Push abs Push_To_Storage
				Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(abs(m_Divergence.at<float>(row, col)));
				Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(abs(m_Projection.at<float>(row, col)));
				Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(abs(m_BodyPartProjection.at<float>(row, col)));
			}
			Person[Person_Index].Without[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_Vorticity.at<Vec3f>(row, col)[1]);
			Person[Person_Index].Neck_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_Rotation.at<Vec3f>(row, col)[1]);
			Person[Person_Index].Body_Ref[Grid_Index].Each_Cell[Cell_Index].Cell_FB.push_back(m_BodyPartRotation.at<Vec3f>(row, col)[1]);
		}
	}
}

//Up Down cell - Identifies Up and Down motion of the pixel based on orientation, view and populates the proper motion feature storage 
void Process_Kinect::Up_Down_Joint(double pixel_Orientation, int Person_Index, int Joint_Index, int View, int Orientation_State, float Stick_P, float Stick_Rx, float Stick_Ry, float Stick_Rz)
{
	//printf("UD\n");
	if (Orientation_State == 1)
	{
		if ((pixel_Orientation >= -135) & (pixel_Orientation <= -45))
		{
			//Up - Push abs Push_To_Storage
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_UD.push_back(abs(Stick_P));
		}
		else if ((pixel_Orientation >= 45) & (pixel_Orientation <= 135))
		{
			//Down - Push -1 * abs Push_To_Storage
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_UD.push_back(-1 * abs(Stick_P));
		}
		Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_UD.push_back(Stick_Ry);
	}
	else if (Orientation_State == 2)
	{
		if ((pixel_Orientation >= -135) & (pixel_Orientation <= -45))
		{
			//Down - Push -1 * abs Push_To_Storage
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_UD.push_back(-1 * abs(Stick_P));
		}
		else if ((pixel_Orientation >= 45) & (pixel_Orientation <= 135))
		{
			//Up - Push abs Push_To_Storage
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_UD.push_back(abs(Stick_P));
		}
		Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_UD.push_back(Stick_Ry);
	}
	else if (Orientation_State == 3)
	{
		if ((pixel_Orientation > -45) & (pixel_Orientation < 45))
		{
			//Down - Push -1 * abs Push_To_Storage
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_UD.push_back(-1 * abs(Stick_P));
		}
		else if ((pixel_Orientation >= -180) & (pixel_Orientation < -135))
		{
			//Up - Push abs Push_To_Storage
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_UD.push_back(abs(Stick_P));
		}
		else if ((pixel_Orientation > 135) & (pixel_Orientation <= 180))
		{
			//Up - Push abs Push_To_Storage
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_UD.push_back(abs(Stick_P));
		}
		Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_UD.push_back(Stick_Rx);
	}
	else if (Orientation_State == 4)
	{
		if ((pixel_Orientation > -45) & (pixel_Orientation < 45))
		{
			//Up - Push abs Push_To_Storage
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_UD.push_back(abs(Stick_P));
		}
		else if ((pixel_Orientation >= -180) & (pixel_Orientation < -135))
		{
			//Down - Push -1 * abs Push_To_Storage
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_UD.push_back(-1 * abs(Stick_P));
		}
		else if ((pixel_Orientation > 135) & (pixel_Orientation <= 180))
		{
			//Down - Push -1 * abs Push_To_Storage
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_UD.push_back(-1 * abs(Stick_P));
		}
		Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_UD.push_back(Stick_Rx);
	}
}
//Left Right Cell - Identifies Left and Right motion of the pixel based on orientation, view and populates the proper motion feature storage 
void Process_Kinect::Left_Right_Joint(double pixel_Orientation, int Person_Index, int Joint_Index, int View, int Orientation_State, float Stick_P, float Stick_Rx, float Stick_Ry, float Stick_Rz)
{
	//printf("LR\n");
	if (Orientation_State == 1)
	{
		if (View == 1)
		{
			if ((pixel_Orientation > -45) & (pixel_Orientation < 45))
			{
				//Left - Push -1 * abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(abs(Stick_P));
			}
			else if ((pixel_Orientation >= -180) & (pixel_Orientation < -135))
			{
				//Right - Push abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(-1 * abs(Stick_P));
			}
			else if ((pixel_Orientation > 135) & (pixel_Orientation <= 180))
			{
				//Right - Push abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(-1 * abs(Stick_P));
			}
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(Stick_Rx);
		}
		else if (View == 2)
		{
			if ((pixel_Orientation > -45) & (pixel_Orientation < 45))
			{
				//Right - Push -1 * abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(-1 * abs(Stick_P));
			}
			else if ((pixel_Orientation >= -180) & (pixel_Orientation < -135))
			{
				//Left - Push abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(abs(Stick_P));
			}
			else if ((pixel_Orientation > 135) & (pixel_Orientation <= 180))
			{
				//Left - Push abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(abs(Stick_P));
			}
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(Stick_Rx);
		}
		else if (View >= 3)
		{
			//Only motion in z- direction
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(Stick_Rz);
		}
	}
	else if (Orientation_State == 2)
	{
		if (View == 1)
		{
			if ((pixel_Orientation > -45) & (pixel_Orientation < 45))
			{
				//Left - Push -1 * abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(-1 * abs(Stick_P));
			}
			else if ((pixel_Orientation >= -180) & (pixel_Orientation < -135))
			{
				//Right - Push abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(abs(Stick_P));
			}
			else if ((pixel_Orientation > 135) & (pixel_Orientation <= 180))
			{
				//Right - Push abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(abs(Stick_P));
			}
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(Stick_Rx);
		}
		else if (View == 2)
		{
			if ((pixel_Orientation > -45) & (pixel_Orientation < 45))
			{
				//Right - Push -1 * abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(abs(Stick_P));
			}
			else if ((pixel_Orientation >= -180) & (pixel_Orientation < -135))
			{
				//Left - Push abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(-1 * abs(Stick_P));
			}
			else if ((pixel_Orientation > 135) & (pixel_Orientation <= 180))
			{
				//Left - Push abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(-1 * abs(Stick_P));
			}
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(Stick_Rx);
		}
		else if (View >= 3)
		{
			//Only motion in z- direction
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(Stick_Rz);
		}
	}
	else if (Orientation_State == 3)
	{
		if (View == 1)
		{
			if ((pixel_Orientation >= -135) & (pixel_Orientation <= -45))
			{
				//Left - Push abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(abs(Stick_P));
			}
			else if ((pixel_Orientation >= 45) & (pixel_Orientation <= 135))
			{
				//Right - Push -1 * abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(-1 * abs(Stick_P));
			}
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(Stick_Ry);
		}
		else if (View == 2)
		{
			if ((pixel_Orientation >= -135) & (pixel_Orientation <= -45))
			{
				//Right - Push -1 * abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(-1 * abs(Stick_P));
			}
			else if ((pixel_Orientation >= 45) & (pixel_Orientation <= 135))
			{
				//Left - Push abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(abs(Stick_P));
			}
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(Stick_Ry);
		}
		else if (View >= 3)
		{
			//Only motion in z- direction
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(Stick_Rz);
		}
	}
	else if (Orientation_State == 4)
	{
		if (View == 1)
		{
			if ((pixel_Orientation >= -135) & (pixel_Orientation <= -45))
			{
				//Right - Push -1 * abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(-1 * abs(Stick_P));
			}
			else if ((pixel_Orientation >= 45) & (pixel_Orientation <= 135))
			{
				//Left - Push abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(abs(Stick_P));
			}
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(Stick_Ry);
		}
		else if (View == 2)
		{
			if ((pixel_Orientation >= -135) & (pixel_Orientation <= -45))
			{
				//Left - Push abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(abs(Stick_P));
			}
			else if ((pixel_Orientation >= 45) & (pixel_Orientation <= 135))
			{
				//Right - Push -1 * abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(-1 * abs(Stick_P));
			}
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(Stick_Ry);
		}
		else if (View >= 3)
		{
			//Only motion in z- direction
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_LR.push_back(Stick_Rz);
		}
	}
}

//Gaze Joint - Identifies Forward and Backward motion of the pixel based on orientation, view and populates the proper motion feature storage 
void Process_Kinect::Gaze_Joint(double pixel_Orientation, int Person_Index, int Joint_Index, int View, int Orientation_State, float Stick_P, float Stick_Rx, float Stick_Ry, float Stick_Rz)
{
	//printf("Gaze\n");
	if (Orientation_State <= 2)
	{
		if (View <= 2)
		{
			//Only motion in z- direction
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(Stick_Rz);
		}
		else if (View == 3)
		{
			if ((pixel_Orientation > -45) & (pixel_Orientation < 45))
			{
				//Back - Push -1 * abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(-1 * abs(Stick_P));
			}
			else if ((pixel_Orientation >= -180) & (pixel_Orientation < -135))
			{
				//Forward - Push abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(abs(Stick_P));
			}
			else if ((pixel_Orientation > 135) & (pixel_Orientation <= 180))
			{
				//Forward - Push abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(abs(Stick_P));
			}
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(Stick_Rx);
		}
		else if (View == 4)
		{
			if ((pixel_Orientation > -45) & (pixel_Orientation < 45))
			{
				//Forward - Push abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(abs(Stick_P));
			}
			else if ((pixel_Orientation >= -180) & (pixel_Orientation < -135))
			{
				//Back - Push -1 *  abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(-1 * abs(Stick_P));
			}
			else if ((pixel_Orientation > 135) & (pixel_Orientation <= 180))
			{
				//Back - Push -1 *  abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(-1 * abs(Stick_P));
			}
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(Stick_Rx);
		}
	}
	else if (Orientation_State == 3)
	{
		if (View <= 2)
		{
			//Only motion in z- direction
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(Stick_Rz);
		}
		else if (View == 3)
		{
			if ((pixel_Orientation >= -135) & (pixel_Orientation <= -45))
			{
				//Back - Push -1 * abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(-1 * abs(Stick_P));
			}
			else if ((pixel_Orientation >= 45) & (pixel_Orientation <= 135))
			{
				//Forward - Push abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(abs(Stick_P));
			}
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(Stick_Ry);
		}
		else if (View == 4)
		{
			if ((pixel_Orientation >= -135) & (pixel_Orientation <= -45))
			{
				//Forward - Push abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(abs(Stick_P));
			}
			else if ((pixel_Orientation >= 45) & (pixel_Orientation <= 135))
			{
				//Back - Push -1 * abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(-1 * abs(Stick_P));
			}
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(Stick_Ry);
		}
	}
	else if (Orientation_State == 4)
	{
		if (View <= 2)
		{
			//Only motion in z- direction
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(Stick_Rz);
		}
		else if (View == 3)
		{
			if ((pixel_Orientation >= -135) & (pixel_Orientation <= -45))
			{
				//Forward - Push abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(abs(Stick_P));
			}
			else if ((pixel_Orientation >= 45) & (pixel_Orientation <= 135))
			{
				//Back - Push -1 * abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(-1 * abs(Stick_P));
			}
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(Stick_Ry);
		}
		else if (View == 4)
		{
			if ((pixel_Orientation >= -135) & (pixel_Orientation <= -45))
			{
				//Back - Push -1 * abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(-1 * abs(Stick_P));
			}
			else if ((pixel_Orientation >= 45) & (pixel_Orientation <= 135))
			{
				//Forward - Push abs Push_To_Storage
				Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(abs(Stick_P));
			}
			Person[Person_Index].Joint_Neck_Ref[Joint_Index].Joint_FB.push_back(Stick_Ry);
		}
	}
}


void SceneFlow(cv::Mat Color_Prev, cv::Mat Color_Curr, cv::Mat Depth_Prev, cv::Mat Depth_Curr, int bSEL, static int Npyr = 2, static int Wext = 1, static int Witer = 5, static int WW = 2, static int stepNR = 1, static float maxZ = 100, float alfa = 10)
{

}

