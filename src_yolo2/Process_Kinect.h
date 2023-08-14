#pragma once

#include "AgentControl.h"
#include "ProtectedClient.h"
#include "Inputs_types.h"

#include "ActionRecognitionService.h"
#include "Inputs_constants.h"

#define _WINSOCKAPI_ //before windows.h conflicts with winsock2
#define NOMINMAX
#include <Windows.h>
#include <Kinect.h>
#include <math.h>
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
//#include "opencv2\core\mat.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2/video/tracking.hpp"
#include "BaseKinectCV.h"
#include <inttypes.h>

#include <Eigen/Core>
#include <Eigen/Cholesky>

#define PI 3.14159265
#define NO_OF_CELLS 3
#define NO_OF_ORIENTATIONS 3
#define NO_OF_GRIDS 3
#define NO_OF_JOINTS 20
#define NO_OF_ACTIONS 3
#define EXTEND_FG 20

//Feature Index defines
#define UN_OFFSET 162
#define UW_OFFSET 324
#define WB_OFFSET 486
#define WN_OFFSET 594
#define WW_OFFSET 702

#define WJOINT_OFFSET 180 // For WH Normalized
#define UNWJOINT_OFFSET 300 // For WH Unnormalized
using namespace cv;
using namespace Eigen;
//Weighted Histogram
typedef struct
{
	//Weighted Without - WW
	//Weighted Neck    - WN
	//Weighted BodyRef - WB
	float WW_Hist[NO_OF_GRIDS*NO_OF_ORIENTATIONS*NO_OF_CELLS * 2  * 2];
	float WN_Hist[NO_OF_GRIDS*NO_OF_ORIENTATIONS*NO_OF_CELLS * 2 * 2];
	float WB_Hist[NO_OF_GRIDS*NO_OF_ORIENTATIONS*NO_OF_CELLS * 2 * 2];
	float WJoint_Hist[NO_OF_JOINTS*NO_OF_ORIENTATIONS * 2];
	float UNWJoint_Hist[NO_OF_JOINTS*NO_OF_ORIENTATIONS * 2];
}Weighted_Hist;
//Unweighted Histogram
typedef struct
{
	//Unweighted Without - UW
	//Unweighted Neck    - UN
	//Unweighted BodyRef - UB
	float UW_Hist[NO_OF_GRIDS*NO_OF_ORIENTATIONS*NO_OF_CELLS * 2 * 3];
	float UN_Hist[NO_OF_GRIDS*NO_OF_ORIENTATIONS*NO_OF_CELLS * 2 * 3];
	float UB_Hist[NO_OF_GRIDS*NO_OF_ORIENTATIONS*NO_OF_CELLS * 2 * 3];
	float UJoint_Hist[NO_OF_JOINTS*NO_OF_ORIENTATIONS * 3];
}Unweighted_Hist;

//Action Histogram
typedef struct
{
	//One Unweighted Histogram & 
	//One Weighted Histogram for each person
	//For 6 kinect skeletons, 6 of these 
	//histograms have to be created.
	Unweighted_Hist UWH;
	Weighted_Hist   WH;
}Motion_Hist;

class KinectBody
{
public:
	//KinectBody();
	//~KinectBody();
	//KinectBody(void) : IsMainBody(false){}
	KinectBody(cv::Point ptJoints[JointType_Count], cv::Vec3f vJoints[JointType_Count],
		TrackingState nStates[JointType_Count], double Orientation, int view, int x_min, int x_max, int y_min, int y_max, bool valid_neck);

public:
	cv::Point	Joints2D[JointType_Count];		// note the joint array is indexed by joint type
	cv::Vec3f	Joints3D[JointType_Count];		// that is, use it like Joints2D[Joint_Type], rather than Joints2D[i]

	bool		IsMainBody;
	UINT64		BodyIndex;
	double		BodyOrientation;
	int			BodyView;
	int			Foreground_x_lim[2];
	int			Foreground_y_lim[2];
	TrackingState	JointStates[JointType_Count];
	bool		neck_valid;
};

//Struct for 3D scene flow velocity
typedef struct
{
	//x,y,z velocities
	cv::Mat Vx;
	cv::Mat Vy;
	cv::Mat Vz;

	//Indicates success of SceneFlow computation function
	int SFsuccess;
}SF_3DVelcoties;

//Struct for 3D Velocity storage for histogram
//Cell_Storage
typedef struct
{
	//Cell Storage for
	//Up-Down, Forward-Backward, Left-Right
	std::vector<float> Cell_UD;
	std::vector<float> Cell_FB;
	std::vector<float> Cell_LR;

}Cell;

//Grid_Storage
typedef struct
{
	//Each Grid contains 6 Cells
	Cell Each_Cell[6];

}Grid;

//Joint_Storage
typedef struct
{
	//Joint Storage for
	//Up-Down, Forward-Backward, Left-Right
	std::vector<float> Joint_UD;
	std::vector<float> Joint_FB;
	std::vector<float> Joint_LR;

}Motion_Joint;

//Person_Storage
typedef struct
{
	//Each person contains 3 Grids each for Without referenced, Neck referenced and Body referenced
	Grid Without[3];
	Grid Neck_Ref[3];
	Grid Body_Ref[3];
	Motion_Joint Joint_Neck_Ref[JointType_Count];

}Person_Features;

class Process_Kinect : public BaseKinectCV
{
public:
	Process_Kinect(bool Thrift_use);
	~Process_Kinect();

public:
	void		Init(void);
	bool		Update(void);
	void		Close(void);
	//SF_3DVelcoties SceneFlow(cv::Mat Color_Prev, cv::Mat Color_Curr, cv::Mat Depth_Prev, cv::Mat Depth_Curr, int bSEL, static int Npyr = 2, static int Wext = 1, static int Witer = 5, static int WW = 2, static int stepNR = 1, static float maxZ = 100, float alfa = 10); //Scene Flow Function
	void SceneFlow(cv::Mat Color_Prev, cv::Mat Color_Curr, cv::Mat Depth_Prev, cv::Mat Depth_Curr, int bSEL, static int Npyr = 2, static int Wext = 1, static int Witer = 5, static int WW = 2, static int stepNR = 1, static float maxZ = 100, float alfa = 10); //Scene Flow Function

private:
	HRESULT		Nui_Init(void);
	HRESULT		InitializeDefaultSensor(void);
	void		Nui_Clear(void);
	bool		Nui_NextColorFrame(void);
	bool		Nui_NextDepthFrame(void);
	bool		Nui_NextBodyFrame(void);
	bool		Nui_NextLabelFrame(void);
	void		Nui_DrawFrame(void);
	void		Nui_DrawBody(cv::Mat &mtxRGBCanvas, KinectBody& body);
	void		Nui_DrawBone(cv::Mat &mtxRGBCanvas, const KinectBody& body, JointType joint0, JointType joint1);
	cv::Point	Nui_ProjectDepthToColorPixel(cv::Vec3f v);
	int			detectFaces(Mat frame);
	void        ELM_Initialize(void);
	//void		ComputeKinematicFeatures();
	void		ComputeStickKinematicFeatures();
	void		ConvertWorldMotionToBodySpace(); //Converts Kinematic motion features to Body centric space
	void		Up_Down_Cell(double pixel_Orientation, int Person_Index, int Grid_Index, int Cell_Index, int View, int Orientation_State, int row, int col);
	void		Left_Right_Cell(double pixel_Orientation, int Person_Index, int Grid_Index, int Cell_Index, int View, int Orientation_State, int row, int col);
	void		Gaze_Cell(double pixel_Orientation, int Person_Index, int Grid_Index, int Cell_Index, int View, int Orientation_State, int row, int col);
	void		Up_Down_Joint(double pixel_Orientation, int Person_Index, int Joint_Index, int View, int Orientation_State, float Stick_P, float Stick_Rx, float Stick_Ry, float Stick_Rz);
	void		Left_Right_Joint(double pixel_Orientation, int Person_Index, int Joint_Index, int View, int Orientation_State, float Stick_P, float Stick_Rx, float Stick_Ry, float Stick_Rz);
	void		Gaze_Joint(double pixel_Orientation, int Person_Index, int Joint_Index, int View, int Orientation_State, float Stick_P, float Stick_Rx, float Stick_Ry, float Stick_Rz);
	void        Compute_Action_Histograms(); //Compute Pose-invaiant action histograms using Kinematic motion features in Body centric space
	void		Clear_Vectors(void);
	imi::ActionType::type Map_inttoActionType_type(int);
	imi::ActionType::type Map_inttoActionType_small(int);
	//int			Nui_FindMainBody(std::vector<KinectBody>& vecKinectBodies);
	//bool		Nui_FindHandMask(cv::Mat &mtxDepthImg, const KinectBody& mainbody, cv::Mat &mtxMask, bool bActive[2]);

public:
	virtual cv::Point		ProjectToPixel(cv::Vec3f v);
	virtual cv::Vec3f		BackProjectPixel(cv::Point xp, double z);
	SF_3DVelcoties SFlow;
	//std::vector< std::vector<double> > User_actions;
	std::vector<imi::UserAction> User_actions;
	bool	isUseThrift;
	bool	Action_Detected;
private:
	// it is possible to use IMultiSourceFrameReader instead of all the separate frame readers
	// however, we find the frame rate of IMultiSourceFrameReader is much lower in run-time

	IKinectSensor*			m_pKinectSensor;
	IDepthFrameReader*      m_pDepthFrameReader;
	IColorFrameReader*      m_pColorFrameReader;
	IBodyFrameReader*       m_pBodyFrameReader;
	IBodyIndexFrameReader*	m_pBodyIndexReader;
	ICoordinateMapper*      m_pCoordinateMapper;
	uint64_t		Skeleton_TrackingID[6];
	Person_Features Person[6]; // Storage for motion features in body centric space - Multiple person
	Motion_Hist     Action_Features[6]; //Histogram features in body centric space - Multiple person
	int             m_nhn,m_ndims,m_Jointndims;
	MatrixXd        inW, bias, outW;
	MatrixXd        Stick_inW, Stick_bias, Stick_outW;
	float			m_recognition_threshold;
	bool			Skel_Present;
	int				m_NoSkeletons;

	ProtectedClient<imi::ActionRecognitionServiceClient> *ActionManager;
	//WAITABLE_HANDLE			m_hFrameEvents[4];
protected:
	CascadeClassifier face_cascade;
private:
	void PixeltoBodyPartLabel(std::vector<KinectBody>& vecKinectBodies);
	void Populate_Cell_Numbering(int No_of_Cells);
	std::vector<KinectBody>	m_vecKinectBodies;
	std::vector<KinectBody>	m_Prev_vecKinectBodies;
	std::vector< std::vector<int> > Foreground_points;
	//cv::Mat					m_mtxHandMask;
	//bool					m_bHandActive[2];

	// Safe release for interfaces
	template<class Interface>
	inline void SafeRelease(Interface *& pInterfaceToRelease)
	{
		if (pInterfaceToRelease != NULL)
		{
			pInterfaceToRelease->Release();
			pInterfaceToRelease = NULL;
		}
	}
};

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

inline cv::Vec3b GetDistinctColor(int n)
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
	unsigned char R = (nColourValues[n] & 0xFF0000) >> 16;
	unsigned char G = (nColourValues[n] & 0x00FF00) >> 8;
	unsigned char B = nColourValues[n] & 0x0000FF;
	return cv::Vec3b(B, G, R);
}

