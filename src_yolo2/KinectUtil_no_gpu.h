#ifndef __APP__
#define __APP__

#include <Windows.h>
#include <Kinect.h>
#include <opencv2/opencv.hpp>
#include "ThriftTools.hpp"
#include "ProtectedClient.h"
#include "ObjectDetectionService.h"
#include "Inputs_constants.h"

#include <vector>

#include <wrl/client.h>
#include "detection_layer.h"
#include "network.h"
using namespace Microsoft::WRL;

typedef struct {
	bool eventBackhome;
	bool eventTakeBagOff;
	bool eventCannotPutBaginChair;
} eventFlag;


class KinectBody
{
public:
	//KinectBody();
	//~KinectBody();
	//KinectBody(void) : IsMainBody(false){}
	KinectBody(cv::Vec3f vJoints[JointType_Count], TrackingState nStates[JointType_Count]);

public:
	cv::Point	Joints2D[JointType_Count];		// note the joint array is indexed by joint type
	cv::Vec3f	Joints3D[JointType_Count];		// that is, use it like Joints2D[Joint_Type], rather than Joints2D[i]

	bool		IsMainBody;
	//	UINT64		BodyIndex;
	//	double		BodyOrientation;
	//	int			BodyView;
	//	int			Foreground_x_lim[2];
	//	int			Foreground_y_lim[2];
	TrackingState	JointStates[JointType_Count];
	//	bool		neck_valid;
};


class KinectUtil
{
public:
	// Sensor
	ComPtr<IKinectSensor> kinect;

	// Reader
	ComPtr<IDepthFrameReader> depthFrameReader;
	ComPtr<IColorFrameReader> colorFrameReader;
	ComPtr<IBodyIndexFrameReader> BodyIndexReader;
	ComPtr<IBodyFrameReader> BodyFrameReader;

	ComPtr<IDepthFrame> depthFrame;
	ComPtr<IColorFrame> colorFrame;
	ComPtr<IBodyFrame> bodyFrame;

	DepthSpacePoint* m_pDepthCoordinates;


	// Depth Buffer
	std::vector<UINT16> depthBuffer;
	int depthWidth;
	int depthHeight;
	unsigned int depthBytesPerPixel;
	cv::Mat depthMat;
	cv::Mat depthMat8bit;//>>5

	// Color Buffer
	std::vector<BYTE> colorBuffer;
	int colorWidth;
	int colorHeight;
	unsigned int colorBytesPerPixel;
	cv::Mat colorMat;

	// Body Buffer
	IBody* ppBodies[BODY_COUNT];

	cv::Mat i_RgbTodepthForshow;//show 8bit
	cv::Mat i_RgbTodepth;//calculate the distance 16bit

	std::vector<KinectBody>	m_vecKinectBodies;

	network net;
	detection_layer l;
	image **alphabet;
	char **names;
	list *options;
	char *name_list;
	int i;

	float thresh;

public:
	// Constructor
	KinectUtil();

	KinectUtil(char *datacfg, char *namelist, char *cfgfile, char *weightfile, float thresh, ProtectedClient<imi::ObjectDetectionServiceClient>* client);
	KinectUtil(char *datacfg, char *namelist, char *cfgfile, char *weightfile, float thresh);

	// Destructor
	~KinectUtil();

	// Processing
	void run();

private:
	// Initialize
	void initialize(char *datacfg, char *namelist, char *cfgfile, char *weightfile);

	// Initialize Sensor
	inline void initializeSensor();

	// Coordinate Mapper
	ComPtr<ICoordinateMapper> coordinateMapper;

	// Initialize Depth
	inline void initializeDepth();

	inline void initializeColor();

	inline void initializeBodyIndex();

	inline void initializeBodyFrame();

	// Finalize
	void finalize();

	// Update Data
	void update();

	// Update Depth
	inline void updateDepth();

	inline void updateColor();

	inline void updateBodyFrame();

	// Show Data
	void show();

	// Show Depth
	inline void showDepth();
	inline void showColor();
	inline void drawDepth();
	inline void detection();
	float GetImgAvg(cv::Mat img);
	float GetImgAvg(cv::Mat img, int thr);
	void userRGB2Depth();
	void caculateXYZinCameraSpace(object *RecObects, int objectNumPerFrame);
	void caculateXY(object RecObects, float *x, float *y, int thr);
	int otsuThreshold(cv::Mat frame);

	int maxObjectNum;
	int objectNumPerFrame;
	ProtectedClient<imi::ObjectDetectionServiceClient>* objectClient;
	bool isUseThrift;
	std::vector< ::imi::ObjectInfo> current_objects;
	void initializeChecking();
	void updateNew(std::vector<::imi::ObjectInfo> & Objects);
	bool hasNew;
	void checkObjects(std::vector<::imi::ObjectInfo> & Objects);

	cv::Mat DrawSkeletonFrame(cv::Mat);
	void DrawBody(cv::Mat &mtxRGBCanvas, KinectBody& body);
	void DrawBone(cv::Mat &mtxRGBCanvas, const KinectBody& body, JointType joint0, JointType joint1);
	cv::Point ProjectDepthToColorPixel(cv::Vec3f v);
	image showImg(image im);
	void write_infor_to_txt(object *RecObects, int *objectNumPerFrame);

};

#endif // __APP__