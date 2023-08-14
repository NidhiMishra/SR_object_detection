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


#include <pcl\point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include "kcf.h"
#include <string>

using namespace Microsoft::WRL;

typedef struct {
	bool eventBackhome;
	bool eventTakeBagOff;
	bool eventTakeABook;
	bool eventRemindWater;
	bool eventTakeACup;
	bool eventTakeABottle;
	bool eventSeatInChair;
	bool eventForgetBottle;
	bool eventDrinkWater;
	bool eventPersonLeaving;
} eventFlag;

typedef struct {
	bool cupflag;
	bool bottleflag;
	bool bowlflag;
	bool wineglassflag;
	bool phoneflag;
	bool bookflag;
} objFlag;

enum objectDetectionEvent{ General = 0, ForgetBehavie = 1, Grasp = 2, Person_objects = 3, Demo_home = 4, Demo_what = 5};

class KinectBody
{
public:
	//KinectBody();
	//~KinectBody();
	//KinectBody(void) : IsMainBody(false){}
	KinectBody(cv::Point ptJoints[JointType_Count], cv::Vec3f vJoints[JointType_Count], TrackingState nStates[JointType_Count]);

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
	ComPtr<IBodyIndexFrame> bodyIndexFrame;

	ComPtr<IFrameDescription> pFrameDescription;

	DepthSpacePoint* m_pDepthCoordinates;
	

	// Depth Buffer
	std::vector<UINT16> depthBuffer;
	int depthWidth;
	int depthHeight;
	unsigned int depthBytesPerPixel;
	cv::Mat depthMat;
	cv::Mat depthMat8bit;//>>5

	// Depth Buffer for Grasping
	std::vector<UINT16> depthBufferGrasping;

	// Color Buffer
	std::vector<BYTE> colorBuffer;
	int colorWidth;
	int colorHeight;
	unsigned int colorBytesPerPixel;
	cv::Mat colorMat;

	// Body Buffer
	IBody* ppBodies[BODY_COUNT];

	// Body Index
	cv::Mat bodyLabels;

	cv::Mat i_RgbTodepthForshow;//show 8bit
	cv::Mat i_RgbTodepth;//calculate the distance 16bit
	cv::Mat i_RgbTodepthForGrasping;
	cv::Mat i_PersonIdx;

	std::vector<KinectBody>	m_vecKinectBodies;

	network net;
	detection_layer l;
	image **alphabet;
	char **names;
	list *options;
	char *name_list;
	int i;

	float thresh;

	int frame;
	int trackingInterval;

	std::vector<KCF_Tracker> trackers;
	std::vector<object> tracking_objects;

	eventFlag eventFlagForDemo;
	int tmpDemoflag;
	int frameIdx;
	int demoIdx;
	int controlflag;

	objFlag objectFlagForDemoWhatitis;

	//IplImage *disp_main;//for show

public:
	// Constructor
	KinectUtil();

	KinectUtil(char *datacfg, char *namelist, char *cfgfile, char *weightfile, float thresh, ProtectedClient<imi::ObjectDetectionServiceClient>* client);
	KinectUtil(char *datacfg, char *namelist, char *cfgfile, char *weightfile, float thresh);

	// Destructor
	~KinectUtil();

	// Processing
	void run();
	void run(objectDetectionEvent objEvent);

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
	//void update(objectDetectionEvent objEvent);

	// Update Depth
	inline void updateDepth();

	inline void updateColor();

	inline void updateBodyFrame();

	inline void updateBodyIndex();

	// Show Data
	//void show();
	void show(objectDetectionEvent objEvent);

	// Show Depth
	inline void showDepth();
	inline void showColor();
	inline void drawDepth();
	//inline void drawDepth(objectDetectionEvent objEvent);
	inline void detection(objectDetectionEvent objEvent);
	inline void InitialTracker(object *RecObects, int objectNumPerFrame);
	inline void test_tracker_img(object *RecObects, int *objectNumPerFrame);
	inline void voice(std::string str);

	void objectDetectionLocal(object *RecObects, int *objectNumPerFrame, int width, int height, cv::Point selectedPosition);

	std::string object2str(object *RecObects, int objectNumPerFrame, objectDetectionEvent objEvent);
	float GetImgAvg(cv::Mat img);
	float GetImgAvg(cv::Mat img, int thr);
	void userRGB2Depth();
	void caculateXYZinCameraSpace(object *RecObects, int objectNumPerFrame, image im, unsigned char flag, objectDetectionEvent objEvent);
	void show_grasp(image im, char **names, image **alphabet, int classes, object *RecObects, int objectNumPerFrame);
	void labelOBjPixels(image im, image img_backup, cv::Mat imageROI8Bit, object b, int thr);
	void objectBelong2Person(object *RecObects, int objectNumPerFrame);
	void caculateXY(object b, float *centreX, float *centreY, float *topX, float *topY, float *bottomX, float *bottomY, float *leftX, float *leftY, float *rightX, float *rightY, int thr);
	int otsuThreshold(cv::Mat frame);
	void desk_seg(float thr);

	cv::Mat colorImgFilterbyDistance(cv::Mat colorMat, float distance, cv::Mat RGB2DepthMat);

	int maxObjectNum;
	int objectNumPerFrame;
	ProtectedClient<imi::ObjectDetectionServiceClient>* objectClient;
	bool isUseThrift;
	std::vector< ::imi::ObjectInfo> current_objects;
	void initializeChecking();
	void updateNew(std::vector<::imi::ObjectInfo> & Objects);
	bool hasNew;
	void checkObjects(std::vector<::imi::ObjectInfo> & Objects);

	cv::Mat DrawSkeletonFrame(cv::Mat, int Id);
	void DrawBody(cv::Mat &mtxRGBCanvas, KinectBody& body);
	void DrawBone(cv::Mat &mtxRGBCanvas, const KinectBody& body, JointType joint0, JointType joint1);
	cv::Point ProjectDepthToColorPixel(cv::Vec3f v);
	void skeletonShow(image im, int Id);
	void write_infor_to_txt(object *RecObects, int *objectNumPerFrame);
	void write_infor_to_txt_grasp(object *RecObects, int *objectNumPerFrame);
	void write_infor_to_txt_general_obj(object *RecObects, int objectNumPerFrame);
	void write_infor_to_txt_left_handheld_obj(object *RecObects, int objectNumPerFrame);
	void write_infor_to_txt_right_handheld_obj(object *RecObects, int objectNumPerFrame);
	void write_infor_to_txt_carried_obj(object *RecObects, int objectNumPerFrame);
	cv::Point ProjectToPixel(cv::Vec3f v);

};

#endif // __APP__