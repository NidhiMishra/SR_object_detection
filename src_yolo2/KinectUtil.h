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

class KinectUtil
{
public:
	// Sensor
	ComPtr<IKinectSensor> kinect;

	// Reader
	ComPtr<IDepthFrameReader> depthFrameReader;
	ComPtr<IColorFrameReader> colorFrameReader;

	// Depth Buffer
	std::vector<UINT16> depthBuffer;
	int depthWidth;
	int depthHeight;
	unsigned int depthBytesPerPixel;
	cv::Mat depthMat;

	// Color Buffer
	std::vector<BYTE> colorBuffer;
	int colorWidth;
	int colorHeight;
	unsigned int colorBytesPerPixel;
	cv::Mat colorMat;

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
	void write_infor_to_txt(object *RecObects, int *objectNumPerFrame);

private:
	// Initialize
	void initialize(char *datacfg, char *namelist, char *cfgfile, char *weightfile);

	// Initialize Sensor
	inline void initializeSensor();

	// Coordinate Mapper
	ComPtr<ICoordinateMapper> coordinateMapper;

	// Initialize Depth
	inline void initializeDepth();

	inline void KinectUtil::initializeColor();

	// Finalize
	void finalize();

	// Update Data
	void update();

	// Update Depth
	inline void updateDepth();

	inline void KinectUtil::updateColor();

	// Show Data
	void show();

	// Show Depth
	inline void showDepth();
	inline void KinectUtil::showColor();
	inline void KinectUtil::drawDepth();
	inline void KinectUtil::detection();
	float KinectUtil::GetImgAvg(cv::Mat img);

	int maxObjectNum;
	int objectNumPerFrame;
	ProtectedClient<imi::ObjectDetectionServiceClient>* objectClient;
	bool isUseThrift;
	std::vector< ::imi::ObjectInfo> current_objects;
	void initializeChecking();
	void updateNew(std::vector<::imi::ObjectInfo> & Objects);
	bool hasNew;
	void checkObjects(std::vector<::imi::ObjectInfo> & Objects);
};

#endif // __APP__