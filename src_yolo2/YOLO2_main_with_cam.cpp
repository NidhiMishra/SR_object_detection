// YoloKinect1.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "ThriftTools.hpp"
#include "ProtectedClient.h"
#include "ObjectDetectionService.h"
#include "Inputs_constants.h"

#include <iostream>
#include <sstream>


#include "KinectUtil_with_cam.h"
#include <fstream>
std::ofstream out("log.txt");

extern "C" detectBoxes *GlobleObjBoxes = (detectBoxes *)calloc(100, sizeof(detectBoxes));
extern "C" int GlobleObjBoxesNum = 0;
extern "C" char* str2virtualHuman = "";


int main(int argc, char* argv[])
{
	try {
		char* cfgfile = new char[2046];
		char* weightfile = new char[2046];
		char* numObject = new char[20];
		char* useThrift = new char[20];
		char *datacfg = new char[2046];
		char *namelist = new char[2046];
		char *threshStr = new char[10];
		char *eventName = new char[10];

		cfgfile = "..\\..\\..\\cfg\\yolo.cfg";
		weightfile = "..\\..\\..\\weights\\yolo.weights";
		datacfg = "data\\coco.data";
		namelist = "data\\names.list";
		bool isUseThrift = 0;

		float thresh = 0.24;

		objectDetectionEvent objEvent = Demo_what; // Demo_what, Grasp //Demo_what, General, ForgetBehavie, Grasp，Demo_home, Person_objects

		if (argc > 2)
		{
			cfgfile = argv[1];
			weightfile = argv[2];
			datacfg = argv[3];
			namelist = argv[4];
			thresh = atof(argv[5]);
			eventName = argv[6];
			if (strcmp(eventName, "objectDectection") == 0)objEvent = Demo_what;
			else if (strcmp(eventName, "Grasp") == 0)objEvent = Grasp;

		}

		GlobleObjBoxes[0].Obj.objClass = -10;		

#ifdef i2p
		if (isUseThrift) {
			ProtectedClient<imi::ObjectDetectionServiceClient>* ObjectDetectionClient;
			ObjectDetectionClient = new ProtectedClient<imi::ObjectDetectionServiceClient>("localhost", imi::g_Inputs_constants.DEFAULT_OBJECT_SERVICE_PORT);
			KinectUtil *mKinectUtil = new KinectUtil(datacfg, namelist, cfgfile, weightfile, thresh, ObjectDetectionClient);
			mKinectUtil->run(objEvent);
		}
		else {
			KinectUtil *mKinectUtil = new KinectUtil(datacfg, namelist, cfgfile, weightfile, thresh);
			mKinectUtil->run(objEvent);
		}
#else
		KinectUtil *mKinectUtil = new KinectUtil(datacfg, namelist, cfgfile, weightfile, thresh);
		mKinectUtil->run(objEvent);
#endif
	}
	catch (std::exception& ex) {
		out << ex.what() << std::endl;
		std::cout << ex.what() << std::endl;
	}


	return 0;
}

