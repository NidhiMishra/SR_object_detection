// YoloKinect1.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "ThriftTools.hpp"
#include "ProtectedClient.h"
#include "ObjectDetectionService.h"
#include "Inputs_constants.h"

#include <iostream>
#include <sstream>

#include "KinectUtil_speech_api.h"
#include <fstream>
std::ofstream out("log.txt");


extern "C" detectBoxes *GlobleObjBoxes = (detectBoxes *)calloc(100, sizeof(detectBoxes));
extern "C" int GlobleObjBoxesNum = 0;

int main(int argc, char* argv[])
{
	try {
		char* cfgfile = new char[2046];
		char* weightfile = new char[2046];
		char* numObject = new char[20];
		char* useThrift = new char[20];
		char *datacfg = new char[2046];
		char *namelist = new char[2046];

		//numObject = "100";
		cfgfile = "..\\..\\..\\cfg\\yolo.cfg";
		weightfile = "..\\..\\..\\weights\\yolo.weights";
		datacfg = "data\\coco.data";
		namelist = "data\\names.list";
		bool isUseThrift = 1;

		float thresh = 0.15;

		for (int next = 1; next < argc; next += 2)
		{
			if (strcmp(argv[next], "-cfgfile") == 0)
			{
				strcpy(cfgfile, argv[next + 1]);
				fprintf(stdout, "\t-cfgfile %s\n", cfgfile);
			}
			else if (strcmp(argv[next], "-weightfile") == 0)
			{
				strcpy(weightfile, argv[next + 1]);
				fprintf(stdout, "\t-weightfile %s\n", weightfile);
			}
			else if (strcmp(argv[next], "-numObject") == 0)
			{
				strcpy(numObject, argv[next + 1]);
				fprintf(stdout, "\t-numObject %s\n", numObject);
			}
			else if (strcmp(argv[next], "-useThrift") == 0)
			{
				strcpy(useThrift, argv[next + 1]);
				fprintf(stdout, "\t-useThrift %s\n", useThrift);
				isUseThrift = (atoi(useThrift) != 0);
			}

		}

		GlobleObjBoxes[0].Obj.objClass = -10;		

		if (isUseThrift) {
			ProtectedClient<imi::ObjectDetectionServiceClient>* ObjectDetectionClient;
			ObjectDetectionClient = new ProtectedClient<imi::ObjectDetectionServiceClient>("localhost", imi::g_Inputs_constants.DEFAULT_OBJECT_SERVICE_PORT);
			KinectUtil *mKinectUtil = new KinectUtil(datacfg, namelist, cfgfile, weightfile, thresh, ObjectDetectionClient);
			mKinectUtil->run();
		}
		else {
			KinectUtil *mKinectUtil = new KinectUtil(datacfg, namelist, cfgfile, weightfile, thresh);
			mKinectUtil->run();
		}

	}
	catch (std::exception& ex) {
		out << ex.what() << std::endl;
		std::cout << ex.what() << std::endl;
	}


	return 0;
}

