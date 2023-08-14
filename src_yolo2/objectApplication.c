#include "image.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/version.hpp"
#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio_c.h"
#endif
#endif

extern detectBoxes *GlobleObjBoxes;
extern int GlobleObjBoxesNum;
extern char* str2virtualHuman;

void object_category_init(showName *objectName, int *objectNumPerFrame, int objEvent)
{
	

	//showName *objectName = calloc(*num_Obj, sizeof(showName));
	//for grasping
	if (objEvent == 1 || objEvent == 4)//for story
	{
		*objectNumPerFrame = 10;
		objectName[0].showName = "cup";
		objectName[0].objName = "cup";
		objectName[1].showName = "book";
		objectName[1].objName = "book";
		objectName[2].showName = "handbag";
		objectName[2].objName = "handbag";
		objectName[3].showName = "backpack";
		objectName[3].objName = "backpack";
		objectName[4].showName = "bottle";
		objectName[4].objName = "bottle";
		objectName[5].showName = "cell phone";
		objectName[5].objName = "cell phone";
		objectName[6].showName = "person";
		objectName[6].objName = "person";
		objectName[7].showName = "chair";
		objectName[7].objName = "chair";
		objectName[8].showName = "tvmonitor";
		objectName[8].objName = "tvmonitor";
		objectName[9].showName = "laptop";
		objectName[9].objName = "laptop";

		//objectName[7].showName = "wine glass";
		//objectName[7].objName = "wine glass";
		//objectName[8].showName = "remote";
		//objectName[8].objName = "remote";
	}
	else if (objEvent == 2)//for grasping
	{
		*objectNumPerFrame = 4;
		objectName[0].showName = "cup";
		objectName[0].objName = "cup";
		objectName[1].showName = "bottle";
		objectName[1].objName = "bottle";
		objectName[2].showName = "bowl";
		objectName[2].objName = "bowl";	
		objectName[3].showName = "wine glass";
		objectName[3].objName = "wine glass";	
		//objectName[4].showName = "mouse";
		//objectName[4].objName = "mouse";
	}
	else if (objEvent == 5)//What is it
	{
		*objectNumPerFrame = 13;
		objectName[0].showName = "cup";
		objectName[0].objName = "cup";
		objectName[1].showName = "bottle";
		objectName[1].objName = "bottle";
		objectName[2].showName = "book";
		objectName[2].objName = "book";
		objectName[3].showName = "wine glass";
		objectName[3].objName = "wine glass";
		objectName[4].showName = "cellphone";
		objectName[4].objName = "cell phone"; 
		objectName[5].showName = "fork";
		objectName[5].objName = "fork";
		objectName[6].showName = "handbag";
		objectName[6].objName = "handbag";
		objectName[7].showName = "backpack";
		objectName[7].objName = "backpack";
		objectName[8].showName = "umbrella";
		objectName[8].objName = "umbrella";
		objectName[9].showName = "tie";
		objectName[9].objName = "tie";
		objectName[10].showName = "suitcase";
		objectName[10].objName = "suitcase";
		objectName[11].showName = "pencil";
		objectName[11].objName = "pencil";
		objectName[12].showName = "pen";
		objectName[12].objName = "pen";
	}


	//return objectName;
}

void objectFilterUsingObjectCategory(object *RecObects, int *objectNumPerFrame, int objEvent)
{
	if (objEvent == 0 || objEvent == 3) return;
	int num_Obj = 100;
	showName *objectName = calloc(num_Obj, sizeof(showName));
	object_category_init(objectName, &num_Obj, objEvent);
	int N = *objectNumPerFrame;
	object *temp = calloc(100, sizeof(object));
	int idx = 0;
	for (int i = 0; i < num_Obj; i++)
		for (int j = 0; j < N; j++)
		{
			if (strcmp(RecObects[j].name, objectName[i].objName) == 0)
			{
				strcpy(RecObects[j].name, objectName[i].showName);
				temp[idx] = RecObects[j];
				idx++;
				//break;
			}
		}
	for (int i = 0; i < idx; i++)
	{
		RecObects[i] = temp[i];
	}
	*objectNumPerFrame = idx;
	free(temp);
	free(objectName);
}

void objectFilterUsingPersonId(object *RecObects, int *objectNumPerFrame, int *pIdx, int *pCount)
{
	int personIdx[6] = { 255, 255, 255, 255, 255, 255 };
	int personCount = 0;
	int objectNum = *objectNumPerFrame;
	unsigned char flag = 0;
	object b;
	int resevedObjectNum = 0;
	object resevedObject[100];
	for (int i = 0; i < objectNum; i++)
	{
		b = RecObects[i];
		if ((b.bodyId >= 1 && b.bodyId <= 6) && strcmp(b.name, "person") == 0)
		{
			flag = 0;
			resevedObject[resevedObjectNum] = b;
			resevedObjectNum++;

			for (int pIdx = 0; pIdx < personCount; pIdx++)
			{
				if (b.bodyId == personIdx[pIdx]){ flag = 1;	}
			}
			if (flag == 0)
			{
				personIdx[personCount] = b.bodyId;
				personCount++;
			}
		}
	}
	//for one person
	for (int i = 0; i < personCount; i++)
		pIdx[i] = personIdx[i];
	*pCount = personCount;
	for (int i = 0; i < resevedObjectNum; i++)
		RecObects[i] = resevedObject[i];
	*objectNumPerFrame = resevedObjectNum;
}

void objectFilterSpecialID(object *RecObects, int *objectNumPerFrame, int personID)
{
	int objectNum = *objectNumPerFrame;
	unsigned char flag = 0;
	object b;
	int resevedObjectNum = 0;
	object resevedObject[100];
	for (int i = 0; i < objectNum; i++)
	{
		b = RecObects[i];
		if (b.bodyId == personID && strcmp(b.name, "person") != 0 && personID != 255)
		{
			resevedObject[resevedObjectNum] = b;
			resevedObjectNum++;
		}
	}
	for (int i = 0; i < resevedObjectNum; i++)
		RecObects[i] = resevedObject[i];
	*objectNumPerFrame = resevedObjectNum;
}

void distanceFilter(object *RecObects, int *objectNumPerFrame, float distanceThreshold)
{
	if (distanceThreshold == 0) return;
	int num_Obj = 100;
	int N = *objectNumPerFrame;
	object *temp = calloc(100, sizeof(object));
	int idx = 0;
	for (int i = 0; i < N; i++)
	{
		if (RecObects[i].CameraZ < distanceThreshold)
		{
			temp[idx] = RecObects[i];
			idx++;
		}
			
	}
	for (int i = 0; i < idx; i++)
	{
		RecObects[i] = temp[i];
	}
	*objectNumPerFrame = idx;
	free(temp);
}

void removeRepeatedName(detectBoxes *RecObects, int *objectNumPerFrame, showName *objectName, int num_Obj)
{
	int idx = 0;
	detectBoxes *temp = calloc(100, sizeof(detectBoxes));

	for (int i = 0; i < num_Obj; i++)
	{
		for (int j = 0; j < *objectNumPerFrame; j++)
		{
			if (strcmp(objectName[i].objName, RecObects[j].Obj.name) == 0)
			{
				strcpy(RecObects[j].Obj.name, objectName[i].showName);
				temp[idx] = RecObects[j];
				idx++;
				break;
			}
		}
	}
	for (int i = 0; i < idx; i++)
	{
		RecObects[i] = temp[i];
	}
	*objectNumPerFrame = idx;

	free(temp);
}



void send2VirtualHuman(detectBoxes *RecObects, int *objectNum, char* strID)
{
	int num_Obj = 100;
	showName *objectName = calloc(num_Obj, sizeof(showName));
	init_Show_Obj(objectName, &num_Obj);
	int N = *objectNum;

	if (N > 0)
	{
		if (strcmp(strID, str2virtualHuman) != 0)
		{
			str2virtualHuman = strID;
			removeRepeatedName(RecObects, objectNum, objectName, num_Obj);
			printf(str2virtualHuman);
			printf("\n");
		}
		else
		{
			*objectNum = 0;
		}
		if (strcmp(strID, "") == 0)*objectNum = 0;

	}


	free(objectName);

}

void object_vote_mutilframe(object *RecObects, int *objectNumPerFrame)
{
	int i, j, k, l;

	int maxBoxesNum = 100;
	detectBoxes *ObjBoxes = calloc(maxBoxesNum, sizeof(detectBoxes));
	detectBoxes *ShowObjBoxes = calloc(maxBoxesNum, sizeof(detectBoxes));
	int BoxesNum = *objectNumPerFrame;
	int ShowObjBoxesNum = 0;

	int appearNumMax = 2;
	int negativeAppearNumMax = 0;
	float thresh_boxes_frames = 0.10;

	int num_Obj = 100;
	showName *objectName = calloc(num_Obj, sizeof(showName));
	init_Show_Obj(objectName, &num_Obj);

	for (i = 0; i < BoxesNum; i++)
	{
		ObjBoxes[i].Obj = RecObects[i];

	}

	if (BoxesNum > 0 && GlobleObjBoxes[0].Obj.objClass == -10)//for the first frame
	{
		for (i = 0; i < BoxesNum; i++)
		{
			GlobleObjBoxes[i] = ObjBoxes[i];
			GlobleObjBoxes[i].appearFrameNum = 1;
			GlobleObjBoxesNum++;
		}
	}
	else if (BoxesNum > 0 && GlobleObjBoxes[0].Obj.objClass != -10)//for other frames
	{
		int oldGlobleObjBoxesNum = GlobleObjBoxesNum;

		for (i = 0; i < BoxesNum; i++)
		{
			for (j = 0; j < oldGlobleObjBoxesNum; j++)
			{
				box b1, b2;
				b1.x = ObjBoxes[i].Obj.x; b1.y = ObjBoxes[i].Obj.y; b1.w = ObjBoxes[i].Obj.w; b1.h = ObjBoxes[i].Obj.h;
				b2.x = GlobleObjBoxes[j].Obj.x; b2.y = GlobleObjBoxes[j].Obj.y; b2.w = GlobleObjBoxes[j].Obj.w; b2.h = GlobleObjBoxes[j].Obj.h;

				// find the object reappeared.
				if (ObjBoxes[i].Obj.objClass == GlobleObjBoxes[j].Obj.objClass && box_iou(b1, b2) >= thresh_boxes_frames)
				{
					if (GlobleObjBoxes[j].appearFrameNum <= appearNumMax)
						GlobleObjBoxes[j].appearFrameNum++;
					GlobleObjBoxes[j].Obj.h = 0.5 * (GlobleObjBoxes[j].Obj.h + ObjBoxes[i].Obj.h);
					GlobleObjBoxes[j].Obj.w = 0.5 * (GlobleObjBoxes[j].Obj.w + ObjBoxes[i].Obj.w);
					GlobleObjBoxes[j].Obj.x = 0.5 * (GlobleObjBoxes[j].Obj.x + ObjBoxes[i].Obj.x);
					GlobleObjBoxes[j].Obj.y = 0.5 * (GlobleObjBoxes[j].Obj.y + ObjBoxes[i].Obj.y);
					GlobleObjBoxes[j].Obj.flagBelong2Person = ObjBoxes[i].Obj.flagBelong2Person;
					GlobleObjBoxes[j].flagReappear = 1;
					break;
				}
			}
			if (j == oldGlobleObjBoxesNum)// find the new object.
			{
				GlobleObjBoxes[GlobleObjBoxesNum] = ObjBoxes[i];
				GlobleObjBoxes[GlobleObjBoxesNum].appearFrameNum = 1;
				GlobleObjBoxes[GlobleObjBoxesNum].flagNew = 1;
				GlobleObjBoxes[GlobleObjBoxesNum].flagReappear = 1;
				GlobleObjBoxesNum++;
			}
		}

		for (i = 0; i < oldGlobleObjBoxesNum; i++)// decrease the frequence of object which does not reappear.
		{
			if (GlobleObjBoxes[i].flagReappear != 1 && GlobleObjBoxes[i].appearFrameNum > negativeAppearNumMax)
			{
				GlobleObjBoxes[i].appearFrameNum--;
				if (GlobleObjBoxes[i].appearFrameNum <= negativeAppearNumMax)
					GlobleObjBoxes[i].flagDelete = 1;
			}
			if (GlobleObjBoxes[i].flagReappear == 1)
			{
				GlobleObjBoxes[i].flagReappear = 0;
			}
		}

		//remove the inexistent objects
		oldGlobleObjBoxesNum = GlobleObjBoxesNum;
		for (i = 0; i < oldGlobleObjBoxesNum; i++)
		{
			if (GlobleObjBoxes[i].flagDelete == 1)
			{
				for (j = i; j < oldGlobleObjBoxesNum - 1; j++)
				{
					GlobleObjBoxes[j] = GlobleObjBoxes[j + 1];
				}
				GlobleObjBoxesNum--;
			}
		}
	}

	for (i = 0; i < GlobleObjBoxesNum; i++)//show the objects (appearing 5 times)
	{
		if (GlobleObjBoxes[i].appearFrameNum >= appearNumMax)
		{
			ShowObjBoxes[ShowObjBoxesNum] = GlobleObjBoxes[i];
			ShowObjBoxesNum++;
		}
	}

	for (i = 0; i < ShowObjBoxesNum; i++)
	{
		RecObects[i] = ShowObjBoxes[i].Obj;
	}
	*objectNumPerFrame = ShowObjBoxesNum;

	free(ObjBoxes);
	free(ShowObjBoxes);
	free(objectName);
}


void object_reminder(image im, char **names, image **alphabet, int classes, object *RecObects, int *objectNumPerFrame)
{
	int i, j, k, l;

	int maxBoxesNum = 100;
	detectBoxes *ObjBoxes = calloc(maxBoxesNum, sizeof(detectBoxes));
	detectBoxes *ShowObjBoxes = calloc(maxBoxesNum, sizeof(detectBoxes));
	int BoxesNum = *objectNumPerFrame;
	int ShowObjBoxesNum = 0;

	int appearNumMax = 6;
	int negativeAppearNumMax = -3;
	float thresh_boxes_frames = 0.10;

	detectBoxes *NewObjBoxes;
	NewObjBoxes = calloc(100, sizeof(detectBoxes));
	int NewObjBoxesNum = 0;

	int num_Obj = 100;
	showName *objectName = calloc(num_Obj, sizeof(showName));
	init_Show_Obj(objectName, &num_Obj);

	for (i = 0; i < BoxesNum; i++)
	{
		ObjBoxes[i].Obj = RecObects[i];

	}

		if (BoxesNum > 0 && GlobleObjBoxes[0].Obj.objClass == -10)//for the first frame
	{
	for (i = 0; i < BoxesNum; i++)
	{
	GlobleObjBoxes[i] = ObjBoxes[i];
	GlobleObjBoxes[i].appearFrameNum = 1;
	GlobleObjBoxesNum++;
	}
	}
		else if (BoxesNum > 0 && GlobleObjBoxes[0].Obj.objClass != -10)//for other frames
	{
	int oldGlobleObjBoxesNum = GlobleObjBoxesNum;

	for (i = 0; i < BoxesNum; i++)
	{
	for (j = 0; j < oldGlobleObjBoxesNum; j++)
	{
		box b1, b2;
		b1.x = ObjBoxes[i].Obj.x; b1.y = ObjBoxes[i].Obj.y; b1.w = ObjBoxes[i].Obj.w; b1.h = ObjBoxes[i].Obj.h;
		b2.x = GlobleObjBoxes[j].Obj.x; b2.y = GlobleObjBoxes[j].Obj.y; b2.w = GlobleObjBoxes[j].Obj.w; b2.h = GlobleObjBoxes[j].Obj.h;

	// find the object reappeared.
		if (ObjBoxes[i].Obj.objClass == GlobleObjBoxes[j].Obj.objClass && box_iou(b1, b2) >= thresh_boxes_frames)
	{
	if (GlobleObjBoxes[j].appearFrameNum <= appearNumMax)
	GlobleObjBoxes[j].appearFrameNum++;
	GlobleObjBoxes[j].Obj.h = 0.5 * (GlobleObjBoxes[j].Obj.h + ObjBoxes[i].Obj.h);
	GlobleObjBoxes[j].Obj.w = 0.5 * (GlobleObjBoxes[j].Obj.w + ObjBoxes[i].Obj.w);
	GlobleObjBoxes[j].Obj.x = 0.5 * (GlobleObjBoxes[j].Obj.x + ObjBoxes[i].Obj.x);
	GlobleObjBoxes[j].Obj.y = 0.5 * (GlobleObjBoxes[j].Obj.y + ObjBoxes[i].Obj.y);
	GlobleObjBoxes[j].flagReappear = 1;
	break;
	}
	}
	if (j == oldGlobleObjBoxesNum)// find the new object.
	{
	GlobleObjBoxes[GlobleObjBoxesNum] = ObjBoxes[i];
	GlobleObjBoxes[GlobleObjBoxesNum].appearFrameNum = 1;
	GlobleObjBoxes[GlobleObjBoxesNum].flagNew = 1;
	GlobleObjBoxes[GlobleObjBoxesNum].flagReappear = 1;
	GlobleObjBoxesNum++;
	}
	}

	for (i = 0; i < oldGlobleObjBoxesNum; i++)// decrease the frequence of object which does not reappear.
	{
	if (GlobleObjBoxes[i].flagReappear != 1 && GlobleObjBoxes[i].appearFrameNum > negativeAppearNumMax)
	{
	GlobleObjBoxes[i].appearFrameNum--;
	if (GlobleObjBoxes[i].appearFrameNum <= negativeAppearNumMax)
	GlobleObjBoxes[i].flagDelete = 1;
	}
	if (GlobleObjBoxes[i].flagReappear == 1)
	{
	GlobleObjBoxes[i].flagReappear = 0;
	}
	}

	//remove the inexistent objects
	oldGlobleObjBoxesNum = GlobleObjBoxesNum;
	for (i = 0; i < oldGlobleObjBoxesNum; i++)
	{
	if (GlobleObjBoxes[i].flagDelete == 1)
	{
	for (j = i; j < oldGlobleObjBoxesNum - 1; j++)
	{
	GlobleObjBoxes[j] = GlobleObjBoxes[j + 1];
	}
	GlobleObjBoxesNum--;
	}
	}
	}

	for (i = 0; i < GlobleObjBoxesNum; i++)//show the objects (appearing 5 times)
	{
	if (GlobleObjBoxes[i].appearFrameNum > 5)
	{
	ShowObjBoxes[ShowObjBoxesNum] = GlobleObjBoxes[i];
	ShowObjBoxesNum++;
	}
	}
	for (i = 0; i < ShowObjBoxesNum; i++)//the object labelled as new objects.
	{
	if (ShowObjBoxes[i].flagNew == 1 && checkObjLabel(names[ShowObjBoxes[i].Obj.objClass], objectName, num_Obj))
	{
	NewObjBoxes[NewObjBoxesNum] = ShowObjBoxes[i];
	NewObjBoxesNum++;
	}
	}

	draw_object_boxes(im, NewObjBoxes, NewObjBoxesNum, names, alphabet, classes);
	//draw_object_boxes(im, GlobleObjBoxes, GlobleObjBoxesNum, names, alphabet, classes);
	//draw_text_box(GlobleObjBoxes, GlobleObjBoxesNum, NewObjBoxes, NewObjBoxesNum, names, objectName, num_Obj);
	char* strID = NULL;
	strID = draw_text_box(GlobleObjBoxes, GlobleObjBoxesNum, NewObjBoxes, NewObjBoxesNum, names, objectName, num_Obj);

	send2VirtualHuman(NewObjBoxes, &NewObjBoxesNum, strID);

	for (i = 0; i < NewObjBoxesNum; i++)
	{
		RecObects[i] = NewObjBoxes[i].Obj;
	}
	*objectNumPerFrame = NewObjBoxesNum;


	NewObjBoxesNum = 0;
	free(ObjBoxes);
	free(ShowObjBoxes);
	free(objectName);
	free(NewObjBoxes);
}

void draw_even_message(image im, detectBoxes *ObjBoxes, int BoxesNum, image **alphabet, int classes)
{
	int i = 0, j = 0;
	float prob = 0;
	box b;
	int class;
	float distance;
	float X, Y;
	char *strID = "";
	int index = 0;
	detectBoxes *TempObject = calloc(100, sizeof(detectBoxes));

	CvFont font;

	cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.5f, 1.5f, 0, 2, CV_AA);//设置显示的字体
	IplImage *pImg = cvCreateImage(cvSize(960, 128), IPL_DEPTH_8U, 3);
	for (int y = 0; y < 128; ++y){
		for (int x = 0; x < 960; ++x){
			for (int k = 0; k < 3; ++k){
				pImg->imageData[y*pImg->widthStep + x * 3 + k] = 0;
			}
		}
	}

		for (i = 0; i < BoxesNum; i++)
		{
			if (ObjBoxes[i].Obj.flagBelong2Person == 1)
			{
				TempObject[index] = ObjBoxes[i];
				index++;
			}

		}

	
		if (index == 1)
		{
			strID = "You take a ";
			strID = StrJoin(strID, TempObject[0].Obj.name);
			strID = StrJoin(strID, "!");
		}
		else if (index > 1)
		{
			strID = "You take ";
			for (i = 0; i < index - 2; i++)
			{
				strID = StrJoin(strID, TempObject[i].Obj.name);
				strID = StrJoin(strID, ", ");
			}
			strID = StrJoin(strID, TempObject[index - 2].Obj.name);
			strID = StrJoin(strID, " and ");
			strID = StrJoin(strID, TempObject[index - 1].Obj.name);
			strID = StrJoin(strID, "!");
		}
		cvPutText(pImg, strID, cvPoint(50, 65), &font, CV_RGB(255, 255, 255));
		cvNamedWindow("MyImg", CV_WINDOW_AUTOSIZE);
		cvShowImage("MyImg", pImg);
		cvMoveWindow("MyImg", 0, 540);
		cvReleaseImage(&pImg);
		free(TempObject);

}

void draw_even_message_demo_home(int flag)
{
	char *strID = "";
	char *strID1 = "";
	//detectBoxes *TempObject = calloc(100, sizeof(detectBoxes));

	CvFont font;

	cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.5f, 1.5f, 0, 1.5, CV_AA);//设置显示的字体
	IplImage *pImg = cvCreateImage(cvSize(960, 128), IPL_DEPTH_8U, 3);
	//IplImage *pImg1 = cvCreateImage(cvSize(960, 80), IPL_DEPTH_8U, 3);
	for (int y = 0; y < 128; ++y){
		for (int x = 0; x < 960; ++x){
			for (int k = 0; k < 3; ++k){
				pImg->imageData[y*pImg->widthStep + x * 3 + k] = 0;
			//	pImg1->imageData[y*pImg1->widthStep + x * 3 + k] = 0;
			}
		}
	}
	if (flag == 0)
	{
		strID = " ";
	}
	else if (flag == 1)
	{
		strID = "Welcome back home! Tom.";
		//strID1 = "Welcome back home! Tom.";
	}
	else if (flag == 2)
	{
		strID = "Please take off your bag. You can put it on the desk";
	}
	else if (flag == 3)
	{
		strID = "There is a chair on your right. You can sit there and read your book.";
	}
	else if (flag == 4)
	{
		strID = "(Half an hour later) I prepare some drinks for you.";
	}
	else if (flag == 5)
	{
		strID = "This is a bottle of pure water.";
	}
	else if (flag == 6)
	{
		strID = "This is a cup of green tee.";
	}
	else if (flag == 7)
	{
		strID = "(Two hours later) Hi, Tom. You have spend two hours on reading. ";
		strID1 = "Maybe you'd better go out to have a rest.";
	}
	else if (flag == 8)
	{
		strID = "You forget your bag!";
	}
	else if (flag == 9)
	{
		strID = "Have a good time!";
	}

	cvPutText(pImg, strID, cvPoint(50, 50), &font, CV_RGB(255, 255, 255));
	cvPutText(pImg, strID1, cvPoint(50, 95), &font, CV_RGB(255, 255, 255));
	cvNamedWindow("MyImg", CV_WINDOW_AUTOSIZE);
	cvShowImage("MyImg", pImg);
	cvMoveWindow("MyImg", 0, 540);

	//cvPutText(pImg1, strID1, cvPoint(50, 30), &font, CV_RGB(255, 255, 255));
	//cvNamedWindow("MyImg1", CV_WINDOW_AUTOSIZE);
	//cvShowImage("MyImg1", pImg1);
	//cvMoveWindow("MyImg1", 0, 605);
	cvReleaseImage(&pImg);
	//cvReleaseImage(&pImg1);
	////free(TempObject);

}


void draw_object_distance(image im, detectBoxes *ObjBoxes, int BoxesNum, image **alphabet, int classes)
{
	int i = 0;
	float prob = 0;
	box b;
	int class;
	float distance;
	float X, Y;
	float Height, Width;
	char dis[10];
	char xCor[10], yCor[10];
	char HStr[10], WStr[10];
	for (i = 0; i < BoxesNum; i++)
	{
		prob = ObjBoxes[i].Obj.prob;
		b.x = ObjBoxes[i].Obj.x;
		b.y = ObjBoxes[i].Obj.y;
		b.w = ObjBoxes[i].Obj.w;
		b.h = ObjBoxes[i].Obj.h;
		class = ObjBoxes[i].Obj.objClass;
		distance = ObjBoxes[i].Obj.CameraZ;
		X = ObjBoxes[i].Obj.CameraX;
		Y = ObjBoxes[i].Obj.CameraY;
		Height = ObjBoxes[i].Obj.CameraHeight;
		Width = ObjBoxes[i].Obj.CameraWidth;
		sprintf(dis, "z = %3.2f m", distance);
		sprintf(xCor, "x = %3.2f m", X);
		sprintf(yCor, "y = %3.2f m", Y);
		sprintf(HStr, "h = %3.2f m", Height);
		sprintf(WStr, "w = %3.2f m", Width);

		int width = im.h * .012;

		//printf("%s: %.0f%%\n", names[class], prob * 100);
		int offset = class * 123457 % classes;
		float red = get_color(2, offset, classes);
		float green = get_color(1, offset, classes);
		float blue = get_color(0, offset, classes);
		float rgb[3];

		//width = prob*20+2;

		rgb[0] = red;
		rgb[1] = green;
		rgb[2] = blue;

		int left = (b.x - b.w / 2.)*im.w;
		int right = (b.x + b.w / 2.)*im.w;
		int top = (b.y - b.h / 2.)*im.h;
		int bot = (b.y + b.h / 2.)*im.h;

		if (left < 0) left = 0;
		if (right > im.w - 1) right = im.w - 1;
		if (top < 0) top = 0;
		if (bot > im.h - 1) bot = im.h - 1;

		if (alphabet) {
			image label = get_label(alphabet, dis, (im.h*.03) / 10);
			draw_label(im, top - 4 * width, left, label, rgb);
			label = get_label(alphabet, yCor, (im.h*.03) / 10);
			draw_label(im, top - 9 * width, left, label, rgb);
			label = get_label(alphabet, xCor, (im.h*.03) / 10);
			draw_label(im, top - 14 * width, left, label, rgb);
			label = get_label(alphabet, HStr, (im.h*.03) / 10);
			draw_label(im, top - 19 * width, left, label, rgb);
			label = get_label(alphabet, WStr, (im.h*.03) / 10);
			draw_label(im, top - 24 * width, left, label, rgb);
		}

	}
}

void object_show(image im, char **names, image **alphabet, int classes, object *RecObects, int *objectNumPerFrame)
{
	int i, j, k, l;

	int maxBoxesNum = 100;
	detectBoxes *ObjBoxes = calloc(maxBoxesNum, sizeof(detectBoxes));
	int BoxesNum = *objectNumPerFrame;

	for (i = 0; i < BoxesNum; i++)
	{
		ObjBoxes[i].Obj = RecObects[i];
	}

	draw_object_boxes(im, ObjBoxes, BoxesNum, names, alphabet, classes);
	//List_object_name(im, ObjBoxes, BoxesNum, names, alphabet, classes);
	//draw_object_distance(im, ObjBoxes, BoxesNum, alphabet, classes);
	//draw_even_message(im, ObjBoxes, BoxesNum, alphabet, classes);
	free(ObjBoxes);
}

void object_show_grasp(image im, char **names, image **alphabet, int classes, object *RecObects, int *objectNumPerFrame)
{
	int i, j, k, l;

	int maxBoxesNum = 100;
	detectBoxes *ObjBoxes = calloc(maxBoxesNum, sizeof(detectBoxes));
	int BoxesNum = *objectNumPerFrame;
	int r = 0, c = 10;
	int step = 190;

	for (i = 0; i < BoxesNum; i++)
	{
		ObjBoxes[i].Obj = RecObects[i];
		draw_information(im, ObjBoxes[i], names, alphabet, classes, r, c);
		r += step;
	}	

	//draw_object_boxes(im, ObjBoxes, BoxesNum, names, alphabet, classes);
	//draw_object_distance(im, ObjBoxes, BoxesNum, alphabet, classes);
	//draw_even_message(im, ObjBoxes, BoxesNum, alphabet, classes);
	free(ObjBoxes);
}

/*void object_show_grasp(image im, char **names, image **alphabet, int classes, object *RecObects, int *objectNumPerFrame)
{
	image im_backup = copy_image(im);
	for (int i = 0; i < im.h; i++)
		for (int j = 0; j < im.w; j++)
		{
			set_pixel(im, j, i, 0, 0.4);
			set_pixel(im, j, i, 1, 0.4);
			set_pixel(im, j, i, 2, 0.4);
		}
	//draw_line_width(im, 50, 7 * im.h / 8, im.w-50, 7 * im.h / 8, 1, 255, 255, 255);
	//draw_line_width(im, 50, 7 * im.h / 8 + 1, im.w - 50, 7 * im.h / 8 + 1, 1, 255, 255, 255);

	//draw_line_width(im, 100, 6 * im.h / 8, im.w-100, 6 * im.h / 8, 1, 255, 255, 255);
	//draw_line_width(im, 100, 6 * im.h / 8 + 1, im.w - 100, 6 * im.h / 8 + 1, 1, 255, 255, 255);

	//draw_line_width(im, 150, 5 * im.h / 8, im.w - 150, 5 * im.h / 8, 1, 255, 255, 255);
	//draw_line_width(im, 150, 5 * im.h / 8 + 1, im.w - 150, 5 * im.h / 8 + 1, 1, 255, 255, 255);

	//draw_line_width(im, 200, 4 * im.h / 8, im.w - 200, 4 * im.h / 8, 1, 255, 255, 255);
	//draw_line_width(im, 200, 4 * im.h / 8 + 1, im.w - 200, 4 * im.h / 8 + 1, 1, 255, 255, 255);

	for (int i = 0; i < objectNumPerFrame; i++)
	{
		object b = RecObects[i];


	//im = copy_image(im_backup);
	free_image(im_backup);
}*/

void object_show_person(image im, char **names, image **alphabet, int classes, object *RecObects, int *objectNumPerFrame)
{
	int i, j, k, l;

	int maxBoxesNum = 100;
	detectBoxes *ObjBoxes = calloc(maxBoxesNum, sizeof(detectBoxes));
	int BoxesNum = *objectNumPerFrame;

	for (i = 0; i < BoxesNum; i++)
	{
		ObjBoxes[i].Obj = RecObects[i];
	}

	draw_object_boxes(im, ObjBoxes, BoxesNum, names, alphabet, classes);
	//draw_object_distance(im, ObjBoxes, BoxesNum, alphabet, classes);
	//draw_even_message(im, ObjBoxes, BoxesNum, alphabet, classes);
	free(ObjBoxes);
}

void object_show_person_demo_home(image im, char **names, image **alphabet, int classes, object *RecObects, int *objectNumPerFrame, int flag)
{
	int i, j, k, l;

	int maxBoxesNum = 100;
	detectBoxes *ObjBoxes = calloc(maxBoxesNum, sizeof(detectBoxes));
	int BoxesNum = *objectNumPerFrame;

	for (i = 0; i < BoxesNum; i++)
	{
		ObjBoxes[i].Obj = RecObects[i];
	}

	draw_object_boxes(im, ObjBoxes, BoxesNum, names, alphabet, classes);
	//draw_object_distance(im, ObjBoxes, BoxesNum, alphabet, classes);
	//draw_even_message_demo_home(flag);
	free(ObjBoxes);
}

