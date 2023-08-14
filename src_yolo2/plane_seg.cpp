#include <iostream>
#include <stdio.h>
#include <vector>
#include <time.h>
#include <algorithm>
#include <pcl\point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
//#include <boost/thread/thread.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;
typedef pcl::PointXYZ PointInT;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;
typedef struct{
	float x;
	float y;
	float z;
}Vertex;
using namespace std;
/*boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
viewer->setBackgroundColor (255, 255, 255);
pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
//viewer->addCoordinateSystem (1.0);
viewer->initCameraParameters ();
return (viewer);
}
void Seg_viewer(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,pcl::PointIndices::Ptr inlier_Idx)
{
int i;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_RGB (new pcl::PointCloud<pcl::PointXYZRGB>);
cloud_RGB->points.resize(cloud->points.size());
////////////////////////////////////////////////////////
uint8_t r1(0), g1 = (255), b1 = (0), r2(255), g2 = (0), b2 = (0);
uint32_t rgb1 = (static_cast<uint32_t>(r1) << 16 |
static_cast<uint32_t>(g1) << 8 | static_cast<uint32_t>(b1));
uint32_t rgb2 = (static_cast<uint32_t>(r2) << 16 |
static_cast<uint32_t>(g2) << 8 | static_cast<uint32_t>(b2));
///////////////////////////////////////////////////////
for (i = 0; i <  cloud->points.size(); i++)
{
cloud_RGB->points[i].x = cloud->points[i].x;
cloud_RGB->points[i].y = cloud->points[i].y;
cloud_RGB->points[i].z = cloud->points[i].z;
cloud_RGB->points[i].rgb=*reinterpret_cast<float*>(&rgb1);

}
for (i = 0; i < inlier_Idx->indices.size(); i++)
cloud_RGB->points[inlier_Idx->indices[i]].rgb=*reinterpret_cast<float*>(&rgb2);
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_rgb;
viewer_rgb = rgbVis(cloud_RGB);
while (!viewer_rgb->wasStopped ())
{
viewer_rgb->spinOnce();
}
}*/
void Seg_viewer(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointIndices::Ptr inlier_Idx)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_seg(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);
	vector<int>labels(cloud->points.size(), 0);
	for (int i = 0; i<inlier_Idx->indices.size(); i++)
		labels[inlier_Idx->indices[i]] = 1;
	for (int i = 0; i<cloud->points.size(); i++)
	{
		if (labels[i] == 0)
			cloud_seg->points.push_back(cloud->points[i]);
		else
			cloud_plane->points.push_back(cloud->points[i]);
	}
	pcl::visualization::PCLVisualizer viewer("Seg Viewer");
	viewer.setBackgroundColor(255, 255, 255);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> view_cloud_plane(cloud_plane, 255, 0, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> view_cloud_seg(cloud_seg, 0, 255, 0);
	viewer.addPointCloud(cloud_plane, view_cloud_plane, "plane");
	viewer.addPointCloud(cloud_seg, view_cloud_seg, "seg");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "plane");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "seg");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}

int XYZorPly_Read(string Filename, PointCloudPtr&cloud)
{
	int i, j, k;
	int nXYZ_nums;
	std::vector<Vertex> vXYZ;
	FILE *fp = fopen(Filename.c_str(), "r");
	if (fp == NULL)
	{
		printf("File can't open!\n");
		return -1;
	}
	const char*FILEPATH = Filename.c_str();
	char a = FILEPATH[strlen(FILEPATH) - 1];
	//
	if (a == 'y')
	{
		char str[1024];
		fscanf(fp, "%s\n", &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %d\n", &str, &str, &nXYZ_nums);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %s %s %s\n", &str, &str, &str, &str, &str);
		fscanf(fp, "%s\n", &str);
	}
	else
	{
		fscanf(fp, "%d\n", &nXYZ_nums);
	}
	vXYZ.resize(nXYZ_nums);
	for (i = 0; i < vXYZ.size(); i++)
	{
		fscanf(fp, "%f %f %f\n", &vXYZ[i].x, &vXYZ[i].y, &vXYZ[i].z);
	}
	fclose(fp);
	cloud->width = vXYZ.size();
	cloud->height = 1;
	cloud->is_dense = true;
	cloud->points.resize(cloud->width*cloud->height);
	for (i = 0; i < cloud->points.size(); i++)
	{
		cloud->points[i].x = vXYZ[i].x;
		cloud->points[i].y = vXYZ[i].y;
		cloud->points[i].z = vXYZ[i].z;
	}
	return 0;
}
void Cloud_save(PointCloudPtr cloud, string file_out)
{
	FILE*fp = fopen(file_out.c_str(), "w");
	fprintf(fp, "%d\n", cloud->points.size());
	for (int i = 0; i<cloud->points.size(); i++)
		fprintf(fp, "%f %f %f\n", cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
	fclose(fp);
}
void plane_segmentation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, UINT16 *depthBuffer, UINT16 depthHeight, UINT16 depthWidth)
{
	/****2 inputs: read_file_dir  write_file_dir (all in .xyz format) *****/
	//string file_in = "E:\\NTU\\code\\PCL_seg\\depth.xyz";
	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	//XYZorPly_Read(file_in, cloud);

	//clock_t begin_seg, end_seg;
	//double  seg_cost;
	//begin_seg = clock();
	float distance_thresh = 0.02;//Key parameter, tuned based on application
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(distance_thresh);
	seg.setInputCloud(cloud);
	seg.segment(*inliers, *coefficients);
	//end_seg = clock();
	//seg_cost = (double)(end_seg - begin_seg) / CLOCKS_PER_SEC;
	//printf("Time: %.3f s\n", seg_cost);
	//visualization
	//Seg_viewer(cloud, inliers);
	//save segmented_data
	/*string file_out = "seg.xyz";
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_seg(new pcl::PointCloud<pcl::PointXYZ>);
	vector<int>labels(cloud->points.size(), 0);
	for (int i = 0; i<inliers->indices.size(); i++)
		labels[inliers->indices[i]] = 1;
	for (int i = 0; i<cloud->points.size(); i++)
	{
		if (labels[i] == 0)
			cloud_seg->points.push_back(cloud->points[i]);
	}
	Cloud_save(cloud_seg, file_out);*/
	//return (0);

	vector<int>labels(cloud->points.size(), 0);
	for (int i = 0; i<inliers->indices.size(); i++)
		labels[inliers->indices[i]] = 1;
//	for (int i = 0; i<cloud->points.size(); i++)
//	{
		//if (labels[i] == 1)
		//	depthBuffer[i] = 0;
//		depthBuffer[inliers->indices[i]] = 0;
//	}
	int index = 0;
	for (int i = 0; i < depthHeight*depthWidth; i++)
	{
		if (depthBuffer[i] == 0) continue;
		if (labels[index] == 1)	depthBuffer[i] = 0;
		index++;
	}

}


void plane_segmentation_test()
{
	/****2 inputs: read_file_dir  write_file_dir (all in .xyz format) *****/
	string file_in = "E:\\NTU\\code\\PCL_seg\\1(1).xyz";
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	XYZorPly_Read(file_in, cloud);

	//
	clock_t begin_seg, end_seg;
	double  seg_cost;
	begin_seg = clock();
	float distance_thresh = 0.01;//Key parameter, tuned based on application
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(distance_thresh);
	seg.setInputCloud(cloud);
	seg.segment(*inliers, *coefficients);
	end_seg = clock();
	seg_cost = (double)(end_seg - begin_seg) / CLOCKS_PER_SEC;
	printf("Time: %.3f s\n", seg_cost);
	//visualization
	Seg_viewer(cloud, inliers);
	//save segmented_data
	/*string file_out = "seg.xyz";
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_seg(new pcl::PointCloud<pcl::PointXYZ>);
	vector<int>labels(cloud->points.size(), 0);
	for (int i = 0; i<inliers->indices.size(); i++)
	labels[inliers->indices[i]] = 1;
	for (int i = 0; i<cloud->points.size(); i++)
	{
	if (labels[i] == 0)
	cloud_seg->points.push_back(cloud->points[i]);
	}
	Cloud_save(cloud_seg, file_out);*/
	//return (0);

	/*	vector<int>labels(cloud->points.size(), 0);
	for (int i = 0; i<inliers->indices.size(); i++)
	labels[inliers->indices[i]] = 1;
	for (int i = 0; i<cloud->points.size(); i++)
	{
	if (labels[i] == 1)
	depthBuffer[i] = 0;
	}
	*/
}

