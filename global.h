#pragma once
#define _SECURE_SCL 0 
#define BOOST_TYPEOF_EMULATION

#define TEST
#define ON_GPU
#define CUDA_TEST

#define MSR3DVideo_Ballet 0
#define Poznan_Fencing 1
#define Intel_Kermit 2
#define Technicolor_Painter 3
#define S01_H1 4
#define S02_H2 5
#define S03_H3 6
#define S04_H4 7
#define S05_R1 8
#define S06_R2 9
#define S07_R3 10
#define S08_R4 11
#define S09_A1 12
#define S10_A2 13

#define MAXNUM_11X11 331
#define MAXNUM_9X9 309
#define MAXNUM_7X7 287
#define MAXNUM_5X5 265

#include "stdio.h"
#include "stdlib.h"
#include <iostream>
#include "opencv.hpp"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/eigen.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <math.h>
#include <list>
#include <ctime>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <bitset>
#include <set>
#include <tchar.h>
#include <Windows.h>
#include <psapi.h>
#include <conio.h>

#if defined(ON_GPU) || defined(CUDA_TEST)
#include "CUDA.cuh"
#endif

using namespace pcl;
using namespace std;
using namespace io;
using namespace cv;
using namespace Eigen;
using namespace visualization;

typedef struct {
	Matrix3d m_K;
	Matrix3d m_RotMatrix;
	Matrix3Xd m_Trans;
	Matrix4d m_ProjMatrix;
} CalibStruct;

extern int data_mode, _width, _height, total_num_cameras, total_num_frames;
extern double MinZ, MaxZ, scaleZ;
extern vector<Vector2d> tech_minmaxZ;
extern string path;
extern vector<CalibStruct> m_CalibParams;
extern int proj_mode;

#if defined(ON_GPU) || defined(CUDA_TEST)
extern CUDA CudaGpu;
#endif

class PPC {
private:
	float* geometry;

public:
	//v2
	virtual bool CheckOcclusion(int idx) { return false; }
	virtual Vec3b GetColor(int idx) { return { 0, 0, 0 }; }
	virtual vector<uchar> GetVU() { return { 0 }; }
	virtual vector<uchar> GetY() { return { 0 }; }
	virtual uchar GetrefV() { return 0; }
	virtual uchar GetrefU() { return 0; }
	virtual float* GetGeometry() { return NULL; }
	virtual void SetGeometry(PointXYZRGB p) {}
	virtual void SetGeometry(float* geo) {}
	virtual void SetOcclusion(int idx) {}
	virtual void SetRefColor(Vec3b color, int idx) {}
	virtual void SetRefColor(PointXYZRGB point, int idx) {}
	virtual void SetColor(Vec3b color, int idx) {}
	virtual void SetColor(uchar v_, uchar u_, uchar y_, int idx) {}
	virtual void SetColor(uchar refv, uchar refu, vector<uchar> vu, vector<uchar> y) {}
	virtual void SetOcclusionZero() {}
	virtual void SetZero() {}
	//v2.2
	virtual int GetColorNum() { return 0; }
	virtual uchar GetV() { return 0; }
	virtual uchar GetU() { return 0; }
	virtual vector<bool> GetOcclusion() { return { false }; }
	virtual void SetColor(PointXYZRGB point, int idx) {}
	virtual void SetColor(uchar avrv, uchar avru, vector<uchar> y, vector<bool> occ) {}

};

class PPC_v1 : public PPC {
private:
	float* geometry;
	vector<uchar> Y, U, V;
	vector<bool> occlusion;

public:

	PPC_v1()
	{
		geometry = (float*)malloc(3 * sizeof(float));
		Y = vector<uchar>(total_num_cameras);
		U = vector<uchar>(total_num_cameras);
		V = vector<uchar>(total_num_cameras);
		occlusion = vector<bool>(total_num_cameras, false);
	}

	bool CheckOcclusion(int idx)
	{
		return !occlusion[idx];
	}

	void SetZero()
	{
		for (int cam = 0; cam < total_num_cameras; cam++) {
			occlusion[cam] = false;
			Y[cam] = 0;
			U[cam] = 0;
			V[cam] = 0;
		}
	}

	vector<bool> GetOcclusion() {
		return occlusion;
	}

	Vec3b GetColor(int idx)
	{
		Vec3b vuy;
		vuy[0] = V[idx];
		vuy[1] = U[idx];
		vuy[2] = Y[idx];

		return vuy;
	}

	int GetColorNum() {
		int num = 0;
		for (int i = 0; i < total_num_cameras; i++)
			if (!CheckOcclusion(i)) num++;
		return num;
	}

	float* GetGeometry() {
		return geometry;
	}

	void SetGeometry(PointXYZRGB p) {

		geometry[0] = p.x;
		geometry[1] = p.y;
		geometry[2] = p.z;
	}

	void SetGeometry(float* geo) {
		geometry[0] = geo[0];
		geometry[1] = geo[1];
		geometry[2] = geo[2];
	}

	void SetColor(Vec3b color, int idx)
	{
		Y[idx] = color[0];
		U[idx] = color[1];
		V[idx] = color[2];
		occlusion[idx] = true;
	}

	void SetColor(PointXYZRGB point, int idx) {
		V[idx] = point.r;
		U[idx] = point.g;
		Y[idx] = point.b;
		occlusion[idx] = true;
	}
};

class PPC_v2_1 : public PPC {
private:
	float* geometry;
	uchar refV;
	uchar refU;
	vector<uchar> VU;
	vector<uchar> Y;

public:

	PPC_v2_1()
	{
		// occlusion���� �ʱ�ȭ.
		geometry = (float*)malloc(3 * sizeof(float));
		VU = vector<uchar>(total_num_cameras, uchar(255));
		Y = vector<uchar>(total_num_cameras);
	}

	bool CheckOcclusion(int idx)
	{
		return int(VU[idx]) == 255;
	}

	Vec3b GetColor(int idx)
	{
		Vec3b vuy;
		int firstCamNum = 0;
		for (int i = 0; i < total_num_cameras; i++) {
			if (!CheckOcclusion(i)) {
				firstCamNum = i;
				break;
			}
		}

		uchar v = refV;
		uchar u = refU;
		for (int i = firstCamNum + 1; i <= idx; i++) {
			if (!CheckOcclusion(i)) {
				v += (VU[i] >> 4) - 7;
				u += (uchar(VU[i] << 4) >> 4) - 7;
			}
		}
		vuy[0] = v;
		vuy[1] = u;

		vuy[2] = Y[idx];
		return vuy;
	}

	int GetColorNum() {
		int num = 0;
		for (int i = 0; i < total_num_cameras; i++)
			if (!CheckOcclusion(i)) num++;
		return num;
	}

	///save, load 함수에 쓰임
	vector<uchar> GetVU() {
		return VU;
	}

	vector<uchar> GetY() {
		return Y;
	}

	uchar GetrefV() {
		return refV;
	}

	uchar GetrefU() {
		return refU;
	}

	float* GetGeometry() {
		return geometry;
	}

	void SetGeometry(PointXYZRGB p) {

		geometry[0] = p.x;
		geometry[1] = p.y;
		geometry[2] = p.z;

	}

	void SetGeometry(float* geo) {
		geometry[0] = geo[0];
		geometry[1] = geo[1];
		geometry[2] = geo[2];
	}

	void SetOcclusion(int idx) {
		VU[idx] = uchar(255);
	}

	// color��  YUV
	void SetRefColor(Vec3b color, int idx)
	{
		//cout << (int)color[2] << " " << (int)color[1] << " " << (int)color[0] << endl;
		refV = color[2];
		refU = color[1];
		Y[idx] = color[0];
		VU[idx] = 0;
	}

	// color�� VUY
	void SetRefColor(PointXYZRGB point, int idx)
	{

		refV = point.r;
		refU = point.g;
		Y[idx] = point.b;
		VU[idx] = 0;
	}

	// color�� yuv
	void SetColor(Vec3b color, int idx)
	{
		int firstCamNum = 0;
		for (int i = 0; i < total_num_cameras; i++) {
			if (!CheckOcclusion(i)) {
				firstCamNum = i;
				break;
			}
		}

		uchar v = refV;
		uchar u = refU;
		for (int i = firstCamNum + 1; i < idx; i++) {
			if (!CheckOcclusion(i)) {

				v += (VU[i] >> 4) - 7;
				u += (uchar(VU[i] << 4) >> 4) - 7;

			}
		}
		//if (color[2] - v + 7 > 16)
		VU[idx] = uchar((color[2] - v + 7) << 4) + uchar(color[1] - u + 7);
		Y[idx] = color[0];
	}


	void SetColor(uchar v_, uchar u_, uchar y_, int idx)
	{
		int firstCamNum = 0;
		for (int i = 0; i < total_num_cameras; i++) {
			if (!CheckOcclusion(i)) {
				firstCamNum = i;
				break;
			}
		}

		uchar v = refV;
		uchar u = refU;
		for (int i = firstCamNum + 1; i < idx; i++) {
			if (!CheckOcclusion(i)) {
				v += (VU[i] >> 4) - 7;
				u += (uchar(VU[i] << 4) >> 4) - 7;
			}
		}

		VU[idx] = uchar(uchar((v_ - v + 7) << 4) + (u_ - u + 7));
		Y[idx] = y_;
	}

	void SetColor(uchar refv, uchar refu, vector<uchar> vu, vector<uchar> y) {
		refV = refv;
		refU = refu;
		VU = vu;
		Y = y;
	}
};

class PPC_v2_2 : public PPC {
private:
	float* geometry;
	uchar avrV;
	uchar avrU;
	vector<uchar> Y;
	vector<bool> occlusion;
public:

	PPC_v2_2()
	{
		// occlusion���� �ʱ�ȭ.
		geometry = (float*)malloc(3 * sizeof(float));
		Y = vector<uchar>(total_num_cameras);
		occlusion = vector<bool>(total_num_cameras, false);
	}

	bool CheckOcclusion(int idx)
	{
		return !occlusion[idx];
	}

	uchar GetV() {
		//uchar avrV = new uchar[1];
		return avrV;
	}

	uchar GetU() {
		//uchar* avrU = new uchar[1];
		return avrU;
	}

	vector<uchar> GetY() {
		return Y;
	}

	Vec3b GetColor(int idx)
	{
		Vec3b vuy;
		vuy[0] = avrV;
		vuy[1] = avrU;
		vuy[2] = Y[idx];

		return vuy;
	}

	float* GetGeometry() {
		return geometry;
	}

	vector<bool> GetOcclusion() {
		return occlusion;
	}

	void SetGeometry(PointXYZRGB p) {
		geometry[0] = p.x;
		geometry[1] = p.y;
		geometry[2] = p.z;

	}
	void SetGeometry(float* geo) {
		geometry[0] = geo[0];
		geometry[1] = geo[1];
		geometry[2] = geo[2];
	}

	void SetOcclusion(int idx) {
		occlusion[idx] = true;
	}

	int GetColorNum() {
		int num = 0;
		for (int i = 0; i < total_num_cameras; i++)
			if (occlusion[i]) num++;

		return num;
	}

	// color��  YUV
	void SetColor(Vec3b color, int idx)
	{
		int colorNum = GetColorNum();
		if (colorNum == 0) {
			avrV = color[2];
			avrU = color[1];
		}
		else {
			avrV = uchar(avrV * (colorNum / float(colorNum + 1)) + (color[2] / float(colorNum + 1)));
			avrU = uchar(avrU * (colorNum / float(colorNum + 1)) + (color[1] / float(colorNum + 1)));
		}
		Y[idx] = color[0];
		occlusion[idx] = true;
	}

	//VUY
	void SetColor(PointXYZRGB point, int idx) {
		int colorNum = GetColorNum();
		if (colorNum == 0) {
			avrV = point.r;
			avrU = point.g;
		}
		else {
			avrV = uchar(avrV * (colorNum / float(colorNum + 1)) + (point.r / float(colorNum + 1)));
			avrU = uchar(avrU * (colorNum / float(colorNum + 1)) + (point.g / float(colorNum + 1)));
		}
		Y[idx] = point.b;
		occlusion[idx] = true;
	}

	void SetColor(uchar avrv, uchar avru, vector<uchar> y, vector<bool> occ) {
		avrV = avrv;
		avrU = avru;
		Y = y;
		occlusion = occ;
	}
};

