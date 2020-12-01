#pragma once

#include "global.h"
#include "common.h"

extern int _width, _height, total_num_cameras, total_num_frames;
extern double MinZ, MaxZ, scaleZ;
extern string path;
extern vector<CalibStruct> m_CalibParams;
extern vector<int> camera_order;

void set_parameters(int data_mode, int &color_bits, int &depth_bits, int& mask_size);
void load_matrix_data();
void compute_projection_matrices();
void load_file_name(vector<vector<string>> &color_names, vector<vector<string>> &depth_names);
void load_file_name(vector<string> &color_names_, vector<string> &depth_names_, int depth_bits);
void load_file_name_mode4(vector<vector<string>>& color_names, vector<vector<string>>& depth_names, int refenceView, int furthest_index);
void GetRotationMat(Vector3d& euler, Matrix3d& rotationMat);
void Quaternion2RotationMat(Vector4d& quaternion, Matrix3d& rotationMat);
void Euler2RotationMat(Vector3d& euler, Matrix3d& rotationMat);
void get_RT_data_json(const char* file, vector<Vector3d>& Rotation_vec, vector<Vector3d>& Position_vec, vector<Vector3d>& KFocal_vec, vector<Vector3d>& KPrinciple_vec, int total_num_cameras);
void get_RT_data_json(
	const char* file,
	vector<Vector3d>& Rotation_vec,
	vector<Vector3d>& Position_vec,
	vector<Vector2d>& Depth_vec,
	vector<Vector3d>& KFocal_vec,
	vector<Vector3d>& KPrinciple_vec,
	int total_num_cameras);