#include "set_environment.h"
#include "rapidjson/document.h"     // rapidjson's DOM-style API
#include "rapidjson/prettywriter.h" // for stringify JSON
#include <rapidjson/filereadstream.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdio>

using namespace rapidjson;
using namespace std;
using namespace cv;

#define PI 3.14159265

Vector3d rad2deg(Vector3d radian);
Vector3d deg2rad(Vector3d degree);
string rootDir = "D:\\ETRI\\";

void set_parameters(int data_mode, int& color_bits, int& depth_bits, int & mask_size)
{
	switch (data_mode)
	{
	case MSR3DVideo_Ballet:
		total_num_cameras = 8;
		total_num_frames = 100;

		_width = 1024;
		_height = 768;

		MinZ = 42.0;
		MaxZ = 130.0;

		//scaleZ = 0;
		color_bits = 8;
		depth_bits = 8;

		path = rootDir + "MSR3DVideo-Ballet";

		break;

	case Poznan_Fencing:
		total_num_cameras = 10;
		total_num_frames = 250;

		_width = 1920;
		_height = 1080;


		MinZ = 3.5;
		MaxZ = 7.0;
		//scaleZ = 0;

		color_bits = 8;
		depth_bits = 16;

		path = rootDir + "Poznan_Fencing";

		break;

	case Intel_Kermit:
		total_num_cameras = 13;
		//total_num_cameras = 10;
		total_num_frames = 300;

		_width = 1920;
		_height = 1080;

		MinZ = 0.3;
		MaxZ = 1.62;
		//scaleZ = 0;

		color_bits = 10;
		depth_bits = 16;

		path = rootDir + "Intel_Kermit";

		break;

	case Technicolor_Painter:
		total_num_cameras = 16;
		//total_num_cameras = 10;
		total_num_frames = 372;

		_width = 2048;
		_height = 1088;

		MinZ = 1.773514;
		MaxZ = 5.300389;
		//scaleZ = 0;

		color_bits = 10;
		depth_bits = 16;

		path = rootDir + "Technicolor_Painter";
		break;

	case S01_H1:case S02_H2:case S03_H3:case S04_H4:
		//total_num_cameras = 21 * 21;
		//total_num_cameras = 49; //121
		total_num_cameras = mask_size * mask_size;
		total_num_frames = 1;

		_width = 3840;
		_height = 2160;

		MinZ = 0;
		MaxZ = 0;
		scaleZ = 10000;

		color_bits = 10;
		depth_bits = 16;

		if (data_mode == S01_H1) path = rootDir + "S01_H1";
		else if (data_mode == S02_H2) path = rootDir + "S02_H2";
		else if (data_mode == S03_H3) path = rootDir + "S03_H3";
		else if (data_mode == S04_H4) path = rootDir + "S04_H4";
		break;
	case S05_R1:case S06_R2:case S07_R3:case S08_R4:
		total_num_cameras = mask_size * mask_size;
		total_num_frames = 1;

		_width = 3840;
		_height = 2160;

		MinZ = 0;
		MaxZ = 0;
		scaleZ = 1000;

		color_bits = 10;
		depth_bits = 16;

		if (data_mode == S05_R1) path = rootDir + "S05_R1";
		else if (data_mode == S06_R2) path = rootDir + "S06_R2";
		else if (data_mode == S07_R3) path = rootDir + "S07_R3";
		else if (data_mode == S08_R4) path = rootDir + "S08_R4";
		break;

	case S09_A1:case S10_A2:

		total_num_cameras = mask_size * mask_size;
		total_num_frames = 1;

		_width = 3840;
		_height = 2160;

		MinZ = 0;
		MaxZ = 0;
		scaleZ = 10000;

		color_bits = 10;
		depth_bits = 16;

		if (data_mode == S09_A1) path = rootDir + "S09_A1";
		else if (data_mode == S10_A2) path = rootDir + "S10_A2";
		break;
	default:
		cerr << "Wrong data_mode!!!" << endl;
		exit(0);
	}
}

void load_matrix_data()
{

	if (data_mode == MSR3DVideo_Ballet || data_mode == Poznan_Fencing) {
		string matrix_path = path + "\\*.txt";

		intptr_t matrix_handle;

		struct _finddata_t matrix_fd;

		if ((matrix_handle = _findfirst(matrix_path.c_str(), &matrix_fd)) == -1L)
			cout << "No file in directory!" << endl;

		string matrixfile;
		matrixfile = path + "\\" + matrix_fd.name;
		ifstream openFile(matrixfile);

		if (!openFile.is_open())
		{
			cerr << "Failed to open " << endl;
			exit(EXIT_FAILURE);
		}

		double col0, col1, col2, col3;
		int row_count = 0;
		int camera_idx = 0;
		string buffer;
		vector<CalibStruct> temp_CalibParams(total_num_cameras);

		Matrix3Xd temp(3, 1);

		temp << 0, 0, 0;

		for (int camera_idx = 0; camera_idx < total_num_cameras; camera_idx++)
			temp_CalibParams[camera_idx].m_Trans = temp;

		while (!openFile.eof() && total_num_cameras != camera_idx)
		{
			getline(openFile, buffer);

			//   get intrinsics
			while (openFile >> col0 >> col1 >> col2)
			{
				temp_CalibParams[camera_idx].m_K(row_count, 0) = col0;
				temp_CalibParams[camera_idx].m_K(row_count, 1) = col1;
				temp_CalibParams[camera_idx].m_K(row_count, 2) = col2;

				if (row_count > 1) {
					row_count = 0;
					break;
				}
				row_count++;
			}

			// skip distortion coefficient
			getline(openFile, buffer);
			getline(openFile, buffer);
			if (data_mode == Poznan_Fencing) getline(openFile, buffer);

			// get extrinsics
			while (openFile >> col0 >> col1 >> col2 >> col3)
			{
				temp_CalibParams[camera_idx].m_RotMatrix(row_count, 0) = col0;
				temp_CalibParams[camera_idx].m_RotMatrix(row_count, 1) = col1;
				temp_CalibParams[camera_idx].m_RotMatrix(row_count, 2) = col2;
				temp_CalibParams[camera_idx].m_Trans(row_count, 0) = col3;

				if (row_count > 1) {
					row_count = 0;
					break;
				}
				row_count++;
			}

			if (data_mode) {
				temp_CalibParams[camera_idx].m_Trans = -1 * temp_CalibParams[camera_idx].m_RotMatrix * temp_CalibParams[camera_idx].m_Trans;
			}

			getline(openFile, buffer);
			getline(openFile, buffer);

			camera_idx++;
		}

		m_CalibParams = temp_CalibParams;

		if (data_mode) {
			int ref = total_num_cameras / 2;
			Matrix3d refR = m_CalibParams[ref].m_RotMatrix;
			Matrix3Xd refT(3, 1);
			refT = m_CalibParams[ref].m_Trans;

			Matrix3Xd refRT(3, 4);
			refRT.col(0) = m_CalibParams[ref].m_RotMatrix.col(0);
			refRT.col(1) = m_CalibParams[ref].m_RotMatrix.col(1);
			refRT.col(2) = m_CalibParams[ref].m_RotMatrix.col(2);
			refRT.col(3) = m_CalibParams[ref].m_Trans.col(0);

			Matrix4d refRT4x4;
			refRT4x4.row(0) = refRT.row(0);
			refRT4x4.row(1) = refRT.row(1);
			refRT4x4.row(2) = refRT.row(2);
			refRT4x4.row(3) << 0, 0, 0, 1;

			for (int cam_num = 0; cam_num < total_num_cameras; cam_num++)
			{
				m_CalibParams[cam_num].m_Trans = m_CalibParams[cam_num].m_RotMatrix * (-refR.inverse() * refT) + m_CalibParams[cam_num].m_Trans;
				m_CalibParams[cam_num].m_RotMatrix *= refR.inverse();

				m_CalibParams[cam_num].m_ProjMatrix = compute_projection_matrices(cam_num);
			}
		}

		openFile.close();
		_findclose(matrix_handle);


	}
	else if (data_mode == Intel_Kermit || data_mode == Technicolor_Painter) {
		string matrix_path;
		vector<CalibStruct> temp_CalibParams(total_num_cameras);
		if (data_mode == Intel_Kermit)
			matrix_path = path + "\\IntelKermit.json";

		else if (data_mode == Technicolor_Painter)
			matrix_path = path + "\\TechnicolorPainter.json";

		char fileName[100];
		strcpy(fileName, matrix_path.c_str());

		vector<Vector3d> R_vec;
		vector<Vector3d> P_vec;
		vector<Vector2d> DR_vec;
		vector<Vector3d> KF_vec;
		vector<Vector3d> KP_vec;
		if (data_mode == Intel_Kermit /*|| mode == Technicolor_Painter*/)
			get_RT_data_json(fileName, R_vec, P_vec, KF_vec, KP_vec, total_num_cameras);
		else if (data_mode == Technicolor_Painter)
			get_RT_data_json(fileName, R_vec, P_vec, DR_vec, KF_vec, KP_vec, total_num_cameras);

		for (int camera_idx = 0; camera_idx < total_num_cameras; camera_idx++) {
			temp_CalibParams[camera_idx].m_K(0, 0) = KF_vec[camera_idx][0];
			temp_CalibParams[camera_idx].m_K(1, 1) = KF_vec[camera_idx][1];
			temp_CalibParams[camera_idx].m_K(0, 2) = KP_vec[camera_idx][0];
			temp_CalibParams[camera_idx].m_K(1, 2) = KP_vec[camera_idx][1];
			temp_CalibParams[camera_idx].m_K(2, 2) = 1.0;

			//Quaternion2RotationMat(R_vec[camera_idx], temp_CalibParams[camera_idx].m_RotMatrix);
			//temp_CalibParams[camera_idx].m_RotMatrix = Matrix3d::Identity();
			Euler2RotationMat(R_vec[camera_idx], temp_CalibParams[camera_idx].m_RotMatrix);
			//GetRotationMat(R_vec[camera_idx], temp_CalibParams[camera_idx].m_RotMatrix);

			//cout << "V[" << camera_idx << "] T :: " << P_vec[camera_idx] << endl<<endl;

		 //      cout << "V[" << camera_idx << "] R :: " << R_vec[camera_idx] << endl<<endl;

			Vector3d temp;
			temp << P_vec[camera_idx][1], P_vec[camera_idx][2], P_vec[camera_idx][0];

			P_vec[camera_idx] = temp;
			temp_CalibParams[camera_idx].m_Trans = P_vec[camera_idx];
			//temp_CalibParams[camera_idx].m_Trans = -1 * temp_CalibParams[camera_idx].m_RotMatrix.transpose() * P_vec[camera_idx];
		}

		m_CalibParams = temp_CalibParams;

		int ref = total_num_cameras / 2;
		Matrix3d refR = m_CalibParams[ref].m_RotMatrix;
		Matrix3Xd refT(3, 1);
		refT = m_CalibParams[ref].m_Trans;

		Matrix3Xd refRT(3, 4);
		refRT.col(0) = m_CalibParams[ref].m_RotMatrix.col(0);
		refRT.col(1) = m_CalibParams[ref].m_RotMatrix.col(1);
		refRT.col(2) = m_CalibParams[ref].m_RotMatrix.col(2);
		refRT.col(3) = m_CalibParams[ref].m_Trans.col(0);

		Matrix4d refRT4x4;
		refRT4x4.row(0) = refRT.row(0);
		refRT4x4.row(1) = refRT.row(1);
		refRT4x4.row(2) = refRT.row(2);
		refRT4x4.row(3) << 0, 0, 0, 1;

		for (int cam_num = 0; cam_num < total_num_cameras; cam_num++)
		{
			m_CalibParams[cam_num].m_Trans = m_CalibParams[cam_num].m_RotMatrix * (-refR.inverse() * refT) + m_CalibParams[cam_num].m_Trans;
			m_CalibParams[cam_num].m_RotMatrix *= refR.inverse();

			m_CalibParams[cam_num].m_ProjMatrix = compute_projection_matrices(cam_num);
		}

		if (data_mode == Technicolor_Painter) {
			tech_minmaxZ = DR_vec;
		
		}
	}
	else if (data_mode >= S01_H1)
	{
		vector<CalibStruct> temp_CalibParams(total_num_cameras);
		vector<Vector3d> R_vec;
		vector<Vector3d> P_vec;

		string filename = "cam_pose.txt";

		string matrixfile;
		matrixfile = path + "\\" + filename;
		ifstream openFile(matrixfile);

		if (!openFile.is_open())
		{
			cerr << "Failed to open " << endl;
			exit(EXIT_FAILURE);
		}

		string buffer;
		//R
		getline(openFile, buffer);
		while (!openFile.eof())
		{
			getline(openFile, buffer);
			std::string delimiter = "\t";
			//cout << "line:: " << buffer << endl;

			size_t pos = 0;
			std::string token;
			int token_int = 0;

			Vector3d R_ = { 0, 0, 0 };
			Vector3d P_ = { 0, 0, 0 };

			int var_idx = 0;
			while ((pos = buffer.find(delimiter)) != std::string::npos) {
				token = buffer.substr(0, pos);

				token_int = stoi(token);
				switch (var_idx)
				{
				case 0:
					break;
				case 1://Px
					P_[0] = token_int;
					break;
				case 2://Py
					P_[1] = token_int;
					break;
				case 3 ://Pz
					P_[2] = token_int;
					break;
				case 4://Rx
					R_[0] = token_int;
					break;
				case 5://Ry
					R_[1] = token_int;
					break;
				case 6://Rz
					R_[2] = token_int;
					break;
				}
				//std::cout << token_int << std::endl;

				buffer.erase(0, pos + delimiter.length());
				var_idx++;
			}

			// cm to m
			P_ *= 0.01;

			P_vec.push_back(P_);
			R_vec.push_back(R_);
		}

		if (camera_order[0] != 0) {
			for (int i = 0; i < total_num_cameras; i++)
			{
				int camera_idx = camera_order[i];

				float f = 12.6037245f;//12.604f
				float w = 36.f;//36.0f;
				float h = 20.25f;//20.0f;

				temp_CalibParams[i].m_K(0, 0) = f * (_width / w);
				temp_CalibParams[i].m_K(0, 1) = 0;
				temp_CalibParams[i].m_K(0, 2) = _width / 2.f;
				temp_CalibParams[i].m_K(1, 0) = 0;
				temp_CalibParams[i].m_K(1, 1) = f * (_height / h);
				temp_CalibParams[i].m_K(1, 2) = _height / 2.f;
				temp_CalibParams[i].m_K(2, 0) = 0;
				temp_CalibParams[i].m_K(2, 1) = 0;
				temp_CalibParams[i].m_K(2, 2) = 1; //homo

				//R: Euler 2 R_3*3
				Euler2RotationMat(R_vec[camera_idx], temp_CalibParams[i].m_RotMatrix);

				temp_CalibParams[i].m_Trans = -1 * temp_CalibParams[i].m_RotMatrix.transpose() * P_vec[camera_idx];

			}

			m_CalibParams = temp_CalibParams;

			int ref = 0;
			Matrix3d refR = m_CalibParams[ref].m_RotMatrix;
			Matrix3Xd refT(3, 1);
			refT = m_CalibParams[ref].m_Trans;

			Matrix3Xd refRT(3, 4);
			refRT.col(0) = m_CalibParams[ref].m_RotMatrix.col(0);
			refRT.col(1) = m_CalibParams[ref].m_RotMatrix.col(1);
			refRT.col(2) = m_CalibParams[ref].m_RotMatrix.col(2);
			refRT.col(3) = m_CalibParams[ref].m_Trans.col(0);

			Matrix4d refRT4x4;
			refRT4x4.row(0) = refRT.row(0);
			refRT4x4.row(1) = refRT.row(1);
			refRT4x4.row(2) = refRT.row(2);
			refRT4x4.row(3) << 0, 0, 0, 1;

			for (int cam_num = 0; cam_num < total_num_cameras; cam_num++)
			{
				m_CalibParams[cam_num].m_Trans = m_CalibParams[cam_num].m_RotMatrix * (-refR.inverse() * refT) + m_CalibParams[cam_num].m_Trans;
				m_CalibParams[cam_num].m_RotMatrix *= refR.inverse();

				m_CalibParams[cam_num].m_ProjMatrix = compute_projection_matrices(cam_num);
			}
		}
		else {

			for (int camera_idx = 0; camera_idx < total_num_cameras; camera_idx++)
			{

				float f = 12.6037245f;//12.604f
				float w = 36.f;//36.0f;
				float h = 20.25f;//20.0f;

				temp_CalibParams[camera_idx].m_K(0, 0) = f * (_width / w);
				temp_CalibParams[camera_idx].m_K(0, 1) = 0;
				temp_CalibParams[camera_idx].m_K(0, 2) = _width / 2.f;
				temp_CalibParams[camera_idx].m_K(1, 0) = 0;
				temp_CalibParams[camera_idx].m_K(1, 1) = f * (_height / h);
				temp_CalibParams[camera_idx].m_K(1, 2) = _height / 2.f;
				temp_CalibParams[camera_idx].m_K(2, 0) = 0;
				temp_CalibParams[camera_idx].m_K(2, 1) = 0;
				temp_CalibParams[camera_idx].m_K(2, 2) = 1;//homo

				//R: Euler 2 R_3*3
				Euler2RotationMat(R_vec[camera_idx], temp_CalibParams[camera_idx].m_RotMatrix);

				//temp_CalibParams[camera_idx].m_Trans = P_vec[camera_idx];
				temp_CalibParams[camera_idx].m_Trans = -1 * temp_CalibParams[camera_idx].m_RotMatrix.transpose() * P_vec[camera_idx];
			
			}

			m_CalibParams = temp_CalibParams;

			int ref = total_num_cameras / 2;
			Matrix3d refR = m_CalibParams[ref].m_RotMatrix;
			Matrix3Xd refT(3, 1);
			refT = m_CalibParams[ref].m_Trans;

			Matrix3Xd refRT(3, 4);
			refRT.col(0) = m_CalibParams[ref].m_RotMatrix.col(0);
			refRT.col(1) = m_CalibParams[ref].m_RotMatrix.col(1);
			refRT.col(2) = m_CalibParams[ref].m_RotMatrix.col(2);
			refRT.col(3) = m_CalibParams[ref].m_Trans.col(0);

			Matrix4d refRT4x4;
			refRT4x4.row(0) = refRT.row(0);
			refRT4x4.row(1) = refRT.row(1);
			refRT4x4.row(2) = refRT.row(2);
			refRT4x4.row(3) << 0, 0, 0, 1;

			for (int cam_num = 0; cam_num < total_num_cameras; cam_num++)
			{
				m_CalibParams[cam_num].m_Trans = m_CalibParams[cam_num].m_RotMatrix * (-refR.inverse() * refT) + m_CalibParams[cam_num].m_Trans;
				m_CalibParams[cam_num].m_RotMatrix *= refR.inverse();

				m_CalibParams[cam_num].m_ProjMatrix = compute_projection_matrices(cam_num);
			}

		}

	}

}

void Quaternion2RotationMat(Vector4d& quaternion, Matrix3d& rotationMat)
{
	rotationMat(0, 0) = 1.0 - 2.0 * quaternion(2) * quaternion(2) - 2.0 * quaternion(3) * quaternion(3);
	rotationMat(0, 1) = 2.0 * quaternion(1) * quaternion(2) - 2.0 * quaternion(3) * quaternion(0);
	rotationMat(0, 2) = 2.0 * quaternion(1) * quaternion(3) + 2.0 * quaternion(2) * quaternion(0);
	rotationMat(1, 0) = 2.0 * quaternion(1) * quaternion(2) + 2.0 * quaternion(3) * quaternion(0);
	rotationMat(1, 1) = 1.0 - 2.0 * quaternion(1) * quaternion(1) - 2.0 * quaternion(3) * quaternion(3);
	rotationMat(1, 2) = 2.0 * quaternion(2) * quaternion(3) - 2.0 * quaternion(1) * quaternion(0);
	rotationMat(2, 0) = 2.0 * quaternion(1) * quaternion(3) - 2.0 * quaternion(2) * quaternion(0);
	rotationMat(2, 1) = 2.0 * quaternion(2) * quaternion(3) + 2.0 * quaternion(1) * quaternion(0);
	rotationMat(2, 2) = 1.0 - 2.0 * quaternion(1) * quaternion(1) - 2.0 * quaternion(2) * quaternion(2);

	rotationMat = rotationMat.transpose();
}

void GetRotationMat(Vector3d& euler, Matrix3d& rotationMat)
{
	double sx = sin(euler(0));
	double cx = cos(euler(0));
	double sy = sin(euler(1));
	double cy = cos(euler(1));
	double sz = sin(euler(2));
	double cz = cos(euler(2));

	Matrix3d Rx, Ry, Rz;
	Rx <<
		1, 0, 0,
		0, cx, -1.0*sx,
		0, sx, cx;
	Ry <<
		cy, 0, sy,
		0, 1, 0,
		-1.0*sy, 0, cy;
	Rz <<
		cz, -1.0*sz, 0,
		sz, cz, 0,
		0, 0, 1;
	rotationMat = Rx * Ry * Rz;
}

void Euler2RotationMat(Vector3d& euler, Matrix3d& rotationMat)
{
	//double sh = sin(euler(2));
	//double ch = cos(euler(2));
	//double sa = sin(euler(1));
	//double ca = cos(euler(1));
	//double sb = sin(euler(0));
	//double cb = cos(euler(0)) ;
	//rotationMat(0, 0) = ch * ca;
	//rotationMat(0, 1) = -1.0 * ch * sa * cb + sh * sb;
	//rotationMat(0, 2) = ch * sa * sb + sh * cb;
	//rotationMat(1, 0) = sa;
	//rotationMat(1, 1) = ca * cb;
	//rotationMat(1, 2) = -1.0 * ca * sb;
	//rotationMat(2, 0) = -1.0 * sh * ca;
	//rotationMat(2, 1) = sh * sa * cb + ch * sb;
	//rotationMat(2, 2) = -1.0 * sh * sa * sb + ch * cb;

	euler = deg2rad(euler);

	//rotationMat = rotationMat.transpose();
	if (data_mode == S05_R1 || data_mode == S06_R2) {

		// Euler
		double ch = cos(euler(0));
		double sh = sin(euler(0));
		double ca = cos(euler(1));
		double sa = sin(euler(1));
		double cb = cos(euler(2));
		double sb = sin(euler(2));

		rotationMat(1, 0) = ch * ca;
		rotationMat(1, 1) = sh * sb - ch * sa * cb;
		rotationMat(1, 2) = ch * sa * sb + sh * cb;
		rotationMat(2, 0) = sa;
		rotationMat(2, 1) = ca * cb;
		rotationMat(2, 2) = -ca * sb;
		rotationMat(0, 0) = -sh * ca;
		rotationMat(0, 1) = sh * sa * cb + ch * sb;
		rotationMat(0, 2) = -sh * sa * sb + ch * cb;


		//Matrix3d rotationMat2;
		//rotationMat2(0, 0) = ch * ca;
		//rotationMat2(0, 1) = sh * sb - ch * sa*cb;
		//rotationMat2(0, 2) = ch * sa*sb + sh * cb;
		//rotationMat2(1, 0) = sa;
		//rotationMat2(1, 1) = ca * cb;
		//rotationMat2(1, 2) = -ca * sb;
		//rotationMat2(2, 0) = -sh * ca;
		//rotationMat2(2, 1) = sh * sa*cb + ch * sb;
		//rotationMat2(2, 2) = -sh * sa*sb + ch * cb;

		//cout << rotationMat << endl << endl;
		//cout << rotationMat.inverse() << endl << endl;
		//cout << rotationMat2 << endl << endl;
		//cout << rotationMat2.inverse() << endl << endl;
		//cout << endl << endl;

		//double sina = sin(euler(0));
		//double cosa = cos(euler(0));
		//double sinb = sin(euler(1));
		//double cosb = cos(euler(1));
		//double sinc = sin(euler(2));
		//double cosc = cos(euler(2));


		//// Euler XYZ
		//rotationMat(0, 0) = cosb * cosc;
		//rotationMat(0, 1) = sina * sinb * cosc - cosa * sinc;
		//rotationMat(0, 2) = cosa * sinb * cosc + sina * sinc;

		//rotationMat(1, 0) = cosb * sinc;
		//rotationMat(1, 1) = sina * sinb * sinc + cosa * cosc;
		//rotationMat(1, 2) = cosa * sinb * sinc - sina * cosc;

		//rotationMat(2, 0) = -1.0 * sinb;
		//rotationMat(2, 1) = sina * cosb;
		//rotationMat(2, 2) = cosa * cosb;

		//rotationMat(0, 0) = cosb * cosc;
		//rotationMat(0, 1) = -cosb*sinc;
		//rotationMat(0, 2) = sinb;
		//rotationMat(1, 0) = cosa*sinc+sina*sinb*sinc;
		//rotationMat(1, 1) = cosa * cosc - sina * sinb * sinc;
		//rotationMat(1, 2) = -sina*cosb;
		//rotationMat(2, 0) = sina * sinc-cosa*sinb*cosc;
		//rotationMat(2, 1) = sina * cosc+cosa*sinb*sinc;
		//rotationMat(2, 2) = cosa * cosb;

	}


	else {

		double sinc = sin(euler(2)); //sinc
		double cosc = cos(euler(2)); //cosc
		double sinb = sin(euler(1)); //
		double cosb = cos(euler(1)); //cosb
		double sina = sin(euler(0)); //
		double cosa = cos(euler(0)); //

		//Euler ZYX
		rotationMat(0, 0) = cosb * cosa;
		rotationMat(0, 1) = sinc * sinb * cosa - cosc * sina;
		rotationMat(0, 2) = cosc * sinb * cosa + sinc * sina;

		rotationMat(1, 0) = cosb * sina;
		rotationMat(1, 1) = sinc * sinb * sina + cosc * cosa;
		rotationMat(1, 2) = cosc * sinb * sina - sinc * cosa;

		rotationMat(2, 0) = -1.0 * sinb;
		rotationMat(2, 1) = sinc * cosb;
		rotationMat(2, 2) = cosc * cosb;
	}


	//Matrix3d Rz, Ry, Rx;
	//Rz << cy, -sy, 0,
	//   sy, cy, 0,
	//   0, 0, 1;
	//Ry << cp, 0, sp,
	//   0, 1, 0,
	//   -sp, 0, cp;
	//Rx << 1, 0, 0,
	//   0, cr, -sr,
	//   0, sr, cr;
	//
	//Matrix3d temp = Rz * Ry;
	//rotationMat = temp * Rx;

	//cout << "Euler ZYX�� ��ȯ ��� :: " << endl;

	//cout << rotationMat << endl;
	//cout << endl;


	//// Rodriguez
	//Mat rod_output;
	//Mat rod_input(3,1,CV_64F);
	//rod_input.at<double>(0, 0) = euler(2); 
	//rod_input.at<double>(1, 0) = euler(1);
	//rod_input.at<double>(2, 0) = euler(0);

	//Rodrigues(rod_input, rod_output);

	//for (int i = 0; i < 3; i++) {
	//   for (int j= 0; j < 3; j++) {
	//      rotationMat(i, j) = rod_output.at<double>(i, j);
	//   }
	//}


}

void compute_projection_matrices()
{
	Matrix3d inMat;
	Matrix3Xd exMat(3, 4);

	for (int cam_idx = 0; cam_idx < total_num_cameras; cam_idx++)
	{
		// The intrinsic matrix
		inMat = m_CalibParams[cam_idx].m_K;

		// The extrinsic matrix
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				exMat(i, j) = m_CalibParams[cam_idx].m_RotMatrix(i, j);

		for (int i = 0; i < 3; i++)
			exMat(i, 3) = m_CalibParams[cam_idx].m_Trans(i, 0);

		// Multiply the intrinsic matrix by the extrinsic matrix to find our projection matrix
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 4; j++) {
				m_CalibParams[cam_idx].m_ProjMatrix(i, j) = 0.0;

				for (int k = 0; k < 3; k++)
					m_CalibParams[cam_idx].m_ProjMatrix(i, j) += inMat(i, k) * exMat(k, j);
			}

		m_CalibParams[cam_idx].m_ProjMatrix(3, 0) = 0.0;
		m_CalibParams[cam_idx].m_ProjMatrix(3, 1) = 0.0;
		m_CalibParams[cam_idx].m_ProjMatrix(3, 2) = 0.0;
		m_CalibParams[cam_idx].m_ProjMatrix(3, 3) = 1.0;
	}

}

void load_file_name(
	vector<vector<string>> &color_names,
	vector<vector<string>> &depth_names)
{
	string cam_path = path + "\\cam";

	intptr_t color_handle, depth_handle;

	struct _finddata_t color_fd, depth_fd;

	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++)
	{
		string color_path = cam_path + to_string(cam_num) + "\\*.jpg";
		string depth_path = cam_path + to_string(cam_num) + "\\*.png";

		color_handle = _findfirst(color_path.c_str(), &color_fd);
		depth_handle = _findfirst(depth_path.c_str(), &depth_fd);

		for (int frame_num = 0; frame_num < total_num_frames; frame_num++)
		{
			color_names[cam_num][frame_num] = color_fd.name;
			depth_names[cam_num][frame_num] = depth_fd.name;

			_findnext(color_handle, &color_fd);
			_findnext(depth_handle, &depth_fd);
		}
	}

	_findclose(color_handle);
	_findclose(depth_handle);
}

void load_file_name(
	vector<string> &color_names_,
	vector<string> &depth_names_,
	int depth_bits)
{
	intptr_t color_handle, depth_handle;

	struct _finddata_t color_fd, depth_fd;

	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++)
	{
		string color_path, depth_path;
		switch (data_mode) {
		case Poznan_Fencing:
			color_path = path + "\\*cam" + to_string(cam_num) + "_tex*.yuv";
			if (depth_bits == 8) depth_path = path + "\\*cam" + to_string(cam_num) + "_depth*.yuvcut.yuv";
			else depth_path = path + "\\*cam" + to_string(cam_num) + "_depth*.yuv";
			break;

		case Intel_Kermit:
			color_path = path + "\\v" + to_string(cam_num + 1) + "_tex*.yuv";
			depth_path = path + "\\v" + to_string(cam_num + 1) + "_dep*.yuv";
			break;

		case Technicolor_Painter:
			color_path = path + "\\v" + to_string(cam_num) + "_tex*.yuv";
			depth_path = path + "\\v" + to_string(cam_num) + "_dep*.yuv";
			break;
		
		}

		color_handle = _findfirst(color_path.c_str(), &color_fd);
		depth_handle = _findfirst(depth_path.c_str(), &depth_fd);

		color_names_[cam_num] = color_fd.name;
		depth_names_[cam_num] = depth_fd.name;

		_findnext(color_handle, &color_fd);
		_findnext(depth_handle, &depth_fd);
	}

	_findclose(color_handle);
	_findclose(depth_handle);
}

void load_file_name_mode4(
	vector<vector<string>>& color_names,
	vector<vector<string>>& depth_names,
	int referenceView,
	int furthest_index)
{
	string cam_path = path;
	intptr_t color_handle, depth_handle;

	int cnt = 0;
	struct _finddata_t color_fd, depth_fd;

	string color_path = cam_path + "\\RGB\\col" + "\\*.png";
	string depth_path = cam_path + "\\RGB\\dep" + "\\*.png";

	color_handle = _findfirst(color_path.c_str(), &color_fd);
	depth_handle = _findfirst(depth_path.c_str(), &depth_fd);

	int temp_num;
	if (referenceView == 220 && data_mode >= S01_H1) temp_num = referenceView + furthest_index;
	else temp_num = total_num_cameras;

	vector<string> temp_vec;
	temp_vec.resize(total_num_frames);
	color_names.resize(temp_num, temp_vec);
	depth_names.resize(temp_num, temp_vec);

	if (referenceView == 220) {

		for (int cam_num = 0; cam_num < referenceView + furthest_index; cam_num++) //MAXNUM_11X11
		{
			for (int frame_num = 0; frame_num < total_num_frames; frame_num++)
			{
				color_names[cam_num][frame_num] = color_fd.name;
				depth_names[cam_num][frame_num] = depth_fd.name;

				_findnext(color_handle, &color_fd);
				_findnext(depth_handle, &depth_fd);
			}
		}
	}
	else {
		for (int cam_num = 0; cam_num < total_num_cameras; cam_num++)
		{
			for (int frame_num = 0; frame_num < total_num_frames; frame_num++)
			{
				color_names[cam_num][frame_num] = color_fd.name;
				depth_names[cam_num][frame_num] = depth_fd.name;

				_findnext(color_handle, &color_fd);
				_findnext(depth_handle, &depth_fd);
			}
		}
	}
	//for (int cam_num = 0; cam_num < total_num_cameras; cam_num++)
	//{
	//   cout << color_names[cam_num][0] << endl;
	//}

	_findclose(color_handle);
	_findclose(depth_handle);
}

void get_RT_data_json(const char* file, vector<Vector3d>& Rotation_vec, vector<Vector3d>& Position_vec, vector<Vector3d>& KFocal_vec, vector<Vector3d>& KPrinciple_vec, int total_num_cameras)
{
	const char* fileName = file;

	FILE* pFile = fopen(fileName, "rb");
	char buffer_str[65536];
	FileReadStream is(pFile, buffer_str, sizeof(buffer_str));
	Document d;
	d.ParseStream<0, UTF8<>, FileReadStream>(is);

	vector<Vector3d> R_vec;
	vector<Vector3d> P_vec;
	vector<Vector3d> KF_vec;
	vector<Vector3d> KP_vec;
	Vector3d R_vec_temp;
	Vector3d P_vec_temp;
	Vector3d KF_vec_temp;
	Vector3d KP_vec_temp;
	String R_name("Rotation");
	String P_name("Position");
	String KF_name("Focal");
	String KP_name("Principle_point");

	const Value& cameras = d["cameras"];
	//assert(attributes.IsArray());

	for (SizeType i = 0; i < cameras.Size(); i++)
	{
		const Value& attribute = cameras[i];
		assert(attribute.IsObject());

		for (Value::ConstMemberIterator itr = attribute.MemberBegin(); itr != attribute.MemberEnd(); ++itr)
		{
			int type_num = 6;

			switch (itr->value.GetType())
			{
			case 0://!< null
				break;
			case 1://!< false
				break;
			case 2://!< true
				break;
			case 3://!< object
				break;
			case 4://!< array

				if ((R_name.compare(itr->name.GetString()) == 0))//Rotation
				{
					for (SizeType i = 0; i < itr->value.Size(); i++)
					{
						R_vec_temp[i] = itr->value[i].GetDouble();
						//cout << R_vec_temp[i] << endl;
					}
					R_vec.push_back(R_vec_temp);
				}
				else if ((P_name.compare(itr->name.GetString()) == 0))//Position
				{
					for (SizeType i = 0; i < itr->value.Size(); i++)
					{
						P_vec_temp[i] = itr->value[i].GetDouble();
						//cout << P_vec_temp[i] << endl;
					}
					P_vec.push_back(P_vec_temp);
				}
				else if ((KF_name.compare(itr->name.GetString()) == 0))//Position
				{
					for (SizeType i = 0; i < itr->value.Size(); i++)
					{
						KF_vec_temp[i] = itr->value[i].GetDouble();
						//cout << P_vec_temp[i] << endl;
					}
					KF_vec.push_back(KF_vec_temp);
				}
				else if ((KP_name.compare(itr->name.GetString()) == 0))//Position
				{
					for (SizeType i = 0; i < itr->value.Size(); i++)
					{
						KP_vec_temp[i] = itr->value[i].GetDouble();
						//cout << P_vec_temp[i] << endl;
					}
					KP_vec.push_back(KP_vec_temp);
				}

				break;
			case 5://!< string
			   cout << itr->name.GetString() << " : " << itr->value.GetString() << endl;
				break;
			case 6://!< number
			   cout << itr->name.GetString() << " : " << itr->value.GetInt() << endl;
				break;

			}
		}
	}

	Rotation_vec = R_vec;
	Position_vec = P_vec;
	KFocal_vec = KF_vec;
	KPrinciple_vec = KP_vec;
}

void get_RT_data_json(
	const char* file,
	vector<Vector3d>& Rotation_vec,
	vector<Vector3d>& Position_vec,
	vector<Vector2d>& Depth_vec,
	vector<Vector3d>& KFocal_vec,
	vector<Vector3d>& KPrinciple_vec,
	int total_num_cameras)
{
	const char* fileName = file;

	FILE* pFile = fopen(fileName, "rb");
	char buffer_str[65536];
	FileReadStream is(pFile, buffer_str, sizeof(buffer_str));
	Document d;
	d.ParseStream<0, UTF8<>, FileReadStream>(is);

	vector<Vector3d> R_vec;
	vector<Vector3d> P_vec;
	vector<Vector2d> DR_vec;
	vector<Vector3d> KF_vec;
	vector<Vector3d> KP_vec;
	Vector3d R_vec_temp;
	Vector3d P_vec_temp;
	Vector2d DR_vec_temp;
	Vector3d KF_vec_temp;
	Vector3d KP_vec_temp;
	String R_name("Rotation");
	String P_name("Position");
	String DR_name("Depth_range");
	String KF_name("Focal");
	String KP_name("Principle_point");

	const Value& cameras = d["cameras"];
	//assert(attributes.IsArray());

	for (SizeType i = 0; i < cameras.Size(); i++)
	{
		const Value& attribute = cameras[i];
		assert(attribute.IsObject());

		for (Value::ConstMemberIterator itr = attribute.MemberBegin(); itr != attribute.MemberEnd(); ++itr)
		{
			int type_num = 6;

			switch (itr->value.GetType())
			{
			case 4://!< array

				if ((R_name.compare(itr->name.GetString()) == 0))//Rotation
				{
					for (SizeType i = 0; i < itr->value.Size(); i++)
					{
						R_vec_temp[i] = itr->value[i].GetDouble();
						//cout << R_vec_temp[i] << endl;
					}
					R_vec.push_back(R_vec_temp);
				}
				else if ((P_name.compare(itr->name.GetString()) == 0))//Position
				{
					for (SizeType i = 0; i < itr->value.Size(); i++)
					{
						P_vec_temp[i] = itr->value[i].GetDouble();
						//cout << P_vec_temp[i] << endl;
					}
					P_vec.push_back(P_vec_temp);
				}
				else if ((DR_name.compare(itr->name.GetString()) == 0))//Depth range
				{
					for (SizeType i = 0; i < itr->value.Size(); i++)
					{
						DR_vec_temp[i] = itr->value[i].GetDouble();
						//cout << P_vec_temp[i] << endl;
					}
					DR_vec.push_back(DR_vec_temp);
				}
				else if ((KF_name.compare(itr->name.GetString()) == 0))//Position
				{
					for (SizeType i = 0; i < itr->value.Size(); i++)
					{
						KF_vec_temp[i] = itr->value[i].GetDouble();
						//cout << P_vec_temp[i] << endl;
					}
					KF_vec.push_back(KF_vec_temp);
				}
				else if ((KP_name.compare(itr->name.GetString()) == 0))//Position
				{
					for (SizeType i = 0; i < itr->value.Size(); i++)
					{
						KP_vec_temp[i] = itr->value[i].GetDouble();
						//cout << P_vec_temp[i] << endl;
					}
					KP_vec.push_back(KP_vec_temp);
				}
				break;
			default:
				break;
			}
		}
	}

	Rotation_vec = R_vec;
	Position_vec = P_vec;
	Depth_vec = DR_vec;
	KFocal_vec = KF_vec;
	KPrinciple_vec = KP_vec;
}

Vector3d rad2deg(Vector3d radian)
{
	Vector3d output;
	for (int i = 0; i < 3; i++)
	{
		output[i] = radian[i] * 180.0 / PI;
	}
	return output;
}
Vector3d deg2rad(Vector3d degree)
{
	Vector3d output;
	for (int i = 0; i < 3; i++)
	{
		output[i] = degree[i] * PI / 180.0;
	}
	return output;
}