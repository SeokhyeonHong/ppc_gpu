#include "common.h"
#define _CRT_SECURE_NO_WARNINGS

vector<int> make_camOrder(int refView, int mask_size, map<int, int> &LookUpTable) {
	vector<int> camera_order;
	if (refView == 220 && data_mode >= 4) {
		int temp = 0;
		camera_order.push_back(refView);

		for (int i = 1; i < total_num_cameras / 2 + 1; i++) {
			if (i % mask_size == mask_size / 2 + 1) temp += 22 - mask_size; 
			else temp++;
			camera_order.push_back(refView - temp);
			camera_order.push_back(refView + temp);
		}

		for (int i = 0; i < total_num_cameras; i++) {
			LookUpTable.insert(make_pair(camera_order[i], i));
		}

		int cnt = 0;
		for (map<int, int>::iterator it = LookUpTable.begin(); it != LookUpTable.end(); it++) {
			it->second = cnt++;
		}
	}
	else {
		for (int i = 0; i < total_num_cameras; i++)
			camera_order.push_back(i);
	}

	return camera_order;
}

double depth_level_2_Z(unsigned char d)
{
	double z;

	z = 1.0 / ((d / 255.0) * (1.0 / MinZ - 1.0 / MaxZ) + 1.0 / MaxZ);

	return z;
}

double depth_level_2_Z_s(unsigned short d)
{
	double z;

	z = 1.0 / ((d / 65535.0) * (1.0 / MinZ - 1.0 / MaxZ) + 1.0 / MaxZ);

	return z;
}

double depth_level_2_Z_s(unsigned short d, int camera)
{
	double z;

	z = 1.0 / ((d / 65535.0) * (1.0 / tech_minmaxZ[camera][0] - 1.0 / tech_minmaxZ[camera][1]) + 1.0 / tech_minmaxZ[camera][1]);

	return z;
}

double depth_level_2_Z_s_direct(unsigned short d)
{
	double z;


	z = (double)d / scaleZ;

	return z;
}

void projection_UVZ_2_XY_PC(
	Matrix4d projMatrix,
	double u,
	double v,
	double z,
	double* x,
	double* y)
{
	double c0, c1, c2;

	c0 = z * projMatrix(0, 2) + projMatrix(0, 3);
	c1 = z * projMatrix(1, 2) + projMatrix(1, 3);
	c2 = z * projMatrix(2, 2) + projMatrix(2, 3);

	v = (double)_height - v - 1.0;

	*y = u * (c1 * projMatrix(2, 0) - projMatrix(1, 0) * c2)
		+ v * (c2 * projMatrix(0, 0) - projMatrix(2, 0) * c0)
		+ projMatrix(1, 0) * c0
		- c1 * projMatrix(0, 0);

	*y /= v * (projMatrix(2, 0) * projMatrix(0, 1) - projMatrix(2, 1) * projMatrix(0, 0))
		+ u * (projMatrix(1, 0) * projMatrix(2, 1) - projMatrix(1, 1) * projMatrix(2, 0))
		+ projMatrix(0, 0) * projMatrix(1, 1)
		- projMatrix(1, 0) * projMatrix(0, 1);

	*x = (*y) * (projMatrix(0, 1) - projMatrix(2, 1) * u) + c0 - c2 * u;

	*x /= projMatrix(2, 0) * u - projMatrix(0, 0);
}

double MVG(
	Matrix3d K,
	Matrix3d R_wc,
	Matrix3Xd t_wc,
	int x,
	int y,
	double Z,
	double* X,
	double* Y)
{
	double X_cam = (x - K(0, 2)) * (Z / K(0, 0));
	double Y_cam = (y - K(1, 2)) * (Z / K(1, 1));

	//cam coord
	Matrix3Xd C_cam(3, 1);
	C_cam(0, 0) = X_cam;
	C_cam(1, 0) = Y_cam;
	C_cam(2, 0) = Z;

	//assuming R, t as matrix world to cam
	Matrix3Xd C_world(3, 1);
	C_world = R_wc.inverse() * (C_cam - t_wc);
	*X = C_world(0, 0);
	*Y = C_world(1, 0);

	return C_world(2, 0);
}

bool confirm_point(
	int camera,
	PointXYZRGB p,
	vector<Mat> color_imgs)
{
	int u, v;
	int blue, green, red;

	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++)
	{
		if (camera == cam_num) continue;

		projection_XYZ_2_UV(m_CalibParams[cam_num].m_ProjMatrix, p.x, p.y, -p.z, u, v);

		if (u < 0 || v < 0 || u >= _width || v >= _height) return true;

		blue = abs((int)color_imgs[cam_num].at<Vec3b>(v, u)[0] - (int)p.b);
		green = abs((int)color_imgs[cam_num].at<Vec3b>(v, u)[1] - (int)p.g);
		red = abs((int)color_imgs[cam_num].at<Vec3b>(v, u)[2] - (int)p.r);

		if (blue + green + red < 20) return true;
	}

	return false;
}

Mat cvt_yuv2bgr(
	string name,
	int frame,
	int type,
	bool is_yuv)
{
	int numOfPixel = _width * _height;

	//  file I/O
	FILE* fp = fopen(name.c_str(), "rb");
	if (fp == NULL) {
		perror("encoder file open fail");
		exit(0);
	}

	if (!is_yuv) {
		short* buffer = new short[numOfPixel];
		Mat img(_height, _width, CV_16UC3);

		for (int i = 0; i < frame; i++)
			fread(buffer, 2, numOfPixel, fp);

		// byte data �о����
		fread(buffer, 2, numOfPixel, fp);

		// y ä�� Mat�� �����
		for (int y = 0; y < _height; y++)
			for (int x = 0; x < _width; x++) {
				img.at<Vec3s>(y, x)[0] = buffer[y * _width + x];
				img.at<Vec3s>(y, x)[1] = buffer[y * _width + x];
				img.at<Vec3s>(y, x)[2] = buffer[y * _width + x];
			}

		delete[] buffer;

		return img;
	}

	if (type == 8) {
		char* buffer_y = new char[numOfPixel];
		char* buffer_u = new char[numOfPixel / 4];
		char* buffer_v = new char[numOfPixel / 4];

		Mat img_y(_height, _width, CV_8UC1);
		Mat img_u(_height / 2, _width / 2, CV_8UC1);
		Mat img_v(_height / 2, _width / 2, CV_8UC1);
		Mat img_yuv[3];
		Mat img;

		for (int i = 0; i < frame; i++) {
			fread(buffer_y, 1, numOfPixel, fp);
			fread(buffer_u, 1, numOfPixel / 4, fp);
			fread(buffer_v, 1, numOfPixel / 4, fp);
		}

		// byte data �о����
		fread(buffer_y, 1, numOfPixel, fp);
		fread(buffer_u, 1, numOfPixel / 4, fp);
		fread(buffer_v, 1, numOfPixel / 4, fp);

		// y ä�� Mat�� �����
		for (int y = 0; y < _height; y++) {
			for (int x = 0; x < _width; x++) {
				img_y.at<uchar>(y, x) = buffer_y[y * _width + x];
			}
		}

		// u, v ä�� Mat�� �����
		for (int y = 0; y < _height / 2; y++) {
			for (int x = 0; x < _width / 2; x++) {
				img_u.at<uchar>(y, x) = buffer_u[y * (_width / 2) + x];
				img_v.at<uchar>(y, x) = buffer_v[y * (_width / 2) + x];
			}
		}

		// 3�� ¥�� Mat array�� �� ä�� Mat �Ҵ�.
		// u, v ä���� resize ����.
		img_yuv[0] = img_y;
		resize(img_u, img_yuv[1], Size(_width, _height));
		resize(img_v, img_yuv[2], Size(_width, _height));

		// �ϳ��� ���� ä�� Mat�� �����.
		merge(img_yuv, 3, img);

		// ������ ��ȯ.
		cvtColor(img, img, CV_YUV2BGR);

		delete[] buffer_y;
		delete[] buffer_u;
		delete[] buffer_v;

		// ���⼭ 10bits ��ȯ�ϸ� �ɵ�

		return img;
	}
	else if (type == 10) {
		short* buffer_y = new short[numOfPixel];
		short* buffer_u = new short[numOfPixel / 4];
		short* buffer_v = new short[numOfPixel / 4];

		Mat img_y(_height, _width, CV_8U);
		Mat img_u(_height / 2, _width / 2, CV_8U);
		Mat img_v(_height / 2, _width / 2, CV_8U);
		Mat img_yuv[3];
		Mat img;

		for (int i = 0; i < frame; i++) {
			fread(buffer_y, 2, numOfPixel, fp);
			fread(buffer_u, 2, numOfPixel / 4, fp);
			fread(buffer_v, 2, numOfPixel / 4, fp);
		}

		// byte data �о����
		fread(buffer_y, 2, numOfPixel, fp);
		fread(buffer_u, 2, numOfPixel / 4, fp);
		fread(buffer_v, 2, numOfPixel / 4, fp);

		// y ä�� Mat�� �����
		for (int y = 0; y < _height; y++) {
			for (int x = 0; x < _width; x++) {
				img_y.at<uchar>(y, x) = uchar(buffer_y[y * _width + x] * 255.0 / 1023.0);
			}
		}

		// u, v ä�� Mat�� �����
		for (int y = 0; y < _height / 2; y++) {
			for (int x = 0; x < _width / 2; x++) {
				img_u.at<uchar>(y, x) = uchar(buffer_u[y * (_width / 2) + x] * 255.0 / 1023.0);
				img_v.at<uchar>(y, x) = uchar(buffer_v[y * (_width / 2) + x] * 255.0 / 1023.0);
			}
		}

		// 3�� ¥�� Mat array�� �� ä�� Mat �Ҵ�.
		// u, v ä���� resize ����.
		img_yuv[0] = img_y;
		resize(img_u, img_yuv[1], Size(_width, _height));
		resize(img_v, img_yuv[2], Size(_width, _height));

		// �ϳ��� ���� ä�� Mat�� �����.
		merge(img_yuv, 3, img);

		// ������ ��ȯ.
		cvtColor(img, img, CV_YUV2BGR);

		delete[] buffer_y;
		delete[] buffer_u;
		delete[] buffer_v;

		return img;
	}
	else {
		short* buffer_y = new short[numOfPixel];
		short* buffer_u = new short[numOfPixel / 4];
		short* buffer_v = new short[numOfPixel / 4];

		Mat img_y(_height, _width, CV_16UC1);
		Mat img_u(_height / 2, _width / 2, CV_16UC1);
		Mat img_v(_height / 2, _width / 2, CV_16UC1);
		Mat img_yuv[3];
		Mat img;

		for (int i = 0; i < frame; i++) {
			fread(buffer_y, 2, numOfPixel, fp);
			fread(buffer_u, 2, numOfPixel / 4, fp);
			fread(buffer_v, 2, numOfPixel / 4, fp);
		}

		// byte data �о����
		fread(buffer_y, 2, numOfPixel, fp);
		fread(buffer_u, 2, numOfPixel / 4, fp);
		fread(buffer_v, 2, numOfPixel / 4, fp);

		// y ä�� Mat�� �����
		for (int y = 0; y < _height; y++) {
			for (int x = 0; x < _width; x++) {
				img_y.at<ushort>(y, x) = buffer_y[y * _width + x];
			}
		}

		// u, v ä�� Mat�� �����
		for (int y = 0; y < _height / 2; y++) {
			for (int x = 0; x < _width / 2; x++) {
				img_u.at<ushort>(y, x) = buffer_u[y * (_width / 2) + x];
				img_v.at<ushort>(y, x) = buffer_v[y * (_width / 2) + x];
			}
		}

		// 3�� ¥�� Mat array�� �� ä�� Mat �Ҵ�.
		// u, v ä���� resize ����.
		img_yuv[0] = img_y;
		resize(img_u, img_yuv[1], Size(_width, _height));
		resize(img_v, img_yuv[2], Size(_width, _height));

		// �ϳ��� ���� ä�� Mat�� �����.
		merge(img_yuv, 3, img);

		// ������ ��ȯ.
		cvtColor(img, img, CV_YUV2BGR);

		delete[] buffer_y;
		delete[] buffer_u;
		delete[] buffer_v;

		return img;
	}
}

Mat readYUV(
	string name,
	int frame,
	int type)
{
	int numOfPixel = _width * _height;

	//  file I/O
	FILE* fp = fopen(name.c_str(), "rb");
	if (fp == NULL) {
		perror("encoder file open fail");
		exit(0);
	}

	if (type == 8) {
		char* buffer_y = new char[numOfPixel];
		char* buffer_u = new char[numOfPixel / 4];
		char* buffer_v = new char[numOfPixel / 4];

		Mat img_y(_height, _width, CV_8UC1);
		Mat img_u(_height / 2, _width / 2, CV_8UC1);
		Mat img_v(_height / 2, _width / 2, CV_8UC1);
		Mat img_yuv[3];
		Mat img;

		for (int i = 0; i < frame; i++) {
			fread(buffer_y, 1, numOfPixel, fp);
			fread(buffer_u, 1, numOfPixel / 4, fp);
			fread(buffer_v, 1, numOfPixel / 4, fp);
		}

		// byte data �о����
		fread(buffer_y, 1, numOfPixel, fp);
		fread(buffer_u, 1, numOfPixel / 4, fp);
		fread(buffer_v, 1, numOfPixel / 4, fp);

		// y ä�� Mat�� �����
		for (int y = 0; y < _height; y++) {
			for (int x = 0; x < _width; x++) {
				img_y.at<uchar>(y, x) = buffer_y[y * _width + x];
			}
		}

		// u, v ä�� Mat�� �����
		for (int y = 0; y < _height / 2; y++) {
			for (int x = 0; x < _width / 2; x++) {
				img_u.at<uchar>(y, x) = buffer_u[y * (_width / 2) + x];
				img_v.at<uchar>(y, x) = buffer_v[y * (_width / 2) + x];
			}
		}

		// 3�� ¥�� Mat array�� �� ä�� Mat �Ҵ�.
		// u, v ä���� resize ����.
		img_yuv[0] = img_y;
		resize(img_u, img_yuv[1], Size(_width, _height));
		resize(img_v, img_yuv[2], Size(_width, _height));

		// �ϳ��� ���� ä�� Mat�� �����.
		merge(img_yuv, 3, img);

		delete[] buffer_y;
		delete[] buffer_u;
		delete[] buffer_v;

		// ���⼭ 10bits ��ȯ�ϸ� �ɵ�

		return img;
	}
	else if (type == 10) {
		short* buffer_y = new short[numOfPixel];
		short* buffer_u = new short[numOfPixel / 4];
		short* buffer_v = new short[numOfPixel / 4];

		Mat img_y(_height, _width, CV_8U);
		Mat img_u(_height / 2, _width / 2, CV_8U);
		Mat img_v(_height / 2, _width / 2, CV_8U);
		Mat img_yuv[3];
		Mat img;

		for (int i = 0; i < frame; i++) {
			fread(buffer_y, 2, numOfPixel, fp);
			fread(buffer_u, 2, numOfPixel / 4, fp);
			fread(buffer_v, 2, numOfPixel / 4, fp);
		}

		// byte data �о����
		fread(buffer_y, 2, numOfPixel, fp);
		fread(buffer_u, 2, numOfPixel / 4, fp);
		fread(buffer_v, 2, numOfPixel / 4, fp);

		// y ä�� Mat�� �����
		for (int y = 0; y < _height; y++) {
			for (int x = 0; x < _width; x++) {
				img_y.at<uchar>(y, x) = uchar(buffer_y[y * _width + x] * 255.0 / 1023.0);
			}
		}

		// u, v ä�� Mat�� �����
		for (int y = 0; y < _height / 2; y++) {
			for (int x = 0; x < _width / 2; x++) {
				img_u.at<uchar>(y, x) = uchar(buffer_u[y * (_width / 2) + x] * 255.0 / 1023.0);
				img_v.at<uchar>(y, x) = uchar(buffer_v[y * (_width / 2) + x] * 255.0 / 1023.0);
			}
		}

		// 3�� ¥�� Mat array�� �� ä�� Mat �Ҵ�.
		// u, v ä���� resize ����.
		img_yuv[0] = img_y;
		resize(img_u, img_yuv[1], Size(_width, _height));
		resize(img_v, img_yuv[2], Size(_width, _height));

		// �ϳ��� ���� ä�� Mat�� �����.
		merge(img_yuv, 3, img);

		delete[] buffer_y;
		delete[] buffer_u;
		delete[] buffer_v;

		return img;
	}
	else {
		short* buffer_y = new short[numOfPixel];
		short* buffer_u = new short[numOfPixel / 4];
		short* buffer_v = new short[numOfPixel / 4];

		Mat img_y(_height, _width, CV_16UC1);
		Mat img_u(_height / 2, _width / 2, CV_16UC1);
		Mat img_v(_height / 2, _width / 2, CV_16UC1);
		Mat img_yuv[3];
		Mat img;

		for (int i = 0; i < frame; i++) {
			fread(buffer_y, 2, numOfPixel, fp);
			fread(buffer_u, 2, numOfPixel / 4, fp);
			fread(buffer_v, 2, numOfPixel / 4, fp);
		}

		// byte data �о����
		fread(buffer_y, 2, numOfPixel, fp);
		fread(buffer_u, 2, numOfPixel / 4, fp);
		fread(buffer_v, 2, numOfPixel / 4, fp);

		// y ä�� Mat�� �����
		for (int y = 0; y < _height; y++) {
			for (int x = 0; x < _width; x++) {
				img_y.at<ushort>(y, x) = buffer_y[y * _width + x];
			}
		}

		// u, v ä�� Mat�� �����
		for (int y = 0; y < _height / 2; y++) {
			for (int x = 0; x < _width / 2; x++) {
				img_u.at<ushort>(y, x) = buffer_u[y * (_width / 2) + x];
				img_v.at<ushort>(y, x) = buffer_v[y * (_width / 2) + x];
			}
		}

		img_yuv[0] = img_y;
		resize(img_u, img_yuv[1], Size(_width, _height));
		resize(img_v, img_yuv[2], Size(_width, _height));

		// �ϳ��� ���� ä�� Mat�� �����.
		merge(img_yuv, 3, img);

		delete[] buffer_y;
		delete[] buffer_u;
		delete[] buffer_v;

		return img;
	}
}

#ifdef ON_GPU
PointCloud<PointXYZRGB>::Ptr make_PC(
	int camera,
	Mat color_img,
	Mat depth_img)
{
	PointCloud<PointXYZRGB>::Ptr pointcloud(new PointCloud<PointXYZRGB>);
	
	Matrix3d m_RotMatrix_inv = m_CalibParams[camera].m_RotMatrix.inverse();
	
	int numpix = _width * _height;
	double* hst_x, * hst_y, * hst_z;
	uchar* hst_b, * hst_g, * hst_r;

	hst_x = (double*)malloc(sizeof(double) * numpix);
	hst_y = (double*)malloc(sizeof(double) * numpix);
	hst_z = (double*)malloc(sizeof(double) * numpix);
	hst_b = (uchar*)malloc(sizeof(uchar) * numpix);
	hst_g = (uchar*)malloc(sizeof(uchar) * numpix);
	hst_r = (uchar*)malloc(sizeof(uchar) * numpix);
	
	CudaGpu.make_PC(color_img, depth_img, data_mode, scaleZ, m_CalibParams[camera].m_K.data(), m_RotMatrix_inv.data(), m_CalibParams[camera].m_Trans.data(), hst_x, hst_y, hst_z, hst_b, hst_g, hst_r);
	for (int i = 0; i < numpix; ++i)
	{
		PointXYZRGB p;

		p.x = hst_x[i];
		p.y = hst_y[i];
		p.z = hst_z[i];

		p.b = hst_b[i];
		p.g = hst_g[i];
		p.r = hst_r[i];
		pointcloud->points.push_back(p);
	}
	
	free(hst_x);
	free(hst_y);
	free(hst_z);
	free(hst_b);
	free(hst_g);
	free(hst_r);

	return pointcloud;
}
#else
PointCloud<PointXYZRGB>::Ptr make_PC(
	int camera,
	Mat color_img,
	Mat depth_img)
{
	PointCloud<PointXYZRGB>::Ptr pointcloud(new PointCloud<PointXYZRGB>);
	for (int y = 0; y < _height; y++) {
		for (int x = 0; x < _width; x++)
		{
			Vec3b d, color;
			Vec3s d_s, color_s;
			double Z, X = 0.0, Y = 0.0;

			switch (data_mode) {
			case 0:
				d = depth_img.at<Vec3b>(y, x);

				Z = depth_level_2_Z(d[0]);
				projection_UVZ_2_XY_PC(m_CalibParams[camera].m_ProjMatrix, x, y, Z, &X, &Y);

				//Z *= (-1);
				break;

			case 1:
			case 2:
			case 3:
				d_s = depth_img.at<Vec3s>(y, x);
				color = color_img.at<Vec3b>(y, x);

				Z = depth_level_2_Z_s(d_s[0]);

				Z = MVG(m_CalibParams[camera].m_K, m_CalibParams[camera].m_RotMatrix, m_CalibParams[camera].m_Trans, x, y, Z, &X, &Y);
				break;
			case 4:
			case 5:
			case 6:
			case 7:
			case 8:
			case 9:
			case 10:
			case 11:
			case 12:
			case 13:
				color = color_img.at<Vec3b>(y, x);
				Z = depth_level_2_Z_s_direct(depth_img.at<ushort>(y, x));
				Z = MVG(m_CalibParams[camera].m_K, m_CalibParams[camera].m_RotMatrix, m_CalibParams[camera].m_Trans, x, y, Z, &X, &Y);
				break;
			}


			PointXYZRGB p;

			p.x = X;
			p.y = Y;
			p.z = Z;

			p.b = color[0];
			p.g = color[1];
			p.r = color[2];

			//if ((data_mode == 8 || data_mode == 9) && depth_img.at<ushort>(y, x) >= 9000) continue;
			pointcloud->points.push_back(p);
		}
	}
	return pointcloud;
}
#endif

void get_color_and_depth_imgs(
	int frame,
	vector<int> camera_order,
	vector<vector<string>> color_names,
	vector<vector<string>> depth_names,
	vector<Mat>& color_imgs,
	vector<Mat>& depth_imgs)
{
	//vector<Mat> imgs(total_num_cameras);
	//vector<Mat> imgs2(total_num_cameras);

	vector<Mat> imgs;
	vector<Mat> imgs2;

	string folder_path;
	if (camera_order[0] == 0) {
		for (int camera = 0; camera < total_num_cameras; camera++)
		{
			//int camera = camera_order[i];

			folder_path = path + "/cam" + to_string(camera) + "/";
			Mat color_img, depth_img;

			if (data_mode == MSR3DVideo_Ballet) {
				folder_path = path + "/cam" + to_string(camera) + "/";
				color_img = imread(folder_path + color_names[camera][frame]);
				depth_img = imread(folder_path + depth_names[camera][frame]);
			}
			else if (data_mode >= S01_H1){
				color_img = imread(path + "/RGB/col/" + color_names[camera][frame]);
				depth_img = imread(path + "/RGB/dep/" + depth_names[camera][frame], IMREAD_ANYDEPTH);
			}

			cvtColor(color_img, color_img, CV_BGR2YUV);
			imgs.push_back(color_img);
			imgs2.push_back(depth_img);
		}
	}
	else {
		for (int i = 0; i < total_num_cameras; i++)
		{
			int camera = camera_order[i];

			folder_path = path + "/cam" + to_string(camera) + "/";
			Mat color_img, depth_img;

			if (data_mode == MSR3DVideo_Ballet) {
				folder_path = path + "/cam" + to_string(camera) + "/";
				color_img = imread(folder_path + color_names[camera][frame]);
				depth_img = imread(folder_path + depth_names[camera][frame]);
			}
			else if (data_mode >= S01_H1) {
				color_img = imread(path + "/RGB/col/" + color_names[camera][frame]);
				depth_img = imread(path + "/RGB/dep/" + depth_names[camera][frame], IMREAD_ANYDEPTH);
			}

			cvtColor(color_img, color_img, CV_BGR2YUV);
			imgs.push_back(color_img);
			imgs2.push_back(depth_img);
		}
	}
	
	color_imgs = imgs;
	depth_imgs = imgs2;
}

void get_color_and_depth_imgs(
	int frame,
	vector<string> color_names_,
	vector<string> depth_names_,
	vector<Mat>& color_imgs,
	vector<Mat>& depth_imgs,
	int color_bits,
	int depth_bits)
{
	vector<Mat> imgs(total_num_cameras);
	vector<Mat> imgs2(total_num_cameras);

	for (int camera = 0; camera < total_num_cameras; camera++)
	{
		Mat color_img, depth_img;

		switch (data_mode) {
		case Poznan_Fencing:
			color_img = readYUV(path + "\\" + color_names_[camera], frame, color_bits);
			if (depth_bits == 8) depth_img = cvt_yuv2bgr(path + "\\" + depth_names_[camera], frame, depth_bits);
			else depth_img = cvt_yuv2bgr(path + "\\" + depth_names_[camera], frame, depth_bits, 0);
			break;

		case Intel_Kermit:
		case Technicolor_Painter:
			color_img = readYUV(path + "\\" + color_names_[camera], frame, color_bits);
			depth_img = cvt_yuv2bgr(path + "\\" + depth_names_[camera], frame, depth_bits);

			break;
		}

		imgs[camera] = color_img;
		imgs2[camera] = depth_img;
	}

	color_imgs = imgs;
	depth_imgs = imgs2;
}

void find_min_max(
	PointCloud<PointXYZRGB>::Ptr source_PC,
	vector<float>& min,
	vector<float>& max)
{
	for (int p = 0; p < 3; p++)
	{
		min[p] = FLT_MAX;

		max[p] = FLT_MIN;
	}

	for (int i = 0; i < source_PC->points.size(); i++)
	{
		if (source_PC->points[i].x < min[0]) min[0] = source_PC->points[i].x;
		if (source_PC->points[i].y < min[1]) min[1] = source_PC->points[i].y;
		if (source_PC->points[i].z < min[2]) min[2] = source_PC->points[i].z;

		if (source_PC->points[i].x > max[0]) max[0] = source_PC->points[i].x;
		if (source_PC->points[i].y > max[1]) max[1] = source_PC->points[i].y;
		if (source_PC->points[i].z > max[2]) max[2] = source_PC->points[i].z;
	}
}

void find_min_max(
	vector<PointCloud<PointXYZRGB>::Ptr> vec_PC,
	vector<float>& min,
	vector<float>& max)
{
	for (int p = 0; p < 3; p++)
	{
		min[p] = FLT_MAX;

		max[p] = FLT_MIN;
	}

	for (int pc_idx = 0; pc_idx < vec_PC.size(); pc_idx++) {
		for (int i = 0; i < vec_PC[pc_idx]->points.size(); i++)
		{
			if (vec_PC[pc_idx]->points[i].x < min[0]) min[0] = vec_PC[pc_idx]->points[i].x;
			if (vec_PC[pc_idx]->points[i].y < min[1]) min[1] = vec_PC[pc_idx]->points[i].y;
			if (vec_PC[pc_idx]->points[i].z < min[2]) min[2] = vec_PC[pc_idx]->points[i].z;

			if (vec_PC[pc_idx]->points[i].x > max[0]) max[0] = vec_PC[pc_idx]->points[i].x;
			if (vec_PC[pc_idx]->points[i].y > max[1]) max[1] = vec_PC[pc_idx]->points[i].y;
			if (vec_PC[pc_idx]->points[i].z > max[2]) max[2] = vec_PC[pc_idx]->points[i].z;
		}
	}
}

void find_min_max(
	vector<PPC*> source_PC,
	vector<float>& min,
	vector<float>& max)
{
	for (int p = 0; p < 3; p++)
	{
		min[p] = FLT_MAX;

		max[p] = FLT_MIN;
	}

	for (int i = 0; i < source_PC.size(); i++)
	{
		float* geo = source_PC[i]->GetGeometry();
		if (geo[0] < min[0]) min[0] = geo[0];
		if (geo[1] < min[1]) min[1] = geo[1];
		if (geo[2] < min[2]) min[2] = geo[2];

		if (geo[0] > max[0]) max[0] = geo[0];
		if (geo[1] > max[1]) max[1] = geo[1];
		if (geo[2] > max[2]) max[2] = geo[2];
	}
}

void view_PC(PointCloud<PointXYZRGB>::Ptr pointcloud)
{
	int v1 = 0;

	PCLVisualizer viewer("PC viewer demo");
	viewer.setSize(1280, 1000);
	viewer.createViewPort(0.0, 0.0, 1.0, 1.0, v1);
	viewer.addCoordinateSystem(5.0);
	
	PointCloudColorHandlerRGBField<pcl::PointXYZRGB > rgb_handler(pointcloud);
	viewer.addPointCloud(pointcloud, rgb_handler, "result", v1);
	while (!viewer.wasStopped()) viewer.spinOnce();
}

void view_PC_yuvTorgb(PointCloud<PointXYZRGB>::Ptr pointcloud)
{
	int v1 = 0;

	PCLVisualizer viewer("PC viewer demo");
	viewer.setSize(1280, 1000);
	viewer.createViewPort(0.0, 0.0, 1.0, 1.0, v1);
	viewer.addCoordinateSystem(5.0);

	for (int i = 0; i < pointcloud->points.size(); i++) {
		Mat m(1, 1, CV_8UC3);

		m.at<Vec3b>(0, 0)[0] = pointcloud->points[i].b;
		m.at<Vec3b>(0, 0)[1] = pointcloud->points[i].g;
		m.at<Vec3b>(0, 0)[2] = pointcloud->points[i].r;

		cvtColor(m, m, CV_YUV2BGR);

		pointcloud->points[i].b = m.at<Vec3b>(0, 0)[0];
		pointcloud->points[i].g = m.at<Vec3b>(0, 0)[1];
		pointcloud->points[i].r = m.at<Vec3b>(0, 0)[2];
	}


	PointCloudColorHandlerRGBField<pcl::PointXYZRGB > rgb_handler(pointcloud);
	viewer.addPointCloud(pointcloud, rgb_handler, "result", v1);
	while (!viewer.wasStopped()) viewer.spinOnce();
}

void view_PC(PointCloud<PointXYZRGB>::Ptr pointcloud, int cam_idx)
{
	int v1 = 0;

	PCLVisualizer viewer("PC viewer demo");
	viewer.setSize(1280, 1000);
	viewer.createViewPort(0.0, 0.0, 1.0, 1.0, v1);
	//viewer.addCoordinateSystem(5.0);

	Matrix3f intrinsic = m_CalibParams[cam_idx].m_K.cast<float>();
	Matrix4f extrinsic = Matrix4f::Identity();
	Matrix3d invRotation = m_CalibParams[cam_idx].m_RotMatrix.transpose();
	Matrix3Xd invTranslation = -invRotation * m_CalibParams[cam_idx].m_Trans;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			extrinsic(i, j) = float(invRotation(i, j));
		}
		extrinsic(i, 3) = float(invTranslation(i, 0));
	}

	viewer.setCameraParameters(intrinsic, extrinsic);

	PointCloudColorHandlerRGBField<pcl::PointXYZRGB > rgb_handler(pointcloud);
	viewer.addPointCloud(pointcloud, rgb_handler, "result", v1);
	while (!viewer.wasStopped()) viewer.spinOnce();
}

void projection(PointCloud<PointXYZRGB>::Ptr pointcloud, int camera, Mat& img, Mat& depthimg, Mat& is_hole_img)
{
	PointCloud<PointXYZRGB>::iterator cloudit;
	PointCloud<PointXYZRGB>::iterator cloudend;

	double X;
	double Y;
	double Z;
	int u;
	int v;

	double dist;
	double w;

	for (cloudit = pointcloud->points.begin(), cloudend = pointcloud->points.end(); cloudit < cloudend; cloudit++) {

		X = cloudit->x;
		Y = cloudit->y;
		Z = cloudit->z;

		//Z = Z;//-Z

		w = projection_XYZ_2_UV(
			m_CalibParams[camera].m_ProjMatrix,
			X,
			Y,
			Z,
			u,
			v);

		dist = find_point_dist(w, camera);

		if ((u < 0) || (v < 0) || (u > _width - 1) || (v > _height - 1)) continue;

		if (depthimg.at<double>(v, u) == -1)
			depthimg.at<double>(v, u) = dist;
		else
		{
			if (dist < depthimg.at<double>(v, u))
				depthimg.at<double>(v, u) = dist;

			else continue;
		}

		img.at<Vec3b>(v, u)[0] = ushort(cloudit->b);
		img.at<Vec3b>(v, u)[1] = ushort(cloudit->g);
		img.at<Vec3b>(v, u)[2] = ushort(cloudit->r);
	}

	for (int v = 0; v < _height; v++){
		for (int u = 0; u < _width; u++) {
			if (depthimg.at<double>(v, u) == -1)
				is_hole_img.at<uchar>(v, u) = 1;
			else
				is_hole_img.at<uchar>(v, u) = 0;
		}
	}
}

void projection_bypoint(PointXYZRGB p, int camera, Mat& img, Mat& dist_img, Mat& is_hole_img)
{
	int u;
	int v;

	double dist;
	double w;


	double X = p.x;
	double Y = p.y;
	double Z = p.z;

	w = projection_XYZ_2_UV(
		m_CalibParams[camera].m_ProjMatrix,
		X,
		Y,
		Z,
		u,
		v);

	dist = find_point_dist(w, camera);

	if ((u < 0) || (v < 0) || (u > _width - 1) || (v > _height - 1)) return;

	if (dist_img.at<double>(v, u) == -1) {
		dist_img.at<double>(v, u) = dist;
		is_hole_img.at<uchar>(v, u) = 0;
	}
	else
	{
		if (dist < dist_img.at<double>(v, u))
			dist_img.at<double>(v, u) = dist;

		else return;
	}

	img.at<Vec3b>(v, u)[0] = uchar(p.b);
	img.at<Vec3b>(v, u)[1] = uchar(p.g);
	img.at<Vec3b>(v, u)[2] = uchar(p.r);
}

double det(double mat[3][3])
{
	double D = 0;

	D = mat[0][0] * ((mat[1][1] * mat[2][2]) - (mat[2][1] * mat[1][2]))
		- mat[0][1] * (mat[1][0] * mat[2][2] - mat[2][0] * mat[1][2])
		+ mat[0][2] * (mat[1][0] * mat[2][1] - mat[2][0] * mat[1][1]);

	return D;
}

// BGR to Gray
void printPSNRWithBlackPixel_RGB(
	Mat orig_img,
	Mat proj_img)
{
	cout << "PSNR with hole." << endl;

	float mse, psnr, tmp = 0;
	float sum = 0;
	int cnt = 0;

	cvtColor(orig_img, orig_img, COLOR_BGR2GRAY);
	cvtColor(proj_img, proj_img, COLOR_BGR2GRAY);

	int n = 0;

	for (int v = 0; v < _height; v++)
		for (int u = 0; u < _width; u++) {
			if (proj_img.at<uchar>(v, u) == 0) {
				n++;
			}
			tmp = orig_img.at<uchar>(v, u) - proj_img.at<uchar>(v, u);
			cnt++;
			sum += tmp * tmp;
		}

	mse = sum / cnt;
	psnr = 10 * log10(255 * 255 / mse);

	cout << "PSNR : " << psnr << endl;
	cout << "number of black pixels : " << n << endl;
}

// BGR 채널별 따로
void printPSNRWithBlackPixel_RGB(
	vector<Mat> orig_imgs,
	vector<Mat> proj_imgs,
	vector<Mat> is_hole_filled_imgs,
	vector<float>& psnrs_b,
	vector<float>& psnrs_g,
	vector<float>& psnrs_r,
	vector<int>& num_holes)
{
	cout << "PSNR with hole." << endl;
	//////////////////////////
	float avgPSNR_b= 0, avgPSNR_g = 0, avgPSNR_r = 0.0;
	float avgNumofPixel_b = 0, avgNumofPixel_g = 0, avgNumofPixel_r = 0.0;

	vector<float> PSNR_vec_b, PSNR_vec_g, PSNR_vec_r;
	vector<int> hole_num_vec;

	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++)
	{
		float mse_b = 0, psnr_b = 0, tmp_b = 0;
		float mse_g = 0, psnr_g = 0, tmp_g = 0;
		float mse_r = 0, psnr_r = 0, tmp_r = 0;

		float sum_b = 0, sum_g = 0, sum_r = 0;
		int cnt_b = 0, cnt_g = 0, cnt_r = 0;

		Mat bgr_orig[3];
		Mat bgr_proj[3];

		//Mat orig_ = orig_imgs[cam_num];
		//Mat proj_ = proj_imgs[cam_num];

		Mat orig_ = orig_imgs[cam_num].clone();
		Mat proj_ = proj_imgs[cam_num].clone();


		cvtColor(orig_, orig_, COLOR_YUV2BGR);
		cvtColor(proj_, proj_, COLOR_YUV2BGR);

		split(orig_, bgr_orig);
		split(proj_, bgr_proj);

		int n = 0;

		for (int v = 0; v < _height; v++)
			for (int u = 0; u < _width; u++) {

				if(is_hole_filled_imgs[cam_num].at<uchar>(v, u) ==1)
				//if (bgr_proj[0].at<uchar>(v, u) == 0 && bgr_proj[1].at<uchar>(v, u) == 0 && bgr_proj[2].at<uchar>(v, u) == 0) {
					n++;
				

				tmp_b = bgr_orig[0].at<uchar>(v, u) - bgr_proj[0].at<uchar>(v, u);
				cnt_b++;
				sum_b += tmp_b * tmp_b;

				tmp_g = bgr_orig[1].at<uchar>(v, u) - bgr_proj[1].at<uchar>(v, u);
				cnt_g++;
				sum_g += tmp_g * tmp_g;

				tmp_r = bgr_orig[2].at<uchar>(v, u) - bgr_proj[2].at<uchar>(v, u);
				cnt_r++;
				sum_r += tmp_r * tmp_r;
			}

		mse_b = sum_b / cnt_b;
		psnr_b = 10 * log10(255 * 255 / mse_b);

		mse_g = sum_g / cnt_g;
		psnr_g = 10 * log10(255 * 255 / mse_g);

		mse_r = sum_r / cnt_r;
		psnr_r = 10 * log10(255 * 255 / mse_r);

		PSNR_vec_b.push_back(psnr_b);
		PSNR_vec_g.push_back(psnr_g);
		PSNR_vec_r.push_back(psnr_r);
		hole_num_vec.push_back(n);

		avgPSNR_b += psnr_b;
		avgNumofPixel_b += n;

		avgPSNR_g += psnr_g;
		avgNumofPixel_g += n;

		avgPSNR_r += psnr_r;
		avgNumofPixel_r += n;

		num_holes.push_back(n);
		psnrs_b.push_back(psnr_b);
		psnrs_g.push_back(psnr_g);
		psnrs_r.push_back(psnr_r);
	}

	cout << "num of holes ::::::::::::::::" << endl;

	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++) {
		cout << "cam" << cam_num << " : " << num_holes[cam_num] << endl;
	}

	cout << "PSNR with black pixel ::::::::::::::::::" << endl;

	cout << "B channel :::::::::::::::" << endl;
	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++) {
		cout << "cam" << cam_num << " : " << psnrs_b[cam_num] << endl;
	}

	cout << "G channel :::::::::::::::" << endl;
	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++) {
		cout << "cam" << cam_num << " : " << psnrs_g[cam_num] << endl;
	}

	cout << "R channel :::::::::::::::" << endl;
	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++) {
		cout << "cam" << cam_num << " : " << psnrs_r[cam_num] << endl;
	}


	avgPSNR_b /= total_num_cameras;
	avgNumofPixel_b /= total_num_cameras;

	avgPSNR_g /= total_num_cameras;
	avgNumofPixel_g /= total_num_cameras;

	avgPSNR_r /= total_num_cameras;
	avgNumofPixel_r /= total_num_cameras;
}

void calcPSNRWithBlackPixel_RGB_per_viewpoint(
	int cam_num,
	Mat orig_img,
	Mat proj_img,
	Mat is_hole_filled_img,
	vector<float>& psnrs_b,
	vector<float>& psnrs_g,
	vector<float>& psnrs_r,
	vector<int>& num_holes)
{
	//////////////////////////

	float mse_b = 0, psnr_b = 0, tmp_b = 0;
	float mse_g = 0, psnr_g = 0, tmp_g = 0;
	float mse_r = 0, psnr_r = 0, tmp_r = 0;

	float sum_b = 0, sum_g = 0, sum_r = 0;
	int cnt_b = 0, cnt_g = 0, cnt_r = 0;

	Mat bgr_orig[3];
	Mat bgr_proj[3];

	Mat orig_ = orig_img.clone();
	Mat proj_ = proj_img.clone();

	cvtColor(orig_, orig_, COLOR_YUV2BGR);
	cvtColor(proj_, proj_, COLOR_YUV2BGR);

	split(orig_, bgr_orig);
	split(proj_, bgr_proj);

	int n = 0;

	for (int v = 0; v < _height; v++)
		for (int u = 0; u < _width; u++) {

			if (is_hole_filled_img.at<uchar>(v, u) == 1)
				//if (bgr_proj[0].at<uchar>(v, u) == 0 && bgr_proj[1].at<uchar>(v, u) == 0 && bgr_proj[2].at<uchar>(v, u) == 0) {
				n++;

			tmp_b = bgr_orig[0].at<uchar>(v, u) - bgr_proj[0].at<uchar>(v, u);
			cnt_b++;
			sum_b += tmp_b * tmp_b;

			tmp_g = bgr_orig[1].at<uchar>(v, u) - bgr_proj[1].at<uchar>(v, u);
			cnt_g++;
			sum_g += tmp_g * tmp_g;

			tmp_r = bgr_orig[2].at<uchar>(v, u) - bgr_proj[2].at<uchar>(v, u);
			cnt_r++;
			sum_r += tmp_r * tmp_r;
		}

	mse_b = sum_b / cnt_b;
	psnr_b = 10 * log10(255 * 255 / mse_b);

	mse_g = sum_g / cnt_g;
	psnr_g = 10 * log10(255 * 255 / mse_g);

	mse_r = sum_r / cnt_r;
	psnr_r = 10 * log10(255 * 255 / mse_r);

	num_holes.push_back(n);
	psnrs_b.push_back(psnr_b);
	psnrs_g.push_back(psnr_g);
	psnrs_r.push_back(psnr_r);
}

// BGR to Gray
void printPSNRWithoutBlackPixel_RGB(
	Mat orig_img,
	Mat proj_img)
{
	cout << "PSNR without black hole." << endl;

	float mse, psnr, tmp = 0;
	float sum = 0;
	int cnt = 0;

	cvtColor(orig_img, orig_img, COLOR_BGR2GRAY);
	cvtColor(proj_img, proj_img, COLOR_BGR2GRAY);

	int n = 0;

	for (int v = 0; v < _height; v++)
		for (int u = 0; u < _width; u++) {
			if (proj_img.at<uchar>(v, u) == 0) {
				n++;
			}
			else {
				tmp = orig_img.at<uchar>(v, u) - proj_img.at<uchar>(v, u);
				cnt++;
				sum += tmp * tmp;
			}
		}

	mse = sum / cnt;
	psnr = 10 * log10(255 * 255 / mse);

	cout << "PSNR : " << psnr << endl;
	cout << "number of black pixels : " << n << endl;
}

// BGR each channel
void printPSNRWithoutBlackPixel_RGB(
	vector<Mat> orig_imgs,
	vector<Mat> proj_imgs,
	vector<Mat>is_hole_proj_imgs,
	vector<float>& psnrs_b,
	vector<float>& psnrs_g,
	vector<float>& psnrs_r,
	vector<int>& num_holes)
{
	cout << "PSNR without black hole." << endl;
	float avgPSNR_b = 0, avgPSNR_g = 0, avgPSNR_r = 0.0;
	float avgNumofPixel_b = 0, avgNumofPixel_g = 0, avgNumofPixel_r = 0.0;

	vector<float> PSNR_vec_b, PSNR_vec_g, PSNR_vec_r;
	vector<int> hole_num_vec;

	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++)
	{
		float mse_b = 0, psnr_b = 0, tmp_b = 0;
		float mse_g = 0, psnr_g = 0, tmp_g = 0;
		float mse_r = 0, psnr_r = 0, tmp_r = 0;

		float sum_b = 0, sum_g = 0, sum_r = 0;
		int cnt_b = 0, cnt_g = 0, cnt_r = 0;

		Mat bgr_orig[3];
		Mat bgr_proj[3];

		//Mat orig_ = orig_imgs[cam_num];
		//Mat proj_ = proj_imgs[cam_num];

		Mat orig_ = orig_imgs[cam_num].clone();
		Mat proj_ = proj_imgs[cam_num].clone();

		cvtColor(orig_, orig_, COLOR_YUV2BGR);
		cvtColor(proj_, proj_, COLOR_YUV2BGR);

		split(orig_, bgr_orig);
		split(proj_, bgr_proj);

		int n = 0;

		for (int v = 0; v < _height; v++)
			for (int u = 0; u < _width; u++) {

				if (is_hole_proj_imgs[cam_num].at<bool>(v, u))
				//if (bgr_proj[0].at<uchar>(v, u) == 0 && bgr_proj[1].at<uchar>(v, u) == 0 && bgr_proj[2].at<uchar>(v, u) == 0) {
					n++;
				

				else {
					tmp_b = bgr_orig[0].at<uchar>(v, u) - bgr_proj[0].at<uchar>(v, u);
					cnt_b++;
					sum_b += tmp_b * tmp_b;

					tmp_g = bgr_orig[1].at<uchar>(v, u) - bgr_proj[1].at<uchar>(v, u);
					cnt_g++;
					sum_g += tmp_g * tmp_g;

					tmp_r = bgr_orig[2].at<uchar>(v, u) - bgr_proj[2].at<uchar>(v, u);
					cnt_r++;
					sum_r += tmp_r * tmp_r;

				}
			}

		mse_b = sum_b / cnt_b;
		psnr_b = 10 * log10(255 * 255 / mse_b);

		mse_g = sum_g / cnt_g;
		psnr_g = 10 * log10(255 * 255 / mse_g);

		mse_r = sum_r / cnt_r;
		psnr_r = 10 * log10(255 * 255 / mse_r);

		PSNR_vec_b.push_back(psnr_b);
		PSNR_vec_g.push_back(psnr_g);
		PSNR_vec_r.push_back(psnr_r);
		hole_num_vec.push_back(n);

		avgPSNR_b += psnr_b;
		avgNumofPixel_b += n;

		avgPSNR_g += psnr_g;
		avgNumofPixel_g += n;

		avgPSNR_r += psnr_r;
		avgNumofPixel_r += n;

		num_holes.push_back(n);
		psnrs_b.push_back(psnr_b);
		psnrs_g.push_back(psnr_g);
		psnrs_r.push_back(psnr_r);
	}

	cout << "num of holes ::::::::::::::::" << endl;

	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++) {
		cout << "cam" << cam_num << " : " << num_holes[cam_num] << endl;
	}

	cout << "PSNR with black pixel ::::::::::::::::::" << endl;

	cout << "B channel :::::::::::::::" << endl;
	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++) {
		cout << "cam" << cam_num << " : " << psnrs_b[cam_num] << endl;
	}

	cout << "G channel :::::::::::::::" << endl;
	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++) {
		cout << "cam" << cam_num << " : " << psnrs_g[cam_num] << endl;
	}

	cout << "R channel :::::::::::::::" << endl;
	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++) {
		cout << "cam" << cam_num << " : " << psnrs_r[cam_num] << endl;
	}

	avgPSNR_b /= total_num_cameras;
	avgNumofPixel_b /= total_num_cameras;

	avgPSNR_g /= total_num_cameras;
	avgNumofPixel_g /= total_num_cameras;

	avgPSNR_r /= total_num_cameras;
	avgNumofPixel_r /= total_num_cameras;
}

void calcPSNRWithoutBlackPixel_RGB_per_viewpoint(
	int cam_num,
	Mat orig_img,
	Mat proj_img,
	Mat is_hole_proj_img,
	vector<float>& psnrs_b,
	vector<float>& psnrs_g,
	vector<float>& psnrs_r,
	vector<int>& num_holes)
{

	float mse_b = 0, psnr_b = 0, tmp_b = 0;
	float mse_g = 0, psnr_g = 0, tmp_g = 0;
	float mse_r = 0, psnr_r = 0, tmp_r = 0;

	float sum_b = 0, sum_g = 0, sum_r = 0;
	int cnt_b = 0, cnt_g = 0, cnt_r = 0;

	Mat bgr_orig[3];
	Mat bgr_proj[3];

	//Mat orig_ = orig_imgs[cam_num];
	//Mat proj_ = proj_imgs[cam_num];

	Mat orig_ = orig_img.clone();
	Mat proj_ = proj_img.clone();

	cvtColor(orig_, orig_, COLOR_YUV2BGR);
	cvtColor(proj_, proj_, COLOR_YUV2BGR);

	split(orig_, bgr_orig);
	split(proj_, bgr_proj);

	int n = 0;

	for (int v = 0; v < _height; v++)
		for (int u = 0; u < _width; u++) {

			if (is_hole_proj_img.at<bool>(v, u))
				//if (bgr_proj[0].at<uchar>(v, u) == 0 && bgr_proj[1].at<uchar>(v, u) == 0 && bgr_proj[2].at<uchar>(v, u) == 0) {
				n++;

			else {
				tmp_b = bgr_orig[0].at<uchar>(v, u) - bgr_proj[0].at<uchar>(v, u);
				cnt_b++;
				sum_b += tmp_b * tmp_b;

				tmp_g = bgr_orig[1].at<uchar>(v, u) - bgr_proj[1].at<uchar>(v, u);
				cnt_g++;
				sum_g += tmp_g * tmp_g;

				tmp_r = bgr_orig[2].at<uchar>(v, u) - bgr_proj[2].at<uchar>(v, u);
				cnt_r++;
				sum_r += tmp_r * tmp_r;

			}
		}

	mse_b = sum_b / cnt_b;
	psnr_b = 10 * log10(255 * 255 / mse_b);

	mse_g = sum_g / cnt_g;
	psnr_g = 10 * log10(255 * 255 / mse_g);

	mse_r = sum_r / cnt_r;
	psnr_r = 10 * log10(255 * 255 / mse_r);

	num_holes.push_back(n);
	psnrs_b.push_back(psnr_b);
	psnrs_g.push_back(psnr_g);
	psnrs_r.push_back(psnr_r);

}

void printPSNR(
	vector<float> psnrs_b,
	vector<float> psnrs_g,
	vector<float> psnrs_r,
	vector<int>& num_holes) {


	cout << "num of holes ::::::::::::::::" << endl;

	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++) {
		cout << "cam" << cam_num << " : " << num_holes[cam_num] << endl;
	}

	cout << "PSNR with black pixel ::::::::::::::::::" << endl;

	cout << "B channel :::::::::::::::" << endl;
	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++) {
		cout << "cam" << cam_num << " : " << psnrs_b[cam_num] << endl;
	}

	cout << "G channel :::::::::::::::" << endl;
	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++) {
		cout << "cam" << cam_num << " : " << psnrs_g[cam_num] << endl;
	}

	cout << "R channel :::::::::::::::" << endl;
	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++) {
		cout << "cam" << cam_num << " : " << psnrs_r[cam_num] << endl;
	}

}

// 3채널 통째로.
void printPSNRWithoutBlackPixel_2(
	vector<Mat> orig_imgs,
	vector<Mat> proj_imgs,
	vector<float>& psnrs,
	vector<int>& num_holes)
{
	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++)
	{
		double mse = 0, sum = 0;
		int cnt = 0;
		float tmp = 0;

		Mat yuv_orig[3];
		Mat yuv_proj[3];

		split(orig_imgs[cam_num], yuv_orig);
		split(proj_imgs[cam_num], yuv_proj);

		int n = 0;

		for (int v = 0; v < _height; v++)
			for (int u = 0; u < _width; u++) {

				if (yuv_proj[0].at<uchar>(v, u) == 0 && yuv_proj[1].at<uchar>(v, u) == 0 && yuv_proj[2].at<uchar>(v, u) == 0)
					n++;

				else {
					tmp = yuv_orig[0].at<uchar>(v, u) - yuv_proj[0].at<uchar>(v, u);
					cnt++;
					sum += tmp * tmp;

					tmp = yuv_orig[1].at<uchar>(v, u) - yuv_proj[1].at<uchar>(v, u);
					cnt++;
					sum += tmp * tmp;

					tmp = yuv_orig[2].at<uchar>(v, u) - yuv_proj[2].at<uchar>(v, u);
					cnt++;
					sum += tmp * tmp;

					/* cout << (int)yuv_orig[0].at<uchar>(v, u) << " " << (int)yuv_proj[0].at<uchar>(v, u) << endl;
					 cout << (int)yuv_orig[1].at<uchar>(v, u) << " " << (int)yuv_proj[1].at<uchar>(v, u) << endl;
					 cout << (int)yuv_orig[2].at<uchar>(v, u) << " " << (int)yuv_proj[2].at<uchar>(v, u) << endl;
					 cout << "======" << endl;*/

				}
			}

		mse = sum / cnt;
		float psnr = 10 * log10(255 * 255 / mse);

		num_holes.push_back(n);
		psnrs.push_back(psnr);
	}

	cout << "num of holes ::::::::::::::::" << endl;

	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++) {
		cout << "cam" << cam_num << " : " << num_holes[cam_num] << endl;
	}

	cout << "PSNR without black pixel ::::::::::::::::::" << endl;

	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++) {
		cout << "cam" << cam_num << " : " << psnrs[cam_num] << endl;
	}
}

// 3채널 통째로.
void printPSNRWithBlackPixel_2(
	vector<Mat> orig_imgs,
	vector<Mat> proj_imgs,
	vector<float>& psnrs)
{
	float avgPSNR = 0.0;
	float avgNumofPixel = 0.0;

	vector<float> PSNR_vec;
	vector<int> hole_num_vec;

	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++)
	{
		double mse = 0, sum = 0;
		int cnt = 0;
		float tmp = 0;

		Mat yuv_orig[3];
		Mat yuv_proj[3];

		split(orig_imgs[cam_num], yuv_orig);
		split(proj_imgs[cam_num], yuv_proj);

		int n = 0;

		for (int v = 0; v < _height; v++)
			for (int u = 0; u < _width; u++) {

				if (yuv_proj[0].at<uchar>(v, u) == 0 && yuv_proj[1].at<uchar>(v, u) == 0 && yuv_proj[2].at<uchar>(v, u) == 0) {
					n++;
				}

				tmp = yuv_orig[0].at<uchar>(v, u) - yuv_proj[0].at<uchar>(v, u);
				cnt++;
				sum += tmp * tmp;

				tmp = yuv_orig[1].at<uchar>(v, u) - yuv_proj[1].at<uchar>(v, u);
				cnt++;
				sum += tmp * tmp;

				tmp = yuv_orig[2].at<uchar>(v, u) - yuv_proj[2].at<uchar>(v, u);
				cnt++;
				sum += tmp * tmp;
			}

		mse = sum / cnt;
		float psnr = 10 * log10(255 * 255 / mse);

		PSNR_vec.push_back(psnr);
		hole_num_vec.push_back(n);

		avgPSNR += psnr;
		avgNumofPixel += n;

		psnrs.push_back(psnr);
	}
	cout << "PSNR with black pixel ::::::::::::::::::" << endl;

	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++) {
		cout << "cam" << cam_num << " : " << psnrs[cam_num] << endl;
	}

	avgPSNR /= total_num_cameras;
	avgNumofPixel /= total_num_cameras;
}

void back_projection(
	PointCloud<PointXYZRGB>::Ptr pointcloud,
	int camera,
	Mat& img,
	Mat& is_hole_img,
	int nNeighbor)
{

	if (nNeighbor != 4 && nNeighbor != 8 && nNeighbor != 12 && nNeighbor != 20 && nNeighbor != 24) {
		cerr << "Wrong neighbor number!" << endl;
		exit(0);
	}

	vector<vector<vector<PointXYZRGB>>> point_storage(img.rows, vector<vector<PointXYZRGB>>(img.cols));
	vector<vector<vector<double>>> point_dist_storage(img.rows, vector<vector<double>>(img.cols));

	Matrix<double, 3, 1> camera_center;
	Matrix<double, 4, 1> world_2d_;
	Matrix<double, 4, 4> RT;

	Mat geo_mat(Size(img.cols, img.rows), CV_64FC3);

	Vec3d temp1, temp2;

	double X;
	double Y;
	double Z;
	int u;
	int v;

	double dist;
	double w;

	for (int i = 0; i < pointcloud->points.size(); i++)
	{
		PointXYZRGB temp_point;

		X = pointcloud->points[i].x;
		Y = pointcloud->points[i].y;
		Z = pointcloud->points[i].z;

		//fencing ballet
		//Z = -Z;
		w = projection_XYZ_2_UV(
			m_CalibParams[camera].m_ProjMatrix,
			X,
			Y,
			Z,
			u,
			v);

		dist = find_point_dist(w, camera);

		if ((u < 0) || (v < 0) || (u > _width - 1) || (v > _height - 1)) continue;

		temp_point.x = pointcloud->points[i].x;
		temp_point.y = pointcloud->points[i].y;
		temp_point.z = pointcloud->points[i].z;
		temp_point.r = pointcloud->points[i].r;
		temp_point.g = pointcloud->points[i].g;
		temp_point.b = pointcloud->points[i].b;

		point_storage[v][u].push_back(temp_point);
		point_dist_storage[v][u].push_back(dist);

	}

	///////////////////////////////////////////////////////////////////// BACK PROJECTION
	camera_center = -(m_CalibParams[camera].m_RotMatrix.transpose()) * m_CalibParams[camera].m_Trans;

	double cx = m_CalibParams[camera].m_K(2);
	double cy = m_CalibParams[camera].m_K(5);
	double focal_length = m_CalibParams[camera].m_K(0);

	RT << m_CalibParams[camera].m_RotMatrix(0, 0), m_CalibParams[camera].m_RotMatrix(0, 1), m_CalibParams[camera].m_RotMatrix(0, 2), m_CalibParams[camera].m_Trans(0),
		m_CalibParams[camera].m_RotMatrix(1, 0), m_CalibParams[camera].m_RotMatrix(1, 1), m_CalibParams[camera].m_RotMatrix(1, 2), m_CalibParams[camera].m_Trans(1),
		m_CalibParams[camera].m_RotMatrix(2, 0), m_CalibParams[camera].m_RotMatrix(2, 1), m_CalibParams[camera].m_RotMatrix(2, 2), m_CalibParams[camera].m_Trans(2),
		0, 0, 0, 1;


	for (int i = 0; i < img.cols; i++) {
		for (int j = 0; j < img.rows; j++) {
			world_2d_ << (j - cx), (i - cy), focal_length, 1;
			world_2d_ = RT.inverse() * world_2d_;

			geo_mat.at<Vec3d>(j, i)[0] = world_2d_(0);
			geo_mat.at<Vec3d>(j, i)[1] = world_2d_(1);
			geo_mat.at<Vec3d>(j, i)[2] = world_2d_(2);
		}
	}

	for (int i = 0; i < img.cols; i++) {
		for (int j = 0; j < img.rows; j++) {
			double minTanAngle = DBL_MAX;
			double minDepth = DBL_MAX;

			Vec3b temp_color;

			bool is_hole = 1;

			temp2[0] = geo_mat.at<Vec3d>(j, i)[0] - camera_center(0);
			temp2[1] = geo_mat.at<Vec3d>(j, i)[1] - camera_center(1);
			temp2[2] = geo_mat.at<Vec3d>(j, i)[2] - camera_center(2);

			if (point_storage[j][i].size() > 0) {
				vector<PointXYZRGB> point_vec = point_storage[j][i];
				for (int k = 0; k < point_vec.size(); k++) {
					PointXYZRGB pt = point_vec[k];
					double dist = point_dist_storage[j][i][k];

					if (minDepth > dist) {
						minDepth = dist;
						temp_color[0] = pt.b;
						temp_color[1] = pt.g;
						temp_color[2] = pt.r;
						is_hole = 0;
					}
				}
			}
			else {
				if (nNeighbor >= 4) {
					if (j > 0) {
						vector<PointXYZRGB> point_vec = point_storage[j - 1][i];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j - 1][i][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
								is_hole = 0;
							}
						}
					}

					if (i > 0) {
						vector<PointXYZRGB> point_vec = point_storage[j][i - 1];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j][i - 1][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
								is_hole = 0;
							}
						}
					}

					if (j < img.rows - 1) {
						vector<PointXYZRGB> point_vec = point_storage[j + 1][i];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j + 1][i][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
								is_hole = 0;
							}
						}
					}

					if (i < img.cols - 1) {
						vector<PointXYZRGB> point_vec = point_storage[j][i + 1];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j][i + 1][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
								is_hole = 0;
							}
						}
					}
				}
				/*
				if (nNeighbor >= 8) {
					if (j > 0 && i > 0) {
						vector<PointXYZRGB> point_vec = point_storage[j - 1][i - 1];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j - 1][i - 1][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
							}
						}
					}

					if (j > 0 && i < img.cols - 1) {
						vector<PointXYZRGB> point_vec = point_storage[j - 1][i + 1];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j - 1][i + 1][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
							}
						}
					}

					if (j < img.rows - 1 && i > 0) {
						vector<PointXYZRGB> point_vec = point_storage[j + 1][i - 1];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j + 1][i - 1][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
							}
						}
					}

					if (j < img.rows - 1 && i < img.cols - 1) {
						vector<PointXYZRGB> point_vec = point_storage[j + 1][i + 1];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j + 1][i + 1][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
							}
						}
					}
				}
				if (nNeighbor >= 12) {
					if (j > 1) {
						vector<PointXYZRGB> point_vec = point_storage[j - 2][i];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j - 2][i][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
							}
						}
					}

					if (i > 1) {
						vector<PointXYZRGB> point_vec = point_storage[j][i - 2];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j][i - 2][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
							}
						}
					}

					if (j < img.rows - 2) {
						vector<PointXYZRGB> point_vec = point_storage[j + 2][i];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j + 2][i][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
							}
						}
					}

					if (i < img.cols - 2) {
						vector<PointXYZRGB> point_vec = point_storage[j][i + 2];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j][i + 2][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
							}
						}
					}
				}
				if (nNeighbor >= 20) {
					if (j > 1 && i > 0) {
						vector<PointXYZRGB> point_vec = point_storage[j - 2][i - 1];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j - 2][i - 1][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
							}
						}
					}

					if (j > 1 && i < img.cols - 1) {
						vector<PointXYZRGB> point_vec = point_storage[j - 2][i + 1];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j - 2][i + 1][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
							}
						}
					}

					if (i < img.cols - 2 && j > 0) {
						vector<PointXYZRGB> point_vec = point_storage[j - 1][i + 2];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j - 1][i + 2][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
							}
						}
					}

					if (i < img.cols - 2 && j < img.rows - 1) {
						vector<PointXYZRGB> point_vec = point_storage[j + 1][i + 2];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j + 1][i + 2][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
							}
						}
					}

					if (j < img.rows - 2 && i < img.cols - 1) {
						vector<PointXYZRGB> point_vec = point_storage[j + 2][i + 1];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j + 2][i + 1][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
							}
						}
					}

					if (j < img.rows - 2 && i > 0) {
						vector<PointXYZRGB> point_vec = point_storage[j + 2][i - 1];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j + 2][i - 1][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
							}
						}
					}

					if (i > 1 && j < img.rows - 1) {
						vector<PointXYZRGB> point_vec = point_storage[j + 1][i - 2];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j + 1][i - 2][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
							}
						}
					}

					if (i > 1 && j > 0) {
						vector<PointXYZRGB> point_vec = point_storage[j - 1][i - 2];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j - 1][i - 2][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
							}
						}
					}
				}
				if (nNeighbor >= 24) {
					if (j > 1 && i > 1) {
						vector<PointXYZRGB> point_vec = point_storage[j - 2][i - 2];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j - 2][i - 2][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
							}
						}
					}

					if (j > 1 && i < img.cols - 2) {
						vector<PointXYZRGB> point_vec = point_storage[j - 2][i + 2];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j - 2][i + 2][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
							}
						}
					}

					if (j < img.rows - 2 && i < img.cols - 2) {
						vector<PointXYZRGB> point_vec = point_storage[j + 2][i + 2];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j + 2][i + 2][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
							}
						}
					}

					if (j < img.rows - 2 && i > 1) {
						vector<PointXYZRGB> point_vec = point_storage[j + 2][i - 2];
						for (int k = 0; k < point_vec.size(); k++) {
							PointXYZRGB pt = point_vec[k];
							double dist = point_dist_storage[j + 2][i - 2][k];

							temp1[0] = pt.x - camera_center(0);
							temp1[1] = pt.y - camera_center(1);
							temp1[2] = pt.z - camera_center(2);

							// double dist = norm(temp1.cross(temp2)) / norm(temp2);
							double tanAngle = abs(norm(temp1.cross(temp2)) / temp1.dot(temp2));

							if (minTanAngle > tanAngle) {
								minTanAngle = tanAngle;
								temp_color[0] = pt.b;
								temp_color[1] = pt.g;
								temp_color[2] = pt.r;
							}
						}
					}
				}
				*/
			}
			img.at<Vec3b>(j, i) = temp_color;
			if(is_hole) is_hole_img.at<uchar>(j, i) = 1;
			else is_hole_img.at<uchar>(j, i) = 0;
		}
	}

}

double projection_XYZ_2_UV(
	Matrix4d projMatrix,
	double x,
	double y,
	double z,
	int& u,
	int& v)
{
	double u_, v_;
	double w;
	u_ = projMatrix(0, 0) * x + projMatrix(0, 1) * y + projMatrix(0, 2) * z + projMatrix(0, 3);
	v_ = projMatrix(1, 0) * x + projMatrix(1, 1) * y + projMatrix(1, 2) * z + projMatrix(1, 3);
	w = projMatrix(2, 0) * x + projMatrix(2, 1) * y + projMatrix(2, 2) * z + projMatrix(2, 3);


	u_ /= w;
	v_ /= w;
	if (!data_mode) {
		v_ = _height - v_ - 1.0;
	}

	u = cvRound(u_);
	v = cvRound(v_);

	return w;
}

double find_point_dist(
	double w,
	int camera)
{
	double numerator=0., denominator=0., dist = 0.;
	double M[3][3];


	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			M[i][j] = m_CalibParams[camera].m_ProjMatrix(i, j);

	for (int i = 0; i < 3; i++)
		denominator = denominator + (M[2][i] * M[2][i]);

	denominator = sqrt(denominator);

	numerator = det(M);


	// sign
	if (numerator < 0) numerator = -1;

	else if (numerator == 0) numerator = 0;

	else numerator = 1;

	numerator = numerator * w;

	if (denominator == 0) cerr << "error" << endl;

	else dist = (numerator / denominator);

	return dist;

}

void RGB_dev(vector<PPC> PPC, vector<vector<float>>& dev_pointnum_percent, vector<float>& point_num_per_color)
{
	for (int point_num = 0; point_num < PPC.size(); point_num++) {

		float avr_r = 0, avr_g = 0, avr_b = 0;
		float avr_r_2 = 0, avr_g_2 = 0, avr_b_2 = 0;
		int cam_number = 0;
		for (int cam = 0; cam < total_num_cameras; cam++) {
			Mat hsv(1, 1, CV_8UC3);
			// ���԰� ��ħ.
			hsv.at<Vec3b>(0, 0) = PPC[point_num].GetColor(cam);
			cvtColor(hsv, hsv, CV_BGR2HSV);

			if (!PPC[point_num].CheckOcclusion(cam)) {
				cam_number++;
				avr_r += (float)PPC[point_num].GetColor(cam)[0];
				avr_g += (float)PPC[point_num].GetColor(cam)[1];
				avr_b += (float)PPC[point_num].GetColor(cam)[2];
				avr_r_2 += avr_r * avr_r;
				avr_g_2 += avr_g * avr_g;
				avr_b_2 += avr_b * avr_b;
			}
		}
		point_num_per_color[cam_number - 1] += 1;

		avr_r /= cam_number;
		avr_g /= cam_number;
		avr_b /= cam_number;
		avr_r_2 /= cam_number;
		avr_g_2 /= cam_number;
		avr_b_2 /= cam_number;

		float dev_r = 0, dev_g = 0, dev_b = 0;


		dev_r = sqrt(avr_r_2 - avr_r * avr_r);
		dev_g = sqrt(avr_g_2 - avr_g * avr_g);
		dev_b = sqrt(avr_b_2 - avr_b * avr_b);

		float avr_dev = (dev_r + dev_g + dev_b) / 3.0;
		if (avr_dev >= 0 && avr_dev < 5) {
			dev_pointnum_percent[cam_number - 1][0] += 1;
		}
		else if (avr_dev >= 5 && avr_dev < 10) {
			dev_pointnum_percent[cam_number - 1][1] += 1;
		}
		else if (avr_dev >= 10 && avr_dev < 15) {
			dev_pointnum_percent[cam_number - 1][2] += 1;
		}
		else if (avr_dev >= 15 && avr_dev < 20) {
			dev_pointnum_percent[cam_number - 1][3] += 1;
		}
		else if (avr_dev >= 20) {
			dev_pointnum_percent[cam_number - 1][4] += 1;
		}
	}

	for (int cam = 0; cam < total_num_cameras; cam++) {
		for (int i = 0; i < 5; i++) {
			dev_pointnum_percent[cam][i] = dev_pointnum_percent[cam][i] / (float)point_num_per_color[cam] * 100;
		}
	}


	int total = 0;

	//fout << "COLOR N���� point ���� :::::::::::::::::" << endl;

	for (int cam = 0; cam < total_num_cameras; cam++) {
		cout << cam + 1 << "��: " << point_num_per_color[cam] << "(" << point_num_per_color[cam] / PPC.size() * 100 << "%)" << endl;
		//fout << cam + 1 << "��: " << point_num_per_color[cam] << "(" << point_num_per_color[cam] / PPC.size() * 100 << "%)" << endl;
		//point_num_per_color[cam] = point_num_per_color[cam] / PPC.size() * 100;
		//cout << cam + 1 << "��: " << point_num_per_color[cam] << "%" << endl;
	}

	for (int cam = 0; cam < total_num_cameras; cam++) {
		cout << cam + 1 << "��: " << dev_pointnum_percent[cam][0] << "/" << dev_pointnum_percent[cam][1] << "/" << dev_pointnum_percent[cam][2] << "/" << dev_pointnum_percent[cam][3] << "/" << dev_pointnum_percent[cam][4] << endl;
		//fout << cam + 1 << "��: " << dev_pointnum_percent[cam][0] << "/" << dev_pointnum_percent[cam][1] << "/" << dev_pointnum_percent[cam][2] << "/" << dev_pointnum_percent[cam][3] << "/" << dev_pointnum_percent[cam][4] << endl;
	}
}

void HSV_dev(vector<PPC*> PPC, vector<vector<float>>& dev_pointnum_percent, vector<float>& point_num_per_color)
{
	for (int point_num = 0; point_num < PPC.size(); point_num++) {
		float avr_h = 0, avr_s = 0, avr_v = 0;
		float avr_h_2 = 0, avr_s_2 = 0, avr_v_2 = 0;
		int cam_number = 0;
		for (int cam = 0; cam < total_num_cameras; cam++) {

			Mat hsv(1, 1, CV_8UC3);
			// ���԰� ��ħ.
			hsv.at<Vec3b>(0, 0) = PPC[point_num]->GetColor(cam);
			cvtColor(hsv, hsv, CV_BGR2HSV);

			if (!PPC[point_num]->CheckOcclusion(cam)) {
				cam_number++;
				avr_h += (float)hsv.at<Vec3b>(0, 0)[0];
				avr_s += (float)hsv.at<Vec3b>(0, 0)[1];
				avr_v += (float)hsv.at<Vec3b>(0, 0)[2];
				avr_h_2 += float((float)hsv.at<Vec3b>(0, 0)[0] * (float)hsv.at<Vec3b>(0, 0)[0]);
				avr_s_2 += float((float)hsv.at<Vec3b>(0, 0)[1] * (float)hsv.at<Vec3b>(0, 0)[1]);
				avr_v_2 += float((float)hsv.at<Vec3b>(0, 0)[2] * (float)hsv.at<Vec3b>(0, 0)[2]);
			}
		}
		point_num_per_color[cam_number - 1] += 1;

		avr_h /= cam_number;
		avr_s /= cam_number;
		avr_v /= cam_number;
		avr_h_2 /= cam_number;
		avr_s_2 /= cam_number;
		avr_v_2 /= cam_number;

		float dev_h = 0, dev_s = 0, dev_v = 0;


		dev_h = sqrt(avr_h_2 - avr_h * avr_h);
		dev_s = sqrt(avr_s_2 - avr_s * avr_s);
		dev_v = sqrt(avr_v_2 - avr_v * avr_v);

		float avr_dev = (dev_h + dev_s + dev_v) / 3.0;


		if (avr_dev >= 0 && avr_dev < 5) {
			dev_pointnum_percent[cam_number - 1][0] += 1;
		}
		else if (avr_dev >= 5 && avr_dev < 10) {
			dev_pointnum_percent[cam_number - 1][1] += 1;
		}
		else if (avr_dev >= 10 && avr_dev < 15) {
			dev_pointnum_percent[cam_number - 1][2] += 1;
		}
		else if (avr_dev >= 15 && avr_dev < 20) {
			dev_pointnum_percent[cam_number - 1][3] += 1;
		}
		else if (avr_dev >= 20) {
			dev_pointnum_percent[cam_number - 1][4] += 1;
		}

	}

	//for (int cam = 0; cam < total_num_cameras; cam++) {
	//   for (int i = 0; i < 5; i++) {
	//      dev_pointnum_percent[cam][i] = dev_pointnum_percent[cam][i] / (float)point_num_per_color[cam] * 100;
	//   }
	//}


	//int total = 0;

	////fout << "COLOR N���� point ���� :::::::::::::::::" << endl;

	//for (int cam = 0; cam < total_num_cameras; cam++) {
	//   cout << cam + 1 << "��: " << point_num_per_color[cam] << "(" << point_num_per_color[cam] / PPC.size() * 100 << "%)" << endl;
	//   //fout << cam + 1 << "��: " << point_num_per_color[cam] << "(" << point_num_per_color[cam] / PPC.size() * 100 << "%)" << endl;
	//   //point_num_per_color[cam] = point_num_per_color[cam] / PPC.size() * 100;
	//   //cout << cam + 1 << "��: " << point_num_per_color[cam] << "%" << endl;
	//}

	//for (int cam = 0; cam < total_num_cameras; cam++) {
	//   cout << cam + 1 << "��: " << dev_pointnum_percent[cam][0] << "/" << dev_pointnum_percent[cam][1] << "/" << dev_pointnum_percent[cam][2] << "/" << dev_pointnum_percent[cam][3] << "/" << dev_pointnum_percent[cam][4] << endl;
	//   //fout << cam + 1 << "��: " << dev_pointnum_percent[cam][0] << "/" << dev_pointnum_percent[cam][1] << "/" << dev_pointnum_percent[cam][2] << "/" << dev_pointnum_percent[cam][3] << "/" << dev_pointnum_percent[cam][4] << endl;
	//}
}

void YUV_dev(vector<PPC*> PPC, vector<vector<float>>& dev_pointnum_percent, vector<float>& point_num_per_color)
{
	for (int point_num = 0; point_num < PPC.size(); point_num++) {
		float avr_y = 0, avr_u = 0, avr_v = 0;
		float avr_y_2 = 0, avr_u_2 = 0, avr_v_2 = 0;
		int cam_number = 0;
		for (int cam = 0; cam < total_num_cameras; cam++) {

			Mat yuv(1, 1, CV_8UC3);
			// ���԰� ��ħ.
			yuv.at<Vec3b>(0, 0) = PPC[point_num]->GetColor(cam);
			cvtColor(yuv, yuv, CV_BGR2YUV);

			if (!PPC[point_num]->CheckOcclusion(cam)) {
				cam_number++;
				avr_y += (float)yuv.at<Vec3b>(0, 0)[0];
				avr_u += (float)yuv.at<Vec3b>(0, 0)[1];
				avr_v += (float)yuv.at<Vec3b>(0, 0)[2];
				avr_y_2 += float((float)yuv.at<Vec3b>(0, 0)[0] * (float)yuv.at<Vec3b>(0, 0)[0]);
				avr_u_2 += float((float)yuv.at<Vec3b>(0, 0)[1] * (float)yuv.at<Vec3b>(0, 0)[1]);
				avr_v_2 += float((float)yuv.at<Vec3b>(0, 0)[2] * (float)yuv.at<Vec3b>(0, 0)[2]);
			}
		}
		point_num_per_color[cam_number - 1] += 1;

		avr_y /= cam_number;
		avr_u /= cam_number;
		avr_v /= cam_number;
		avr_y_2 /= cam_number;
		avr_u_2 /= cam_number;
		avr_v_2 /= cam_number;

		float dev_y = 0, dev_u = 0, dev_v = 0;


		dev_y = sqrt(avr_y_2 - avr_y * avr_y);
		dev_u = sqrt(avr_u_2 - avr_u * avr_u);
		dev_v = sqrt(avr_v_2 - avr_v * avr_v);

		float avr_dev = (dev_y + dev_u + dev_v) / 3.0;


		if (avr_dev >= 0 && avr_dev < 5) {
			dev_pointnum_percent[cam_number - 1][0] += 1;
		}
		else if (avr_dev >= 5 && avr_dev < 10) {
			dev_pointnum_percent[cam_number - 1][1] += 1;
		}
		else if (avr_dev >= 10 && avr_dev < 15) {
			dev_pointnum_percent[cam_number - 1][2] += 1;
		}
		else if (avr_dev >= 15 && avr_dev < 20) {
			dev_pointnum_percent[cam_number - 1][3] += 1;
		}
		else if (avr_dev >= 20) {
			dev_pointnum_percent[cam_number - 1][4] += 1;
		}

	}

	for (int cam = 0; cam < total_num_cameras; cam++) {
		for (int i = 0; i < 5; i++) {
			dev_pointnum_percent[cam][i] = dev_pointnum_percent[cam][i] / (float)point_num_per_color[cam] * 100;
		}
	}

	int total = 0;

	for (int cam = 0; cam < total_num_cameras; cam++) {
		//cout << cam + 1 << "��: " << point_num_per_color[cam] << "(" << point_num_per_color[cam] / PPC.size() * 100 << "%)" << endl;
		//point_num_per_color[cam] = point_num_per_color[cam] / PPC.size() * 100;
		//cout << cam + 1 << "��: " << point_num_per_color[cam] << "%" << endl;
	}

	for (int cam = 0; cam < total_num_cameras; cam++) {
		//cout << cam + 1 << "��: " << dev_pointnum_percent[cam][0] << "/" << dev_pointnum_percent[cam][1] << "/" << dev_pointnum_percent[cam][2] << "/" << dev_pointnum_percent[cam][3] << "/" << dev_pointnum_percent[cam][4] << endl;
	}
}

void YUV_dev2(vector<PPC*> PPC, vector<vector<float>>& dev_pointnum, vector<int>& point_num_per_color, vector<int>& full_color_dev)
{
	cout << "YUV_dev2 method is proceeding ..." << endl;
	float avr_y = 0, avr_u = 0, avr_v = 0;
	float avr_y_2 = 0, avr_u_2 = 0, avr_v_2 = 0;
	int cam_number = 0;
	Mat yuv(1, 1, CV_8UC3);
	float dev_y = 0, dev_u = 0, dev_v = 0;
	float avr_dev;
	
	int zero_num = 0;
	for (int point_num = 0; point_num < PPC.size(); point_num++) {
		avr_y = 0, avr_u = 0, avr_v = 0;
		avr_y_2 = 0, avr_u_2 = 0, avr_v_2 = 0;
		cam_number = 0;
		for (int cam = 0; cam < total_num_cameras; cam++) {
			

			if (!PPC[point_num]->CheckOcclusion(cam)) {
				yuv.at<Vec3b>(0, 0) = PPC[point_num]->GetColor(cam);

				cam_number++;
				avr_y += (float)yuv.at<Vec3b>(0, 0)[2];
				avr_u += (float)yuv.at<Vec3b>(0, 0)[1];
				avr_v += (float)yuv.at<Vec3b>(0, 0)[0];
				avr_y_2 += (float)yuv.at<Vec3b>(0, 0)[2] * (float)yuv.at<Vec3b>(0, 0)[2];
				avr_u_2 += (float)yuv.at<Vec3b>(0, 0)[1] * (float)yuv.at<Vec3b>(0, 0)[1];
				avr_v_2 += (float)yuv.at<Vec3b>(0, 0)[0] * (float)yuv.at<Vec3b>(0, 0)[0];
			}
		}
		if (cam_number == 0) {
			zero_num++;
			continue;
		}
		point_num_per_color[cam_number - 1] += 1;

		avr_y /= cam_number;
		avr_u /= cam_number;
		avr_v /= cam_number;
		avr_y_2 /= cam_number;
		avr_u_2 /= cam_number;
		avr_v_2 /= cam_number;


		dev_y = 0, dev_u = 0, dev_v = 0;

		dev_y = sqrt(avr_y_2 - avr_y * avr_y);
		dev_u = sqrt(avr_u_2 - avr_u * avr_u);
		dev_v = sqrt(avr_v_2 - avr_v * avr_v);

		//full_color => dev => num
		

		if (cam_number == total_num_cameras) {
			if (dev_y >= 0 && dev_y < 5) full_color_dev[0]++;
			else if (dev_y >= 5 && dev_y < 10) full_color_dev[1]++;
			else if (dev_y >= 10 && dev_y < 15) full_color_dev[2]++;
			else if (dev_y >= 15 && dev_y < 20) full_color_dev[3]++;
			else if (dev_y >= 20 && dev_y < 25) full_color_dev[4]++;
			else if (dev_y >= 25 && dev_y < 30) full_color_dev[5]++;
			else if (dev_y >= 30 && dev_y < 35) full_color_dev[6]++;
			else if (dev_y >= 35 && dev_y < 40) full_color_dev[7]++;
			else if (dev_y >= 40 && dev_y < 45) full_color_dev[8]++;
			else if (dev_y >= 45 && dev_y < 50) full_color_dev[9]++;
			else if (dev_y >= 50 && dev_y < 55) full_color_dev[10]++;
			else if (dev_y >= 55 && dev_y < 60) full_color_dev[11]++;
			else if (dev_y >= 60 && dev_y < 65) full_color_dev[12]++;
			else if (dev_y >= 65 && dev_y < 70) full_color_dev[13]++;
			else if (dev_y >= 70 && dev_y < 75) full_color_dev[14]++;
			else if (dev_y >= 75 && dev_y < 80) full_color_dev[15]++;
			else if (dev_y >= 80 && dev_y < 85) full_color_dev[16]++;
			else if (dev_y >= 85 && dev_y < 90) full_color_dev[17]++;
			else if (dev_y >= 90 && dev_y < 95) full_color_dev[18]++;
			else if (dev_y >= 95 && dev_y < 100) full_color_dev[19]++;

		}

		avr_dev = (dev_y + dev_u + dev_v) / 3.0;

	
		dev_pointnum[cam_number - 1][0] += avr_dev;
		dev_pointnum[cam_number - 1][1] += dev_y;
		dev_pointnum[cam_number - 1][2] += dev_u;
		dev_pointnum[cam_number - 1][3] += dev_v;
	}
	cout << "zero num : " << zero_num << endl;
	for (int cam = 0; cam < total_num_cameras; cam++) {
		for (int i = 0; i < 4; i++) {
			if (point_num_per_color[cam] != 0) dev_pointnum[cam][i] = dev_pointnum[cam][i] / (float)point_num_per_color[cam];
		}
	}


	cout << "YUV_dev2 method is done ..." << endl;
}

void YUV_dev3_about_MaxValue(vector<PPC*> PPC, vector<float>& point_num_per_color)
{
	int zero_num = 0;
	for (int point_num = 0; point_num < PPC.size(); point_num++) {
		float avr_y = 0, avr_u = 0, avr_v = 0;
		float avr_y_2 = 0, avr_u_2 = 0, avr_v_2 = 0;
		int cam_number = 0;
		for (int cam = 0; cam < total_num_cameras; cam++) {
			Mat yuv(1, 1, CV_8UC3);

			if (!PPC[point_num]->CheckOcclusion(cam)) {
				yuv.at<Vec3b>(0, 0) = PPC[point_num]->GetColor(cam);

				cam_number++;
				avr_y += (float)yuv.at<Vec3b>(0, 0)[2];
				avr_u += (float)yuv.at<Vec3b>(0, 0)[1];
				avr_v += (float)yuv.at<Vec3b>(0, 0)[0];
				avr_y_2 += (float)yuv.at<Vec3b>(0, 0)[2] * (float)yuv.at<Vec3b>(0, 0)[2];
				avr_u_2 += (float)yuv.at<Vec3b>(0, 0)[1] * (float)yuv.at<Vec3b>(0, 0)[1];
				avr_v_2 += (float)yuv.at<Vec3b>(0, 0)[0] * (float)yuv.at<Vec3b>(0, 0)[0];
			}
		}


		if (cam_number == 0) {
			zero_num++;
			continue;
		}

		avr_y /= cam_number;
		avr_u /= cam_number;
		avr_v /= cam_number;
		avr_y_2 /= cam_number;
		avr_u_2 /= cam_number;
		avr_v_2 /= cam_number;


		float dev_y = 0, dev_u = 0, dev_v = 0;

		dev_y = sqrt(avr_y_2 - avr_y * avr_y);
		dev_u = sqrt(avr_u_2 - avr_u * avr_u);
		dev_v = sqrt(avr_v_2 - avr_v * avr_v);

		if (cam_number == total_num_cameras - 1) {
			if (dev_y >= 0 && dev_y < 5) point_num_per_color[0]++;
			else if (dev_y >= 5 && dev_y < 10) point_num_per_color[1]++;
			else if (dev_y >= 10 && dev_y < 15) point_num_per_color[2]++;
			else if (dev_y >= 15 && dev_y < 20) point_num_per_color[3]++;
			else if (dev_y >= 20 && dev_y < 25) point_num_per_color[4]++;
			else if (dev_y >= 25 && dev_y < 30) point_num_per_color[5]++;
			else if (dev_y >= 30 && dev_y < 35) point_num_per_color[6]++;
			else if (dev_y >= 35 && dev_y < 40) point_num_per_color[7]++;
			else if (dev_y >= 40 && dev_y < 45) point_num_per_color[8]++;
			else if (dev_y >= 45 && dev_y < 50) point_num_per_color[9]++;
			else if (dev_y >= 50 && dev_y < 55) point_num_per_color[10]++;
			else if (dev_y >= 55 && dev_y < 60) point_num_per_color[11]++;
			else if (dev_y >= 60 && dev_y < 65) point_num_per_color[12]++;
			else if (dev_y >= 65 && dev_y < 70) point_num_per_color[13]++;
			else if (dev_y >= 70 && dev_y < 75) point_num_per_color[14]++;
			else if (dev_y >= 75 && dev_y < 80) point_num_per_color[15]++;
			else if (dev_y >= 80 && dev_y < 85) point_num_per_color[16]++;
			else if (dev_y >= 85 && dev_y < 90) point_num_per_color[17]++;
			else if (dev_y >= 90 && dev_y < 95) point_num_per_color[18]++;
			else if (dev_y >= 95 && dev_y < 100) point_num_per_color[19]++;

		}

	}

	cout << "zero num : " << zero_num << endl;


}

void printPSNRWithoutBlackPixel(
	vector<Mat> orig_imgs,
	vector<Mat> proj_imgs,
	vector<float>& psnrs,
	vector<int>& num_holes)
{
	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++)
	{
		float mse, psnr, tmp = 0;
		float sum = 0;
		int cnt = 0;

		cvtColor(orig_imgs[cam_num], orig_imgs[cam_num], COLOR_BGR2GRAY);
		cvtColor(proj_imgs[cam_num], proj_imgs[cam_num], COLOR_BGR2GRAY);

		int n = 0;

		for (int v = 0; v < _height; v++)
			for (int u = 0; u < _width; u++) {

				if (proj_imgs[cam_num].at<uchar>(v, u) == 0) {
					n++;
				}
				else {
					tmp = orig_imgs[cam_num].at<uchar>(v, u) - proj_imgs[cam_num].at<uchar>(v, u);
					cnt++;
					sum += tmp * tmp;
				}
			}

		mse = sum / cnt;
		psnr = 10 * log10(255 * 255 / mse);

		num_holes.push_back(n);
		psnrs.push_back(psnr);
	}

	cout << "num of holes ::::::::::::::::" << endl;
	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++) {
		cout << "cam" << cam_num << " : " << num_holes[cam_num] << endl;
	}

	cout << "PSNR without black pixel ::::::::::::::::::" << endl;
	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++) {
		cout << "cam" << cam_num << " : " << psnrs[cam_num] << endl;
	}
}

void printPSNRWithBlackPixel(
	vector<Mat> orig_imgs,
	vector<Mat> proj_imgs,
	vector<float>& psnrs)
{
	float avgPSNR = 0.0;
	float avgNumofPixel = 0.0;

	vector<float> PSNR_vec;
	vector<int> hole_num_vec;

	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++)
	{
		float mse, psnr, tmp = 0;
		float sum = 0;
		int cnt = 0;

		cvtColor(orig_imgs[cam_num], orig_imgs[cam_num], COLOR_BGR2GRAY);
		cvtColor(proj_imgs[cam_num], proj_imgs[cam_num], COLOR_BGR2GRAY);

		int n = 0;

		for (int v = 0; v < _height; v++)
			for (int u = 0; u < _width; u++) {

				if (proj_imgs[cam_num].at<uchar>(v, u) == 0) {
					n++;
				}
				tmp = orig_imgs[cam_num].at<uchar>(v, u) - proj_imgs[cam_num].at<uchar>(v, u);
				cnt++;
				sum += tmp * tmp;
			}

		mse = sum / cnt;
		psnr = 10 * log10(255 * 255 / mse);

		PSNR_vec.push_back(psnr);
		hole_num_vec.push_back(n);

		avgPSNR += psnr;
		avgNumofPixel += n;

		psnrs.push_back(psnr);
	}
	cout << "PSNR with black pixel ::::::::::::::::::" << endl;
	for (int cam_num = 0; cam_num < total_num_cameras; cam_num++) {
		cout << "cam" << cam_num << " : " << psnrs[cam_num] << endl;
	}


	avgPSNR /= total_num_cameras;
	avgNumofPixel /= total_num_cameras;
}

vector<double> operator-(vector<double> a, double b)
{
	vector<double> retvect;
	for (int i = 0; i < a.size(); i++)
	{
		retvect.push_back(a[i] - b);
	}
	return retvect;
}

vector<double> operator*(vector<double> a, vector<double> b)
{
	vector<double> retvect;
	for (int i = 0; i < a.size(); i++)
	{
		retvect.push_back(a[i] * b[i]);
	}
	return retvect;
}

Matrix4d compute_projection_matrices(int cam_num)
{
	Matrix3Xd camRT(3, 4);
	Matrix4d camP;

	// The extrinsic matrix
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			camRT(i, j) = m_CalibParams[cam_num].m_RotMatrix(i, j);

	for (int i = 0; i < 3; i++)
		camRT(i, 3) = m_CalibParams[cam_num].m_Trans(i, 0);

	// Multiply the intrinsic matrix by the extrinsic matrix to find our projection matrix
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 4; j++) {
			camP(i, j) = 0.0;

			for (int k = 0; k < 3; k++)
				camP(i, j) += m_CalibParams[cam_num].m_K(i, k) * camRT(k, j);
		}

	camP(3, 0) = 0.0;
	camP(3, 1) = 0.0;
	camP(3, 2) = 0.0;
	camP(3, 3) = 1.0;

	return camP;
}

