#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <opencv2/opencv.hpp>
#include <vector>

#define DEBUG_GPU

using namespace cv;

extern "C"
{

	/////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////
	////////////////////// Device Code //////////////////////
	/////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////
	__global__ void makc_PC_GPU(
		cuda::PtrStepSz<uchar3> color_src,
		cuda::PtrStepSz<ushort> depth_src,
		double scaleZ,
		double* K,
		double* R_wc_inv,
		double* t_wc,
		double* dev_x,
		double* dev_y,
		double* dev_z,
		uchar* dev_b,
		uchar* dev_g,
		uchar* dev_r);

	__global__ void perform_projection_GPU(
		int ppc_size,
		int cam_num,
		cuda::PtrStepSz<uchar3> proj_img,
		cuda::PtrStepSz<uchar> is_hole_proj_img,
		cuda::PtrStepSz<double> depth_value_img,
		double* dev_ProjMatrix,
		float* dev_x,
		float* dev_geo_y,
		float* dev_z,
		uchar* dev_color_y,
		uchar* dev_u,
		uchar* dev_v,
		bool* dev_occlusion
		);

	__device__ double depth_level_2_Z_s_direct(ushort d, double scaleZ);

	__device__ double3 MVG(
		double* K,
		double* R_wc_inv,
		double* t_wc,
		int x,
		int y,
		double Z);

	__device__ double find_point_dist(double w, double* projMatrix);

	__device__ double determinant(double mat[3][3]);
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	////////////////////// Host Code //////////////////////
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	class CUDA
	{
	public:
		CUDA(void);
		virtual ~CUDA(void);

		void make_PC(
			Mat color_img,
			Mat depth_img,
			int data_mode,
			double scaleZ,
			double* hst_K,
			double* hst_R_wc_inv,
			double* hst_t_wc,
			double* hst_x,
			double* hst_y,
			double* hst_z,
			uchar* hst_b,
			uchar* hst_g,
			uchar* hst_r);

		void perform_projection(
			Mat sample_mat,
			uchar** proj_data,
			uchar** is_hole_proj_data,
			double** depth_value_data,
			int total_num_cameras,
			double* hst_ProjMatrix,
			int valid_ppc_size,
			float* hst_x,
			float* hst_geo_y,
			float* hst_z,
			uchar* hst_color_y,
			uchar* hst_u,
			uchar* hst_v,
			bool* hst_occlusion);
	};
}
