#include "global.h"
#include "set_environment.h"
#include "common.h"
#include "plenoptic_point_cloud.h"

int data_mode, _width, _height, total_num_cameras, total_num_frames;
double MinZ, MaxZ, scaleZ;
vector<Vector2d> tech_minmaxZ;
string path;
vector<CalibStruct> m_CalibParams;
double version;
vector<int> camera_order;
int proj_mode = 0; //0-projection , 1-backprojection
vector<PPC_v1> ppc_vec;

#ifdef ON_GPU
CUDA CudaGpu;
#endif

int main()
{
	// test
	clock_t start = clock();
	////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////
	////		data_mode 														////
	////			0 - ballet , 1 - fencing, 2 - intel, 3 - tech, 				////
	////			4, 5 - hotel(front) , 6, 7 - hotel(back)					////
	////			8, 9 - restaurant(left), 10, 11 - restaurant(back)			////
	////			12, 13 - Apartment(left)									////
	////		ppc_mode 														////
	////			0 - only incremental										////
	////			1 - incremental + voxelized									////
	////			2 - batch + voxelized										////
	////			3 - modified batch											////
	////		colorspace														////
	////			0 - YUV														////
	////			1 - HSV														////
	////			2 - RGB														////
	////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////

	int ppc_mode, view_num, voxel_div_num, colorspace;

	//fixed variable
	const int referenceView = 220;
	version = 1.0;
	ppc_mode = 3;
	colorspace = 0;

	//unfixed variable
	view_num = 3;
	int max_ppc_size = 100000000;

	
	ppc_vec.resize(max_ppc_size);
	const int furthest_index = (view_num + 1) / 2 * 22;
	
	

#ifndef TEST
	data_mode = 5;
	voxel_div_num = 8192;
	cout << " ============================= " << endl;
	cout << "          data_mode  " << data_mode << endl;
	cout << " ============================= " << endl;
	cout << "          version  " << version << endl;
	cout << " ============================= " << endl;
	cout << "          ppc_mode  " << ppc_mode << endl;
	cout << " ============================= " << endl;
	cout << "          view_num  " << view_num << endl;
	cout << " ============================= " << endl;
	cout << "      voxel_div_num  " << voxel_div_num << endl;
	cout << " ============================= " << endl;
#endif

#ifdef TEST
	vector<int> datas = { 5 };
	for (int data_i = 0; data_i < datas.size(); data_i++) {
		data_mode = datas[data_i];
#endif
		//set information of the data
		int color_bits, depth_bits;
		set_parameters(data_mode, color_bits, depth_bits, view_num);

		//set view order
		map<int, int> camera_order_LookUpTable;
		camera_order = make_camOrder(referenceView, view_num, camera_order_LookUpTable);

		//load camera parameters of each view
		load_matrix_data();

		//compute projection matrices by camera parameters
		compute_projection_matrices();

		vector<string> color_names_(total_num_cameras);
		vector<string> depth_names_(total_num_cameras);

		vector<vector<string>> color_names;
		vector<vector<string>> depth_names;

		cout << "color name resize done .. " << endl << endl;
		//load image names
		if (!data_mode) load_file_name(color_names, depth_names);
		else if (data_mode >= S01_H1) load_file_name_mode4(color_names, depth_names, referenceView, furthest_index);
		else load_file_name(color_names_, depth_names_, depth_bits);

		cout << "load_file_name done .. " << endl << endl;

		Mat blank_c, blank_d;
		Mat temp_8(_height, _width, CV_8UC3, Scalar::all(0));
		Mat temp_16(_height, _width, CV_16UC3, Scalar::all(0));

		vector<Mat> color_imgs(total_num_cameras);
		vector<Mat> depth_imgs(total_num_cameras);
		Mat depth_value_img(_height, _width, CV_64F, -1);

		int frame_num = 1;

#ifdef TEST
		vector<int> voxel_div_nums = { 1024 };
		for (int voxel_i = 0; voxel_i < voxel_div_nums.size(); voxel_i++) {
			voxel_div_num = voxel_div_nums[voxel_i];

			cout << " ============================= " << endl;
			cout << "          view_num  " << view_num << endl;
			cout << " ============================= " << endl;
			cout << "          data_mode  " << data_mode << endl;
			cout << " ============================= " << endl;
			cout << "      voxel_div_num  " << voxel_div_num << endl;
			cout << " ============================= " << endl;

			ofstream fout_data;
			ofstream fout_dev;

			frame_num = 1;
			int cnt = 0;

			string name_mode;
			if (data_mode == 0) name_mode = "ballet";
			else if (data_mode == 1) name_mode = "fencing";
			else if (data_mode == 2) name_mode = "intel";
			else if (data_mode == 3) name_mode = "tech";
			else if (data_mode == 4) name_mode = "hotel1";
			else if (data_mode == 5) name_mode = "hotel2";
			else if (data_mode == 6) name_mode = "hotel3";
			else if (data_mode == 7) name_mode = "hotel4";
			else if (data_mode == 8) name_mode = "rest1";
			else if (data_mode == 9) name_mode = "rest2";
			else if (data_mode == 10) name_mode = "rest3";
			else if (data_mode == 11) name_mode = "rest4";
			else if (data_mode == 12) name_mode = "apart1";
			else if (data_mode == 13) name_mode = "apart2";

			string name_ppc;
			if (ppc_mode == 0) name_ppc = "incre";
			else if (ppc_mode == 1) name_ppc = "increNvoxel";
			else if (ppc_mode == 2) name_ppc = "batch";
			else if (ppc_mode == 3) name_ppc = "modifiedbatch";

			string name_colorspace;
			if (colorspace == 0) name_colorspace = "YUV";
			else if (colorspace == 1) name_colorspace = "HSV";
			else if (colorspace == 2) name_colorspace = "BGR";

			string version_ = to_string(version).substr(0, 3);
			string name_data = "output\\" + version_ + "_" + name_mode + "_" + name_ppc + "_" + name_colorspace + "_" + to_string(voxel_div_num) + "_data.csv";
			string name_dev = "output\\" + version_ + "_" + name_mode + "_" + name_ppc + "_" + name_colorspace + "_" + to_string(voxel_div_num) + "_dev.csv";

			fout_data.open(name_data);
			fout_dev.open(name_dev);

			if (ppc_mode == 1) fout_data << "frame,#PC,depth_threhsold,increPPC,increNvoxPPC,degOfDecreasedPoint,Cube_x_size,Cube_y_size,Cube_z_size,cube_x_size,cube_y_size,cube_z_size,";
			else if (ppc_mode == 3) fout_data << "frame,#PC,formulaicPPC,degOfDecreasedPoint,Cube_x_size,Cube_y_size,Cube_z_size,cube_x_size,cube_y_size,cube_z_size,";
			fout_data << "\n";

			fout_dev << "frame,";
			fout_dev << "\n";

#endif
			PROCESS_MEMORY_COUNTERS_EX g_mc, pmc;
			for (int frame = 0; frame < frame_num; frame++)
			{
#ifdef TEST
				fout_data << frame << ",";
				fout_dev << frame << "\n";
#endif
				GetProcessMemoryInfo(GetCurrentProcess(),
					(PROCESS_MEMORY_COUNTERS*)&g_mc, sizeof(g_mc));

				//get color images and depth images to make ppc
				if (!data_mode || data_mode >= S01_H1) get_color_and_depth_imgs(frame, camera_order, color_names, depth_names, color_imgs, depth_imgs);
				else get_color_and_depth_imgs(frame, color_names_, depth_names_, color_imgs, depth_imgs, color_bits, depth_bits);

				GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
				cout << "get_color_and_depth_imgs Memory Usage : " << (pmc.PrivateUsage - g_mc.PrivateUsage) / (1024 * 1024) << " MB" << endl;
				cout << "get_color_and_depth_imgs done... " << endl << endl;

#ifdef TEST
				fout_data << _width * _height * total_num_cameras << ",";
				int increPPC_size = 0;
#endif

				vector<PPC*> Plen_PC;
				int ppc_number = 0;

				Mat is_hole_temp(_height, _width, CV_8U, Scalar::all(1));
				Mat depth_value_img(_height, _width, CV_64F, -1);

				vector<Mat> projection_imgs(total_num_cameras);
				vector<Mat> filled_imgs(total_num_cameras);
				vector<Mat> is_hole_proj_imgs(total_num_cameras);
				vector<Mat> is_hole_filled_imgs(total_num_cameras);
				vector<Mat> depth_value_imgs(total_num_cameras);

				for (int cam = 0; cam < total_num_cameras; cam++) {
					projection_imgs[cam] = temp_8.clone();
					filled_imgs[cam] = temp_8.clone();
					is_hole_proj_imgs[cam] = is_hole_temp.clone();
					is_hole_filled_imgs[cam] = is_hole_temp.clone();
					depth_value_imgs[cam] = depth_value_img.clone();
				}

				vector<float> psnrs_p, psnrs_h;
				vector<float> psnrs_p_1, psnrs_p_2, psnrs_p_3;
				vector<float> psnrs_h_1, psnrs_h_2, psnrs_h_3;
				vector<int> num_holes_p, num_holes_h;

				vector<float> min(3);
				vector<float> max(3);
				set<unsigned long long>::iterator voxel_iter_in_progress;
				vector<float> Cube_size, cube_size;

				//find min_max of 3D space
				vector<PointCloud<PointXYZRGB>::Ptr> pointclouds;
				set<unsigned long long> valid_cube_indices;

				clock_t t1 = clock();
				find_min_max_3D_space(pointclouds, color_imgs, depth_imgs, min, max);
				clock_t t2 = clock();
				cout << "find_min_max_3D_space time : " << (t2 - t1) / CLOCKS_PER_SEC << endl << endl;

				//find valid voxel data
				clock_t t3 = clock();
				find_valid_voxels(pointclouds, min, max, voxel_div_num, Cube_size, cube_size, valid_cube_indices);
				clock_t t4 = clock();
				cout << "find_valid_voxels time : " << (t4 - t3) / CLOCKS_PER_SEC << endl << endl;

				pointclouds.clear();
				vector<PointCloud<PointXYZRGB>::Ptr>().swap(pointclouds);

				int iteration = 0;
				bool end_ppc_generation = false;
				int cur_ppc_size = 0;
				int total_ppc_size = 0;

#ifdef TEST
				vector<vector<float>> dev_pointnum(total_num_cameras, vector<float>(4, 0));
				vector<int> point_num_per_color(total_num_cameras, 0);
				vector<int> full_color_dev(20, 0);
				float making_ppc_all_time = (t4 - t1) / CLOCKS_PER_SEC;
				float projection_time_per_view = 0.;
#endif
				clock_t t5 = clock();
				while (end_ppc_generation == false) {

					clock_t t13 = clock();
					make_PPC_modified_batch(iteration, max_ppc_size, min, voxel_div_num, color_imgs, depth_imgs, Cube_size, cube_size, valid_cube_indices, end_ppc_generation, cur_ppc_size);
					clock_t t14 = clock();
					cout << "make_PPC_modified_batch time : " << (t14 - t13) / CLOCKS_PER_SEC << endl << endl;

					total_ppc_size += cur_ppc_size;
					iteration++;

					//perform_projection
					clock_t t7, t8;
#ifdef CUDA_TEST
					t7 = clock();
					perform_projection(cur_ppc_size, projection_imgs, is_hole_proj_imgs, depth_value_imgs);
					t8 = clock();
					cout << "projection whole views time : " << (double)(t8 - t7) / CLOCKS_PER_SEC << endl;
					cout << "---------------------------------" << endl;
#else
					for (int cam = 0; cam < total_num_cameras; cam++) {
						cout << cam << "th pointcloud is being projected ..." << endl;
						int nNeighbor = 4;
						int window_size = 2;

						//execute projection of ppc to each view
						t7 = clock();
						perform_projection(cam, cur_ppc_size, projection_imgs[cam], is_hole_proj_imgs[cam], depth_value_imgs[cam]);
						is_hole_filled_imgs[cam] = is_hole_proj_imgs[cam].clone();
						// holefilling_per_viewpoint(projection_imgs[cam], filled_imgs[cam], is_hole_filled_imgs[cam], window_size);
						t8 = clock();

						cout << "projection and hole filling one view time : " << (double)(t8 - t7) / CLOCKS_PER_SEC << endl;
						cout << "---------------------------------" << endl;
					}

#endif
#ifdef TEST
					making_ppc_all_time += (t14 - t13) / CLOCKS_PER_SEC;
					projection_time_per_view += (t8 - t7) / CLOCKS_PER_SEC;

					//YUVdev 계산
					clock_t t11 = clock();
					calc_YUV_stddev_global(cur_ppc_size, dev_pointnum, point_num_per_color, full_color_dev);
					clock_t t12 = clock();
					cout << "calc_YUV_stddev_global time : " << (t12 - t11) / CLOCKS_PER_SEC << endl;
#endif
				}
				clock_t t6 = clock();
				cout << "make ppc and projection final time : " << (t6 - t1) / CLOCKS_PER_SEC << endl << endl;

				printPSNRWithoutBlackPixel_RGB(color_imgs, projection_imgs, is_hole_proj_imgs, psnrs_p_1, psnrs_p_2, psnrs_p_3, num_holes_p);
				printPSNRWithBlackPixel_RGB(color_imgs, filled_imgs, is_hole_filled_imgs, psnrs_h_1, psnrs_h_2, psnrs_h_3, num_holes_h);

#ifdef TEST			
				//save images
				string folder_name_string = "output\\image\\" + name_mode;
				const char* foler_name = folder_name_string.c_str();
				CreateDirectory("output\\image", NULL);
				CreateDirectory(foler_name, NULL);

				clock_t t9 = clock();
				for (int cam = 0; cam < total_num_cameras; cam++) {
					Mat proj_viewImg, filled_viewImg;

					cvtColor(projection_imgs[cam], proj_viewImg, CV_YUV2BGR);
					imwrite("output\\image\\" + name_mode + "\\" + version_ + "_" + name_mode + "_" + name_ppc + "_" + to_string(voxel_div_num) + "_projmode" + to_string(proj_mode) + "_view" + to_string(camera_order_LookUpTable.find(camera_order[cam])->second) + "_proj.png", proj_viewImg);

					cvtColor(filled_imgs[cam], filled_viewImg, CV_YUV2BGR);
					imwrite("output\\image\\" + name_mode + "\\" + version_ + "_" + name_mode + "_" + name_ppc + "_" + to_string(voxel_div_num) + "_projmode" + to_string(proj_mode) + "_view" + to_string(camera_order_LookUpTable.find(camera_order[cam])->second) + "_filled.png", filled_viewImg);

					proj_viewImg.release();
					filled_viewImg.release();
				}
				clock_t t10 = clock();
				cout << "save image time: " << float(t10 - t9) / CLOCKS_PER_SEC << endl << endl;
				for (int cam = 0; cam < total_num_cameras; cam++) {
					for (int i = 0; i < 4; i++) {
						if (point_num_per_color[cam] != 0) dev_pointnum[cam][i] = dev_pointnum[cam][i] / (float)point_num_per_color[cam];
					}
				}

				fout_dev << "pointNum" << "\n";
				for (int i = 0; i < total_num_cameras; i++)
					fout_dev << "color" << i + 1 << "," << point_num_per_color[i] << "\n";
				fout_dev << "\n";

				fout_dev << "Y dev" << "\n";
				for (int i = 0; i < total_num_cameras; i++)
					fout_dev << "color" << i + 1 << "," << dev_pointnum[i][1] << "\n";
				fout_dev << "\n";

				fout_dev << "U dev" << "\n";
				for (int i = 0; i < total_num_cameras; i++)
					fout_dev << "color" << i + 1 << "," << dev_pointnum[i][2] << "\n";
				fout_dev << "\n";

				fout_dev << "V dev" << "\n";
				for (int i = 0; i < total_num_cameras; i++)
					fout_dev << "color" << i + 1 << "," << dev_pointnum[i][3] << "\n";
				fout_dev << "\n";

				fout_dev << "# point per color dev of full color " << "\n";
				for (int i = 0; i < full_color_dev.size(); i++)
					fout_dev << i * 5 << "-" << (i + 1) * 5 << "," << full_color_dev[i] << "\n";
				fout_dev << "\n";

				dev_pointnum.clear();
				point_num_per_color.clear();
				full_color_dev.clear();

				vector<vector<float>>().swap(dev_pointnum);
				vector<int>().swap(point_num_per_color);
				vector<int>().swap(full_color_dev);

				fout_data << total_ppc_size << "," << 100 - ((float)total_ppc_size / (_width * _height * total_num_cameras) * 100) << "%," <<
					Cube_size[0] << "," << Cube_size[1] << "," << Cube_size[2] << "," <<
					cube_size[0] << "," << cube_size[1] << "," << cube_size[2] << ",";

				fout_data << "\n\n";
				if (proj_mode == 0) fout_data << "projection" << "\n";
				else if (proj_mode == 1) fout_data << "back_projection" << "\n";
				fout_data << "making ppc time,projection time per view\n";
				fout_data << making_ppc_all_time << "," << projection_time_per_view << "\n\n";

				map<int, int> num_holes_p_map, num_holes_h_map;
				map<int, float> psnrs_p_1_map, psnrs_p_2_map, psnrs_p_3_map, psnrs_h_1_map, psnrs_h_2_map, psnrs_h_3_map;

				for (int i = 0; i < total_num_cameras; i++) {
					num_holes_p_map.insert(make_pair(camera_order[i], num_holes_p[i]));
					num_holes_h_map.insert(make_pair(camera_order[i], num_holes_h[i]));
					psnrs_p_1_map.insert(make_pair(camera_order[i], psnrs_p_1[i]));
					psnrs_p_2_map.insert(make_pair(camera_order[i], psnrs_p_2[i]));
					psnrs_p_3_map.insert(make_pair(camera_order[i], psnrs_p_3[i]));
					psnrs_h_1_map.insert(make_pair(camera_order[i], psnrs_h_1[i]));
					psnrs_h_2_map.insert(make_pair(camera_order[i], psnrs_h_2[i]));
					psnrs_h_3_map.insert(make_pair(camera_order[i], psnrs_h_3[i]));
				}

				Mat proj_viewImg, filled_viewImg;
				int count_it = 0;

				fout_data << "hole_num\n";
				count_it = 0;
				for (map<int, int>::iterator it = num_holes_p_map.begin(); it != num_holes_p_map.end(); it++) {
					fout_data << "cam" << count_it++ << "," << it->second << "\n";
				}
				count_it = 0;
				fout_data << "\nPSNR_without_hole_R\n";
				for (map<int, float>::iterator it = psnrs_p_1_map.begin(); it != psnrs_p_1_map.end(); it++) {
					fout_data << "cam" << count_it++ << "," << it->second << "\n";
				}
				count_it = 0;
				fout_data << "\nPSNR_without_hole_G\n";
				for (map<int, float>::iterator it = psnrs_p_2_map.begin(); it != psnrs_p_2_map.end(); it++) {
					fout_data << "cam" << count_it++ << "," << it->second << "\n";
				}
				count_it = 0;
				fout_data << "\nPSNR_without_hole_B\n";
				for (map<int, float>::iterator it = psnrs_p_3_map.begin(); it != psnrs_p_3_map.end(); it++) {
					fout_data << "cam" << count_it++ << "," << it->second << "\n";
				}
				count_it = 0;
				fout_data << "\n";

				fout_data << "\nhole_num_after_holefilling\n";
				for (map<int, int>::iterator it = num_holes_h_map.begin(); it != num_holes_h_map.end(); it++) {
					fout_data << "cam" << count_it++ << "," << it->second << "\n";
				}
				count_it = 0;
				fout_data << "\nPSNR_with_hole_R\n";
				for (map<int, float>::iterator it = psnrs_h_1_map.begin(); it != psnrs_h_1_map.end(); it++) {
					fout_data << "cam" << count_it++ << "," << it->second << "\n";
				}
				count_it = 0;
				fout_data << "\nPSNR_with_hole_G\n";
				for (map<int, float>::iterator it = psnrs_h_2_map.begin(); it != psnrs_h_2_map.end(); it++) {
					fout_data << "cam" << count_it++ << "," << it->second << "\n";
				}
				count_it = 0;
				fout_data << "\nPSNR_with_hole_B\n";
				for (map<int, float>::iterator it = psnrs_h_3_map.begin(); it != psnrs_h_3_map.end(); it++) {
					fout_data << "cam" << count_it++ << "," << it->second << "\n";
				}
				count_it = 0;
				fout_data << "\n\n";
				fout_data.close();
				fout_dev.close();

				num_holes_p_map.clear();
				num_holes_h_map.clear();
				psnrs_p_1_map.clear();
				psnrs_p_2_map.clear();
				psnrs_p_3_map.clear();
				psnrs_h_1_map.clear();
				psnrs_h_2_map.clear();
				psnrs_h_3_map.clear();

				map<int, int>().swap(num_holes_p_map);
				map<int, int>().swap(num_holes_h_map);
				map<int, float>().swap(psnrs_p_1_map);
				map<int, float>().swap(psnrs_p_2_map);
				map<int, float>().swap(psnrs_p_3_map);
				map<int, float>().swap(psnrs_h_1_map);
				map<int, float>().swap(psnrs_h_2_map);
				map<int, float>().swap(psnrs_h_3_map);
			}
		}
#endif
	}

	clock_t end = clock();

	double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
	cout << "total time: " << elapsed << endl;
	return 0;
}
