#pragma once

#include <opencv2\core.hpp>
//#include <opencv2\imgcodecs.hpp>
//#include <opencv2\highgui.hpp>
//#include <opencv2\imgproc.hpp>

//#include <iostream>
//#include <fstream>
//#include <future>

#include <string>
#include <vector>

using namespace std;
//using namespace std::experimental::filesystem;

void resize(cv::Mat& mat_in, cv::Mat& mat_out);
cv::Mat resize2(cv::Mat mat_in, cv::Size size_out);
vector<char> read_file(string& file_name);
void write_file(string file_name, vector<uchar>& buffer);
cv::Mat buffer_decode_to_mat(std::vector<char>& buffer);
std::vector<uchar> mat_encode_to_buffer(cv::Mat image);

