#include "Funcs.h"

#include <opencv2\imgcodecs.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

#include <fstream>
#include <future>

void resize(cv::Mat& mat_in, cv::Mat& mat_out)
{
	cv::resize(mat_in, mat_out, mat_out.size(), 0, 0, cv::INTER_LINEAR);
}

cv::Mat resize2(cv::Mat mat_in, cv::Size size_out)
{
	cv::Mat mat_out;
	cv::resize(mat_in, mat_out, size_out, 0, 0, cv::INTER_LINEAR);
	return mat_out;
}

vector<char> read_file(string& file_name)
{
	ifstream file(file_name, std::ios::binary | std::ios::ate);
	streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);
	vector<char> buffer(size);
	file.read(buffer.data(), size);
	file.close();
	return buffer;
}

void write_file(string file_name, vector<uchar>& buffer)
{
	std::ofstream file(file_name, std::ios::binary | std::ios::ate);
	file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
	file.close();
}

cv::Mat buffer_decode_to_mat(std::vector<char>& buffer)
{
	return cv::imdecode(buffer, cv::IMREAD_COLOR);
}

std::vector<uchar> mat_encode_to_buffer(cv::Mat image)
{
	std::vector<uchar> buffer;
	cv::imencode(".jpg", image, buffer);
	return buffer;
}