#include <iostream>
//#include <ios>
#include <iomanip>
#include <chrono>
//#include <vector>
#include <opencv2\core.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <thread>
#include <future>

#include <fstream>
#include <list>
#include <vector>
//#include <deque>



#include <filesystem>

//#include "Funcs.h"
#include "Resizer.h"

using namespace std;
using namespace std::experimental::filesystem;


int main()
{
	
	int width = 240;//600;
	int height = 320;//480;
	std::chrono::time_point<std::chrono::system_clock> start, end;
	int64 microseconds;

	

	Resizer resizer;
	thread thrd(std::ref(resizer));
	start = chrono::system_clock::now();
	list<path> files_in;
	for (auto& f : directory_iterator("Images_in"))
	{
		files_in.push_back(f.path());
		string filename_out = "Images_out2\\" + f.path().stem().string() + "_thumb2" + f.path().extension().string();
		resizer.add_file_to_resize(f.path().string(), filename_out, cv::Size(width, height));
	}
	resizer.setExit();
	thrd.join();
	end = chrono::system_clock::now();
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	cout << "Resized by Resizer (use threads for encode, resize and decode) " << setw(9) << microseconds << endl;

	//cv::Mat image_in;
	//cv::Size size_in;
	
	//--------------------------------------------------------------------------------------------
	//vvvvvv------------------------------------------------------------------------------vvvvvvv
	vector<char> buffer_in;
	vector<uchar> buffer_out;
	cv::Mat image_in;
	cv::Size size_out(width, height);
	int type = image_in.type();
	cv::Mat image_out(size_out, type);

	/*
	start = chrono::system_clock::now();
	for (auto f : files_in)
	{
		buffer_in = read_file(f.string());
		image_in = buffer_decode_to_mat(buffer_in);
		resize(image_in, image_out);
		buffer_out = mat_encode_to_buffer(image_out);
		string filename_out = "Images_out\\" + f.stem().string() + "_thumb" + f.extension().string();
		write_file(filename_out, buffer_out);

		//execut.add_task(read_file);
	}
	end = chrono::system_clock::now();
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	cout << "Resized for " << setw(9) << microseconds << endl;
	//^^^^^^-------------------------------------------------------------------------------^^^^^^^
	//--------------------------------------------------------------------------------------------
	*/


	
	//--------------------------------------------------------------------------------------------
	//vvvvvv------------------------------------------------------------------------------vvvvvvv
	std::chrono::time_point<std::chrono::system_clock> start2, end2;
	int64 read, dec, res, enc, write;


	list<vector<char>> buffers_in;
	list<cv::Mat> images_in;
	list<cv::Mat> images_out;
	list<vector<uchar>> buffers_out;
	start = chrono::system_clock::now();

	start2 = chrono::system_clock::now();
	for (auto f : files_in)
		buffers_in.push_back( read_file(f.string()) );
	end2 = chrono::system_clock::now();
	read = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();

	start2 = chrono::system_clock::now();
	for (auto buf : buffers_in)
		images_in.push_back( buffer_decode_to_mat(buf).clone() );
	end2 = chrono::system_clock::now();
	dec = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();

	start2 = chrono::system_clock::now();
	for (auto m : images_in)
	{
		resize(m, image_out);
		images_out.push_back(image_out.clone());
	}
	end2 = chrono::system_clock::now();
	res = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();

	start2 = chrono::system_clock::now();
	for (auto m : images_out)
		buffers_out.push_back(mat_encode_to_buffer(m));
	end2 = chrono::system_clock::now();
	enc = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();

	start2 = chrono::system_clock::now();
	auto it = files_in.begin();
	for (auto buf : buffers_out)
	{
		string filename_out = "Images_out\\" + (*it).stem().string() + "_thumb" + (*it).extension().string();
		write_file(filename_out, buf);
		it++;
	}
	end2 = chrono::system_clock::now();
	write = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
	cout << "read   == " << setw(9) << read << endl;
	cout << "decode == " << setw(9) << dec << endl;
	cout << "resize == " << setw(9) << res << endl;
	cout << "encode == " << setw(9) << enc << endl;
	cout << "write  == " << setw(9) << write << endl;
	end = chrono::system_clock::now();
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	cout << "Resized for " << setw(9) << microseconds << endl;

	//^^^^^^-------------------------------------------------------------------------------^^^^^^^
	//--------------------------------------------------------------------------------------------
	

	//cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
	//imshow("Display window", image_out); // Show our image inside it.
	//image.
	//cv::waitKey(0); // Wait for a keystroke in the window

	getchar();
	return 0;
}